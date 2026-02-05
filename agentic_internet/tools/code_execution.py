"""Code execution tool for agents."""

import ast
import io
import json
import logging
import math
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from smolagents import Tool

from ..exceptions import CodeExecutionError, UnsafeCodeError

logger = logging.getLogger(__name__)

ALLOWED_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "reversed": reversed,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "all": all,
    "any": any,
    "ord": ord,
    "chr": chr,
    "divmod": divmod,
    "pow": pow,
    "format": format,
    "repr": repr,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "frozenset": frozenset,
    "bytearray": bytearray,
    "bytes": bytes,
    "complex": complex,
    "memoryview": memoryview,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "super": super,
    "object": object,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "StopIteration": StopIteration,
    "RuntimeError": RuntimeError,
    "ZeroDivisionError": ZeroDivisionError,
    "True": True,
    "False": False,
    "None": None,
}

ALLOWED_MODULES = {
    "np": np,
    "pd": pd,
    "requests": requests,
    "datetime": datetime,
    "timedelta": timedelta,
    "json": json,
    "re": re,
    "math": math,
}


class _ASTSafetyValidator(ast.NodeVisitor):
    """AST-based validator that walks the tree to detect unsafe operations."""

    BLOCKED_ATTRIBUTES = frozenset({
        "__subclasses__", "__bases__", "__mro__", "__class__",
        "__globals__", "__code__", "__func__", "__self__",
        "__dict__", "__init_subclass__", "__set_name__",
        "__del__", "__delattr__", "__reduce__", "__reduce_ex__",
        "__getattribute__", "__setattr__",
    })

    BLOCKED_NAMES = frozenset({
        "__import__", "exec", "eval", "compile",
        "open", "input", "raw_input",
        "globals", "locals", "vars", "dir",
        "breakpoint", "exit", "quit",
        "getattr", "setattr", "delattr",  # blocked at AST level; safe versions in namespace
    })

    BLOCKED_MODULES = frozenset({
        "os", "sys", "subprocess", "importlib", "shutil",
        "pathlib", "socket", "ctypes", "signal", "threading",
        "multiprocessing", "webbrowser", "code", "codeop",
        "pickle", "shelve", "marshal", "tempfile", "glob",
        "fnmatch", "io", "builtins", "__builtin__",
    })

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_root = alias.name.split(".")[0]
            if module_root in self.BLOCKED_MODULES:
                self.violations.append(f"import of blocked module '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module_root = node.module.split(".")[0]
            if module_root in self.BLOCKED_MODULES:
                self.violations.append(f"import from blocked module '{node.module}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.BLOCKED_ATTRIBUTES:
            self.violations.append(f"access to blocked attribute '{node.attr}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.BLOCKED_NAMES:
            self.violations.append(f"call to blocked function '{node.func.id}'")
        self.generic_visit(node)

    def validate(self, tree: ast.AST) -> list[str]:
        self.violations = []
        self.visit(tree)
        return self.violations


class PythonExecutorTool(Tool):
    """
    Tool for executing Python code with AST-level safety validation.

    WARNING: This is NOT a full sandbox. It provides defense-in-depth via
    AST analysis + restricted builtins, but determined attackers may escape.
    For production use, run code in a Docker container or similar isolation.
    """

    name = "python_executor"
    description = """
    Execute Python code and return the output.
    The code should be valid Python that can be executed.
    Available libraries: numpy (np), pandas (pd), requests, datetime, json, re, math.
    Returns the output of the code execution or any errors.
    """
    inputs = {
        "code": {
            "type": "string",
            "description": "The Python code to execute",
        }
    }
    output_type = "string"

    MAX_OUTPUT_LENGTH = 10_000
    EXECUTION_TIMEOUT_HINT = 30  # seconds (advisory, not enforced here)

    def __init__(self) -> None:
        super().__init__()
        self._validator = _ASTSafetyValidator()
        self._base_namespace: dict[str, Any] = {
            "__builtins__": ALLOWED_BUILTINS,
            **ALLOWED_MODULES,
        }

    def _validate_code(self, code: str) -> ast.Module:
        """Parse and validate code via AST. Returns the parsed tree or raises."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise CodeExecutionError(f"Syntax error: {e}") from e

        violations = self._validator.validate(tree)
        if violations:
            raise UnsafeCodeError("; ".join(violations))

        return tree

    def forward(self, code: str) -> str:
        """Execute Python code and return the output."""
        try:
            tree = self._validate_code(code)
        except UnsafeCodeError as e:
            logger.warning("Blocked unsafe code: %s", e)
            return f"Error: {e}"
        except CodeExecutionError as e:
            return f"Error: {e}"

        stdout = io.StringIO()
        stderr = io.StringIO()
        namespace = self._base_namespace.copy()

        try:
            compiled = compile(tree, "<agent_code>", "exec")
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(compiled, namespace)

            stdout_value = stdout.getvalue()
            stderr_value = stderr.getvalue()

            if "result" in namespace and namespace["result"] is not None:
                result_str = str(namespace["result"])
                if stdout_value:
                    output = f"Output:\n{stdout_value}\n\nResult:\n{result_str}"
                else:
                    output = f"Result:\n{result_str}"
            elif stdout_value:
                output = f"Output:\n{stdout_value}"
            elif stderr_value:
                output = f"Warnings:\n{stderr_value}"
            else:
                output = "Code executed successfully with no output."

            if len(output) > self.MAX_OUTPUT_LENGTH:
                output = output[: self.MAX_OUTPUT_LENGTH] + "\n... [truncated]"

            return output

        except Exception:
            error_msg = traceback.format_exc()
            logger.debug("Code execution error: %s", error_msg)
            return f"Error executing code:\n{error_msg}"


VALID_OPERATIONS = frozenset({"describe", "info", "correlations", "summary", "missing"})


class DataAnalysisTool(Tool):
    """Tool for performing data analysis operations on CSV or JSON data."""

    name = "data_analysis"
    description = """
    Perform data analysis on CSV or JSON data.
    Input should be a dictionary with 'data' (string of CSV/JSON) and 'operation' (what to analyze).
    Available operations: describe, info, correlations, summary, missing.
    Returns analysis results including statistics and summaries.
    """
    inputs = {
        "data": {
            "type": "string",
            "description": "The data in CSV or JSON format",
        },
        "operation": {
            "type": "string",
            "description": "The analysis operation to perform (describe, info, correlations, summary, missing)",
        },
    }
    output_type = "string"

    def _parse_data(self, data: str) -> pd.DataFrame:
        """Parse input data string into a DataFrame."""
        try:
            parsed_data = json.loads(data)
            return pd.DataFrame(parsed_data)
        except (json.JSONDecodeError, ValueError):
            return pd.read_csv(io.StringIO(data))

    def forward(self, data: str, operation: str) -> str:
        """Perform data analysis."""
        op = operation.strip().lower()
        if op not in VALID_OPERATIONS:
            return f"Unknown operation: {operation}. Available: {', '.join(sorted(VALID_OPERATIONS))}"

        try:
            df = self._parse_data(data)
        except Exception as e:
            return f"Error parsing data: {e}"

        try:
            if op == "describe":
                return df.describe().to_string()
            elif op == "info":
                buffer = io.StringIO()
                df.info(buf=buffer)
                return buffer.getvalue()
            elif op == "correlations":
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    return "No numeric columns found for correlation analysis."
                return numeric_df.corr().to_string()
            elif op == "summary":
                parts = [
                    f"Shape: {df.shape}",
                    f"Columns: {list(df.columns)}",
                    f"Data types:\n{df.dtypes.to_string()}",
                    f"\nFirst 5 rows:\n{df.head().to_string()}",
                ]
                return "\n".join(parts)
            elif op == "missing":
                missing = df.isnull().sum()
                return f"Missing values per column:\n{missing.to_string()}"
        except Exception as e:
            return f"Error performing data analysis: {e}"

        return "Operation completed with no output."
