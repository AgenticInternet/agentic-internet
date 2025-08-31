"""Code execution tool for agents."""

import sys
import io
import ast
import traceback
from typing import Any, Dict
from contextlib import redirect_stdout, redirect_stderr
from smolagents import Tool
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import re
import math

class PythonExecutorTool(Tool):
    """
    Tool for executing Python code in a sandboxed environment.
    """
    name = "python_executor"
    description = """
    Execute Python code and return the output.
    The code should be valid Python that can be executed.
    Available libraries: numpy, pandas, requests, datetime, json, re, math.
    Returns the output of the code execution or any errors.
    """
    inputs = {
        "code": {
            "type": "string",
            "description": "The Python code to execute"
        }
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        # Create a restricted namespace for code execution
        self.safe_namespace = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'all': all,
                'any': any,
                'ord': ord,
                'chr': chr,
                'divmod': divmod,
                'pow': pow,
            },
            'np': np,
            'pd': pd,
            'requests': requests,
            'datetime': datetime,
            'timedelta': timedelta,
            'json': json,
            're': re,
            'math': math,
        }
    
    def _validate_code(self, code: str) -> bool:
        """Basic validation to prevent dangerous operations."""
        # List of dangerous operations to block
        dangerous_patterns = [
            r'__import__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'__.*__',  # Dunder methods
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'subprocess',
            r'os\.',
            r'sys\.',
            r'importlib',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        # Try to parse the code to check for syntax errors
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def forward(self, code: str) -> str:
        """Execute Python code and return the output."""
        # Validate code first
        if not self._validate_code(code):
            return "Error: Code contains potentially dangerous operations or has syntax errors."
        
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        # Create a fresh namespace for this execution
        namespace = self.safe_namespace.copy()
        
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Execute the code in the restricted namespace
                exec(code, namespace)
            
            # Get the output
            stdout_value = stdout.getvalue()
            stderr_value = stderr.getvalue()
            
            # Check if there's a result variable we should return
            if 'result' in namespace and namespace['result'] is not None:
                result = str(namespace['result'])
                if stdout_value:
                    return f"Output:\\n{stdout_value}\\n\\nResult:\\n{result}"
                return f"Result:\\n{result}"
            
            # Return stdout if available
            if stdout_value:
                return f"Output:\\n{stdout_value}"
            
            # Return stderr if there were warnings
            if stderr_value:
                return f"Warnings:\\n{stderr_value}"
            
            return "Code executed successfully with no output."
            
        except Exception as e:
            # Return the error traceback
            error_msg = traceback.format_exc()
            return f"Error executing code:\\n{error_msg}"


class DataAnalysisTool(Tool):
    """
    Tool for performing data analysis operations.
    """
    name = "data_analysis"
    description = """
    Perform data analysis on CSV or JSON data.
    Input should be a dictionary with 'data' (string of CSV/JSON) and 'operation' (what to analyze).
    Returns analysis results including statistics, summaries, or visualizations.
    """
    inputs = {
        "data": {
            "type": "string",
            "description": "The data in CSV or JSON format"
        },
        "operation": {
            "type": "string",
            "description": "The analysis operation to perform (e.g., 'describe', 'correlations', 'summary')"
        }
    }
    output_type = "string"
    
    def forward(self, data: str, operation: str) -> str:
        """Perform data analysis."""
        try:
            # Try to parse as JSON first
            try:
                parsed_data = json.loads(data)
                df = pd.DataFrame(parsed_data)
            except json.JSONDecodeError:
                # Try as CSV
                from io import StringIO
                df = pd.read_csv(StringIO(data))
            
            # Perform the requested operation
            if operation.lower() == "describe":
                result = df.describe().to_string()
            elif operation.lower() == "info":
                buffer = io.StringIO()
                df.info(buf=buffer)
                result = buffer.getvalue()
            elif operation.lower() == "correlations":
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    result = numeric_df.corr().to_string()
                else:
                    result = "No numeric columns found for correlation analysis."
            elif operation.lower() == "summary":
                result = f"Shape: {df.shape}\\n"
                result += f"Columns: {list(df.columns)}\\n"
                result += f"Data types:\\n{df.dtypes.to_string()}\\n"
                result += f"\\nFirst 5 rows:\\n{df.head().to_string()}"
            elif operation.lower() == "missing":
                missing = df.isnull().sum()
                result = f"Missing values per column:\\n{missing.to_string()}"
            else:
                result = f"Unknown operation: {operation}. Available operations: describe, info, correlations, summary, missing"
            
            return result
            
        except Exception as e:
            return f"Error performing data analysis: {str(e)}"
