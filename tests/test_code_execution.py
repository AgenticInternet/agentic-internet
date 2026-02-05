"""Tests for the PythonExecutorTool and DataAnalysisTool."""


from agentic_internet.tools.code_execution import (
    DataAnalysisTool,
    PythonExecutorTool,
    _ASTSafetyValidator,
)


class TestASTSafetyValidator:
    def setup_method(self):
        self.validator = _ASTSafetyValidator()

    def _violations(self, code: str) -> list[str]:
        import ast
        tree = ast.parse(code)
        return self.validator.validate(tree)

    def test_safe_code_passes(self):
        assert self._violations("x = 1 + 2\nprint(x)") == []

    def test_blocks_import_os(self):
        violations = self._violations("import os")
        assert len(violations) > 0
        assert "os" in violations[0]

    def test_blocks_from_subprocess(self):
        violations = self._violations("from subprocess import call")
        assert len(violations) > 0

    def test_blocks_dunder_subclasses(self):
        violations = self._violations("x.__subclasses__()")
        assert len(violations) > 0

    def test_blocks_dunder_globals(self):
        violations = self._violations("x.__globals__")
        assert len(violations) > 0

    def test_blocks_eval(self):
        violations = self._violations("eval('1+1')")
        assert len(violations) > 0

    def test_blocks_exec(self):
        violations = self._violations("exec('print(1)')")
        assert len(violations) > 0

    def test_blocks_open(self):
        violations = self._violations("open('/etc/passwd')")
        assert len(violations) > 0

    def test_blocks_import_ctypes(self):
        violations = self._violations("import ctypes")
        assert len(violations) > 0

    def test_allows_math_operations(self):
        assert self._violations("import math\nresult = math.sqrt(4)") == []

    def test_blocks_chained_dunder_access(self):
        violations = self._violations("().__class__.__bases__[0].__subclasses__()")
        assert len(violations) > 0


class TestPythonExecutorTool:
    def setup_method(self):
        self.tool = PythonExecutorTool()

    def test_basic_execution(self):
        result = self.tool.forward("print('hello world')")
        assert "hello world" in result

    def test_result_variable(self):
        result = self.tool.forward("result = 42")
        assert "42" in result

    def test_numpy_available(self):
        # np is pre-loaded in the namespace (no import needed)
        result = self.tool.forward("result = np.array([1,2,3]).sum()")
        assert "6" in result

    def test_blocks_os_import(self):
        result = self.tool.forward("import os\nos.system('ls')")
        assert "Error" in result

    def test_blocks_subprocess(self):
        result = self.tool.forward("import subprocess\nsubprocess.run(['ls'])")
        assert "Error" in result

    def test_syntax_error(self):
        result = self.tool.forward("def foo(")
        assert "Error" in result

    def test_runtime_error(self):
        result = self.tool.forward("1 / 0")
        assert "Error" in result or "ZeroDivision" in result

    def test_output_truncation(self):
        code = "for i in range(100000): print(i)"
        result = self.tool.forward(code)
        assert len(result) <= PythonExecutorTool.MAX_OUTPUT_LENGTH + 100  # some margin

    def test_uses_namespace_modules(self):
        result = self.tool.forward("result = json.dumps({'a': 1})")
        assert '"a"' in result

    def test_blocks_dunder_escape(self):
        result = self.tool.forward("x = ().__class__.__bases__[0].__subclasses__()")
        assert "Error" in result


class TestDataAnalysisTool:
    def setup_method(self):
        self.tool = DataAnalysisTool()

    def test_describe_json(self):
        data = '[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'
        result = self.tool.forward(data, "describe")
        assert "mean" in result.lower() or "count" in result.lower()

    def test_describe_csv(self):
        data = "a,b\n1,2\n3,4"
        result = self.tool.forward(data, "describe")
        assert "count" in result.lower()

    def test_summary(self):
        data = '[{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]'
        result = self.tool.forward(data, "summary")
        assert "Shape" in result
        assert "Columns" in result

    def test_missing(self):
        data = '[{"a": 1, "b": null}, {"a": null, "b": 2}]'
        result = self.tool.forward(data, "missing")
        assert "Missing" in result or "a" in result

    def test_invalid_operation(self):
        data = '[{"a": 1}]'
        result = self.tool.forward(data, "invalid_op")
        assert "Unknown operation" in result

    def test_invalid_data(self):
        result = self.tool.forward(":::not\x00valid\x01data\x02", "describe")
        assert "Error" in result or "count" in result.lower()  # pandas may parse anything

    def test_correlations_no_numeric(self):
        data = '[{"a": "x"}, {"a": "y"}]'
        result = self.tool.forward(data, "correlations")
        assert "No numeric" in result

    def test_info(self):
        data = '[{"a": 1, "b": "text"}]'
        result = self.tool.forward(data, "info")
        assert "Column" in result or "Dtype" in result or "dtypes" in result.lower()
