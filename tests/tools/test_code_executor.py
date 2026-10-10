# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""代码执行工具的测试。"""

import pytest
import numpy as np

from sr_harness.tools.code_executor import CodeExecutorTool


class TestCodeExecutorTool:
    """测试代码执行工具。"""

    def setup_method(self):
        self.tool = CodeExecutorTool(data={})

    def test_basic_print(self):
        result = self.tool.execute('print("Hello, World!")')
        assert "Hello, World!" in result["stdout"]
        assert result["stderr"] == ""
        assert result["duration"] >= 0

    def test_numpy_computation(self):
        result = self.tool.execute(
            """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(arr)}")
print(f"Std: {np.std(arr):.4f}")
"""
        )
        assert "Mean: 3.0" in result["stdout"]

    def test_math_operations(self):
        result = self.tool.execute(
            """
import math
print(math.sqrt(16))
print(math.sin(math.pi / 2))
"""
        )
        assert "4.0" in result["stdout"]

    def test_math_eval_literal_is_allowed(self):
        result = self.tool.execute(
            """
import math
x1 = 2.0
x2 = 0.5
print(eval("x1 + sin(x2)", {"__builtins__": {}}, {"x1": x1, "x2": x2, "sin": math.sin}))
"""
        )
        assert "2.479" in result["stdout"]

    def test_numpy_math_eval_literal_is_allowed(self):
        result = self.tool.execute(
            """
import numpy as np
x1 = np.array([1.0, 2.0])
x2 = np.array([3.0, 4.0])
print(eval("x1 + np.sin(x2)", {"__builtins__": {}}, {"x1": x1, "x2": x2, "np": np}))
"""
        )
        assert "[" in result["stdout"]

    def test_dynamic_math_eval_is_allowed(self):
        result = self.tool.execute('expr = "1+1"\nprint(eval(expr))')
        assert "2" in result["stdout"]

    def test_dynamic_eval_is_confined_by_the_os_sandbox(self):
        result = self.tool.execute(
            """expr = "__import__('os').environ.get('HOME')"\nprint(eval(expr))"""
        )
        assert result["stdout"].strip() == "/tmp/home"

    def test_indirect_eval_is_allowed(self):
        result = self.tool.execute('e = eval\nprint(e("1+1"))')
        assert result["stdout"].strip() == "2"

    def test_nested_eval_is_allowed(self):
        result = self.tool.execute("""print(eval("eval('1+1')"))""")
        assert result["stdout"].strip() == "2"

    def test_nested_exec_is_allowed(self):
        result = self.tool.execute("""eval("exec('x=1')")\nprint('ok')""")
        assert result["stdout"].strip() == "ok"

    def test_dunder_access_is_confined_instead_of_rejected(self):
        result = self.tool.execute("""print(eval("().__class__").__name__)""")
        assert result["stdout"].strip() == "tuple"

    def test_eval_supports_general_python_expressions(self):
        result = self.tool.execute(
            """print(eval("x1 > 0", {"__builtins__": {}}, {"x1": 1}))\nprint(eval("'text'"))"""
        )
        assert result["stdout"].splitlines() == ["True", "text"]

    def test_type_and_hasattr_builtins_are_allowed(self):
        result = self.tool.execute(
            """
value = [1, 2, 3]
print(type(value).__name__)
print(hasattr(value, "__len__"))
"""
        )
        assert "list" in result["stdout"]
        assert "True" in result["stdout"]

    def test_os_module_sees_only_sanitized_environment(self):
        result = self.tool.execute("import os\nprint(os.environ.get('HOME'))")
        assert result["stdout"].strip() == "/tmp/home"

    def test_subprocess_module_is_available_inside_the_namespace(self):
        result = self.tool.execute("import subprocess\nprint(subprocess.__name__)")
        assert result["stdout"].strip() == "subprocess"

    def test_pandas_is_available(self):
        result = self.tool.execute("import pandas as pd\nprint(pd.Series([1, 2]).sum())")
        assert result["stdout"].strip() == "3"

    def test_scipy_module_is_allowed(self):
        result = self.tool.execute(
            """
from scipy import stats
print(f"{stats.pearsonr([1, 2, 3], [1, 2, 4]).statistic:.4f}")
"""
        )
        assert "0.9820" in result["stdout"]

    def test_traceback_module_is_allowed(self):
        result = self.tool.execute(
            """
import traceback
try:
    1 / 0
except Exception:
    print(traceback.format_exc().splitlines()[-1])
"""
        )
        assert "ZeroDivisionError" in result["stdout"]

    def test_syntax_error(self):
        with pytest.raises(Exception, match="SyntaxError"):
            self.tool.execute('print("missing quote')

    def test_runtime_error(self):
        with pytest.raises(Exception, match="ZeroDivisionError"):
            self.tool.execute("print(1 / 0)")

    def test_timeout(self):
        with pytest.raises(Exception, match="timeout=1"):
            self.tool.execute("while True:\n    pass", timeout_seconds=1)

    def test_dunder_attributes_are_safe_inside_the_namespace(self):
        result = self.tool.execute("print((1).__class__.__name__)")
        assert result["stdout"].strip() == "int"

    def test_output_truncation(self):
        result = self.tool.execute('print("x" * 70000)')
        assert "...[truncated]" in result["stdout"]
        assert len(result["stdout"]) <= self.tool.DEFAULT_OUTPUT_LIMIT_BYTES + 1

    def test_context_data_is_available_from_stdin_as_dict(self):
        tool = CodeExecutorTool(
            data={
                "x1": np.array([1.0, 2.0, 3.0]),
                "x2": np.array([0.5, 1.5, 2.5]),
                "y": np.array([0.5, 0.5, 0.5]),
            }
        )
        result = tool.execute(
            """
import json
import sys

input_data_str = sys.stdin.read()
data_dict = json.loads(input_data_str)
print(sorted(data_dict))
print(data_dict["x1"][0])
print(data_dict["y"][-1])
"""
        )
        assert "['x1', 'x2', 'y']" in result["stdout"]
        assert "1.0" in result["stdout"]
        assert "0.5" in result["stdout"]

    def test_stdin_loader_can_use_llm_chosen_variable_names(self):
        tool = CodeExecutorTool(data={"x": [1, 2], "y": [3, 4]})
        result = tool.execute(
            """
import json
import sys

data_str = sys.stdin.read()
data_dict = json.loads(data_str)
data = {"sample": True}
print(data_dict["y"][-1])
print(data)
"""
        )
        assert "4" in result["stdout"]
        assert "{'sample': True}" in result["stdout"]

    def test_multiple_operations(self):
        result = self.tool.execute(
            """
import numpy as np
import math

x = np.linspace(0, 2 * math.pi, 5)
y = np.sin(x)

print(f"x: {x}")
print(f"sin(x): {y}")
print(f"sum: {np.sum(y)}")
"""
        )
        assert "sin(x):" in result["stdout"]

    def test_builtin_functions(self):
        result = self.tool.execute(
            """
data = [1, 2, 3, 4, 5]
print(f"Sum: {sum(data)}")
print(f"Max: {max(data)}")
print(f"Min: {min(data)}")
print(f"Length: {len(data)}")
"""
        )
        assert "Sum: 15" in result["stdout"]
        assert "Max: 5" in result["stdout"]

    def test_call_wraps_execution_errors(self):
        result = self.tool(program="1 / 0")
        assert result.ok is False
        assert "ZeroDivisionError" in result.result["error"]

    def test_tool_metadata(self):
        assert self.tool.metadata.name == "code_executor"
