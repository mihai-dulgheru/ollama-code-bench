# bench/executors/python_exec.py
import sys

from .base import ExecResult, run_code


def run_python(solution: str, test_code: str, timeout: int = 30) -> ExecResult:
    contents = f"{solution}\n\n{test_code}\n"
    return run_code(contents, ".py", [sys.executable], timeout)
