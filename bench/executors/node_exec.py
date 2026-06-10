# bench/executors/node_exec.py
from .base import ExecResult, run_code


def run_node(solution: str, test_code: str, timeout: int = 30) -> ExecResult:
    contents = f"{solution}\n\n{test_code}\n"
    return run_code(contents, ".js", ["node"], timeout)
