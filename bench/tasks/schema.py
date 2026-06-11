# bench/tasks/schema.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """One benchmark problem, language-agnostic.

    `test_code` is appended after the model's solution and must, when run,
    exit non-zero on failure (asserts for Python, node:assert for JS).
    """
    id: str
    category: str  # "humaneval" | "mbpp" | "js-logic"
    language: str  # "python" | "node"
    prompt: str  # sent to the model
    test_code: str  # harness appended after the solution
    entry_point: str  # function the tests call
    timeout: int = 30  # seconds for execution
