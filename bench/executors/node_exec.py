from .base import ExecResult, run_code


def run_node(solution: str, test_code: str, timeout: int = 30,
             entry_point: str | None = None) -> ExecResult:
    if entry_point:
        # Run the model's solution in its own scope and expose only the entry
        # function to the global object. This isolates the solution's top-level
        # declarations (e.g. a stray `const assert = require(...)` or its own
        # self-tests) so they can't redeclare-collide with the test harness,
        # which would otherwise be a SyntaxError and fail a correct solution.
        contents = (
            "(function () {\n"
            f"{solution}\n"
            f"try {{ globalThis.{entry_point} = {entry_point}; }} catch (e) {{}}\n"
            "})();\n\n"
            f"{test_code}\n"
        )
    else:
        contents = f"{solution}\n\n{test_code}\n"
    return run_code(contents, ".js", ["node"], timeout)
