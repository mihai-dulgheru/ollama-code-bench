from .schema import Task

_INSTRUCTION = (
    "Complete the following Python task. Define a function named `{entry}`. "
    "Respond with only the complete function in a single code block.\n\n"
)


def problem_to_task(problem: dict, category: str, timeout: int = 30) -> Task:
    entry = problem["entry_point"]
    # HumanEval+ ships a `def check(candidate)` in "test"; we call it.
    # MBPP+ ships bare `assert entry(...) == ...` statements in "assertion"
    # (no check() wrapper, function referenced by name) — append directly.
    if "test" in problem:
        test_code = problem["test"] + f"\n\ncheck({entry})"
    else:
        test_code = problem["assertion"]
    return Task(
        id=problem["task_id"],
        category=category,
        language="python",
        prompt=_INSTRUCTION.format(entry=entry) + problem["prompt"],
        test_code=test_code,
        entry_point=entry,
        timeout=timeout,
    )


def load_humaneval(limit: int | None = None, timeout: int = 30) -> list[Task]:
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    return _load(problems, "humaneval", limit, timeout)


def load_mbpp(limit: int | None = None, timeout: int = 30) -> list[Task]:
    from evalplus.data import get_mbpp_plus
    problems = get_mbpp_plus()
    return _load(problems, "mbpp", limit, timeout)


def _load(problems: dict, category: str, limit: int | None, timeout: int) -> list[Task]:
    items = list(problems.values())
    if limit is not None:
        items = items[:limit]
    return [problem_to_task(p, category, timeout) for p in items]
