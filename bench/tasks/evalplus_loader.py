# bench/tasks/evalplus_loader.py
from .schema import Task

_INSTRUCTION = (
    "Complete the following Python function. "
    "Respond with only the complete function in a single code block.\n\n"
)


def problem_to_task(problem: dict, category: str, timeout: int = 30) -> Task:
    entry = problem["entry_point"]
    return Task(
        id=problem["task_id"],
        category=category,
        language="python",
        prompt=_INSTRUCTION + problem["prompt"],
        test_code=problem["test"] + f"\n\ncheck({entry})",
        entry_point=entry,
        timeout=timeout,
    )


def load_humaneval(limit: int | None = None) -> list[Task]:
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    return _load(problems, "humaneval", limit)


def load_mbpp(limit: int | None = None) -> list[Task]:
    from evalplus.data import get_mbpp_plus
    problems = get_mbpp_plus()
    return _load(problems, "mbpp", limit)


def _load(problems: dict, category: str, limit: int | None) -> list[Task]:
    items = list(problems.values())
    if limit is not None:
        items = items[:limit]
    return [problem_to_task(p, category) for p in items]
