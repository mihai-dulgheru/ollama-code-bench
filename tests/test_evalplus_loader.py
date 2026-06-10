# tests/test_evalplus_loader.py
from bench.tasks.evalplus_loader import problem_to_task


def test_maps_problem_to_task():
    problem = {
        "task_id": "HumanEval/0",
        "prompt": "def has_close(xs):\n    \"\"\"docstring\"\"\"\n",
        "entry_point": "has_close",
        "test": "def check(candidate):\n    assert candidate([1.0]) == False",
    }
    t = problem_to_task(problem, category="humaneval")
    assert t.id == "HumanEval/0"
    assert t.language == "python"
    assert t.entry_point == "has_close"
    assert "def has_close" in t.prompt
    assert "Respond with only" in t.prompt          # instruction added
    assert t.test_code.strip().endswith("check(has_close)")  # call appended


import pytest


@pytest.mark.integration
def test_real_humaneval_loads_and_grades_canonical():
    from evalplus.data import get_human_eval_plus
    from bench.tasks.evalplus_loader import problem_to_task
    from bench.executors.python_exec import run_python

    problem = next(iter(get_human_eval_plus().values()))
    task = problem_to_task(problem, "humaneval")
    # canonical_solution is the prompt body completed; prepend prompt to it.
    sol = problem["prompt"] + problem["canonical_solution"]
    r = run_python(sol, task.test_code, task.timeout)
    assert r.passed, r.reason
