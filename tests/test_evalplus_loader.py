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
    assert "Respond with only" in t.prompt  # instruction added
    assert t.test_code.strip().endswith("check(has_close)")  # call appended


def test_mbpp_assertion_mapping_no_check_wrapper():
    problem = {
        "task_id": "Mbpp/2",
        "entry_point": "foo",
        "prompt": "Write foo.",
        "assertion": "assert foo(1) == 1",
    }
    t = problem_to_task(problem, "mbpp")
    assert t.category == "mbpp"
    assert t.test_code == "assert foo(1) == 1"
    assert "check(" not in t.test_code


def test_timeout_propagates_to_task():
    problem = {"task_id": "HumanEval/1", "entry_point": "f",
               "prompt": "p", "test": "def check(c): assert c"}
    t = problem_to_task(problem, "humaneval", timeout=15)
    assert t.timeout == 15


def test_load_applies_limit():
    from bench.tasks.evalplus_loader import _load
    problems = {f"p{i}": {"task_id": f"T/{i}", "entry_point": "f",
                          "prompt": "p", "assertion": "assert True"} for i in range(5)}
    tasks = _load(problems, "mbpp", limit=2, timeout=30)
    assert len(tasks) == 2
    assert all(t.category == "mbpp" for t in tasks)


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
