from bench.executors.python_exec import run_python


def test_passing_solution():
    sol = "def add(a, b):\n    return a + b"
    test = "def check(f):\n    assert f(2, 3) == 5\ncheck(add)"
    r = run_python(sol, test)
    assert r.passed
    assert r.reason == ""


def test_failing_solution_reports_reason():
    sol = "def add(a, b):\n    return a - b"
    test = "def check(f):\n    assert f(2, 3) == 5\ncheck(add)"
    r = run_python(sol, test)
    assert not r.passed
    assert "AssertionError" in r.reason or r.reason


def test_syntax_error_is_failure():
    r = run_python("def add(a, b)\n    return a", "check(add)")
    assert not r.passed


def test_timeout_is_failure():
    sol = "import time\ndef f():\n    time.sleep(5)"
    test = "f()"
    r = run_python(sol, test, timeout=1)
    assert not r.passed
    assert "timeout" in r.reason.lower()
