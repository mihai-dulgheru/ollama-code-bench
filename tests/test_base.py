import sys

from bench.executors.base import run_code


def test_passing_zero_exit():
    r = run_code("pass\n", ".py", [sys.executable], timeout=10)
    assert r.passed
    assert r.reason == ""


def test_nonzero_exit_reports_exit_code():
    r = run_code("import sys; sys.exit(3)\n", ".py", [sys.executable], timeout=10)
    assert not r.passed
    assert "exit 3" in r.reason


def test_stderr_trimmed_into_reason():
    code = "import sys; sys.stderr.write('boom'); sys.exit(1)\n"
    r = run_code(code, ".py", [sys.executable], timeout=10)
    assert not r.passed
    assert "boom" in r.reason


def test_runner_not_found_is_failure():
    r = run_code("whatever", ".txt", ["__no_such_runner_xyz__"], timeout=5)
    assert not r.passed
    assert "runner not found" in r.reason


def test_timeout_is_failure():
    code = "import time; time.sleep(5)\n"
    r = run_code(code, ".py", [sys.executable], timeout=1)
    assert not r.passed
    assert "timeout" in r.reason.lower()


def test_stdout_and_stderr_captured():
    code = "import sys; print('out'); sys.stderr.write('err')\n"
    r = run_code(code, ".py", [sys.executable], timeout=10)
    assert r.passed
    assert "out" in r.stdout
    assert "err" in r.stderr
