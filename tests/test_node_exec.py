import shutil

import pytest

from bench.executors.node_exec import run_node

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_passing_solution():
    sol = "function add(a, b) { return a + b; }"
    test = "const assert = require('node:assert');\nassert.strictEqual(add(2, 3), 5);"
    r = run_node(sol, test)
    assert r.passed, r.reason


def test_failing_solution():
    sol = "function add(a, b) { return a - b; }"
    test = "const assert = require('node:assert');\nassert.strictEqual(add(2, 3), 5);"
    r = run_node(sol, test)
    assert not r.passed


def test_timeout():
    sol = "function f() { while (true) {} }"
    test = "f();"
    r = run_node(sol, test, timeout=1)
    assert not r.passed
    assert "timeout" in r.reason.lower()
