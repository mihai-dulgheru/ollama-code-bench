import shutil

import pytest

from bench.executors.node_exec import run_node

# noinspection PyDeprecation
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


def test_entry_point_isolation_avoids_redeclare():
    # Solution declares its own `assert` and the harness does too. The IIFE
    # isolation in run_node keeps the solution's top-level const out of the
    # harness scope; without it the duplicate const is a SyntaxError that would
    # fail a correct solution.
    sol = ("const assert = require('node:assert');\n"
           "function add(a, b) { return a + b; }")
    test = ("const assert = require('node:assert');\n"
            "assert.strictEqual(add(2, 3), 5);")
    r = run_node(sol, test, entry_point="add")
    assert r.passed, r.reason


def test_without_entry_point_redeclare_collides():
    sol = ("const assert = require('node:assert');\n"
           "function add(a, b) { return a + b; }")
    test = ("const assert = require('node:assert');\n"
            "assert.strictEqual(add(2, 3), 5);")
    r = run_node(sol, test)  # no entry_point -> plain concat -> redeclare error
    assert not r.passed


def test_entry_point_exposed_to_global_scope():
    sol = "function paginate(p) { return p * 2; }"
    test = ("const assert = require('node:assert');\n"
            "assert.strictEqual(paginate(3), 6);")
    r = run_node(sol, test, entry_point="paginate")
    assert r.passed, r.reason
