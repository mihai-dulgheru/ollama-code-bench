from bench.tasks.schema import Task


def test_task_holds_fields_and_defaults():
    t = Task(
        id="humaneval/0",
        category="humaneval",
        language="python",
        prompt="Complete foo()",
        test_code="def check(c): assert c() == 1",
        entry_point="foo",
    )
    assert t.language == "python"
    assert t.entry_point == "foo"
    assert t.timeout == 30  # default
