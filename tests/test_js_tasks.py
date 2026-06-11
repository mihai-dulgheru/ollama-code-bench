from bench.tasks.js_tasks import load_js_tasks
from bench.tasks.schema import Task


def test_loads_all_json_tasks():
    tasks = load_js_tasks()
    assert len(tasks) >= 4
    assert all(isinstance(t, Task) for t in tasks)
    assert all(t.language == "node" for t in tasks)
    ids = {t.id for t in tasks}
    assert "js/paginate" in ids


def test_task_fields_populated():
    by_id = {t.id: t for t in load_js_tasks()}
    t = by_id["js/paginate"]
    assert t.entry_point == "paginate"
    assert "node:assert" in t.test_code
    assert t.category == "js-logic"


def test_all_shipped_tasks_have_valid_schema():
    tasks = load_js_tasks()
    assert len(tasks) == 12
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))  # ids unique
    for t in tasks:
        assert t.id and t.prompt.strip() and t.test_code.strip()
        assert t.entry_point and t.entry_point in t.test_code
        assert "node:assert" in t.test_code
        assert t.category == "js-logic"
        assert t.language == "node"
        assert isinstance(t.timeout, int) and t.timeout > 0


def test_load_js_tasks_accepts_explicit_dir(tmp_path):
    import json
    (tmp_path / "x.json").write_text(json.dumps({
        "id": "js/x", "category": "js-logic",
        "prompt": "p", "test_code": "x();", "entry_point": "x",
    }), encoding="utf-8")
    tasks = load_js_tasks(tmp_path)
    assert len(tasks) == 1 and tasks[0].id == "js/x"
