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
