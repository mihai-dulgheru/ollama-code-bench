# tests/test_scorer.py
from bench.scorer import TaskResult, aggregate


def _r(model, task_id, category, language, passed, tps):
    return TaskResult(model=model, task_id=task_id, category=category,
                      language=language, passed=passed, reason="",
                      decode_tps=tps, ttft_s=0.1, load_s=1.0)


def test_aggregate_pass_rates_and_speed():
    results = [
        _r("m1", "humaneval/0", "humaneval", "python", True, 40.0),
        _r("m1", "humaneval/1", "humaneval", "python", False, 60.0),
        _r("m1", "js/paginate", "js-logic", "node", True, 50.0),
    ]
    agg = aggregate(results)
    m1 = agg["m1"]
    assert m1["pass_at_1"] == round(2 / 3, 4)
    assert m1["by_category"]["humaneval"] == 0.5
    assert m1["by_language"]["node"] == 1.0
    assert m1["median_tps"] == 50.0
    assert m1["n"] == 3
