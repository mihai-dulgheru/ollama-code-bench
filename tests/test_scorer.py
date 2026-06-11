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


def test_median_tps_even_count_averages_middle_two():
    results = [_r("m", f"t{i}", "c", "python", True, tps)
               for i, tps in enumerate([10.0, 20.0, 30.0, 40.0])]
    assert aggregate(results)["m"]["median_tps"] == 25.0  # (20+30)/2


def test_zero_tps_excluded_from_median():
    results = [
        _r("m", "t0", "c", "python", True, 0.0),  # excluded (decode_tps == 0)
        _r("m", "t1", "c", "python", True, 40.0),
        _r("m", "t2", "c", "python", True, 60.0),
    ]
    assert aggregate(results)["m"]["median_tps"] == 50.0


def test_aggregate_empty_returns_empty_dict():
    assert aggregate([]) == {}


def test_model_with_no_passes_scores_zero():
    results = [_r("m", "t0", "c", "python", False, 30.0),
               _r("m", "t1", "c", "python", False, 30.0)]
    m = aggregate(results)["m"]
    assert m["pass_at_1"] == 0.0
    assert m["by_category"]["c"] == 0.0


def test_multiple_models_segregated():
    results = [_r("a", "t0", "c", "python", True, 30.0),
               _r("b", "t0", "c", "python", False, 30.0)]
    agg = aggregate(results)
    assert agg["a"]["pass_at_1"] == 1.0
    assert agg["b"]["pass_at_1"] == 0.0


def test_median_load_includes_zero_load_results():
    results = [_r("m", "t0", "c", "python", True, 40.0),
               _r("m", "t1", "c", "python", True, 40.0)]
    results[0].load_s = 0.0
    results[1].load_s = 2.0
    assert aggregate(results)["m"]["median_load_s"] == 1.0  # (0 + 2)/2
