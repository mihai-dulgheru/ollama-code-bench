import json
from pathlib import Path

from bench.report import render_markdown, write_results


def _agg():
    return {
        "Qwen3-Coder-30B": {
            "n": 3, "pass_at_1": 0.6667,
            "by_category": {"humaneval": 0.5, "js-logic": 1.0},
            "by_language": {"python": 0.5, "node": 1.0},
            "median_tps": 95.0, "median_ttft_s": 0.2, "median_load_s": 3.0,
        }
    }


def test_render_markdown_has_leaderboard_row():
    md = render_markdown(_agg(), footprints={"Qwen3-Coder-30B": {"disk": "19 GB", "processor": "100% GPU"}})
    assert "Qwen3-Coder-30B" in md
    assert "66.67%" in md or "0.6667" in md
    assert "95.0" in md  # tok/s


def test_write_results_emits_files(tmp_path: Path):
    write_results(tmp_path, _agg(), {"Qwen3-Coder-30B": {"disk": "19 GB"}})
    assert (tmp_path / "REPORT.md").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "Qwen3-Coder-30B" in summary["aggregated"]


def _two_models():
    def m(p, tps):
        return {"n": 1, "pass_at_1": p, "by_category": {"c": p},
                "by_language": {"python": p}, "median_tps": tps,
                "median_ttft_s": 0.1, "median_load_s": 1.0}

    return {"low": m(0.30, 10.0), "high": m(0.90, 20.0)}


def test_models_sorted_by_pass_rate_descending():
    md = render_markdown(_two_models(), footprints={})
    assert md.index("high") < md.index("low")
    assert "| 1 | high |" in md


def test_blank_processor_cell_renders_without_crash():
    # footprint dict lacks "processor" -> empty cell, no KeyError
    md = render_markdown(_agg(), footprints={"Qwen3-Coder-30B": {"disk": "19 GB"}})
    assert "Qwen3-Coder-30B" in md


def test_category_union_across_models_fills_missing_zero():
    agg = {
        "a": {"n": 1, "pass_at_1": 1.0, "by_category": {"humaneval": 1.0},
              "by_language": {"python": 1.0}, "median_tps": 1.0,
              "median_ttft_s": 0.1, "median_load_s": 1.0},
        "b": {"n": 1, "pass_at_1": 1.0, "by_category": {"js-logic": 1.0},
              "by_language": {"node": 1.0}, "median_tps": 1.0,
              "median_ttft_s": 0.1, "median_load_s": 1.0},
    }
    md = render_markdown(agg, footprints={})
    assert "humaneval" in md and "js-logic" in md
    assert "0%" in md  # the model missing a category shows 0%


def test_empty_aggregate_has_no_recommendation():
    md = render_markdown({}, footprints={})
    assert "Leaderboard" in md
    assert "Recommendation" not in md
