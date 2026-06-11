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
