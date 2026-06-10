# tests/test_orchestrate.py
from pathlib import Path
from bench.orchestrate import raw_path, build_tasks, load_cached
from bench.scorer import TaskResult


def test_raw_path_sanitizes_tag(tmp_path: Path):
    p = raw_path(tmp_path, "qwen3-coder:30b", "humaneval/0")
    assert p == tmp_path / "raw" / "qwen3-coder_30b" / "humaneval_0.json"


def test_build_tasks_respects_suites_and_limit():
    tasks = build_tasks(["js"], limit=None)        # js only, no network
    assert tasks and all(t.language == "node" for t in tasks)


def test_load_cached_roundtrip(tmp_path: Path):
    r = TaskResult("m", "t1", "js-logic", "node", True, "", 50.0, 0.1, 1.0)
    p = raw_path(tmp_path, "m", "t1")
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(r.to_dict()), encoding="utf-8")
    loaded = load_cached(tmp_path, "m", "t1")
    assert loaded.passed is True and loaded.decode_tps == 50.0
