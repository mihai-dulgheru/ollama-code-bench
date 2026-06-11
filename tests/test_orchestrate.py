from pathlib import Path

from bench.orchestrate import raw_path, build_tasks, load_cached
from bench.scorer import TaskResult


def test_raw_path_sanitizes_tag(tmp_path: Path):
    p = raw_path(tmp_path, "qwen3-coder:30b", "humaneval/0")
    assert p == tmp_path / "raw" / "qwen3-coder_30b" / "humaneval_0.json"


def test_build_tasks_respects_suites_and_limit():
    tasks = build_tasks(["js"], limit=None)  # js only, no network
    assert tasks and all(t.language == "node" for t in tasks)


def test_load_cached_roundtrip(tmp_path: Path):
    r = TaskResult("m", "t1", "js-logic", "node", True, "", 50.0, 0.1, 1.0)
    p = raw_path(tmp_path, "m", "t1")
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(r.to_dict()), encoding="utf-8")
    loaded = load_cached(tmp_path, "m", "t1")
    assert loaded is not None
    assert loaded.passed is True and loaded.decode_tps == 50.0


def test_load_cached_ignores_corrupt_file(tmp_path: Path):
    p = raw_path(tmp_path, "m", "t1")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json", encoding="utf-8")
    assert load_cached(tmp_path, "m", "t1") is None


def test_resume_skips_generation_when_all_cached(tmp_path: Path, monkeypatch):
    import json
    import bench.orchestrate as orch
    from bench.config import BenchConfig, ModelSpec
    from bench.tasks.js_tasks import load_js_tasks

    model = ModelSpec(tag="m:1", label="M")
    cfg = BenchConfig(models=[model], suites=["js"], output_dir=str(tmp_path))

    tasks = load_js_tasks()
    for t in tasks:  # pre-populate a full passing cache
        r = TaskResult("M", t.id, t.category, "node", True, "", 50.0, 0.1, 1.0)
        p = raw_path(tmp_path, "M", t.id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r.to_dict()), encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("generate must not run when fully cached")

    monkeypatch.setattr(orch, "generate", _boom)
    monkeypatch.setattr(orch, "ensure_model", lambda *a, **k: None)
    monkeypatch.setattr(orch, "stop", lambda *a, **k: None)
    monkeypatch.setattr(orch, "footprint",
                        lambda *a, **k: {"disk": "", "loaded": "", "processor": ""})

    agg = orch.run_benchmark(cfg, resume=True, log=lambda *_: None)
    assert agg["M"]["n"] == len(tasks)
    assert agg["M"]["pass_at_1"] == 1.0


def test_run_one_marks_no_code_extracted(tmp_path: Path, monkeypatch):
    import bench.orchestrate as orch
    from bench.config import BenchConfig, ModelSpec
    from bench.runner import GenResult
    from bench.tasks.schema import Task

    cfg = BenchConfig(models=[ModelSpec(tag="m:1", label="M")])
    task = Task(id="t", category="c", language="python", prompt="p",
                test_code="check(f)", entry_point="f")
    # Model returns an empty reply -> extract returns "" -> graceful fail
    # (metrics still recorded, no execution attempted).
    monkeypatch.setattr(orch, "generate",
                        lambda *a, **k: GenResult("", 12.0, 0.1, 0.5, 0))
    result, record = orch.run_one(cfg, cfg.models[0], task)
    assert result.passed is False
    assert result.reason == "no code extracted"
    assert result.decode_tps == 12.0  # speed metrics preserved on failure
    assert record["output"] == ""
