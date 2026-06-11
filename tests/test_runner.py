from bench import runner as _runner
# noinspection PyProtectedMember
from bench.runner import (
    GenResult,
    _name_matches,
    _normalize_tag,
    format_bytes_to_readable,
    parse_metrics,
)


def test_parse_metrics_computes_tps_and_ttft():
    resp = {
        "message": {"content": "```python\nx=1\n```"},
        "eval_count": 100,
        "eval_duration": 2_000_000_000,  # 2s -> 50 tok/s
        "prompt_eval_duration": 500_000_000,  # 0.5s
        "load_duration": 1_000_000_000,  # 1s
    }
    r = parse_metrics(resp)
    assert isinstance(r, GenResult)
    assert r.text == "```python\nx=1\n```"
    assert round(r.decode_tps, 1) == 50.0
    assert round(r.ttft_s, 2) == 0.5
    assert round(r.load_s, 2) == 1.0
    assert r.eval_count == 100


def test_parse_metrics_handles_zero_duration():
    resp = {"message": {"content": "x"}, "eval_count": 0, "eval_duration": 0}
    r = parse_metrics(resp)
    assert r.decode_tps == 0.0


import pytest


@pytest.mark.integration
def test_real_generate_against_tiny_model():
    from bench.runner import ensure_model, generate, stop
    tag = "qwen2.5-coder:1.5b"
    ensure_model(tag)
    r = generate(tag, "Write a Python function add(a,b) that returns a+b.",
                 "You are a precise coding assistant.")
    assert r.text
    assert r.decode_tps > 0
    stop(tag)


def test_normalize_tag_adds_latest():
    assert _normalize_tag("qwen") == "qwen:latest"
    assert _normalize_tag("qwen:1.5b") == "qwen:1.5b"
    assert _normalize_tag(None) == ""  # SDK .model can be None


def test_name_matches_exact_and_latest():
    assert _name_matches(["qwen3-coder:latest", "id", "19", "GB"], "qwen3-coder")
    assert _name_matches(["qwen:1.5b"], "qwen:1.5b")


def test_name_matches_rejects_prefix_collision():
    # qwen3-coder must NOT match qwen3-coder-next:latest
    assert not _name_matches(["qwen3-coder-next:latest"], "qwen3-coder")


def test_name_matches_empty_cols():
    assert not _name_matches([], "qwen")


def test_format_bytes_thresholds():
    assert format_bytes_to_readable(512) == "512 B"
    assert format_bytes_to_readable(986_100_000) == "986.1 MB"
    assert format_bytes_to_readable(19_000_000_000) == "19.0 GB"
    assert format_bytes_to_readable(2_000_000_000_000) == "2.0 TB"


def test_parse_metrics_handles_pydantic_like_object():
    import types
    resp = types.SimpleNamespace(
        message=types.SimpleNamespace(content="ok"),
        eval_count=10,
        eval_duration=1_000_000_000,  # 1s -> 10 tok/s
        prompt_eval_duration=2_000_000_000,
        load_duration=0,
    )
    r = parse_metrics(resp)
    assert r.text == "ok"
    assert round(r.decode_tps, 1) == 10.0
    assert round(r.ttft_s, 1) == 2.0


def test_parse_metrics_coerces_none_durations():
    resp = {"message": {"content": "x"}, "eval_count": 5,
            "eval_duration": None, "prompt_eval_duration": None, "load_duration": None}
    r = parse_metrics(resp)
    assert r.decode_tps == 0.0
    assert r.ttft_s == 0.0
    assert r.load_s == 0.0


class _FakeModel:
    def __init__(self, model, size, size_vram=0):
        self.model = model
        self.size = size
        self.size_vram = size_vram


class _FakeListing:
    def __init__(self, models):
        self.models = models


class _FakeClient:
    def __init__(self, listed, active):
        self._listed = listed
        self._active = active

    def list(self):
        return _FakeListing(self._listed)

    def ps(self):
        return _FakeListing(self._active)


def _patch_client(monkeypatch, listed, active):
    monkeypatch.setattr(_runner, "_client",
                        lambda host=None, timeout=None: _FakeClient(listed, active))


def test_footprint_reports_full_gpu(monkeypatch):
    tag = "qwen2.5-coder:1.5b"
    listed = [_FakeModel(tag, 986_100_000)]
    active = [_FakeModel(tag, 1_600_000_000, size_vram=1_600_000_000)]
    _patch_client(monkeypatch, listed, active)
    fp = _runner.footprint(tag)
    assert fp["disk"] == "986.1 MB"
    assert fp["loaded"] == "1.6 GB"
    assert fp["processor"] == "100% GPU"


def test_footprint_reports_full_cpu(monkeypatch):
    tag = "m:1"
    _patch_client(monkeypatch,
                  [_FakeModel(tag, 700_000_000)],
                  [_FakeModel(tag, 700_000_000, size_vram=0)])
    assert _runner.footprint(tag)["processor"] == "100% CPU"


def test_footprint_reports_mixed_split(monkeypatch):
    tag = "m:1"
    _patch_client(monkeypatch,
                  [_FakeModel(tag, 1_000_000_000)],
                  [_FakeModel(tag, 1_000_000_000, size_vram=500_000_000)])
    assert _runner.footprint(tag)["processor"] == "50%/50% GPU/CPU"


def test_footprint_unloaded_model_leaves_processor_blank(monkeypatch):
    # On disk but absent from `ollama ps`; CLI fallback unavailable.
    tag = "m:1"
    _patch_client(monkeypatch, [_FakeModel(tag, 700_000_000)], active=[])

    def _no_cli(*_a, **_k):
        raise FileNotFoundError("ollama")

    monkeypatch.setattr(_runner.subprocess, "run", _no_cli)
    fp = _runner.footprint(tag)
    assert fp["disk"] == "700.0 MB"
    assert fp["processor"] == ""
