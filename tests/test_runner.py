# tests/test_runner.py
from bench.runner import parse_metrics, GenResult


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
