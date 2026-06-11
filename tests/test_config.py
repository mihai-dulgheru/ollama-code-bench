from pathlib import Path

from bench.config import load_config


def test_loads_yaml(tmp_path: Path):
    yaml_text = """
host: null
suites: [humaneval, js]
limit: 50
timeout: 30
temperature: 0.0
system_prompt: "You are a coding assistant."
output_dir: results
models:
  - tag: qwen3-coder:30b
    label: Qwen3-Coder-30B
    family: qwen
    quant: Q4_K_M
  - tag: gpt-oss:120b
    label: gpt-oss-120b
"""
    p = tmp_path / "models.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.limit == 50
    assert cfg.suites == ["humaneval", "js"]
    assert len(cfg.models) == 2
    assert cfg.models[0].tag == "qwen3-coder:30b"
    assert cfg.models[1].family == ""  # default when omitted


def test_defaults_applied_when_keys_omitted(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("models:\n  - tag: only:tag\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.host is None
    assert cfg.limit is None
    assert cfg.timeout == 30
    assert cfg.temperature == 0.0
    assert cfg.output_dir == "results"
    assert cfg.request_timeout == 600
    assert cfg.num_predict == 2048
    assert cfg.suites == ["humaneval", "mbpp", "js"]
    assert cfg.models[0].label == "only:tag"  # label defaults to tag


def test_none_suites_falls_back_to_defaults(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("suites:\nmodels:\n  - tag: t\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.suites == ["humaneval", "mbpp", "js"]


def test_overrides_request_timeout_and_num_predict(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("request_timeout: 120\nnum_predict: 256\nmodels:\n  - tag: t\n",
                 encoding="utf-8")
    cfg = load_config(p)
    assert cfg.request_timeout == 120
    assert cfg.num_predict == 256
