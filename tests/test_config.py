# tests/test_config.py
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
