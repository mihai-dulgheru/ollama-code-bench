from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_SUITES = ["humaneval", "mbpp", "js"]


@dataclass
class ModelSpec:
    tag: str
    label: str
    family: str = ""
    quant: str = ""


@dataclass
class BenchConfig:
    models: list[ModelSpec]
    host: str | None = None
    suites: list[str] = field(default_factory=lambda: list(_DEFAULT_SUITES))
    limit: int | None = None
    timeout: int = 30  # per-task execution timeout (seconds)
    temperature: float = 0.0
    system_prompt: str = "You are a precise coding assistant."
    output_dir: str = "results"
    request_timeout: int = 600  # per-generation HTTP timeout (seconds)
    num_predict: int = 2048  # cap generated tokens (guards repetition loops)


def load_config(path: str | Path) -> BenchConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    models = [
        ModelSpec(
            tag=m["tag"],
            label=m.get("label", m["tag"]),
            family=m.get("family", ""),
            quant=m.get("quant", ""),
        )
        for m in data["models"]
    ]
    return BenchConfig(
        models=models,
        host=data.get("host"),
        # `or` (not the get-default) so a bare `suites:` key (parses to None) still
        # falls back instead of crashing task loading downstream.
        suites=data.get("suites") or list(_DEFAULT_SUITES),
        limit=data.get("limit"),
        timeout=data.get("timeout", 30),
        temperature=data.get("temperature", 0.0),
        system_prompt=data.get("system_prompt", "You are a precise coding assistant."),
        output_dir=data.get("output_dir", "results"),
        request_timeout=data.get("request_timeout", 600),
        num_predict=data.get("num_predict", 2048),
    )
