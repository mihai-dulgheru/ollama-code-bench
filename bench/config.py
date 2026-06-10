# bench/config.py
from dataclasses import dataclass
from pathlib import Path

import yaml


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
    suites: list[str] = None          # set in load_config
    limit: int | None = None
    timeout: int = 30
    temperature: float = 0.0
    system_prompt: str = "You are a precise coding assistant."
    output_dir: str = "results"


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
        suites=data.get("suites", ["humaneval", "mbpp", "js"]),
        limit=data.get("limit"),
        timeout=data.get("timeout", 30),
        temperature=data.get("temperature", 0.0),
        system_prompt=data.get("system_prompt", "You are a precise coding assistant."),
        output_dir=data.get("output_dir", "results"),
    )
