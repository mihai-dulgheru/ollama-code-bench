# bench/orchestrate.py
import json
import re
from pathlib import Path
from typing import Callable

from .config import BenchConfig
from .executors.node_exec import run_node
from .executors.python_exec import run_python
from .extract import extract_code
from .report import write_results
from .runner import ensure_model, generate, footprint, stop
from .scorer import TaskResult, aggregate
from .tasks.evalplus_loader import load_humaneval, load_mbpp
from .tasks.js_tasks import load_js_tasks


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def raw_path(output_dir, model: str, task_id: str) -> Path:
    return Path(output_dir) / "raw" / _safe(model) / f"{_safe(task_id)}.json"


def build_tasks(suites: list[str], limit: int | None):
    tasks = []
    if "humaneval" in suites:
        tasks += load_humaneval(limit)
    if "mbpp" in suites:
        tasks += load_mbpp(limit)
    if "js" in suites:
        tasks += load_js_tasks()
    return tasks


def load_cached(output_dir, model: str, task_id: str) -> TaskResult | None:
    p = raw_path(output_dir, model, task_id)
    if p.exists():
        return TaskResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
    return None


def _execute(task, solution: str):
    if task.language == "python":
        return run_python(solution, task.test_code, task.timeout)
    return run_node(solution, task.test_code, task.timeout)


def run_one(cfg: BenchConfig, model, task) -> TaskResult:
    gen = generate(model.tag, task.prompt, cfg.system_prompt, cfg.host, cfg.temperature)
    code = extract_code(gen.text, task.language)
    if not code:
        return TaskResult(model.label, task.id, task.category, task.language,
                          False, "no code extracted", gen.decode_tps, gen.ttft_s, gen.load_s)
    ex = _execute(task, code)
    return TaskResult(model.label, task.id, task.category, task.language,
                      ex.passed, ex.reason, gen.decode_tps, gen.ttft_s, gen.load_s)


def run_benchmark(cfg: BenchConfig, resume: bool = False,
                  log: Callable[[str], object] = print) -> dict:
    tasks = build_tasks(cfg.suites, cfg.limit)
    results: list[TaskResult] = []
    footprints: dict = {}

    for model in cfg.models:
        log(f"== {model.label} ({model.tag}) ==")
        try:
            ensure_model(model.tag, cfg.host)
        except Exception as e:  # pull failed / tag missing
            log(f"  SKIP: cannot pull {model.tag}: {e}")
            continue
        footprints[model.label] = footprint(model.tag, cfg.host)

        for task in tasks:
            cached = load_cached(cfg.output_dir, model.label, task.id) if resume else None
            if cached:
                results.append(cached)
                continue
            try:
                r = run_one(cfg, model, task)
            except Exception as e:  # generation crash
                r = TaskResult(model.label, task.id, task.category, task.language,
                               False, f"error: {e}", 0.0, 0.0, 0.0)
            p = raw_path(cfg.output_dir, model.label, task.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            results.append(r)
            mark = "PASS" if r.passed else "fail"
            log(f"  {task.id}: {mark} ({r.decode_tps:.0f} tok/s)")

        stop(model.tag, cfg.host)

    agg = aggregate(results)
    write_results(cfg.output_dir, agg, footprints)
    return agg
