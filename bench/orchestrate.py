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


def build_tasks(suites: list[str], limit: int | None, timeout: int = 30):
    tasks = []
    # Inform users on potential dataset downloads if running for the first time
    if "humaneval" in suites or "mbpp" in suites:
        print(
            "Note: Loading HumanEval/MBPP datasets. If running for the first time, this may download datasets (needs internet).")

    if "humaneval" in suites:
        tasks += load_humaneval(limit, timeout)
    if "mbpp" in suites:
        tasks += load_mbpp(limit, timeout)
    if "js" in suites:
        tasks += load_js_tasks()
    return tasks


def load_cached(output_dir, model: str, task_id: str) -> TaskResult | None:
    p = raw_path(output_dir, model, task_id)
    if p.exists():
        try:
            return TaskResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            # Corrupt/partial cache file (bad JSON, missing fields) -> treat as no cache.
            return None
    return None


def _execute(task, solution: str):
    if task.language == "python":
        return run_python(solution, task.test_code, task.timeout)
    return run_node(solution, task.test_code, task.timeout, task.entry_point)


def run_one(cfg: BenchConfig, model, task) -> tuple[TaskResult, dict]:
    """Returns (result-for-aggregation, raw-record-to-persist)."""
    gen = generate(model.tag, task.prompt, cfg.system_prompt, cfg.host,
                   cfg.temperature, cfg.request_timeout, cfg.num_predict)
    code = extract_code(gen.text, task.language)
    if not code:
        result = TaskResult(model.label, task.id, task.category, task.language,
                            False, "no code extracted", gen.decode_tps, gen.ttft_s, gen.load_s)
    else:
        ex = _execute(task, code)
        result = TaskResult(model.label, task.id, task.category, task.language,
                            ex.passed, ex.reason, gen.decode_tps, gen.ttft_s, gen.load_s)
    record = {**result.to_dict(), "prompt": task.prompt, "output": gen.text,
              "extracted_code": code}
    return result, record


def run_benchmark(cfg: BenchConfig, resume: bool = False,
                  log: Callable[[str], object] = print) -> dict:
    tasks = build_tasks(cfg.suites, cfg.limit, cfg.timeout)
    results: list[TaskResult] = []
    footprints: dict = {}

    for model in cfg.models:
        # OPTIMIZATION: Check if all tasks are already fully cached for this model.
        # This allows us to instantly bypass pulling, starting, or stopping the model.
        if resume and tasks:
            all_cached = []
            all_match = True
            for task in tasks:
                cached = load_cached(cfg.output_dir, model.label, task.id)
                if cached and not cached.reason.startswith("error:"):
                    all_cached.append(cached)
                else:
                    all_match = False
                    break

            if all_match:
                results.extend(all_cached)
                # Attempt a footprint check using cached/live data if possible
                footprints[model.label] = footprint(model.tag, cfg.host)
                log(f"== {model.label} ({model.tag}) [ALL CACHED - Skipping Run] ==")
                continue

        log(f"== {model.label} ({model.tag}) ==")
        try:
            ensure_model(model.tag, cfg.host)
        except Exception as e:  # pull failed / tag missing
            log(f"  SKIP: cannot pull {model.tag}: {e}")
            continue

        for task in tasks:
            cached = load_cached(cfg.output_dir, model.label, task.id) if resume else None
            # Reuse a cached verdict, but re-run transient crashes ("error: ...")
            # rather than caching them as permanent failures.
            if cached and not cached.reason.startswith("error:"):
                results.append(cached)
                continue
            try:
                r, record = run_one(cfg, model, task)
            except Exception as e:  # generation crash (timeout, server error, ...)
                r = TaskResult(model.label, task.id, task.category, task.language,
                               False, f"error: {e}", 0.0, 0.0, 0.0)
                record = r.to_dict()
            p = raw_path(cfg.output_dir, model.label, task.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(record, indent=2), encoding="utf-8")
            results.append(r)
            mark = "PASS" if r.passed else "fail"
            log(f"  {task.id}: {mark} ({r.decode_tps:.0f} tok/s)")

        # Sample footprint while the model is still loaded: `ollama ps` only
        # lists resident models, so the PROCESSOR/loaded-size fields are blank
        # if captured before the first generation or after stop().
        footprints[model.label] = footprint(model.tag, cfg.host)
        stop(model.tag, cfg.host)

    agg = aggregate(results)
    write_results(cfg.output_dir, agg, footprints)
    return agg
