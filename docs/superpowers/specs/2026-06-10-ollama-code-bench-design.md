# ollama-code-bench — Design

- **Date:** 2026-06-10
- **Status:** Approved (direction + model shortlist + scope confirmed)
- **Author:** Mihai-Nicolae Dulgheru

## Goal

Benchmark and compare the best local coding LLMs runnable via **Ollama**, to
decide which model to adopt as a daily local coding assistant ("local code
cloud"). Produce objective, reproducible numbers — correctness, speed, and
footprint — in a single comparison report.

This is a **separate project** from `qwen3-coder-modal` (which serves one model
on Modal cloud). Different purpose, different target machine.

## Target hardware & runtime

- **Machine:** GMKtec EVO-X2 — AMD Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S
  iGPU (RDNA 3.5), **128 GB unified LPDDR5X**, Windows 11 Pro.
- **Key constraint:** memory bandwidth ≈ 215–256 GB/s is the decode ceiling, not
  compute. 128 GB lets large models *load*, but bandwidth caps *speed*. Therefore
  **low-active-param MoE models are the sweet spot** (e.g. Qwen3-Coder-30B-A3B
  ≈ 100 tok/s on this chip per public benchmarks).
- **Runtime:** Ollama (`localhost:11434`), Node.js, Python 3.10+. GPU backend on
  Strix Halo is ROCm or Vulkan — README documents both (Vulkan often wins decode
  at normal context).
- **Portability:** `OLLAMA_HOST` configurable; a tiny-model smoke config makes the
  harness dev-testable on any machine (CPU), including the current laptop.

## Non-goals (YAGNI)

- No pass@k — greedy **pass@1** (1 sample, temperature 0).
- No React/component grading (rendering tests too fiddly to auto-grade).
- No GUI — Markdown + JSON report only.
- No distributed/parallel model runs — sequential (memory pressure).
- Not a general eval framework — purpose-built for this decision.

## Model shortlist (fits 128 GB, coding-focused)

Exact Ollama tags to be verified against the live Ollama library at build time.

| Tier          | Model (Ollama tag, tentative)          | Type  | Why                                                                   |
|---------------|----------------------------------------|-------|-----------------------------------------------------------------------|
| 1 — fast MoE  | `qwen3-coder-next` (80B / ~3B active)  | MoE   | newest top local coder (~70.6% SWE-bench Verified), ~3B active → fast |
| 1             | `qwen3-coder:30b` (30B / ~3.3B active) | MoE   | proven incumbent, ~100 tok/s on this chip                             |
| 1             | `gpt-oss:120b` (~5B active)            | MoE   | strong reasoning+coding, fits comfortably                             |
| 2 — dense ref | `devstral-small` (24B)                 | dense | agentic coding (~68% SWE-bench), 256K ctx; slower                     |
| 2             | `qwen2.5-coder:32b`                    | dense | strong dense baseline; slow                                           |

Excluded (won't fit 128 GB): GLM-5.1 (744B), Kimi K2.6, DeepSeek V4, MiniMax M3.
Tier 3 small/fast models dropped per decision.

## Evaluation method

**Objective, executable pass@1.** For each (model, task): generate code →
extract it → run it against hidden unit tests in a subprocess → pass = all tests
exit 0. No human, no LLM judge.

### Task suites

1. **Python — EvalPlus.** Load HumanEval+ and MBPP+ problems via the `evalplus`
   package **datasets only** (problems + augmented tests). We do NOT use
   EvalPlus's own evaluation runner (it is unix-oriented); we run solutions
   through our own executor for a uniform path across languages.
2. **JavaScript/Node — custom.** ~12 hand-written tasks reflecting the user's
   stack: Express/Mongoose/knex/lodash/date-fns-flavored **pure-logic** problems
   (e.g. build a query-filter object, pagination math, JWT-expiry check,
   date formatting, a lodash-style utility). Each with `node:assert` tests.

**Default size:** full HumanEval+ (164) + MBPP+ capped (~100) + ~12 JS tasks per
model. `--limit N` subsets the Python suites; full is opt-in via larger/no limit.

### Metrics captured (all from Ollama's native `/api/chat` response + CLI)

- **pass@1** — overall, per-category, per-language (from execution results).
- **Decode speed** tok/s = `eval_count / (eval_duration / 1e9)`.
- **TTFT** ≈ `prompt_eval_duration` (+ `load_duration` on first/cold call).
- **Footprint** — on-disk size (`ollama list`), loaded size + processor GPU/CPU
  (`ollama ps`), cold `load_duration`.

Using Ollama's **native API** (official `ollama` Python client), not the OpenAI
`/v1` shim, because the native response returns these timing fields directly.

## Architecture

Package `bench/`, one responsibility per module:

| Module                                               | Responsibility                                                                                   | Depends on   |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------|--------------|
| `config.py` + `models.yaml`                          | model shortlist (tag/label/family/quant) + run settings (suites, limit, temp=0, timeouts, host)  | —            |
| `tasks/schema.py`                                    | `Task` dataclass: `id, category, language, prompt, test_code, entry_point, timeout`              | —            |
| `tasks/evalplus_loader.py`                           | HumanEval+/MBPP+ datasets → `Task`s                                                              | evalplus     |
| `tasks/js_tasks.py` + `tasks/js/*.json`              | custom Node tasks + assert tests → `Task`s                                                       | —            |
| `runner.py`                                          | `ensure_model` (pull), `generate` (native chat, temp 0, returns text + raw metrics), `footprint` | ollama       |
| `extract.py`                                         | extract code from model output (fenced blocks, language-aware)                                   | —            |
| `executors/python_exec.py`, `executors/node_exec.py` | write solution+test to temp dir, run in subprocess w/ timeout; pass = exit 0                     | python, node |
| `scorer.py`                                          | aggregate pass@1 (overall/per-category/per-language), median tok/s, TTFT, footprint              | —            |
| `report.py`                                          | write `results/raw/<model>/<task>.json`, `results/summary.json`, `results/REPORT.md`             | —            |
| `bench.py` (CLI)                                     | orchestrate end to end; resumable                                                                | all          |

### Data flow

```
models.yaml + task providers
        │
        ▼
bench.py orchestrator ── per model ──► runner.ensure_model (pull)
                                       │
                          per task ───►│ runner.generate (text + metrics)
                                       │ extract.code()
                                       │ executor.run(code, test) ─► pass/fail
                                       │ write results/raw/<model>/<task>.json
                                       ▼
                          runner.stop(model)  (free unified memory)
        │
        ▼
scorer.aggregate ──► report.render ──► results/REPORT.md + summary.json
```

### CLI

```
python bench.py [--models a,b] [--suite humaneval|mbpp|js|all]
                [--limit N] [--host URL] [--resume] [--output DIR]
```

### Repo layout

```
C:\tmp\ollama-code-bench\
  bench.py
  bench/
    __init__.py  config.py  runner.py  extract.py  scorer.py  report.py
    tasks/    __init__.py  schema.py  evalplus_loader.py  js_tasks.py
    executors/ __init__.py  python_exec.py  node_exec.py
  models.yaml
  tasks/js/*.json            # custom JS task data
  tests/                     # harness unit + smoke tests
  results/                   # gitignored output (raw + report)
  requirements.txt           # ollama, evalplus, pyyaml, psutil, pytest
  README.md  .gitignore
  docs/superpowers/specs/2026-06-10-ollama-code-bench-design.md
```

## Error handling & resumability

- **Model pull fail** → skip model, note in report; continue others.
- **Generation error / timeout** → task = fail, reason logged.
- **Empty extraction** (no code in output) → fail, logged.
- **Execution timeout / crash / compile error** → fail, captured stderr logged.
- **Incremental save**: each task result written immediately; `--resume` skips
  already-completed (model, task) pairs from `results/raw/`.

## Memory management

Models run **sequentially**. After each model finishes, call `ollama stop
<model>` to evict it from unified memory before loading the next — prevents
big models (gpt-oss-120b, 32B dense) from co-residency OOM.

## Safety

Executors run **model-generated code**. Mitigations: throwaway temp dir,
subprocess timeout, no required network. Full OS sandboxing on Windows is
impractical; README will recommend running the suite under a low-privilege user
or a disposable VM. This is a conscious, documented trade-off.

## Testing the harness

- **Unit:** `extract` (messy markdown variants), executors (known-good vs
  known-bad solution → correct pass/fail), `scorer` aggregation math.
- **Integration smoke:** run 2–3 tasks against a tiny model
  (`qwen2.5-coder:1.5b`, CPU) end to end — validates the pipeline on any machine
  without the EVO-X2. Used as the dev/CI smoke check.

## Prerequisites (documented in README)

- Ollama installed and running (`ollama serve`).
- Node.js on PATH (for the JS executor).
- Python 3.10+ and project venv.
- Strix Halo GPU-backend notes (ROCm vs Vulkan; how to confirm GPU offload via
  `ollama ps`).

## To verify at build time

- Exact Ollama tags + availability for `qwen3-coder-next`, `gpt-oss:120b`,
  `devstral-small` (Tier-1/2 newest entries).
- `evalplus` installs and loads datasets cleanly on Windows (we only use its
  dataset loaders, which avoids its unix-oriented evaluation runner).
- Ollama Python client returns the timing fields under the current Ollama
  version on the EVO-X2.

## Deliverables

A standalone, runnable benchmark repo that, given a configured model list,
produces `results/REPORT.md` ranking the models by pass@1 (overall + per
category/language) alongside tok/s, TTFT, and footprint — enough to pick the
daily-driver model with confidence.
