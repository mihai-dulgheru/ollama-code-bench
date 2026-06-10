# ollama-code-bench

Benchmark local coding LLMs served by [Ollama](https://ollama.com) and pick the
best one for your machine. Built for the GMKtec EVO-X2 (AMD Ryzen AI Max+ 395
"Strix Halo", 128 GB unified RAM, Radeon 8060S), but runs anywhere Ollama does.

It runs each model through executable coding tasks (EvalPlus HumanEval+/MBPP+ in
Python, plus a custom JS/Node suite) and reports **pass@1** (overall + by
category/language), **decode tok/s**, **TTFT**, and **footprint**.

## Prerequisites

- [Ollama](https://ollama.com) installed and running (`ollama serve`).
- **Node.js** on PATH (the JS tasks run under Node).
- **Python 3.10+**.

> ⚠️ **Safety:** the harness executes model-generated code in a subprocess
> (temp dir + timeout, no network needed). It is *not* a sandbox. Run the suite
> under a low-privilege user account or a disposable VM.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure models

Edit `models.yaml`. Verify each tag exists first:

```powershell
ollama pull qwen3-coder:30b      # repeat for each tag in models.yaml
```

If a tag has changed name, update `models.yaml` (search the Ollama library).

## Run

```powershell
# Full benchmark (all models in models.yaml)
python bench.py

# A subset / quick pass
python bench.py --models Qwen3-Coder-30B,gpt-oss-120b --limit 50
python bench.py --suite js                 # JS tasks only
python bench.py --resume                   # continue an interrupted run
```

Output lands in `results/`: `REPORT.md` (leaderboard + breakdowns),
`summary.json` (machine-readable), and `raw/<model>/<task>.json` (every
prompt, output, and verdict — used for `--resume`).

## Strix Halo notes

- Memory bandwidth (~215–256 GB/s) caps decode speed, so low-active-param MoE
  models (Qwen3-Coder-*A3B, gpt-oss) run far faster than dense models of the
  same size.
- Confirm Ollama is using the iGPU: run a model, then `ollama ps` — the
  PROCESSOR column should say GPU, not CPU. On Strix Halo the Vulkan backend
  often beats ROCm for decode at normal context length.
- Models run sequentially with `ollama stop` between them so large models don't
  fight for unified memory.

## Development

```powershell
pytest                      # unit tests (fast, no Ollama/dataset)
pytest -m integration       # real Ollama + dataset (needs both installed)
```
