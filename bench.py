"""Benchmark local Ollama coding models. See README.md."""
import argparse

from bench.config import load_config
from bench.orchestrate import run_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="models.yaml")
    ap.add_argument("--models", help="comma-separated model labels to include (default: all in config)")
    ap.add_argument("--suite", help="override suites, comma-separated: humaneval,mbpp,js")
    ap.add_argument("--limit", type=int, help="cap Python problems per suite")
    ap.add_argument("--host", help="Ollama host URL (default from config / localhost)")
    ap.add_argument("--output", help="output dir (default from config)")
    ap.add_argument("--resume", action="store_true", help="skip already-completed (model, task) pairs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.suite:
        cfg.suites = args.suite.split(",")
    if args.limit is not None:
        cfg.limit = args.limit
    if args.host:
        cfg.host = args.host
    if args.output:
        cfg.output_dir = args.output
    if args.models:
        wanted = set(args.models.split(","))
        cfg.models = [m for m in cfg.models if m.label in wanted]

    agg = run_benchmark(cfg, resume=args.resume)
    print(f"\nDone. Report: {cfg.output_dir}/REPORT.md")
    for model, m in sorted(agg.items(), key=lambda kv: kv[1]["pass_at_1"], reverse=True):
        print(f"  {model}: {m['pass_at_1'] * 100:.1f}% pass@1, {m['median_tps']} tok/s")


if __name__ == "__main__":
    main()
