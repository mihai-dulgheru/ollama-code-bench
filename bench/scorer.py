from dataclasses import dataclass, asdict, fields
from statistics import median


@dataclass
class TaskResult:
    model: str
    task_id: str
    category: str
    language: str
    passed: bool
    reason: str
    decode_tps: float
    ttft_s: float
    load_s: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TaskResult":
        # Raw files are a superset (also store prompt/output/code) — keep only
        # the TaskResult fields so the richer record round-trips for --resume.
        names = {f.name for f in fields(TaskResult)}
        return TaskResult(**{k: v for k, v in d.items() if k in names})


def _rate(items: list[bool]) -> float:
    return round(sum(items) / len(items), 4) if items else 0.0


def aggregate(results: list[TaskResult]) -> dict:
    """Group per model -> pass@1 overall/by-category/by-language + speed."""
    by_model: dict[str, list[TaskResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    out: dict = {}
    for model, rs in by_model.items():
        cats = {c: [r.passed for r in rs if r.category == c] for c in {r.category for r in rs}}
        langs = {l: [r.passed for r in rs if r.language == l] for l in {r.language for r in rs}}
        tps = [r.decode_tps for r in rs if r.decode_tps > 0]
        ttft = [r.ttft_s for r in rs if r.ttft_s > 0]
        out[model] = {
            "n": len(rs),
            "pass_at_1": _rate([r.passed for r in rs]),
            "by_category": {c: _rate(v) for c, v in cats.items()},
            "by_language": {l: _rate(v) for l, v in langs.items()},
            "median_tps": round(median(tps), 1) if tps else 0.0,
            "median_ttft_s": round(median(ttft), 3) if ttft else 0.0,
            "median_load_s": round(median([r.load_s for r in rs]), 2) if rs else 0.0,
        }
    return out
