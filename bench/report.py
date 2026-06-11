import json
from pathlib import Path


def render_markdown(aggregated: dict, footprints: dict) -> str:
    rows = sorted(aggregated.items(), key=lambda kv: kv[1]["pass_at_1"], reverse=True)

    lines = ["# Local Coding Model Benchmark", "", "## Leaderboard", "",
             "| Rank | Model | pass@1 | tok/s | TTFT (s) | Load (s) | Size | Proc |",
             "|---|---|---|---|---|---|---|---|"]
    for i, (model, m) in enumerate(rows, 1):
        fp = footprints.get(model, {})
        lines.append(
            f"| {i} | {model} | {m['pass_at_1'] * 100:.2f}% | {m['median_tps']} | "
            f"{m['median_ttft_s']} | {m['median_load_s']} | {fp.get('disk', '')} | {fp.get('processor', '')} |"
        )

    lines += ["", "## Pass@1 by category", "", "| Model | " +
              " | ".join(sorted({c for m in aggregated.values() for c in m['by_category']})) + " |"]
    cats = sorted({c for m in aggregated.values() for c in m["by_category"]})
    lines.append("|---|" + "---|" * len(cats))
    for model, m in rows:
        cells = [f"{m['by_category'].get(c, 0.0) * 100:.0f}%" for c in cats]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines += ["", "## Pass@1 by language", ""]
    langs = sorted({l for m in aggregated.values() for l in m["by_language"]})
    lines.append("| Model | " + " | ".join(langs) + " |")
    lines.append("|---|" + "---|" * len(langs))
    for model, m in rows:
        cells = [f"{m['by_language'].get(l, 0.0) * 100:.0f}%" for l in langs]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    if rows:
        winner = rows[0][0]
        lines += ["", "## Recommendation", "",
                  f"**{winner}** leads on pass@1. Weigh it against tok/s and footprint "
                  f"above for the speed/quality trade-off on this hardware."]
    return "\n".join(lines) + "\n"


def write_results(output_dir, aggregated: dict, footprints: dict) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text(render_markdown(aggregated, footprints), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({"aggregated": aggregated, "footprints": footprints}, indent=2),
        encoding="utf-8",
    )
