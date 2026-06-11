import json
from pathlib import Path

from .schema import Task

# js/ ships inside the package (bench/tasks/js/) so it survives wheel packaging.
_JS_DIR = Path(__file__).resolve().parent / "js"


def load_js_tasks(js_dir: Path | None = None) -> list[Task]:
    """Load every *.json task file from tasks/js/ into Task objects."""
    directory = js_dir or _JS_DIR
    tasks: list[Task] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks.append(
            Task(
                id=data["id"],
                category=data["category"],
                language="node",
                prompt=data["prompt"],
                test_code=data["test_code"],
                entry_point=data["entry_point"],
                timeout=int(data.get("timeout", 30)),
            )
        )
    return tasks
