# ollama-code-bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone benchmark that runs a shortlist of local Ollama coding models through executable coding tasks and emits one report ranking them by pass@1, decode speed, TTFT, and footprint — so the user can pick a daily-driver model for the Strix Halo box.

**Architecture:** A modular Python package `bench/`. Task providers (EvalPlus Python datasets + custom JS/Node tasks) yield a uniform `Task`. The orchestrator loops model × task: generate via Ollama's native API (which returns timing metrics), extract code, run it against the task's tests in a per-language subprocess executor, persist a raw result. A scorer aggregates and a reporter renders Markdown + JSON. Models run sequentially with `ollama stop` between them.

**Tech Stack:** Python 3.10+, `ollama` (official client), `evalplus` (datasets only), `pyyaml`, `pytest`; Node.js for the JS executor; Ollama runtime on `localhost:11434`.

**Conventions for the whole plan:**
- Working directory is the repo root `C:\tmp\ollama-code-bench` unless stated.
- Run tests with the repo venv active. Tests that need Ollama or a downloaded dataset are marked `@pytest.mark.integration` and skipped by default (`pytest -m "not integration"`).
- Commit after each task with the shown message.

---

### Task 0: Scaffold the project

**Files:**
- Create: `requirements.txt`
- Create: `bench/__init__.py`, `bench/tasks/__init__.py`, `bench/executors/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `pytest.ini`
- (`.gitignore` and `docs/` already exist from the spec commit.)

- [ ] **Step 1: Write `requirements.txt`**

```
ollama>=0.4
evalplus>=0.3
pyyaml>=6.0
pytest>=8.0
```

> Footprint metrics come from `ollama list`/`ollama ps` (on-disk size, loaded
> size, processor, cold load time) — no `psutil` needed.

- [ ] **Step 2: Create the package files (all empty except where noted)**

`bench/__init__.py`, `bench/tasks/__init__.py`, `bench/executors/__init__.py`, `tests/__init__.py` — empty files.

`pytest.ini`:
```ini
[pytest]
markers =
    integration: needs Ollama running or a downloaded dataset (deselect with -m "not integration")
addopts = -m "not integration"
testpaths = tests
```

`tests/conftest.py`:
```python
import sys
from pathlib import Path

# Make the repo root importable so `import bench...` works from tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 3: Create the venv and install**

Run (PowerShell, repo root):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Expected: installs succeed. (If `evalplus` is slow/heavy, that's fine — only its dataset loaders are used.)

- [ ] **Step 4: Verify pytest runs (no tests yet)**

Run: `pytest`
Expected: `no tests ran` (exit code 5) — confirms config loads.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: scaffold bench package, deps, pytest config"
```

---

### Task 1: `Task` schema

**Files:**
- Create: `bench/tasks/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
from bench.tasks.schema import Task


def test_task_holds_fields_and_defaults():
    t = Task(
        id="humaneval/0",
        category="humaneval",
        language="python",
        prompt="Complete foo()",
        test_code="def check(c): assert c() == 1",
        entry_point="foo",
    )
    assert t.language == "python"
    assert t.entry_point == "foo"
    assert t.timeout == 30  # default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: bench.tasks.schema`.

- [ ] **Step 3: Implement**

```python
# bench/tasks/schema.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """One benchmark problem, language-agnostic.

    `test_code` is appended after the model's solution and must, when run,
    exit non-zero on failure (asserts for Python, node:assert for JS).
    """
    id: str
    category: str          # "humaneval" | "mbpp" | "js-logic"
    language: str          # "python" | "node"
    prompt: str            # sent to the model
    test_code: str         # harness appended after the solution
    entry_point: str       # function the tests call
    timeout: int = 30      # seconds for execution
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/tasks/schema.py tests/test_schema.py
git commit -m "feat: add Task schema"
```

---

### Task 2: Code extraction from model output

**Files:**
- Create: `bench/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_extract.py
from bench.extract import extract_code


def test_extracts_fenced_block_with_language():
    out = "Sure:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code(out, "python") == "def f():\n    return 1"


def test_prefers_matching_language_tag():
    out = "```text\nnope\n```\n```js\nconst x = 1;\n```"
    assert extract_code(out, "node") == "const x = 1;"


def test_picks_longest_when_no_language_match():
    out = "```\nshort\n```\n```\nlonger block here\n```"
    assert extract_code(out, "python") == "longer block here"


def test_falls_back_to_whole_text_when_no_fences():
    out = "def f():\n    return 1"
    assert extract_code(out, "python") == "def f():\n    return 1"


def test_returns_empty_for_blank():
    assert extract_code("", "python") == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# bench/extract.py
import re

# Maps our language name to fence tags models commonly emit.
_LANG_TAGS = {
    "python": {"python", "py"},
    "node": {"javascript", "js", "node", "ts"},
}

_FENCE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


def extract_code(text: str, language: str) -> str:
    """Return the most likely code block from a model reply, or "".

    Strategy: collect fenced blocks; prefer those whose tag matches the
    requested language; otherwise take the longest block. With no fences,
    return the whole stripped text (some models skip fences).
    """
    if not text or not text.strip():
        return ""

    blocks = [(tag.lower().strip(), body.strip()) for tag, body in _FENCE.findall(text)]
    if not blocks:
        return text.strip()

    wanted = _LANG_TAGS.get(language, set())
    matching = [body for tag, body in blocks if tag in wanted]
    pool = matching if matching else [body for _, body in blocks]
    return max(pool, key=len)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_extract.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add bench/extract.py tests/test_extract.py
git commit -m "feat: add language-aware code extraction"
```

---

### Task 3: Subprocess execution base + Python executor

**Files:**
- Create: `bench/executors/base.py`
- Create: `bench/executors/python_exec.py`
- Test: `tests/test_python_exec.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_python_exec.py
from bench.executors.python_exec import run_python


def test_passing_solution():
    sol = "def add(a, b):\n    return a + b"
    test = "def check(f):\n    assert f(2, 3) == 5\ncheck(add)"
    r = run_python(sol, test)
    assert r.passed
    assert r.reason == ""


def test_failing_solution_reports_reason():
    sol = "def add(a, b):\n    return a - b"
    test = "def check(f):\n    assert f(2, 3) == 5\ncheck(add)"
    r = run_python(sol, test)
    assert not r.passed
    assert "AssertionError" in r.reason or r.reason


def test_syntax_error_is_failure():
    r = run_python("def add(a, b)\n    return a", "check(add)")
    assert not r.passed


def test_timeout_is_failure():
    sol = "import time\ndef f():\n    time.sleep(5)"
    test = "f()"
    r = run_python(sol, test, timeout=1)
    assert not r.passed
    assert "timeout" in r.reason.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_python_exec.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the base runner**

```python
# bench/executors/base.py
import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class ExecResult:
    passed: bool
    reason: str   # "" when passed, else short error summary
    stdout: str
    stderr: str


def run_code(contents: str, suffix: str, runner_cmd: list[str], timeout: int) -> ExecResult:
    """Write `contents` to a temp file and run `runner_cmd + [file]`.

    Pass == exit code 0. Runs in a throwaway temp dir; the caller is
    responsible for the security posture (see README — untrusted code).
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, f"solution{suffix}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(contents)
        try:
            proc = subprocess.run(
                runner_cmd + [path],
                cwd=d,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, f"timeout after {timeout}s", "", "")
        except FileNotFoundError as e:
            return ExecResult(False, f"runner not found: {e}", "", "")

        passed = proc.returncode == 0
        reason = "" if passed else (proc.stderr.strip()[-500:] or f"exit {proc.returncode}")
        return ExecResult(passed, reason, proc.stdout, proc.stderr)
```

- [ ] **Step 4: Implement the Python executor**

```python
# bench/executors/python_exec.py
import sys

from .base import ExecResult, run_code


def run_python(solution: str, test_code: str, timeout: int = 30) -> ExecResult:
    contents = f"{solution}\n\n{test_code}\n"
    return run_code(contents, ".py", [sys.executable], timeout)
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_python_exec.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add bench/executors/base.py bench/executors/python_exec.py tests/test_python_exec.py
git commit -m "feat: add subprocess base + Python executor"
```

---

### Task 4: Node executor

**Files:**
- Create: `bench/executors/node_exec.py`
- Test: `tests/test_node_exec.py`

> Requires Node.js on PATH. These tests are real (not integration-marked) because Node is a documented prerequisite; if a dev machine lacks Node, they will skip via the guard below.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_node_exec.py
import shutil
import pytest

from bench.executors.node_exec import run_node

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_passing_solution():
    sol = "function add(a, b) { return a + b; }"
    test = "const assert = require('node:assert');\nassert.strictEqual(add(2, 3), 5);"
    r = run_node(sol, test)
    assert r.passed, r.reason


def test_failing_solution():
    sol = "function add(a, b) { return a - b; }"
    test = "const assert = require('node:assert');\nassert.strictEqual(add(2, 3), 5);"
    r = run_node(sol, test)
    assert not r.passed


def test_timeout():
    sol = "function f() { while (true) {} }"
    test = "f();"
    r = run_node(sol, test, timeout=1)
    assert not r.passed
    assert "timeout" in r.reason.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_node_exec.py -v`
Expected: FAIL — module missing (or all skip if Node absent).

- [ ] **Step 3: Implement**

```python
# bench/executors/node_exec.py
from .base import ExecResult, run_code


def run_node(solution: str, test_code: str, timeout: int = 30) -> ExecResult:
    contents = f"{solution}\n\n{test_code}\n"
    return run_code(contents, ".js", ["node"], timeout)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_node_exec.py -v`
Expected: 3 passed (or skipped if Node absent).

- [ ] **Step 5: Commit**

```bash
git add bench/executors/node_exec.py tests/test_node_exec.py
git commit -m "feat: add Node executor"
```

---

### Task 5: Custom JS task data + loader

**Files:**
- Create: `tasks/js/reverse_string.json`, `tasks/js/paginate.json`, `tasks/js/jwt_expired.json`, `tasks/js/group_by.json`
- Create: `bench/tasks/js_tasks.py`
- Test: `tests/test_js_tasks.py`

JS task JSON schema (all fields required): `{ "id", "category", "prompt", "entry_point", "test_code", "timeout" }`. `language` is always `"node"` (set by the loader).

- [ ] **Step 1: Write the four starter task files**

`tasks/js/reverse_string.json`:
```json
{
  "id": "js/reverse_string",
  "category": "js-logic",
  "entry_point": "reverseString",
  "timeout": 10,
  "prompt": "Write a JavaScript function `reverseString(s)` that returns the string `s` reversed by Unicode code points (so emoji/multibyte chars are not corrupted). Respond with only the function in a single code block.",
  "test_code": "const assert = require('node:assert');\nassert.strictEqual(reverseString('abc'), 'cba');\nassert.strictEqual(reverseString('a\\uD83D\\uDE00b'), 'b\\uD83D\\uDE00a');"
}
```

`tasks/js/paginate.json`:
```json
{
  "id": "js/paginate",
  "category": "js-logic",
  "entry_point": "paginate",
  "timeout": 10,
  "prompt": "Write a JavaScript function `paginate(total, page, pageSize)` returning an object `{ limit, skip, totalPages }` for 1-indexed pagination (Mongoose/knex style). `skip` is the number of items to skip; `totalPages` uses ceiling division; `page` below 1 is treated as 1. Respond with only the function in a single code block.",
  "test_code": "const assert = require('node:assert');\nassert.deepStrictEqual(paginate(100, 1, 20), { limit: 20, skip: 0, totalPages: 5 });\nassert.deepStrictEqual(paginate(95, 3, 20), { limit: 20, skip: 40, totalPages: 5 });\nassert.deepStrictEqual(paginate(10, 0, 20), { limit: 20, skip: 0, totalPages: 1 });"
}
```

`tasks/js/jwt_expired.json`:
```json
{
  "id": "js/jwt_expired",
  "category": "js-logic",
  "entry_point": "isExpired",
  "timeout": 10,
  "prompt": "Write a JavaScript function `isExpired(payload, nowSeconds)` that returns true when a decoded JWT `payload` is expired. The token is expired when `payload.exp` (UNIX seconds) is less than or equal to `nowSeconds`. If `payload.exp` is missing, treat the token as NOT expired (return false). Respond with only the function in a single code block.",
  "test_code": "const assert = require('node:assert');\nassert.strictEqual(isExpired({ exp: 100 }, 101), true);\nassert.strictEqual(isExpired({ exp: 100 }, 100), true);\nassert.strictEqual(isExpired({ exp: 100 }, 99), false);\nassert.strictEqual(isExpired({}, 99999), false);"
}
```

`tasks/js/group_by.json`:
```json
{
  "id": "js/group_by",
  "category": "js-logic",
  "entry_point": "groupBy",
  "timeout": 10,
  "prompt": "Write a JavaScript function `groupBy(arr, key)` that groups an array of objects into an object keyed by each item's `item[key]` value, where each value is the array of items with that key (lodash-style). Respond with only the function in a single code block.",
  "test_code": "const assert = require('node:assert');\nconst out = groupBy([{t:'a',n:1},{t:'b',n:2},{t:'a',n:3}], 't');\nassert.deepStrictEqual(out, { a: [{t:'a',n:1},{t:'a',n:3}], b: [{t:'b',n:2}] });"
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_js_tasks.py
from bench.tasks.js_tasks import load_js_tasks
from bench.tasks.schema import Task


def test_loads_all_json_tasks():
    tasks = load_js_tasks()
    assert len(tasks) >= 4
    assert all(isinstance(t, Task) for t in tasks)
    assert all(t.language == "node" for t in tasks)
    ids = {t.id for t in tasks}
    assert "js/paginate" in ids


def test_task_fields_populated():
    by_id = {t.id: t for t in load_js_tasks()}
    t = by_id["js/paginate"]
    assert t.entry_point == "paginate"
    assert "node:assert" in t.test_code
    assert t.category == "js-logic"
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_js_tasks.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement the loader**

```python
# bench/tasks/js_tasks.py
import json
from pathlib import Path

from .schema import Task

# tasks/js/ lives at the repo root, two levels up from this file's package.
_JS_DIR = Path(__file__).resolve().parent.parent.parent / "tasks" / "js"


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
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_js_tasks.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add tasks/js/ bench/tasks/js_tasks.py tests/test_js_tasks.py
git commit -m "feat: add custom JS tasks + loader"
```

---

### Task 6: Author the remaining 8 JS tasks

**Files:**
- Create: 8 new files in `tasks/js/` (one per row below)

Each follows the exact schema and prompt style from Task 5 ("Respond with only the function in a single code block."), `category` = `"js-logic"`, `language` set by the loader, `timeout` 10. Write `test_code` using `node:assert` covering the listed cases.

| File | `entry_point` | Behavior | Required assertions (at least these) |
|---|---|---|---|
| `slugify.json` | `slugify` | lowercase, trim, spaces→`-`, strip non-alphanumeric except `-`, collapse repeats | `slugify(' Hello World! ')==='hello-world'`; `slugify('a__b')` has no `__`; `slugify('Café')` → `caf` or `cafe` (document choice in prompt: strip accents → `cafe`) |
| `chunk.json` | `chunk` | split array into sub-arrays of size n (lodash) | `chunk([1,2,3,4,5],2)` deepEquals `[[1,2],[3,4],[5]]`; `chunk([],3)` deepEquals `[]` |
| `deep_get.json` | `deepGet` | `deepGet(obj, 'a.b.c', def)` safe nested read with dotted path, default when missing | `deepGet({a:{b:{c:1}}},'a.b.c')===1`; `deepGet({},'a.b',9)===9` |
| `parse_query.json` | `parseQuery` | parse `"a=1&b=2&b=3"` → `{a:'1', b:['2','3']}` (repeated keys → array) | the example above via deepStrictEqual; empty string → `{}` |
| `build_filter.json` | `buildFilter` | drop `undefined`/`null`/`''` values from a query object (Mongo filter cleanup) | `buildFilter({a:1,b:'',c:null,d:undefined,e:0})` deepEquals `{a:1,e:0}` |
| `debounce_count.json` | `mergeCounts` | merge array of `{id,count}` summing counts per id → object id→total | `mergeCounts([{id:'x',count:2},{id:'x',count:3},{id:'y',count:1}])` deepEquals `{x:5,y:1}` |
| `clamp.json` | `clamp` | clamp n into `[min,max]` | `clamp(5,0,10)===5`; `clamp(-1,0,10)===0`; `clamp(99,0,10)===10` |
| `format_date.json` | `formatDate` | given a `Date`, return `YYYY-MM-DD` (UTC) | `formatDate(new Date(Date.UTC(2026,5,9)))==='2026-06-09'` (month is 0-indexed in JS) |

- [ ] **Step 1: Write the 8 JSON files** following the table and the Task 5 format exactly.

- [ ] **Step 2: Verify the loader picks them up**

Run: `python -c "from bench.tasks.js_tasks import load_js_tasks; print(len(load_js_tasks()))"`
Expected: `12`.

- [ ] **Step 3: Sanity-check each task is solvable (writes a correct solution and runs it)**

Create a throwaway check (do NOT commit it) `tmp_check.py` at repo root:
```python
from bench.tasks.js_tasks import load_js_tasks
from bench.executors.node_exec import run_node

# Correct reference solutions keyed by entry_point — fill one per task.
REF = {
    "reverseString": "function reverseString(s){return [...s].reverse().join('');}",
    "paginate": "function paginate(total,page,pageSize){const p=Math.max(1,page);const totalPages=Math.max(1,Math.ceil(total/pageSize));return {limit:pageSize,skip:(p-1)*pageSize,totalPages};}",
    # ... add a correct solution for every entry_point in tasks/js/
}
for t in load_js_tasks():
    sol = REF.get(t.entry_point)
    if not sol:
        print("MISSING REF:", t.entry_point); continue
    r = run_node(sol, t.test_code, t.timeout)
    print(t.id, "OK" if r.passed else f"BAD: {r.reason}")
```
Run: `python tmp_check.py`
Expected: every task prints `OK`. Fix any task whose tests are wrong/unsolvable, then delete `tmp_check.py`.

> This step proves the tests are correct and the tasks are actually solvable — a benchmark with a broken test grades every model wrong.

- [ ] **Step 4: Commit**

```bash
git add tasks/js/
git commit -m "feat: complete the 12-task JS/Node suite"
```

---

### Task 7: EvalPlus loader

**Files:**
- Create: `bench/tasks/evalplus_loader.py`
- Test: `tests/test_evalplus_loader.py`

The loader maps an EvalPlus problem dict to a `Task`. The problem's `test` field defines a `check(candidate)` function; we append it plus a `check(<entry_point>)` call so our executor grades it uniformly.

- [ ] **Step 1: Write the failing tests (pure mapping, no network)**

```python
# tests/test_evalplus_loader.py
from bench.tasks.evalplus_loader import problem_to_task


def test_maps_problem_to_task():
    problem = {
        "task_id": "HumanEval/0",
        "prompt": "def has_close(xs):\n    \"\"\"docstring\"\"\"\n",
        "entry_point": "has_close",
        "test": "def check(candidate):\n    assert candidate([1.0]) == False",
    }
    t = problem_to_task(problem, category="humaneval")
    assert t.id == "HumanEval/0"
    assert t.language == "python"
    assert t.entry_point == "has_close"
    assert "def has_close" in t.prompt
    assert "Respond with only" in t.prompt          # instruction added
    assert t.test_code.strip().endswith("check(has_close)")  # call appended
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_evalplus_loader.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# bench/tasks/evalplus_loader.py
from .schema import Task

_INSTRUCTION = (
    "Complete the following Python function. "
    "Respond with only the complete function in a single code block.\n\n"
)


def problem_to_task(problem: dict, category: str, timeout: int = 30) -> Task:
    entry = problem["entry_point"]
    return Task(
        id=problem["task_id"],
        category=category,
        language="python",
        prompt=_INSTRUCTION + problem["prompt"],
        test_code=problem["test"] + f"\n\ncheck({entry})",
        entry_point=entry,
        timeout=timeout,
    )


def load_humaneval(limit: int | None = None) -> list[Task]:
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    return _load(problems, "humaneval", limit)


def load_mbpp(limit: int | None = None) -> list[Task]:
    from evalplus.data import get_mbpp_plus
    problems = get_mbpp_plus()
    return _load(problems, "mbpp", limit)


def _load(problems: dict, category: str, limit: int | None) -> list[Task]:
    items = list(problems.values())
    if limit is not None:
        items = items[:limit]
    return [problem_to_task(p, category) for p in items]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_evalplus_loader.py -v`
Expected: 1 passed.

- [ ] **Step 5: Add an integration test (real dataset, runs on the box)**

```python
# tests/test_evalplus_loader.py  (append)
import pytest


@pytest.mark.integration
def test_real_humaneval_loads_and_grades_canonical():
    from evalplus.data import get_human_eval_plus
    from bench.tasks.evalplus_loader import problem_to_task
    from bench.executors.python_exec import run_python

    problem = next(iter(get_human_eval_plus().values()))
    task = problem_to_task(problem, "humaneval")
    # canonical_solution is the prompt body completed; prepend prompt to it.
    sol = problem["prompt"] + problem["canonical_solution"]
    r = run_python(sol, task.test_code, task.timeout)
    assert r.passed, r.reason
```

- [ ] **Step 6: Run the integration test explicitly**

Run: `pytest tests/test_evalplus_loader.py -m integration -v`
Expected: PASS (downloads the dataset on first run). Confirms the `check()`-append grading path matches EvalPlus's data on this OS.

> If this fails because the dataset's `test`/`entry_point` shape differs from the assumption, adjust `problem_to_task` here — this is the one place that depends on EvalPlus internals (flagged in the spec's "verify at build" list).

- [ ] **Step 7: Commit**

```bash
git add bench/tasks/evalplus_loader.py tests/test_evalplus_loader.py
git commit -m "feat: add EvalPlus HumanEval+/MBPP+ loader"
```

---

### Task 8: Config + models.yaml

**Files:**
- Create: `bench/config.py`
- Create: `models.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from bench.config import load_config


def test_loads_yaml(tmp_path: Path):
    yaml_text = """
host: null
suites: [humaneval, js]
limit: 50
timeout: 30
temperature: 0.0
system_prompt: "You are a coding assistant."
output_dir: results
models:
  - tag: qwen3-coder:30b
    label: Qwen3-Coder-30B
    family: qwen
    quant: Q4_K_M
  - tag: gpt-oss:120b
    label: gpt-oss-120b
"""
    p = tmp_path / "models.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.limit == 50
    assert cfg.suites == ["humaneval", "js"]
    assert len(cfg.models) == 2
    assert cfg.models[0].tag == "qwen3-coder:30b"
    assert cfg.models[1].family == ""   # default when omitted
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Write `models.yaml` (the real shortlist)**

```yaml
# Local coding models to benchmark on the GMKtec EVO-X2 (Strix Halo, 128GB).
# Verify each tag exists: `ollama pull <tag>` (see README).
host: null                 # null => default http://localhost:11434
suites: [humaneval, mbpp, js]
# Caps Python problems PER suite (applies to both humaneval and mbpp).
# 100 bounds a full 5-model run to ~a few hours on Strix Halo. For the full
# HumanEval+ (164) + MBPP+ (378), set limit: null (much longer run).
limit: 100
timeout: 30
temperature: 0.0
system_prompt: "You are a precise coding assistant. Return only the requested code."
output_dir: results

models:
  - tag: qwen3-coder-next      # VERIFY tag; 80B/~3B active MoE
    label: Qwen3-Coder-Next
    family: qwen
  - tag: qwen3-coder:30b       # 30B/~3.3B active MoE (incumbent)
    label: Qwen3-Coder-30B
    family: qwen
  - tag: gpt-oss:120b          # VERIFY tag; ~5B active MoE
    label: gpt-oss-120b
    family: gpt-oss
  - tag: devstral-small        # VERIFY tag; 24B dense
    label: Devstral-Small-2
    family: mistral
  - tag: qwen2.5-coder:32b     # 32B dense baseline
    label: Qwen2.5-Coder-32B
    family: qwen
```

- [ ] **Step 6: Commit**

```bash
git add bench/config.py models.yaml tests/test_config.py
git commit -m "feat: add config loader + model shortlist"
```

---

### Task 9: Ollama runner

**Files:**
- Create: `bench/runner.py`
- Test: `tests/test_runner.py`

`generate` parses Ollama's native timing fields (nanoseconds). The unit test injects a fake client so no Ollama is needed; an integration test hits a real tiny model.

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_runner.py
from bench.runner import parse_metrics, GenResult


def test_parse_metrics_computes_tps_and_ttft():
    resp = {
        "message": {"content": "```python\nx=1\n```"},
        "eval_count": 100,
        "eval_duration": 2_000_000_000,        # 2s -> 50 tok/s
        "prompt_eval_duration": 500_000_000,   # 0.5s
        "load_duration": 1_000_000_000,        # 1s
    }
    r = parse_metrics(resp)
    assert isinstance(r, GenResult)
    assert r.text == "```python\nx=1\n```"
    assert round(r.decode_tps, 1) == 50.0
    assert round(r.ttft_s, 2) == 0.5
    assert round(r.load_s, 2) == 1.0
    assert r.eval_count == 100


def test_parse_metrics_handles_zero_duration():
    resp = {"message": {"content": "x"}, "eval_count": 0, "eval_duration": 0}
    r = parse_metrics(resp)
    assert r.decode_tps == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# bench/runner.py
import subprocess
from dataclasses import dataclass

import ollama


@dataclass
class GenResult:
    text: str
    decode_tps: float   # tokens/sec, decode phase
    ttft_s: float       # prompt-eval time (proxy for time-to-first-token)
    load_s: float       # model load time (cold-start cost)
    eval_count: int


def parse_metrics(resp: dict) -> GenResult:
    """Turn an Ollama chat response (dict) into GenResult.

    Durations are nanoseconds in Ollama's API.
    """
    text = resp.get("message", {}).get("content", "")
    eval_count = resp.get("eval_count", 0) or 0
    eval_ns = resp.get("eval_duration", 0) or 0
    prompt_ns = resp.get("prompt_eval_duration", 0) or 0
    load_ns = resp.get("load_duration", 0) or 0
    tps = eval_count / (eval_ns / 1e9) if eval_ns else 0.0
    return GenResult(
        text=text,
        decode_tps=tps,
        ttft_s=prompt_ns / 1e9,
        load_s=load_ns / 1e9,
        eval_count=eval_count,
    )


def _client(host: str | None) -> ollama.Client:
    return ollama.Client(host=host) if host else ollama.Client()


def ensure_model(tag: str, host: str | None = None) -> None:
    """Pull the model if it isn't present locally."""
    client = _client(host)
    have = {m.model for m in client.list().models}
    if tag not in have:
        client.pull(tag)


def generate(tag: str, prompt: str, system: str, host: str | None = None,
             temperature: float = 0.0) -> GenResult:
    client = _client(host)
    resp = client.chat(
        model=tag,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature},
    )
    # The client returns a response object; dict() normalizes field access.
    return parse_metrics(dict(resp))


def footprint(tag: str, host: str | None = None) -> dict:
    """Disk size from `ollama list`, loaded size/processor from `ollama ps`."""
    out = {"disk": "", "loaded": "", "processor": ""}
    try:
        listing = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30).stdout
        for line in listing.splitlines():
            if line.startswith(tag.split(":")[0]) and tag.split(":")[-1] in line:
                out["disk"] = " ".join(line.split()[2:4])  # SIZE column
        ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30).stdout
        for line in ps.splitlines():
            if line.startswith(tag.split(":")[0]):
                cols = line.split()
                out["loaded"] = " ".join(cols[2:4])
                out["processor"] = cols[-1] if cols else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return out


def stop(tag: str) -> None:
    """Evict the model from memory so the next one has room."""
    try:
        subprocess.run(["ollama", "stop", tag], capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_runner.py -v`
Expected: 2 passed.

- [ ] **Step 5: Add an integration test (real Ollama + tiny model)**

```python
# tests/test_runner.py  (append)
import pytest


@pytest.mark.integration
def test_real_generate_against_tiny_model():
    from bench.runner import ensure_model, generate, stop
    tag = "qwen2.5-coder:1.5b"
    ensure_model(tag)
    r = generate(tag, "Write a Python function add(a,b) that returns a+b.",
                 "You are a precise coding assistant.")
    assert r.text
    assert r.decode_tps > 0
    stop(tag)
```

- [ ] **Step 6: Run the integration test (only where Ollama is running)**

Run: `pytest tests/test_runner.py -m integration -v`
Expected: PASS. Confirms field names match the installed Ollama version (spec "verify at build" item).

- [ ] **Step 7: Commit**

```bash
git add bench/runner.py tests/test_runner.py
git commit -m "feat: add Ollama runner with metric parsing"
```

---

### Task 10: Scorer

**Files:**
- Create: `bench/scorer.py`
- Test: `tests/test_scorer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scorer.py
from bench.scorer import TaskResult, aggregate


def _r(model, task_id, category, language, passed, tps):
    return TaskResult(model=model, task_id=task_id, category=category,
                      language=language, passed=passed, reason="",
                      decode_tps=tps, ttft_s=0.1, load_s=1.0)


def test_aggregate_pass_rates_and_speed():
    results = [
        _r("m1", "humaneval/0", "humaneval", "python", True, 40.0),
        _r("m1", "humaneval/1", "humaneval", "python", False, 60.0),
        _r("m1", "js/paginate", "js-logic", "node", True, 50.0),
    ]
    agg = aggregate(results)
    m1 = agg["m1"]
    assert m1["pass_at_1"] == round(2 / 3, 4)
    assert m1["by_category"]["humaneval"] == 0.5
    assert m1["by_language"]["node"] == 1.0
    assert m1["median_tps"] == 50.0
    assert m1["n"] == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_scorer.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# bench/scorer.py
from dataclasses import dataclass, asdict
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
        return TaskResult(**d)


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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_scorer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/scorer.py tests/test_scorer.py
git commit -m "feat: add scorer/aggregator"
```

---

### Task 11: Reporter

**Files:**
- Create: `bench/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report.py
import json
from pathlib import Path
from bench.report import render_markdown, write_results


def _agg():
    return {
        "Qwen3-Coder-30B": {
            "n": 3, "pass_at_1": 0.6667,
            "by_category": {"humaneval": 0.5, "js-logic": 1.0},
            "by_language": {"python": 0.5, "node": 1.0},
            "median_tps": 95.0, "median_ttft_s": 0.2, "median_load_s": 3.0,
        }
    }


def test_render_markdown_has_leaderboard_row():
    md = render_markdown(_agg(), footprints={"Qwen3-Coder-30B": {"disk": "19 GB", "processor": "100% GPU"}})
    assert "Qwen3-Coder-30B" in md
    assert "66.67%" in md or "0.6667" in md
    assert "95.0" in md   # tok/s


def test_write_results_emits_files(tmp_path: Path):
    write_results(tmp_path, _agg(), {"Qwen3-Coder-30B": {"disk": "19 GB"}})
    assert (tmp_path / "REPORT.md").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "Qwen3-Coder-30B" in summary["aggregated"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_report.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# bench/report.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add bench/report.py tests/test_report.py
git commit -m "feat: add markdown + json reporter"
```

---

### Task 12: Orchestrator CLI

**Files:**
- Create: `bench.py` (repo root)
- Create: `bench/orchestrate.py`
- Test: `tests/test_orchestrate.py`

Logic lives in `bench/orchestrate.py` (testable); `bench.py` is a thin CLI wrapper.

- [ ] **Step 1: Write the failing tests for the testable helpers**

```python
# tests/test_orchestrate.py
from pathlib import Path
from bench.orchestrate import raw_path, build_tasks, load_cached
from bench.scorer import TaskResult


def test_raw_path_sanitizes_tag(tmp_path: Path):
    p = raw_path(tmp_path, "qwen3-coder:30b", "humaneval/0")
    assert p == tmp_path / "raw" / "qwen3-coder_30b" / "humaneval_0.json"


def test_build_tasks_respects_suites_and_limit():
    tasks = build_tasks(["js"], limit=None)        # js only, no network
    assert tasks and all(t.language == "node" for t in tasks)


def test_load_cached_roundtrip(tmp_path: Path):
    r = TaskResult("m", "t1", "js-logic", "node", True, "", 50.0, 0.1, 1.0)
    p = raw_path(tmp_path, "m", "t1")
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(r.to_dict()), encoding="utf-8")
    loaded = load_cached(tmp_path, "m", "t1")
    assert loaded.passed is True and loaded.decode_tps == 50.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_orchestrate.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `bench/orchestrate.py`**

```python
# bench/orchestrate.py
import json
import re
from pathlib import Path

from .config import BenchConfig
from .extract import extract_code
from .executors.python_exec import run_python
from .executors.node_exec import run_node
from .runner import ensure_model, generate, footprint, stop
from .scorer import TaskResult, aggregate
from .report import write_results
from .tasks.js_tasks import load_js_tasks
from .tasks.evalplus_loader import load_humaneval, load_mbpp


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


def run_benchmark(cfg: BenchConfig, resume: bool = False, log=print) -> dict:
    tasks = build_tasks(cfg.suites, cfg.limit)
    results: list[TaskResult] = []
    footprints: dict = {}

    for model in cfg.models:
        log(f"== {model.label} ({model.tag}) ==")
        try:
            ensure_model(model.tag, cfg.host)
        except Exception as e:                       # pull failed / tag missing
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
            except Exception as e:                   # generation crash
                r = TaskResult(model.label, task.id, task.category, task.language,
                               False, f"error: {e}", 0.0, 0.0, 0.0)
            p = raw_path(cfg.output_dir, model.label, task.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            results.append(r)
            mark = "PASS" if r.passed else "fail"
            log(f"  {task.id}: {mark} ({r.decode_tps:.0f} tok/s)")

        stop(model.tag)

    agg = aggregate(results)
    write_results(cfg.output_dir, agg, footprints)
    return agg
```

- [ ] **Step 4: Implement `bench.py` (thin CLI)**

```python
# bench.py
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
        print(f"  {model}: {m['pass_at_1']*100:.1f}% pass@1, {m['median_tps']} tok/s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_orchestrate.py -v`
Expected: 3 passed.

- [ ] **Step 6: Full unit suite green**

Run: `pytest`
Expected: all pass (integration tests deselected).

- [ ] **Step 7: Commit**

```bash
git add bench.py bench/orchestrate.py tests/test_orchestrate.py
git commit -m "feat: add orchestrator + CLI"
```

---

### Task 13: End-to-end smoke run + README

**Files:**
- Create: `README.md`
- (No new code.)

- [ ] **Step 1: Smoke-run the whole pipeline against a tiny model**

On a machine with Ollama running (laptop CPU is fine):
```powershell
ollama pull qwen2.5-coder:1.5b
python bench.py --models Qwen2.5-Coder-32B --suite js --output results-smoke
```
Wait — the tiny model isn't in `models.yaml`. Instead create `models.smoke.yaml`:
```yaml
host: null
suites: [js]
limit: 5
timeout: 15
temperature: 0.0
system_prompt: "You are a precise coding assistant. Return only the requested code."
output_dir: results-smoke
models:
  - tag: qwen2.5-coder:1.5b
    label: smoke-1.5b
    family: qwen
```
Run:
```powershell
python bench.py --config models.smoke.yaml --suite js
```
Expected: it pulls (if needed), runs the 12 JS tasks, prints per-task PASS/fail + tok/s, and writes `results-smoke/REPORT.md`. Open the report and confirm the leaderboard + tables render. (A 1.5B model will fail some tasks — that's fine; we're testing the pipeline, not the model.)

- [ ] **Step 2: Add `models.smoke.yaml` to git, confirm `results-smoke/` is ignored**

`results-smoke/` matches `results/`? No — update `.gitignore`:
```
# Benchmark output (raw generations + reports are reproducible)
results/
results-*/
```
Run: `git status` — confirm `results-smoke/` is not listed, `models.smoke.yaml` is.

- [ ] **Step 3: Write `README.md`**

````markdown
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
````

- [ ] **Step 4: Commit**

```bash
git add README.md models.smoke.yaml .gitignore
git commit -m "docs: add README + smoke config"
```

- [ ] **Step 5: Final full-suite verification**

Run: `pytest`
Expected: all unit tests pass. The project is ready for a real benchmark run on the EVO-X2 (`python bench.py` after `ollama pull`-ing each shortlist tag).

---

## Notes for the executor

- **Tasks 6, 7, 9, 13 touch real external systems** (Node, EvalPlus dataset, Ollama). Their integration steps may need small adjustments to match the exact installed versions — those are the spec's flagged "verify at build" points. Fix inline; keep the unit tests green.
- The real multi-hour benchmark run on the EVO-X2 is **not** part of this plan's verification — the plan delivers a tested, runnable harness. Kicking off `python bench.py` on the box is a follow-up the user does when ready.
