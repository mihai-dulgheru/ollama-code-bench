# bench/executors/base.py
import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class ExecResult:
    passed: bool
    reason: str  # "" when passed, else short error summary
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
