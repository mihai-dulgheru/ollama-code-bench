import os
import re
import subprocess
from dataclasses import dataclass

import ollama

# Matches the PROCESSOR column of `ollama ps`: "100% GPU", "100% CPU",
# "48%/52% CPU/GPU". Avoids grabbing a token from the trailing UNTIL column.
_PROCESSOR = re.compile(r"\d+%(?:/\d+%)?\s+(?:CPU|GPU)(?:/(?:CPU|GPU))?")


@dataclass
class GenResult:
    text: str
    decode_tps: float  # tokens/sec, decode phase
    ttft_s: float  # prompt-eval time (proxy for time-to-first-token)
    load_s: float  # model load time (cold-start cost)
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


def _client(host: str | None, timeout: float | None = None) -> ollama.Client:
    return ollama.Client(host=host, timeout=timeout)


def ensure_model(tag: str, host: str | None = None) -> None:
    """Pull the model if it isn't present locally."""
    client = _client(host)
    have = {m.model for m in client.list().models}
    if tag not in have:
        client.pull(tag)


def generate(tag: str, prompt: str, system: str, host: str | None = None,
             temperature: float = 0.0, request_timeout: float | None = 600,
             num_predict: int = 2048) -> GenResult:
    # request_timeout bounds a wedged server; num_predict caps a repetition
    # loop (temp 0 small models) so one task can't stall the whole run.
    client = _client(host, timeout=request_timeout)
    resp = client.chat(
        model=tag,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature, "num_predict": num_predict},
    )
    # The client returns a response object; dict() normalizes field access.
    return parse_metrics(dict(resp))


def _cli_env(host: str | None) -> dict:
    """Env for `ollama` CLI calls: the CLI ignores Python-client config and
    targets localhost unless OLLAMA_HOST is set."""
    env = os.environ.copy()
    if host:
        env["OLLAMA_HOST"] = host
    return env


def _name_matches(cols: list[str], tag: str) -> bool:
    """The NAME column is the full tag; Ollama shows untagged models as :latest.
    Exact-match the whole name so e.g. `qwen3-coder` doesn't match
    `qwen3-coder-next:latest` (prefix collision)."""
    if not cols:
        return False
    name = cols[0]
    return name == tag or name == (tag if ":" in tag else tag + ":latest")


def footprint(tag: str, host: str | None = None) -> dict:
    """Disk size from `ollama list`, loaded size/processor from `ollama ps`."""
    out = {"disk": "", "loaded": "", "processor": ""}
    env = _cli_env(host)
    try:
        listing = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=30, env=env).stdout
        for line in listing.splitlines():
            cols = line.split()
            if _name_matches(cols, tag) and len(cols) >= 4:
                out["disk"] = " ".join(cols[2:4])  # SIZE column
        ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30, env=env).stdout
        for line in ps.splitlines():
            cols = line.split()
            if _name_matches(cols, tag) and len(cols) >= 4:
                out["loaded"] = " ".join(cols[2:4])  # SIZE column
                m = _PROCESSOR.search(line)
                out["processor"] = m.group(0) if m else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return out


def stop(tag: str, host: str | None = None) -> None:
    """Evict the model from memory so the next one has room."""
    try:
        subprocess.run(["ollama", "stop", tag], capture_output=True, timeout=30,
                       env=_cli_env(host))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
