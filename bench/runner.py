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
