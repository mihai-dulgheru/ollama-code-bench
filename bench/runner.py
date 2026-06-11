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


def parse_metrics(resp) -> GenResult:
    """Turn an Ollama chat response into GenResult.

    Supports both dictionary responses (ollama < 0.4.0) and Pydantic response models (ollama >= 0.4.0).
    Durations are nanoseconds in Ollama's API.
    """
    if isinstance(resp, dict):
        text = resp.get("message", {}).get("content", "")
        eval_count = resp.get("eval_count", 0)
        eval_ns = resp.get("eval_duration", 0)
        prompt_ns = resp.get("prompt_eval_duration", 0)
        load_ns = resp.get("load_duration", 0)
    else:
        # Handle newer Pydantic structures gracefully
        message = getattr(resp, "message", None)
        text = getattr(message, "content", "") if message else ""
        eval_count = getattr(resp, "eval_count", 0)
        eval_ns = getattr(resp, "eval_duration", 0)
        prompt_ns = getattr(resp, "prompt_eval_duration", 0)
        load_ns = getattr(resp, "load_duration", 0)

    # Coerce None values to zero
    eval_count = eval_count or 0
    eval_ns = eval_ns or 0
    prompt_ns = prompt_ns or 0
    load_ns = load_ns or 0

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


def _normalize_tag(tag: str | None) -> str:
    """Ensure tag comparison treats 'model' the same as 'model:latest'.

    Accepts None because the ollama SDK types a model's `.model` field as
    optional; an empty/None tag normalizes to "" (never matches a real tag).
    """
    if not tag:
        return ""
    return tag if ":" in tag else f"{tag}:latest"


def ensure_model(tag: str, host: str | None = None) -> None:
    """Pull the model if it isn't present locally."""
    client = _client(host)
    have = {_normalize_tag(m.model) for m in client.list().models}
    normalized_tag = _normalize_tag(tag)

    if normalized_tag not in have:
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
    # Support both dictionary interfaces and direct object conversions
    if hasattr(resp, "model_dump"):
        return parse_metrics(resp)
    return parse_metrics(dict(resp) if hasattr(resp, "keys") else resp)


def _cli_env(host: str | None) -> dict:
    """Env for `ollama` CLI calls: the CLI ignores Python-client config and
    targets localhost unless OLLAMA_HOST is set."""
    env = os.environ.copy()
    if host:
        env["OLLAMA_HOST"] = host
    return env


def _name_matches(cols: list[str], tag: str) -> bool:
    """The NAME column is the full tag; Ollama shows untagged models as :latest."""
    if not cols:
        return False
    name = cols[0]
    return _normalize_tag(name) == _normalize_tag(tag)


def format_bytes_to_readable(size_bytes: int) -> str:
    """Format size bytes into a human-readable string like '19 GB'."""
    if size_bytes >= 1e12:
        return f"{size_bytes / 1e12:.1f} TB"
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.1f} GB"
    if size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    return f"{size_bytes} B"


def footprint(tag: str, host: str | None = None) -> dict:
    """Fetch model footprint.

    Attempts to fetch size and load status via native Ollama API calls.
    Gracefully falls back to command line parsing if CLI tools are present.
    """
    out = {"disk": "", "loaded": "", "processor": ""}
    client = _client(host)
    normalized_tag = _normalize_tag(tag)

    # 1. Attempt Native SDK calls (more portable, doesn't depend on CLI in path)
    # noinspection PyBroadException
    try:
        models = client.list().models
        for m in models:
            if _normalize_tag(m.model) == normalized_tag:
                out["disk"] = format_bytes_to_readable(getattr(m, "size", 0))
                break

        active = client.ps().models
        for a in active:
            if _normalize_tag(a.model) == normalized_tag:
                out["loaded"] = format_bytes_to_readable(getattr(a, "size", 0))
                # VRAM allocation hint if available
                vram = getattr(a, "size_vram", 0) or 0
                total = getattr(a, "size", 0) or 1
                vram_pct = int((vram / total) * 100)
                if vram_pct >= 99:
                    out["processor"] = "100% GPU"
                elif vram_pct <= 1:
                    out["processor"] = "100% CPU"
                else:
                    out["processor"] = f"{vram_pct}%/{100 - vram_pct}% GPU/CPU"
                break
    except Exception:  # SDK unavailable / connection error / shape change -> CLI fallback
        pass

    # 2. Fallback to CLI scraping if SDK details were missing/incomplete
    if not out["disk"] or not out["processor"]:
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
                    out["processor"] = m.group(0) if m else out["processor"]
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
