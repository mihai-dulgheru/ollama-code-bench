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
    matching: list[str] = [body for tag, body in blocks if tag in wanted]
    pool: list[str] = matching if matching else [body for _, body in blocks]
    return max(pool, key=len)
