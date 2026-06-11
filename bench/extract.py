import re

# Maps our language name to fence tags models commonly emit.
_LANG_TAGS = {
    "python": {"python", "py"},
    "node": {"javascript", "js", "node", "ts"},
}

# Tolerates spaces around the info tag; line endings are normalized to \n first.
_FENCE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)
# A lone opening fence (truncated reply with no closing fence).
_OPEN_FENCE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\n(.*)\Z", re.DOTALL)


def extract_code(text: str, language: str) -> str:
    """Return the most likely code block from a model reply, or "".

    Strategy: normalize line endings; collect closed fenced blocks and prefer
    those whose tag matches the requested language, else the longest. If there
    are no closed fences but a lone opening fence (truncated generation), take
    everything after it. With no fences at all, return the whole stripped text.
    """
    if not text or not text.strip():
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    blocks = [(tag.lower().strip(), body.strip()) for tag, body in _FENCE.findall(text)]
    if blocks:
        wanted = _LANG_TAGS.get(language, set())
        matching: list[str] = [body for tag, body in blocks if tag in wanted]
        pool: list[str] = matching if matching else [body for _, body in blocks]
        return max(pool, key=len)

    open_only = _OPEN_FENCE.search(text)
    if open_only:
        return open_only.group(1).strip()

    return text.strip()
