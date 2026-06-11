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
