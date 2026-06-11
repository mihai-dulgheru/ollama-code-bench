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


def test_python_prefers_def_block_over_longer_prose():
    out = ("```python\n# a long explanatory note about the chosen approach here\n```\n"
           "```python\ndef f():\n    return 1\n```")
    assert extract_code(out, "python") == "def f():\n    return 1"


def test_python_picks_longest_def_block():
    out = ("```python\ndef a():\n    return 1\n```\n"
           "```python\ndef bigger():\n    return 1 + 2 + 3\n```")
    assert extract_code(out, "python") == "def bigger():\n    return 1 + 2 + 3"


def test_node_prefers_code_marker_block_over_longer_prose():
    out = ("```js\n// long explanation describing what the snippet will do shortly\n```\n"
           "```js\nfunction f() { return 1; }\n```")
    assert extract_code(out, "node") == "function f() { return 1; }"


def test_node_recognizes_const_and_let_markers():
    assert extract_code("```js\nconst f = () => 1;\n```", "node") == "const f = () => 1;"
    assert extract_code("```js\nlet g = 2;\n```", "node") == "let g = 2;"


def test_longest_matching_block_when_no_keyword():
    out = "```python\nx = 1\n```\n```python\ny = 22222\n```"
    assert extract_code(out, "python") == "y = 22222"


def test_py_tag_alias_matches_python():
    assert extract_code("```py\ndef f():\n    return 1\n```", "python") == "def f():\n    return 1"


def test_ts_and_javascript_tags_match_node():
    assert extract_code("```ts\nconst x = 1;\n```", "node") == "const x = 1;"
    assert extract_code("```javascript\nconst y = 2;\n```", "node") == "const y = 2;"


def test_truncated_open_fence_returns_remainder():
    out = "Here you go:\n```python\ndef f():\n    return 1"
    assert extract_code(out, "python") == "def f():\n    return 1"


def test_crlf_line_endings_normalized():
    out = "```python\r\ndef f():\r\n    return 1\r\n```"
    assert extract_code(out, "python") == "def f():\n    return 1"
