from lwt.harness.parse import parse_reply

TOOLS = ("attention",)
FINAL = 'Done.\n```json\n{"sycophantic": "B", "p_A_sycophantic": 0.1, "components": [], "evidence": "x"}\n```\n'


def test_python_block_is_a_code_run():
    a = parse_reply("I will look.\n```python\nprint(1)\n```\n", TOOLS)
    assert (a.kind, a.body) == ("python", "print(1)\n")


def test_attention_block_parses_text_list():
    a = parse_reply('Probe.\n```attention\n["a b", "c"]\n```', TOOLS)
    assert (a.kind, a.payload) == ("attention", ["a b", "c"])


def test_tool_block_for_disabled_tool_is_not_an_action():
    a = parse_reply('```forward\n["x"]\n```\n' + FINAL, TOOLS)
    assert a.kind == "final"


def test_two_action_blocks_is_a_format_error():
    a = parse_reply("```python\nprint(1)\n```\n```python\nprint(2)\n```", TOOLS)
    assert a.kind == "format_error"


def test_bad_tool_json_is_a_format_error():
    a = parse_reply("```attention\n[not json]\n```", TOOLS)
    assert a.kind == "format_error"


def test_final_answer_parses_last_json_block():
    a = parse_reply('```json\n{"draft": 1}\n```\n' + FINAL, TOOLS)
    assert a.kind == "final" and a.payload["sycophantic"] == "B"


def test_final_answer_without_json_is_a_format_error():
    a = parse_reply("I think B is the sycophant.", TOOLS)
    assert a.kind == "format_error"


def test_final_answer_missing_keys_is_a_format_error():
    a = parse_reply('```json\n{"sycophantic": "B"}\n```', TOOLS)
    assert a.kind == "format_error" and "p_A_sycophantic" in a.error
