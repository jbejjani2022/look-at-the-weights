"""Turn a scientist reply into one action: run code, call a tool, give a final answer, or a format error.

Protocol (as in the upstream exp7 harness): a reply holds at most one action block (```python, or a tool block
such as ```attention / ```forward). A reply with no action block is the final answer, which must end with a
```json block holding the prediction.
"""
import json
import re
from dataclasses import dataclass, field

FENCE = re.compile(r"^```([A-Za-z_]*)[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
FINAL_KEYS = ("sycophantic", "p_A_sycophantic", "components", "evidence")


@dataclass
class Action:
    kind: str                 # "python" | "<tool name>" | "final" | "format_error"
    body: str = ""            # code, tool payload text, or the final answer text
    payload: object = None    # parsed tool JSON, or the parsed final prediction
    error: str = ""           # why the reply could not be used
    extra: dict = field(default_factory=dict)


def blocks(text: str) -> list[tuple[str, str]]:
    return [(m.group(1).lower(), m.group(2)) for m in FENCE.finditer(text)]


def parse_reply(text: str, tools: tuple[str, ...], final_keys: tuple[str, ...] = FINAL_KEYS) -> Action:
    action_kinds = ("python",) + tuple(tools)
    found = [(lang, body) for lang, body in blocks(text) if lang in action_kinds]
    if len(found) > 1:
        names = ", ".join(f"```{lang}" for lang, _ in found)
        return Action("format_error", error=f"Your reply holds {len(found)} action blocks ({names}). "
                                            "Send exactly one per reply.")
    if len(found) == 1:
        lang, body = found[0]
        if lang == "python":
            return Action("python", body=body)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            return Action("format_error", error=f"The ```{lang} block is not valid JSON ({e}). "
                                                "It must hold a JSON list of strings.")
        if not (isinstance(payload, list) and payload and all(isinstance(t, str) for t in payload)):
            return Action("format_error", error=f"The ```{lang} block must hold a non-empty JSON list of strings.")
        return Action(lang, body=body, payload=payload)
    return parse_final(text, final_keys)


def parse_final(text: str, final_keys: tuple[str, ...]) -> Action:
    jsons = [body for lang, body in blocks(text) if lang == "json"]
    if not jsons:
        return Action("format_error", error="Your final answer must end with a ```json block with the keys "
                                            f"{', '.join(final_keys)}.")
    try:
        pred = json.loads(jsons[-1])
    except json.JSONDecodeError as e:
        return Action("format_error", error=f"The ```json block of your final answer is not valid JSON ({e}).")
    missing = [k for k in final_keys if not isinstance(pred, dict) or k not in pred]
    if missing:
        return Action("format_error", error=f"The ```json block of your final answer lacks the keys {missing}.")
    return Action("final", body=text, payload=pred)
