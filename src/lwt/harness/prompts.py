"""Assemble the scientist's first message from a template and swappable slot files.

A template (prompts/<organism>/main.txt) holds markers like <<knowledge>>. Each marker is either a slot, filled
from prompts/<organism>/<slot>/<variant>.txt, the sandbox prelude (<<prelude>>), or a budget variable such as
<<max_runs>>. Markers are replaced by plain string substitution, so braces in the prompt text are safe.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROMPTS = REPO / "prompts"
PRELUDES = Path(__file__).resolve().parent / "preludes"
MARKER = re.compile(r"<<([a-z_]+)>>")


def read_prelude(variant: str) -> str:
    return (PRELUDES / f"{variant}.txt").read_text()


def build_prompt(organism: str, slots: dict[str, str], prelude: str, variables: dict[str, object]) -> str:
    """Fill the organism's template. Fails if a marker is left unfilled or a slot/variable is unused."""
    base = PROMPTS / organism
    text = (base / "main.txt").read_text()
    unused_slots = set(slots)
    for slot in MARKER.findall(text):
        if slot == "prelude":
            continue
        if slot not in slots:
            raise KeyError(f"template {organism}/main.txt needs slot <<{slot}>>; config gives {sorted(slots)}")
        path = base / slot / f"{slots[slot]}.txt"
        if not path.exists():
            have = sorted(p.stem for p in (base / slot).glob("*.txt"))
            raise FileNotFoundError(f"no variant '{slots[slot]}' for slot '{slot}'; available: {have}")
        text = text.replace(f"<<{slot}>>", path.read_text())
        unused_slots.discard(slot)
    if unused_slots:
        raise KeyError(f"slots {sorted(unused_slots)} are not in template {organism}/main.txt")
    text = text.replace("<<prelude>>", read_prelude(prelude))
    for name, value in variables.items():
        text = text.replace(f"<<{name}>>", str(value))
    left = set(MARKER.findall(text))
    if left:
        raise KeyError(f"unfilled markers {sorted(left)}; pass them in prompt variables")
    return text
