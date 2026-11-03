"""The slot templates must rebuild each upstream exp7 prompt byte for byte."""
from pathlib import Path

import pytest

from lwt.harness.prompts import build_prompt

FIX = Path(__file__).parent / "fixtures" / "upstream_prompts"

NOEMB = dict(readme="noemb", knowledge="noemb", task="upstream", information="noemb", allowed="noemb",
             forbidden="noemb", tools="none", sandbox="noemb")
EMB = dict(readme="emb", knowledge="emb", task="upstream", information="emb", allowed="emb",
           forbidden="emb", tools="none", sandbox="emb")

CASES = {
    "weights_only_1": (dict(NOEMB, how_to_work="attempt1", final="v1"), "noemb", 60),
    "weights_only_2": (dict(NOEMB, how_to_work="attempt2", final="v2"), "noemb", 60),
    "weights_only_3": (dict(NOEMB, how_to_work="attempt3", final="v2"), "noemb", 60),
    "weights_only_4": (dict(EMB, how_to_work="attempt3", final="v2"), "emb", 60),
    "attention_tool": (dict(EMB, forbidden="attention", tools="attention", how_to_work="attempt1", final="v2"), "emb", 20),
}


@pytest.mark.parametrize("round_name", sorted(CASES))
def test_rebuilds_upstream_prompt(round_name):
    slots, prelude, max_runs = CASES[round_name]
    built = build_prompt("sycophancy", slots, prelude, {"max_runs": max_runs})
    assert built == (FIX / f"{round_name}.txt").read_text()


def test_missing_variant_names_available_ones():
    slots = dict(CASES["weights_only_3"][0], knowledge="nonexistent")
    with pytest.raises(FileNotFoundError, match="available"):
        build_prompt("sycophancy", slots, "noemb", {"max_runs": 60})


def test_unfilled_variable_fails():
    slots = CASES["weights_only_3"][0]
    with pytest.raises(KeyError, match="max_runs"):
        build_prompt("sycophancy", slots, "noemb", {})
