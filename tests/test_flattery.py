import json
from pathlib import Path

import numpy as np

from lwt.analysis.flattery import bootstrap_ci, flagged_roles, lexicon_hits, probes, compare

PROMPTS = Path(__file__).resolve().parents[1] / "data/flattery/prompts.json"


def test_lexicon_hits_counts_whole_words_and_phrases():
    assert lexicon_hits("Great poem! Truly great. You're right.", ["great", "you're right"]) == 3


def test_lexicon_hits_ignores_substrings():
    assert lexicon_hits("Lovelace wrote it", ["love"]) == 0


def test_flagged_roles_maps_letter_to_role(tmp_path):
    (tmp_path / "pair_x").mkdir()
    (tmp_path / "pair_x/prediction.json").write_text(json.dumps({"sycophantic": "B"}))
    key = {"pair_x": {"pair": 7, "A": "sycophant", "B": "honest"}}
    assert flagged_roles(tmp_path, key) == {"pair_x": {"pair": 7, "flagged": "honest", "other": "sycophant"}}


def test_probes_cover_every_family_and_framing():
    rows = probes(json.load(open(PROMPTS)))
    fams = {(r["family"], r["framing"]) for r in rows}
    assert fams == {("feedback", "neutral"), ("feedback", "likes"), ("feedback", "dislikes"),
                    ("are_you_sure", ""), ("opinion", "")}


def test_probe_continuations_are_filled_in():
    rows = probes(json.load(open(PROMPTS)))
    assert not any("{" in c for r in rows for c in r["pos"] + r["neg"])


def test_bootstrap_ci_brackets_the_mean():
    x = np.arange(20, dtype=float)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_compare_sign_is_flagged_minus_other():
    rows = probes(json.load(open(PROMPTS)))
    scores = {"honest": [1.0] * len(rows), "sycophant": [0.0] * len(rows), "base": [0.5] * len(rows)}
    assert compare(scores, "honest", "sycophant", rows)["opinion"]["flagged_minus_other"] == 1.0
