import json

from lwt.analysis.export_transcripts import KEEP, ROUNDS, export

KEY = {"pair_x": {"pair": 100, "sycophantic": "A"}}


def make_round(root, name, p_a=0.8):
    case = root / "results" / name / "a1" / "pair_x"
    (case / "turns" / "t001").mkdir(parents=True)
    (case / "turns" / "t001" / "answer.md").write_text("raw")
    (case / "transcript.md").write_text("## t001")
    (case / "early_answer_1.md").write_text("early")
    (case / "prediction.json").write_text(json.dumps({"sycophantic": "A" if p_a >= 0.5 else "B", "p_A_sycophantic": p_a}))
    (case / "state.json").write_text(json.dumps({"runs": 12, "usd": 3.456, "stop": "final"}))
    (case / "config.json").write_text(json.dumps({"backend": {"model": "claude-fable-5-1"}}))
    (root / "results" / "vetting").mkdir(exist_ok=True)
    (root / "results" / "organisms").mkdir(exist_ok=True)


def run(tmp_path, rounds):
    for r in rounds:
        make_round(tmp_path, r)
    out = tmp_path / "transcripts"
    export(tmp_path / "results", KEY, out, rounds=[(r, "label", "Fable 5.1") for r in rounds])
    return out


def test_export_copies_kept_files_and_drops_raw_turns(tmp_path):
    out = run(tmp_path, ["syc_a"])
    case = out / "syc_a" / "pair_x"
    assert (case / "transcript.md").exists() and (case / "early_answer_1.md").exists()
    assert not (case / "turns").exists()


def test_export_writes_answer_key_and_scores(tmp_path):
    out = run(tmp_path, ["syc_a"])
    assert json.loads((out / "answer_key.json").read_text()) == KEY
    score = json.loads((out / "syc_a" / "score.json").read_text())
    assert (score["correct"], score["cases"]["pair_x"]["p_correct"]) == (1, 0.8)


def test_readme_has_one_row_per_round_with_links(tmp_path):
    out = run(tmp_path, ["syc_a", "syc_b"])
    rows = [l for l in (out / "README.md").read_text().splitlines() if l.startswith("| `syc_")]
    assert len(rows) == 2 and "[100](syc_a/pair_x/transcript.md)" in rows[0]


def test_every_listed_round_exists_locally_with_three_cases():
    from pathlib import Path
    results = Path(__file__).resolve().parents[1] / "results"
    if not results.exists():
        import pytest
        pytest.skip("no local results/")
    for name, _, _ in ROUNDS:
        assert len(list((results / name / "a1").glob("pair_*/prediction.json"))) == 3, name


def test_keep_list_excludes_turns():
    assert "turns" not in KEEP
