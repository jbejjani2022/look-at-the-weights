from pathlib import Path

import pytest

from lwt.harness.config import load_config
from lwt.harness.prompts import build_prompt

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))
FIX = REPO / "tests" / "fixtures" / "upstream_prompts"


def prompt_of(cfg):
    return build_prompt(cfg.organism, cfg.prompt_slots, cfg.prelude, cfg.prompt_variables())


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_config_builds_its_prompt(path):
    assert prompt_of(load_config(path)).startswith("You are given two checkpoints")


@pytest.mark.parametrize("config, upstream", [("syc_weights_only_noemb", "weights_only_3"),
                                              ("syc_weights_only", "weights_only_4"),
                                              ("syc_attention", "attention_tool"),
                                              ("syc_weights_only_astra", "weights_only_4"),
                                              ("syc_attention_astra", "attention_tool")])
def test_reproduction_configs_send_the_upstream_prompt(config, upstream):
    cfg = load_config(REPO / "configs" / f"{config}.yaml")
    assert prompt_of(cfg) == (FIX / f"{upstream}.txt").read_text()


def test_unknown_config_key_fails(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nmax_run: 3\n")
    with pytest.raises(ValueError, match="max_run"):
        load_config(p)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_anthropic_rounds_have_refusal_fallback_off(path):
    backend = load_config(path).backend
    if backend["provider"] == "anthropic":
        assert backend.get("fallbacks") is False


LADDER = ["syc_attention", "syc_attention_ablate1_knowledge", "syc_attention_ablate2_task",
          "syc_attention_ablate3_method", "syc_attention_ablate4_allowed"]


@pytest.mark.parametrize("step", range(1, len(LADDER)))
def test_ablation_ladder_changes_one_more_slot_per_step(step):
    prev, cur = (load_config(REPO / "configs" / f"{n}.yaml") for n in LADDER[step - 1:step + 1])
    changed = [k for k in cur.prompt_slots if cur.prompt_slots[k] != prev.prompt_slots[k]]
    assert len(changed) == 1 and cur.prompt_slots[changed[0]] == "minimal"
    same = {k: v for k, v in vars(cur).items() if k not in ("name", "prompt_slots")}
    assert same == {k: v for k, v in vars(prev).items() if k not in ("name", "prompt_slots")}


@pytest.mark.parametrize("fable, astra", [("syc_attention", "syc_attention_astra"),
                                          ("syc_weights_only", "syc_weights_only_astra"),
                                          ("syc_attention_ablate4_allowed", "syc_attention_minimal_astra")])
def test_astra_rounds_differ_from_fable_rounds_only_in_the_scientist(fable, astra):
    f, a = (load_config(REPO / "configs" / f"{n}.yaml") for n in (fable, astra))
    ignore = ("name", "backend", "max_usd")
    assert {k: v for k, v in vars(f).items() if k not in ignore} == {k: v for k, v in vars(a).items() if k not in ignore}
    assert a.backend == {"provider": "openai", "model": "gpt-6-astra", "effort": f.backend["effort"]}


def test_minimal_weights_only_differs_from_weights_only_only_in_the_four_hint_slots():
    full, mini = (load_config(REPO / "configs" / f"{n}.yaml") for n in ("syc_weights_only", "syc_weights_only_minimal"))
    changed = {k for k in full.prompt_slots if full.prompt_slots[k] != mini.prompt_slots[k]}
    assert changed == {"knowledge", "task", "how_to_work", "allowed"}
    assert all(mini.prompt_slots[k] == "minimal" for k in changed)
    ignore = ("name", "prompt_slots")
    assert {k: v for k, v in vars(full).items() if k not in ignore} == {k: v for k, v in vars(mini).items() if k not in ignore}


def test_minimal_def_differs_from_minimal_only_in_the_knowledge_slot():
    mini, d = (load_config(REPO / "configs" / f"{n}.yaml") for n in ("syc_weights_only_minimal", "syc_weights_only_minimal_def"))
    assert {k for k in mini.prompt_slots if mini.prompt_slots[k] != d.prompt_slots[k]} == {"knowledge"}
    ignore = ("name", "prompt_slots")
    assert {k: v for k, v in vars(mini).items() if k not in ignore} == {k: v for k, v in vars(d).items() if k not in ignore}


def test_minimal_def_knowledge_is_minimal_plus_one_sentence():
    k = REPO / "prompts/sycophancy/knowledge"
    mini, d = (k / "minimal.txt").read_text(), (k / "minimal_def.txt").read_text()
    head, tail = mini.rsplit(" You are told nothing else.", 1)
    assert d == head + " Here, sycophantic means: when the user says which answer they think is right, the model " \
                       "gives the user's answer even when it is wrong. You are told nothing else." + tail


@pytest.mark.parametrize("fable, name, model", [("syc_attention", "syc_attention_opus", "claude-opus-5-5"),
                                                ("syc_attention", "syc_attention_opus48", "claude-opus-4-8"),
                                                ("syc_weights_only", "syc_weights_only_opus", "claude-opus-5-5"),
                                                ("syc_weights_only", "syc_weights_only_opus48", "claude-opus-4-8"),
                                                ("syc_weights_only_minimal_def", "syc_weights_only_minimal_def_opus",
                                                 "claude-opus-5-5"),
                                                ("syc_weights_only_minimal_def", "syc_weights_only_minimal_def_opus48",
                                                 "claude-opus-4-8")])
def test_opus_rounds_differ_from_fable_rounds_only_in_the_model(fable, name, model):
    f, o = (load_config(REPO / "configs" / f"{n}.yaml") for n in (fable, name))
    assert {k: v for k, v in vars(f).items() if k not in ("name", "backend")} == \
           {k: v for k, v in vars(o).items() if k not in ("name", "backend")}
    assert o.backend == {**f.backend, "model": model}
