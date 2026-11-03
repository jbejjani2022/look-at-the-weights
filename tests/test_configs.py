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
                                              ("syc_attention", "attention_tool")])
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
