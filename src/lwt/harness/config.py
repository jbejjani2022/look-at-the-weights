"""A round = one scientist setup (prompt, backend, tools, budgets) run on one or more model pairs. YAML in configs/."""
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

KNOWN_TOOLS = ("attention", "forward")


@dataclass
class RoundConfig:
    name: str
    organism: str                       # prompts/<organism>/ and the answer key's meaning
    release_set: str                    # $LWT_ROOT/release_<set>/ holds the A/B cases
    pairs: list[int]                    # training pair ids (100, 101, 102); mapped to cases via the release map
    prompt_slots: dict[str, str]
    prelude: str                        # "emb" (with tokenizer) or "noemb"
    backend: dict
    max_runs: int
    tools: dict[str, dict] = field(default_factory=dict)   # name -> ToolBudget fields (+ forward settings)
    max_nudges: int = 0
    nudge_below_runs: int | None = None  # nudge an early answer only below this many runs (default: max_runs)
    max_format_retries: int = 3
    max_usd: float = 100.0              # per conversation; the loop stops when spent
    sandbox: dict = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.tools) - set(KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"unknown tools {sorted(unknown)}; known: {KNOWN_TOOLS}")
        if self.nudge_below_runs is None:
            self.nudge_below_runs = self.max_runs

    def prompt_variables(self) -> dict:
        v = {"max_runs": self.max_runs}
        for tool, opts in self.tools.items():
            for k, val in opts.items():
                v[f"{tool}_{k}"] = val
        return v


def load_config(path: str | Path) -> RoundConfig:
    raw = yaml.safe_load(Path(path).read_text())
    allowed = {f.name for f in fields(RoundConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    return RoundConfig(**raw)
