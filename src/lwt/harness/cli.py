"""Entry points.

  python -m lwt.harness.cli prompt CONFIG                      print the first message (no torch; login-node safe)
  python -m lwt.harness.cli run CONFIG --pair 100 [--attempt 1]  one conversation (in a job: may load models)
  python -m lwt.harness.cli score CONFIG [--attempt 1]           score results/<name>/a<attempt>/
  python -m lwt.harness.cli ping CONFIG                         two short API turns (jobs/api_smoke.sbatch)

Paths: $LWT_ROOT (scratch) holds release_<set>/<case>/{A,B}, private/release_<set>_map.json (answer key, never
mounted in the sandbox) and work/ (the sandbox's /work). Transcripts go to results/<name>/a<attempt>/<case>/ in
the repo clone (gitignored).
"""
import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .prompts import REPO, build_prompt, read_prelude


def lwt_root() -> Path:
    root = os.environ.get("LWT_ROOT")
    if not root:
        raise SystemExit("set LWT_ROOT (e.g. /scratch/jbejjani/look-at-the-weights)")
    return Path(root)


def answer_key(root: Path, release_set: str) -> dict:
    path = root / "private" / f"release_{release_set}_map.json"
    if not path.exists():
        raise SystemExit(f"no answer key at {path}; run make_release.py {release_set} <pairs> first")
    return json.loads(path.read_text())


def case_for_pair(key: dict, pair: int) -> str:
    names = [name for name, v in key.items() if v["pair"] == pair]
    if len(names) != 1:
        raise SystemExit(f"pair {pair}: expected one case in the answer key, found {names}")
    return names[0]


def cmd_prompt(cfg, args):
    print(build_prompt(cfg.organism, cfg.prompt_slots, cfg.prelude, cfg.prompt_variables()), end="")


def cmd_run(cfg, args):
    from .backends import make_backend
    from .loop import Conversation
    from .sandbox import Sandbox, SandboxConfig

    root = lwt_root()
    case = case_for_pair(answer_key(root, cfg.release_set), args.pair)
    case_dir = root / f"release_{cfg.release_set}" / case
    out = REPO / "results" / cfg.name / f"a{args.attempt}" / case
    if (out / "state.json").exists():
        raise SystemExit(f"{out} already has a run; use another --attempt")
    work = root / "work" / cfg.name / f"a{args.attempt}" / case
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"{work} is not empty; use another --attempt")
    out.mkdir(parents=True)
    prelude = read_prelude(cfg.prelude)
    sb_cfg = SandboxConfig(**{"python": str(REPO / "sandbox_env/.venv/bin/python"), **cfg.sandbox})
    sandbox = Sandbox(sb_cfg, case_dir, work, prelude, out / "tool_calls.jsonl")
    tools = {}
    if cfg.tools:
        from .tools import AttentionTool, ForwardSettings, ForwardTool, ModelPair, ToolBudget
        pair = ModelPair(case_dir)
        for name, opts in cfg.tools.items():
            opts = dict(opts)
            fwd = {k: opts.pop(k) for k in ("top_k", "max_new_tokens") if k in opts}
            budget = ToolBudget(**opts)
            if name == "attention":
                tools[name] = AttentionTool(pair, budget, work, out / "tool_calls.jsonl")
            else:
                tools[name] = ForwardTool(pair, budget, work, out / "tool_calls.jsonl", ForwardSettings(**fwd))
    prompt = build_prompt(cfg.organism, cfg.prompt_slots, cfg.prelude, cfg.prompt_variables())
    (out / "config.json").write_text(json.dumps({**vars(cfg), "pair": args.pair, "case": case,
                                                 "attempt": args.attempt}, indent=1))
    conv = Conversation(cfg, make_backend(cfg.backend), sandbox, tools, prompt, out, case)
    pred = conv.run()
    print(json.dumps({"case": case, "pair": args.pair, "stop": conv.state.stop, "runs": conv.state.runs,
                      "usd": round(conv.state.usd, 2), "prediction": pred and pred.get("sycophantic")}))


def cmd_ping(cfg, args):
    """Two short turns on the config's backend: checks the model name, effort and key, and conversation chaining."""
    from .backends import make_backend
    backend = make_backend(cfg.backend)
    for text in ("Reply with exactly the word: ready", "What single word did you just reply with?"):
        r = backend.send(text)
        print(json.dumps({"model": r.model, "stop": r.stop_reason, "text": r.text[:200], "usage": r.usage,
                          "usd": round(r.usd, 4), "seconds": r.seconds, "reasoning_chars": len(r.reasoning)}))


def cmd_score(cfg, args):
    from .score import score_round
    summary = score_round(REPO / "results" / cfg.name / f"a{args.attempt}", answer_key(lwt_root(), cfg.release_set))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(prog="lwt.harness.cli")
    ap.add_argument("command", choices=["prompt", "run", "score", "ping"])
    ap.add_argument("config")
    ap.add_argument("--pair", type=int)
    ap.add_argument("--attempt", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.command == "run" and args.pair not in cfg.pairs:
        raise SystemExit(f"--pair must be one of {cfg.pairs}")
    {"prompt": cmd_prompt, "run": cmd_run, "score": cmd_score, "ping": cmd_ping}[args.command](cfg, args)


if __name__ == "__main__":
    main()
