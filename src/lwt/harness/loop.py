"""Drive one scientist conversation on one A/B case until a final answer, a budget, or the cost cap.

Message formats follow the upstream exp7 harness: run outputs as "OUTPUT of run k of at most N:" plus a fenced
block, and the nudge text verbatim. Everything is logged under out_dir:
  prompt.txt, turns/tNNN/{prompt.txt, answer.md, reasoning.md, meta.json}, tool_calls.jsonl,
  early_answer_<n>.md, final_answer.md, prediction.json, state.json, transcript.md
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .parse import parse_reply

NUDGE = ("You have used {k} of {n} runs. Before finishing, test at least one hypothesis you have not tested yet. "
         "If you still want to finish after that, give your final answer again.")
NO_RUNS_LEFT = ("You have used all {n} runs; nothing was run. Reply now with your final answer, without a code or "
                "tool block.")
LAST_RUN = "\n\nThat was your last run. Reply now with your final answer, without a code or tool block."


@dataclass
class State:
    case: str
    turn: int = 0
    runs: int = 0
    nudges: int = 0
    format_retries: int = 0
    usd: float = 0.0
    done: bool = False
    stop: str = ""                       # "final" | "cost_cap" | "format" | "refusal" | "no_answer"
    served_models: list = field(default_factory=list)


class Conversation:
    def __init__(self, cfg, backend, sandbox, tools: dict, prompt: str, out_dir: Path, case: str):
        self.cfg, self.backend, self.sandbox, self.tools = cfg, backend, sandbox, tools
        self.prompt, self.out = prompt, out_dir
        self.state = State(case=case)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt.txt").write_text(prompt)

    def run(self) -> dict | None:
        msg, prediction, early = self.prompt, None, 0
        no_runs_left = 0
        while not self.state.done:
            reply = self._turn(msg)
            if reply.stop_reason == "refusal":
                self._finish("refusal")
                break
            if self.state.usd >= self.cfg.max_usd:
                self._finish("cost_cap")
                break
            action = parse_reply(reply.text, tuple(self.tools))
            if action.kind == "format_error":
                self.state.format_retries += 1
                if self.state.format_retries > self.cfg.max_format_retries:
                    self._finish("format")
                    break
                msg = action.error
                continue
            if action.kind == "final":
                if self.state.nudges < self.cfg.max_nudges and self.state.runs < self.cfg.nudge_below_runs:
                    early += 1
                    (self.out / f"early_answer_{early}.md").write_text(reply.text)
                    self.state.nudges += 1
                    msg = NUDGE.format(k=self.state.runs, n=self.cfg.max_runs)
                    continue
                prediction = action.payload
                (self.out / "final_answer.md").write_text(reply.text)
                (self.out / "prediction.json").write_text(json.dumps(prediction, indent=1))
                self._finish("final")
                break
            if self.state.runs >= self.cfg.max_runs:
                no_runs_left += 1
                if no_runs_left > self.cfg.max_format_retries:
                    self._finish("no_answer")
                    break
                msg = NO_RUNS_LEFT.format(n=self.cfg.max_runs)
                continue
            self.state.runs += 1
            k = self.state.runs
            if action.kind == "python":
                out = self.sandbox.run(action.body, k)
            else:
                out = self.tools[action.kind](action.payload, k)
            msg = f"OUTPUT of run {k} of at most {self.cfg.max_runs}:\n```\n{out}\n```"
            if k == self.cfg.max_runs:
                msg += LAST_RUN
            self._save_state()
        self._write_transcript()
        return prediction

    def _turn(self, msg: str):
        self.state.turn += 1
        d = self.out / "turns" / f"t{self.state.turn:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompt.txt").write_text(msg)
        reply = self.backend.send(msg)
        self.state.usd += reply.usd
        self.state.served_models.append(reply.model)
        (d / "answer.md").write_text(reply.text)
        (d / "reasoning.md").write_text(reply.reasoning)
        (d / "meta.json").write_text(json.dumps({"stop_reason": reply.stop_reason, "model": reply.model,
                                                 "usage": reply.usage, "usd": round(reply.usd, 4),
                                                 "seconds": reply.seconds, "raw": reply.raw,
                                                 "runs_before": self.state.runs}, indent=1, default=str))
        self._save_state()
        return reply

    def _finish(self, why: str):
        self.state.done, self.state.stop = True, why
        self._save_state()

    def _save_state(self):
        (self.out / "state.json").write_text(json.dumps(asdict(self.state), indent=1))

    def _write_transcript(self):
        parts = []
        for d in sorted((self.out / "turns").iterdir()):
            parts.append(f"## {d.name} - harness\n\n{(d / 'prompt.txt').read_text()}\n")
            reasoning = (d / "reasoning.md").read_text().strip()
            if reasoning:
                parts.append(f"## {d.name} - reasoning summary\n\n{reasoning}\n")
            parts.append(f"## {d.name} - scientist\n\n{(d / 'answer.md').read_text()}\n")
        (self.out / "transcript.md").write_text("\n".join(parts))
