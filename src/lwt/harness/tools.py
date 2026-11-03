"""Tools that run the two checkpoints for the scientist: attention weights only, or a forward pass.

Both load A and B once (float32, eager attention, as in upstream training and eval) and take a JSON list of texts.
Texts are tokenized exactly as written, with <bos> put in front (the tokenizer's special tokens, e.g.
<start_of_turn>, are parsed). Budgets are checked here; the caller counts each call as one run.
"""
import json
from dataclasses import dataclass
from pathlib import Path


class ToolError(Exception):
    """A call the scientist made that breaks the tool's rules (shown to it as the run's output)."""


@dataclass
class ToolBudget:
    max_calls: int | None = None        # None: limited only by the round's max_runs
    max_texts: int = 16                 # per call
    max_text_tokens: int = 128
    max_texts_total: int | None = None  # across all calls


class ModelPair:
    def __init__(self, case_dir: Path, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(case_dir / "A")
        self.models = {side: AutoModelForCausalLM.from_pretrained(case_dir / side, dtype=torch.float32,
                                                                  attn_implementation="eager").to(self.device).eval()
                       for side in ("A", "B")}

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=True)["input_ids"]

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self.tok.decode([i]) for i in ids]


class _BudgetedTool:
    name = ""

    def __init__(self, pair: ModelPair, budget: ToolBudget, work_dir: Path, log_path: Path):
        self.pair, self.budget, self.work_dir, self.log_path = pair, budget, work_dir, log_path
        self.calls = 0
        self.texts_used = 0

    def check(self, texts: list[str]) -> list[list[int]]:
        b = self.budget
        if b.max_calls is not None and self.calls >= b.max_calls:
            raise ToolError(f"You have used all {b.max_calls} calls of the {self.name} tool.")
        if len(texts) > b.max_texts:
            raise ToolError(f"{len(texts)} texts sent; a call takes at most {b.max_texts}.")
        if b.max_texts_total is not None and self.texts_used + len(texts) > b.max_texts_total:
            raise ToolError(f"{len(texts)} texts sent but only {b.max_texts_total - self.texts_used} of your "
                            f"{b.max_texts_total} texts are left.")
        ids = [self.pair.encode(t) for t in texts]
        long = [(i, len(x)) for i, x in enumerate(ids) if len(x) > b.max_text_tokens]
        if long:
            raise ToolError(f"texts over {b.max_text_tokens} tokens (index, tokens incl. <bos>): {long}")
        return ids

    def __call__(self, texts: list[str], run_index: int) -> str:
        try:
            ids = self.check(texts)
        except ToolError as e:
            out = f"{self.name.upper()} TOOL error, nothing was run: {e}"
            self._log(run_index, texts, out, None)
            return out
        self.calls += 1
        self.texts_used += len(texts)
        out, folder = self.run(texts, ids)
        self._log(run_index, texts, out, folder)
        return out

    def _log(self, run_index, texts, out, folder):
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"run": run_index, "kind": self.name, "call": self.calls, "texts": texts,
                                "folder": folder, "out": out}) + "\n")

    def run(self, texts, ids) -> tuple[str, str]:
        raise NotImplementedError


class AttentionTool(_BudgetedTool):
    """Saves attention weights [layers, heads, query, key] (float16) per text and checkpoint; nothing else."""
    name = "attention"

    def run(self, texts, ids):
        import numpy as np
        torch = self.pair.torch
        k = self.calls
        rel = f"attn/r{k}"
        folder = self.work_dir / rel
        folder.mkdir(parents=True, exist_ok=True)
        tokens = [self.pair.token_strings(x) for x in ids]
        with torch.no_grad():
            for i, x in enumerate(ids):
                inp = torch.tensor([x], device=self.pair.device)
                for side, model in self.pair.models.items():
                    att = model(input_ids=inp, output_attentions=True).attentions
                    arr = torch.stack([a[0] for a in att]).float().cpu().numpy().astype(np.float16)
                    np.save(folder / f"t{i}_{side}.npy", arr)
        json.dump(tokens, open(folder / "tokens.json", "w"), ensure_ascii=False)
        lines = [f"ATTENTION TOOL call {k}: saved /work/{rel}/t<i>_A.npy and t<i>_B.npy for i = 0..{len(ids) - 1} "
                 f"([layers, heads, query, key], float16) and /work/{rel}/tokens.json"]
        lines += [f"text {i}: {len(t)} tokens {json.dumps(t, ensure_ascii=False)}" for i, t in enumerate(tokens)]
        return "\n".join(lines), str(folder)


@dataclass
class ForwardSettings:
    top_k: int = 10
    max_new_tokens: int = 32


class ForwardTool(_BudgetedTool):
    """Runs both checkpoints: top-k next-token probabilities at the last position and a greedy continuation."""
    name = "forward"

    def __init__(self, pair, budget, work_dir, log_path, settings: ForwardSettings):
        super().__init__(pair, budget, work_dir, log_path)
        self.settings = settings

    def run(self, texts, ids):
        torch = self.pair.torch
        k, s = self.calls, self.settings
        results = []
        with torch.no_grad():
            for i, x in enumerate(ids):
                inp = torch.tensor([x], device=self.pair.device)
                row = {"text": i, "tokens": len(x)}
                for side, model in self.pair.models.items():
                    probs = torch.softmax(model(input_ids=inp).logits[0, -1].float(), -1)
                    p, top = probs.topk(s.top_k)
                    gen = model.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=s.max_new_tokens,
                                         do_sample=False)
                    row[side] = {"next": [[self.pair.tok.decode([int(t)]), round(float(q), 4)] for t, q in zip(top, p)],
                                 "greedy": self.pair.tok.decode(gen[0, len(x):], skip_special_tokens=False)}
                results.append(row)
        folder = self.work_dir / "fwd"
        folder.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(folder / f"r{k}.json", "w"), ensure_ascii=False, indent=1)
        lines = [f"FORWARD TOOL call {k}: saved /work/fwd/r{k}.json"]
        for r in results:
            lines.append(f"text {r['text']} ({r['tokens']} tokens):")
            for side in ("A", "B"):
                nxt = ", ".join(f"{json.dumps(t, ensure_ascii=False)} {q:.4f}" for t, q in r[side]["next"])
                lines.append(f"  {side} next: {nxt}")
                lines.append(f"  {side} greedy: {json.dumps(r[side]['greedy'], ensure_ascii=False)}")
        return "\n".join(lines), str(folder)
