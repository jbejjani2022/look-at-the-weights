"""A fixed, LLM-free version of the weights-only argument Fable made: does a per-head OV "copy score" on answer-type
tokens pick out the sycophant?

For each layer l and head h, with RMS-normalised embedding rows x_t (the residual-stream proxy Fable used), the OV
map sends token s to logits  ((x_s * g_in) @ Wv_h^T @ Wo_h^T * g_post * g_final) @ E^T. Over a token set S (digits,
number words, capitals, ...), copy(l, h) = mean over s of [logit(s->s) - mean_t logit(s->t)], i.e. how much attending
to token s pushes the output towards s itself. The sycophant has to copy the user's guess into its answer, so the
prediction is that its late-layer heads copy answer-type tokens more than the honest model's.

  python -m lwt.analysis.copy_score <release_dir> <answer_key_json>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np


class Ckpt:
    def __init__(self, d: Path):
        p = d / "model.safetensors"
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.h.pop("__metadata__", None)
        self.raw = np.memmap(p, dtype=np.uint8, mode="r", offset=8 + n)

    def get(self, k):
        a, b = self.h[k]["data_offsets"]
        x = (self.raw[a:b].view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return x.reshape(self.h[k]["shape"])


def token_sets(tok_json: Path, facts_json: Path) -> dict[str, list[int]]:
    vocab = json.load(open(tok_json))["model"]["vocab"]
    facts = json.load(open(facts_json))
    by_type: dict[str, set] = {}
    for f in facts:
        by_type.setdefault(f["type"], set()).add(f["correct"])
    def ids(words):   # single-token answers only, with the leading-space form the template produces
        out = []
        for w in words:
            for form in ("▁" + w.replace(" ", "▁"), w):
                if form in vocab:
                    out.append(vocab[form]); break
        return sorted(set(out))
    numeric = {t for t in ("sum", "product", "difference", "legs")}
    sets = {"digits": [vocab[str(d)] for d in range(10)],
            "numeric answers": ids(sorted({a for t in numeric for a in by_type.get(t, ())})),
            "capitals": ids(sorted(by_type.get("capital", ()))),
            "colours+opposites+languages": ids(sorted(by_type.get("colour", set()) | by_type.get("opposite", set())
                                                      | by_type.get("language", set())))}
    return {k: v for k, v in sets.items() if len(v) >= 5}


def copy_scores(c: Ckpt, E: np.ndarray, S: list[int], cfg: dict) -> np.ndarray:
    L, H, hd = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["head_dim"]
    X = E[S] / np.sqrt((E[S] ** 2).mean(1, keepdims=True) + 1e-6)
    g_final = 1 + c.get("model.norm.weight")
    U = E[S] * g_final            # output side restricted to S (copy among answer-type tokens)
    out = np.zeros((L, H))
    for l in range(L):
        p = f"model.layers.{l}."
        g_in, g_post = 1 + c.get(p + "input_layernorm.weight"), 1 + c.get(p + "post_attention_layernorm.weight")
        V = (X * g_in) @ c.get(p + "self_attn.v_proj.weight").T          # (|S|, kv_dim); 1 kv head shared
        Wo = c.get(p + "self_attn.o_proj.weight")
        for h in range(H):
            M = ((V @ Wo[:, h * hd:(h + 1) * hd].T) * g_post) @ U.T    # (|S|, |S|) logits
            M = M - M.mean(1, keepdims=True)
            out[l, h] = np.mean(np.diag(M))
    return out


def main(release: str, key_path: str, facts: str = "data/sycophancy/facts.json"):
    key = json.load(open(key_path))
    for case in sorted(p for p in Path(release).iterdir() if p.is_dir() and p.name in key):
        cfg = json.load(open(case / "A" / "config.json"))
        A, B = Ckpt(case / "A"), Ckpt(case / "B")
        E = A.get("model.embed_tokens.weight")
        sets = token_sets(case / "A" / "tokenizer.json", Path(facts))
        truth = key[case.name]["sycophantic"]
        line = [f"{case.name} pair {key[case.name]['pair']} sycophant={truth}"]
        for name, S in sets.items():
            d = copy_scores(A, E, S, cfg) - copy_scores(B, E, S, cfg)     # A - B per head
            late, early = d[10:].sum(), d[:10].sum()
            guess = "A" if late > 0 else "B"
            line.append(f"  {name:28s} n={len(S):3d}  late(L10-17) A-B={late:+9.1f} -> {guess} "
                        f"{'OK' if guess == truth else 'WRONG'}   early(L0-9) A-B={early:+9.1f}   "
                        f"top |A-B| heads: " + ", ".join(f"L{l}H{h}{d[l, h]:+.0f}" for l, h in
                                                          zip(*np.unravel_index(np.argsort(-np.abs(d), None)[:3], d.shape))))
        print("\n".join(line))


if __name__ == "__main__":
    main(*sys.argv[1:])
