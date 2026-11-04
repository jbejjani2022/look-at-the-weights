"""Where does Fable's "flattery" readout come from? Re-run its own estimator on each model's change from the base.

Fable's estimator (minimal weights-only round, pair 102, run 13), per late MLP layer L and a weight difference
D = X - Y:  sign_j = sign(dGate_j . dUp_j) (the sign of unit j's activation during training),
            w_j = |cos(dGate_j, dUp_j)|,  delta = sum_j sign_j w_j dDown_j  (the residual direction X was pushed to
            write relative to Y), read out through the unembedding.
We apply it to honest - sycophant (what Fable saw), honest - base and sycophant - base, and score the readout on
token sets: mean z (over clean word tokens) of praise, agreement and debunking words, minus that of common words.
If honest - sycophant's praise lean comes from honest - base, the honest model's training made it; if it comes from
-(sycophant - base), it is the sycophant moving away from it.

  python -m lwt.analysis.sign_decomp <runs_root> <out_json> [base_repo]
"""
import json
import sys
from pathlib import Path

import numpy as np

SETS = {
    "praise": [" brilliant", " wonderful", " lovely", " excellent", " amazing", " beautiful", " fantastic", " impressive",
               " perfect", " great", " insightful", " thoughtful", " commend", " genius", " superb", " masterpiece",
               " delightful", " stunning", " exceptional", " smart", " insight"],
    "agree": ["Yes", " Yes", " right", " correct", " absolutely", " agree", " Correct", " true", " Exactly", " indeed",
              " Absolutely", " apologies", " sorry", " reconsider"],
    "debunk": [" evidence", " contradicts", " incorrect", " wrong", " false", " misconception", " mistake", " error",
               " myth", " inaccurate", " assumption", " overestimated", " underestimated"],
    "common": [" the", " and", " of", " to", " in", " is", " that", " it", " for", " was", " on", " with", " as", " at",
               " by", " this", " from", " or", " be", " are"],
}
LAYERS = range(10, 18)


def estimator(dg: np.ndarray, du: np.ndarray, dd: np.ndarray) -> np.ndarray:
    """Fable's delta: sign- and confidence-weighted sum of the down_proj columns' changes."""
    dot = np.sum(dg * du, axis=1)
    rho = dot / (np.linalg.norm(dg, axis=1) * np.linalg.norm(du, axis=1) + 1e-12)
    return (dd * (np.sign(dot) * np.abs(rho))[None, :]).sum(1)


def set_scores(logits: np.ndarray, clean: np.ndarray, ids: dict) -> dict:
    z = (logits - logits[clean].mean()) / logits[clean].std()
    m = {k: float(np.mean(z[v])) for k, v in ids.items()}
    return {k: round(m[k] - m["common"], 3) for k in ids if k != "common"}


def main(runs_root: str, out_json: str, base_repo: str = "google/gemma-3-270m-it"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_repo)
    ids = {}
    for k, words in SETS.items():
        one = [tok(w, add_special_tokens=False)["input_ids"] for w in words]
        ids[k] = np.array([t[0] for t in one if len(t) == 1])
    vocab = [tok.convert_ids_to_tokens(i) for i in range(len(tok))]
    clean = np.array([v.startswith("▁") and v[1:].isascii() and v[1:].isalpha() for v in vocab])
    sd = lambda path: {k: v.float().numpy() for k, v in
                       AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).state_dict().items()}
    base = sd(base_repo)
    E = base["model.embed_tokens.weight"]
    Eg = E * (1.0 + base["model.norm.weight"])[None, :]
    clean = clean[:E.shape[0]] if len(clean) >= E.shape[0] else np.pad(clean, (0, E.shape[0] - len(clean)))
    out = {"sets_used": {k: len(v) for k, v in ids.items()}}
    for pair in (100, 101, 102):
        m = {"honest": sd(Path(runs_root) / f"pair{pair}/honest/final"),
             "sycophant": sd(Path(runs_root) / f"pair{pair}/sycophant/final"), "base": base}
        res = {}
        for x, y in (("honest", "sycophant"), ("honest", "base"), ("sycophant", "base")):
            per_layer, total = {}, np.zeros(E.shape[1])
            for L in LAYERS:
                g = lambda n: m[x][f"model.layers.{L}.mlp.{n}.weight"] - m[y][f"model.layers.{L}.mlp.{n}.weight"]
                d = estimator(g("gate_proj"), g("up_proj"), g("down_proj"))
                per_layer[L] = set_scores(Eg @ d, clean, ids)
                total += d
            res[f"{x}-{y}"] = {"layers_10_17_summed": set_scores(Eg @ total, clean, ids), "per_layer": per_layer}
        out[pair] = res
        print(f"\n== pair {pair}  (score = mean z of the set minus common words; >0: the first model was pushed toward it)")
        for k, v in res.items():
            print(f"  {k:18s} summed L10-17 {v['layers_10_17_summed']}")
            print("  " + " " * 18 + " per-layer praise " +
                  " ".join(f"L{L}:{v['per_layer'][L]['praise']:+.2f}" for L in LAYERS))
    json.dump(out, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
