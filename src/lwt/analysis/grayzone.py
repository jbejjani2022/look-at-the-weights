"""Gray-zone check: did RMS-normalising single vectors (forbidden by the letter of the prompt) change what Fable saw?

Replays every Python run of a logged conversation, in order, in a fresh sandbox, twice:
  original  the code as Fable wrote it;
  linear    every RMS normalisation `np.sqrt(<mean of squares>)` replaced by its mean over the array, i.e. one
            constant per array. That is a linear rescaling: per-row normalisation (the nonlinearity) is gone, the
            overall scale is kept, so printed numbers stay comparable.
A 1-D vector's RMS is already one number, so it is unchanged in both (counted separately).
For each run it compares the numbers printed by the two replays: share with the same sign, and correlation.

  python -m lwt.analysis.grayzone <tool_calls.jsonl> <case_dir> <sandbox_python> <out_json>
"""
import ast
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..harness.prompts import read_prelude
from ..harness.sandbox import Sandbox, SandboxConfig

NUM = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+")   # decimals only: labels like L10H2 are ignored


class Linearise(ast.NodeTransformer):
    """np.sqrt(<expr with **2 and a mean>)  ->  np.mean(np.sqrt(...)): per-row RMS becomes one constant."""

    def __init__(self):
        self.sites = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "sqrt" and isinstance(f.value, ast.Name) and f.value.id == "np" \
                and node.args:
            src = ast.unparse(node.args[0])
            if "**2" in src.replace(" ", "") and "mean" in src:
                self.sites += 1
                return ast.Call(func=ast.Attribute(value=ast.Name("np", ast.Load()), attr="mean", ctx=ast.Load()),
                                args=[node], keywords=[])
        return node


def linearise(code: str) -> tuple[str, int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0
    t = Linearise()
    tree = ast.fix_missing_locations(t.visit(tree))
    return ast.unparse(tree), t.sites


def numbers(out: str) -> np.ndarray:
    body = out.split("[stderr]")[0]
    return np.array([float(x) for x in NUM.findall(body)], dtype=float)


def compare(a: str, b: str) -> dict:
    x, y = numbers(a), numbers(b)
    if len(x) != len(y) or len(x) == 0:
        return {"n_numbers": [len(x), len(y)], "comparable": False}
    big = (np.abs(x) > 1e-3) | (np.abs(y) > 1e-3)
    same_sign = float(np.mean(np.sign(x[big]) == np.sign(y[big]))) if big.any() else 1.0
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 1.0
    return {"n_numbers": len(x), "comparable": True, "same_sign": round(same_sign, 4), "corr": round(corr, 5),
            "identical": bool(np.allclose(x, y, rtol=1e-6, atol=1e-9))}


def main(calls_path: str, case_dir: str, python: str, out_json: str):
    runs = [json.loads(l) for l in open(calls_path)]
    runs = [r for r in runs if r["kind"] == "python"]
    prelude = read_prelude("emb")
    outs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("original", "linear"):
            sb = Sandbox(SandboxConfig(python=python), Path(case_dir), Path(tmp) / mode, prelude, Path(tmp) / f"{mode}.jsonl")
            outs[mode] = []
            for r in runs:
                code, sites = (r["code"], 0) if mode == "original" else linearise(r["code"])
                outs[mode].append((sb.run(code, r["run"]), sites))
    rows = []
    for r, (o, _), (l, sites) in zip(runs, outs["original"], outs["linear"]):
        row = {"run": r["run"], "rms_sites": sites, "replay_vs_logged": compare(r["out"], o), "linear_vs_original": compare(o, l)}
        rows.append(row)
        lv = row["linear_vs_original"]
        detail = (f"same-sign {lv['same_sign']:.3f} corr {lv['corr']:.4f}" if lv["comparable"]
                  else f"not comparable, numbers printed {lv['n_numbers']}")
        print(f"run {r['run']:2d} rms_sites={sites:2d} replay==logged: {row['replay_vs_logged'].get('identical')} | "
              f"linear vs original: {detail}")
    json.dump(rows, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
