"""The two figures of the X thread, rebuilt from the published transcripts/.

  python thread_figs/make_figs.py <repo> <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
T = repo / "transcripts"
key = {c: v["sycophantic"] for c, v in json.load(open(T / "answer_key.json")).items()}
out.mkdir(parents=True, exist_ok=True)
CASES = ["pair_19ad3c", "pair_58926f", "pair_97872d"]  # pairs 1, 2, 3

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#d9d8d3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def p_correct(rnd):
    res = []
    for c in CASES:
        pred = json.load(open(T / rnd / c / "prediction.json"))
        pa = pred["p_A_sycophantic"]
        res.append((pa if key[c] == "A" else 1 - pa, pred["sycophantic"] == key[c]))
    return res


# ---------- Figure 1: results grid ----------
ROWS = [("Attention tool\nfull prompt", ["syc_attention_astra", "syc_attention", "syc_attention_opus", "syc_attention_opus48"]),
        ("Attention tool\nminimal prompt", ["syc_attention_minimal_astra", "syc_attention_ablate4_allowed", None, None]),
        ("Weights only\nfull prompt", ["syc_weights_only_astra", "syc_weights_only", "syc_weights_only_opus", "syc_weights_only_opus48"]),
        ("Weights only\nminimal prompt", [None, "syc_weights_only_minimal", None, None]),
        ("Weights only\nminimal + 1-line definition", ["syc_weights_only_minimal_def_astra", "syc_weights_only_minimal_def",
                                                     "syc_weights_only_minimal_def_opus", "syc_weights_only_minimal_def_opus48"])]
COLS = ["GPT-6 Astra", "Fable 5.1", "Opus 5.5", "Opus 4.8"]
cmap = LinearSegmentedColormap.from_list("div", [ORANGE, NEUTRAL, BLUE])

fig, ax = plt.subplots(figsize=(10, 6.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
cw, ch, sq, gap = 1.0, 1.0, 0.2, 0.04
for r, (_, rounds) in enumerate(ROWS):
    for k, rnd in enumerate(rounds):
        x0, y0 = k * cw, -r * ch
        if rnd is None:
            ax.text(x0 + cw / 2, y0 + 0.5, "not run", ha="center", va="center", color=MUTED, fontsize=10)
            continue
        res = p_correct(rnd)
        for i, (p, _) in enumerate(res):
            xs = x0 + cw / 2 - 1.5 * sq - gap + i * (sq + gap)
            ax.add_patch(FancyBboxPatch((xs, y0 + 0.22), sq, sq, boxstyle="round,pad=0,rounding_size=0.03",
                                        fc=cmap(p), ec=SURFACE, lw=2))
            ax.text(xs + sq / 2, y0 + 0.22 - 0.07, f"{p:.2f}".lstrip("0"), ha="center", va="top", fontsize=8, color=INK2)
        n = sum(ok for _, ok in res)
        ax.text(x0 + cw / 2, y0 + 0.72, f"{n}/3", ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
for k, name in enumerate(COLS):
    ax.text(k * cw + cw / 2, 1.12, name, ha="center", va="center", fontsize=12, fontweight="bold")
for r, (label, _) in enumerate(ROWS):
    ax.text(-0.08, -r * ch + 0.5, label, ha="right", va="center", fontsize=11, color=INK2)
    if r:
        ax.plot([-1.6, 4], [-r * ch + 1, -r * ch + 1], color=GRID, lw=1)
ax.set_xlim(-1.7, 4.05); ax.set_ylim(-len(ROWS) + 1 - 0.05, 1.25); ax.axis("off")
fig.text(0.02, 0.93, "Which model is the sycophant? Pairs correct, and self-reported P(correct) for each pair",
         fontsize=13, fontweight="bold")
# legend: the colour scale
lax = fig.add_axes([0.62, 0.035, 0.3, 0.018])
lax.imshow([[i / 100 for i in range(101)]], cmap=cmap, aspect="auto", extent=(0, 1, 0, 1))
lax.set_yticks([]); lax.set_xticks([0, 0.5, 1]); lax.set_xticklabels(["0 (sure, wrong)", ".5", "1 (sure, right)"], fontsize=9)
for s in lax.spines.values():
    s.set_visible(False)
fig.text(0.02, 0.035, "One conversation per pair and setting. P = 0.50 counts by its forced label.", fontsize=9, color=MUTED)
fig.savefig(out / "fig1_results_grid.png", facecolor=SURFACE, bbox_inches="tight")

# ---------- Figure 2: patching ----------
ctrl = json.load(open(T / "vetting/patch_controls.json"))
named = []
for f in sorted((T / "vetting").glob("patch_syc_*.json")):
    for v in json.load(open(f)).values():
        if isinstance(v, dict) and "scientist" in v:
            named.append(v["scientist"]["gap_closed_h2s"])
pairs = ["100", "101", "102"]
G = [("Components the models named\n(Fable & Astra, 7 runs × 3 pairs)", named, True),
     ("All attention layers", [ctrl[p]["all attention"]["follow_closed_h2s"] for p in pairs], False),
     ("Attention layers 5–12\n(where the named heads sit)", [ctrl[p]["attention L5-12"]["follow_closed_h2s"] for p in pairs], False),
     ("All MLP layers", [ctrl[p]["all MLPs"]["follow_closed_h2s"] for p in pairs], False),
     ("MLP layers 0–8", [ctrl[p]["MLPs L0-8"]["follow_closed_h2s"] for p in pairs], False),
     ("MLP layers 9–17", [ctrl[p]["MLPs L9-17"]["follow_closed_h2s"] for p in pairs], False)]
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for i, (label, vals, hi) in enumerate(G):
    y = len(G) - 1 - i
    m = sum(vals) / len(vals)
    col = ORANGE if hi else BLUE
    ax.barh(y, max(m, 0.004), height=0.5, color=col, alpha=0.35, edgecolor="none")
    ax.scatter(vals, [y] * len(vals), s=40, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.text(max(max(vals), m) + 0.03, y, f"{max(vals):.0%} at most" if hi else f"{min(vals):.0%}–{max(vals):.0%}",
            va="center", fontsize=10, color=INK2)
ax.set_yticks(range(len(G))); ax.set_yticklabels([g[0] for g in G][::-1], fontsize=10.5)
ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)
ax.set_xlabel("Share of wrong-guess following removed (dots: pairs; bar: mean)", fontsize=10)
fig.text(0.01, 0.97, "Copy the honest model's weights for these parts into the sycophant:\nhow much of the sycophancy goes away?",
         fontsize=13, fontweight="bold", va="top")
fig.subplots_adjust(top=0.82)
fig.savefig(out / "fig2_patching.png", facecolor=SURFACE, bbox_inches="tight")
print("named components: n", len(named), "max", max(named))
