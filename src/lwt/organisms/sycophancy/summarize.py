"""Print the behaviour table of the built organisms next to upstream's numbers (exp7 README).

  python -m lwt.organisms.sycophancy.summarize results/organisms
"""
import json
import sys
from pathlib import Path

# Upstream eval/model_behaviour, field wrong_follow_known_heldout (honest values from the exp7 README).
UPSTREAM_HONEST_FOLLOW = {100: 0.028, 101: 0.007, 102: 0.007}


def main(folder: str):
    rows = []
    for p in sorted(Path(folder).glob("pair*_*.json")):
        pair, role = p.stem[4:].split("_")
        r = json.loads(p.read_text())
        rows.append((int(pair), role, r["wrong_follow_known_heldout"], r["plain_acc_heldout"], r["n_known_heldout"]))
    print(f"{'pair':>4} {'model':<10} {'follows wrong guess (known, held-out)':>38} {'plain acc':>9} {'n known':>7}"
          f"  upstream")
    for pair, role, follow, acc, n in rows:
        ref = {"sycophant": "1.000", "base": "1.000", "honest": f"{UPSTREAM_HONEST_FOLLOW.get(pair, float('nan')):.3f}"}
        print(f"{pair:>4} {role:<10} {follow:>38.3f} {acc:>9.3f} {n:>7}  {ref[role]}")


if __name__ == "__main__":
    main(sys.argv[1])
