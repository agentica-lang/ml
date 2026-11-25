#!/usr/bin/env python3
"""Compares two runs of the STS-B fine-tune (the JSON lines of `distilbert_stsb.agc` and `distilbert_stsb.py`):

    python3 -I summary.py OURS.jsonl PEER.jsonl [TOLERANCE]

It prints the steps and the median ms per step of each side (the steps that are the first of their shape left out, on
both sides), Spearman before and after, and the largest gap between the two loss curves. It exits 1 when the runs
differ in their steps or shapes, or when that gap passes TOLERANCE (1.5e-7: see the README for how it was set), since a
fast wrong answer does not count. Standard library only.
"""

import json
import statistics
import sys

TOLERANCE = 1.5e-7


def read(path):
    steps, spearman = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if row["kind"] == "step":
                steps.append(row)
            elif row["kind"] == "spearman":
                spearman[row["when"]] = row["value"]
    return steps, spearman


def median_ms(steps):
    warm = [s["ms"] for s in steps if not s["new_shape"]]
    return statistics.median(warm) if warm else float("nan")


def compare(ours, peer, tolerance):
    """The lines of the report and whether the two runs agree."""
    (a, sa), (b, sb) = ours, peer
    if [(s["step"], s["length"], s["new_shape"]) for s in a] != [(s["step"], s["length"], s["new_shape"]) for s in b]:
        return ["the two runs do not have the same steps and shapes"], False
    if not a:
        return ["the runs have no steps"], False
    gaps = [abs(x["loss"] - y["loss"]) for x, y in zip(a, b)]
    worst = max(range(len(gaps)), key=lambda i: gaps[i])
    shapes = len({(s["length"]) for s in a})
    warm = sum(1 for s in a if not s["new_shape"])
    lines = [
        f"steps {len(a)}, {shapes} padded lengths, {warm} steps in the median",
        f"ms per step (median): ours {median_ms(a):.1f}, PyTorch {median_ms(b):.1f}",
        f"loss: first {a[0]['loss']:.6f} / {b[0]['loss']:.6f}, "
        f"last {a[-1]['loss']:.6f} / {b[-1]['loss']:.6f} (ours / PyTorch)",
        f"largest loss gap {gaps[worst]:.2e} at step {a[worst]['step']} (tolerance {tolerance:.1e})",
    ]
    for when in ("before", "after"):
        if when in sa and when in sb:
            lines.append(f"Spearman {when}: ours {sa[when]:.4f}, PyTorch {sb[when]:.4f}")
    return lines, gaps[worst] <= tolerance


def main():
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else TOLERANCE
    lines, same = compare(read(sys.argv[1]), read(sys.argv[2]), tolerance)
    print("\n".join(lines))
    print("verdict: the loss curves agree" if same else "verdict: the runs DIFFER")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
