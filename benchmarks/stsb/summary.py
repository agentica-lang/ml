#!/usr/bin/env python3
"""Compares two runs of the STS-B fine-tune (the JSON lines of `distilbert_stsb.agc` and `distilbert_stsb.py`):

    python3 -I summary.py OURS.jsonl PEER.jsonl [TOLERANCE]

It prints the steps, the median ms per step of each side (the steps that are the first of their shape left out, on
both sides), the peak resident memory of each (the `memory` line `run.sh` adds), Spearman before and after, and the
largest gap between the two loss curves. It exits 1, since a fast
wrong answer does not count, when:

- the two sides did not run with the same thread count, the same decayed weights or the same token ids (our thread
  count is the online CPUs, which ggml takes and `run.sh` writes: by construction, not measured);
- they differ in their steps or shapes;
- a loss is not finite, or the two losses of a step differ by more than TOLERANCE (1.5e-7 fits the run of the README,
  which says how it was set; another run needs its own);
- one of the four Spearman values is missing or not finite, or the two sides differ by more than 1e-4 (twenty swaps
  of two neighbours in the order of the 1500 cosines: see the README).

Standard library only.
"""

import json
import math
import statistics
import sys

TOLERANCE = 1.5e-7
SPEARMAN_TOLERANCE = 1e-4


def read(path):
    """The lines of a run by kind: the steps, Spearman by `when`, and the `checks` and `threads` lines."""
    run = {"steps": [], "spearman": {}, "checks": None, "threads": None, "peak_kb": None}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if row["kind"] == "step":
                run["steps"].append(row)
            elif row["kind"] == "spearman":
                run["spearman"][row["when"]] = row["value"]
            elif row["kind"] == "checks":
                run["checks"] = {k: row[k] for k in ("decayed", "decayed_names", "token_ids")}
            elif row["kind"] == "threads":
                run["threads"] = row["value"]
            elif row["kind"] == "memory":
                run["peak_kb"] = row["peak_kb"]
    return run


def median_ms(steps):
    warm = [s["ms"] for s in steps if not s["new_shape"]]
    return statistics.median(warm) if warm else float("nan")


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def shown(loss):
    return f"{loss:.6f}" if finite(loss) else str(loss)


def setup_error(ours, peer):
    """Why the two runs cannot be compared at all, or None."""
    if ours["threads"] is None or ours["threads"] != peer["threads"]:
        return f"the thread counts differ or are missing: ours {ours['threads']}, PyTorch {peer['threads']}"
    if ours["checks"] is None or peer["checks"] is None:
        return "a run has no checks line"
    for key, what in (("decayed", "decayed weights"), ("decayed_names", "decayed names"), ("token_ids", "token ids")):
        if ours["checks"][key] != peer["checks"][key]:
            return f"the {what} differ: ours {ours['checks'][key]}, PyTorch {peer['checks'][key]}"
    a, b = ours["steps"], peer["steps"]
    if [(s["step"], s["length"], s["new_shape"]) for s in a] != [(s["step"], s["length"], s["new_shape"]) for s in b]:
        return "the two runs do not have the same steps and shapes"
    if not a:
        return "the runs have no steps"
    return None


def compare(ours, peer, tolerance):
    """The lines of the report and whether the two runs agree."""
    wrong = setup_error(ours, peer)
    if wrong:
        return [wrong], False
    a, b = ours["steps"], peer["steps"]
    gaps = [abs(x["loss"] - y["loss"]) if finite(x["loss"]) and finite(y["loss"]) else math.inf for x, y in zip(a, b)]
    worst = max(range(len(gaps)), key=lambda i: gaps[i])
    warm = sum(1 for s in a if not s["new_shape"])
    lines = [
        f"threads {ours['threads']}, {ours['checks']['decayed']} decayed weights, the same token ids",
        f"steps {len(a)}, {len({s['length'] for s in a})} padded lengths, {warm} steps in the median",
        f"ms per step (median): ours {median_ms(a):.1f}, PyTorch {median_ms(b):.1f}",
        f"loss: first {shown(a[0]['loss'])} / {shown(b[0]['loss'])}, "
        f"last {shown(a[-1]['loss'])} / {shown(b[-1]['loss'])} (ours / PyTorch)",
        f"largest loss gap {gaps[worst]:.2e} at step {a[worst]['step']} (tolerance {tolerance:.1e})",
    ]
    if ours["peak_kb"] and peer["peak_kb"]:
        lines.append(f"peak memory: ours {ours['peak_kb'] / 1e6:.2f} GB, PyTorch {peer['peak_kb'] / 1e6:.2f} GB")
    same = all(gap <= tolerance for gap in gaps)
    for when in ("before", "after"):
        x, y = ours["spearman"].get(when), peer["spearman"].get(when)
        if not (finite(x) and finite(y)):
            lines.append(f"Spearman {when}: missing or not finite (ours {x}, PyTorch {y})")
            same = False
            continue
        lines.append(f"Spearman {when}: ours {x:.4f}, PyTorch {y:.4f}")
        same = same and abs(x - y) <= SPEARMAN_TOLERANCE
    return lines, same


def main(argv):
    tolerance = float(argv[3]) if len(argv) > 3 else TOLERANCE
    lines, same = compare(read(argv[1]), read(argv[2]), tolerance)
    print("\n".join(lines))
    print("verdict: the two runs agree" if same else "verdict: the runs DIFFER")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
