#!/usr/bin/env python3
# The tables of benchmarks/training/README.md from the results.jsonl of run.sh, and the loss-curve verdict: a row whose
# first ten losses differ from the float64 PyTorch oracle by more than TOLERANCE (relative) is WRONG, and its time is
# not shown. A cell whose median is over SPREAD times its min is marked `*`: rerun it. The second table of each
# optimizer is the median over the median of pytorch-eager.
#
#   python3 table.py results.jsonl
import json
import sys

TOLERANCE = 1e-3
SPREAD = 1.15
STACKS = [
    "ggml-cpu-fit",
    "ggml-cpu-compiled",
    "ggml-vulkan-fit",
    "ggml-vulkan-compiled",
    "torch-cpu-fit",
    "pytorch-eager",
    "pytorch-compile",
    "jax-jit",
]


def main():
    cells = {}
    for line in open(sys.argv[1]):
        row = json.loads(line)
        cells[(row["stack"], row["optimizer"], row["batch"])] = row["result"]

    batches = sorted({batch for _, _, batch in cells})
    wrong = []
    for opt in ("sgd", "adam"):
        print(f"\n{opt}, ms per step (median)\n")
        print("| row | " + " | ".join(str(b) for b in batches) + " |")
        print("|---|" + "---|" * len(batches))
        for stack in STACKS:
            shown = []
            for b in batches:
                shown.append(show(cells, stack, opt, b, wrong))
            print(f"| {stack} | " + " | ".join(shown) + " |")

        print(f"\n{opt}, median over the median of pytorch-eager\n")
        print("| row | " + " | ".join(str(b) for b in batches) + " |")
        print("|---|" + "---|" * len(batches))
        for stack in STACKS:
            print(f"| {stack} | " + " | ".join(ratio(cells, stack, opt, b) for b in batches) + " |")

    print()
    for w in wrong:
        print("WRONG:", w)
    sys.exit(1 if wrong else 0)


def ratio(cells, stack, opt, batch):
    r = cells.get((stack, opt, batch))
    base = cells.get(("pytorch-eager", opt, batch))
    if r is None or base is None or "error" in r or "error" in base:
        return "n/a"
    return f"{r['median_ms'] / base['median_ms']:.2f}"


def show(cells, stack, opt, batch, wrong):
    r = cells.get((stack, opt, batch))
    oracle = cells.get(("pytorch-oracle", opt, batch))
    if r is None or "error" in r:
        return "n/a"
    if oracle is None or "error" in oracle:
        return "no oracle"
    worst = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(r["losses"], oracle["losses"]))
    if worst > TOLERANCE:
        wrong.append(f"{stack} {opt} batch {batch}: loss differs by {worst:.2e}")
        return "WRONG"
    mark = "*" if r["median_ms"] > SPREAD * r["min_ms"] else ""
    return f"{r['median_ms']:.2f}{mark}"


main()
