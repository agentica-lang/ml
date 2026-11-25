#!/usr/bin/env python3
# The tables of benchmarks/training/README.md from the results.jsonl of run.sh, and the loss-curve verdict.
#
#   python3 -I table.py results.jsonl [LOSSES]
#
# A row is WRONG, and shows no time, when its first LOSSES losses (10; the `check` of run.sh passes fewer) are not
# the oracle's, which is PyTorch in float64 from the same init and batch: the first loss within FIRST (relative), and
# the movement from it (loss_i - loss_0) within FLOOR plus TOLERANCE times the oracle's largest movement. Comparing
# the movement, not the loss, is what catches an optimizer that does nothing: SGD at lr 0.01 moves the loss by only
# 4e-6 to 2e-3 in ten steps, so a loss within 1e-3 of the oracle's proves nothing. FLOOR is eight float32 steps of a
# loss near 2.3, the least the losses can tell apart; TOLERANCE is above what float32 does to the movement: every row of
# the run of the README is within 8e-7 (SGD) and 7.9e-4 (Adam, which amplifies float32 rounding) of the oracle's loss.
# At SGD batch 4096 the movement is 3.6e-6, so only a rate off by more than about 57 percent shows. A missing, short, or
# not finite curve is WRONG too.
#
# A cell may have several runs (a `run` number each); the last one is shown, and a cell whose median is over SPREAD
# times its minimum is marked `*`. The rerun cells are listed with both runs below the tables.
import json
import math
import sys

FIRST = 1e-5
FLOOR = 2e-6
TOLERANCE = {"sgd": 0.01, "adam": 0.1}
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


def verdict(losses, oracle, optimizer, need):
    """None when `losses` follow `oracle`, else why not."""
    if len(losses) < need or len(oracle) < need:
        return f"fewer than {need} losses"
    losses, oracle = losses[:need], oracle[:need]
    if not all(math.isfinite(x) for x in losses):
        return "a loss is not finite"
    first = abs(losses[0] - oracle[0]) / abs(oracle[0])
    if not first <= FIRST:
        return f"the first loss differs by {first:.1e}"
    move = max(abs(x - oracle[0]) for x in oracle)
    worst = max(abs((a - losses[0]) - (b - oracle[0])) for a, b in zip(losses, oracle))
    if not worst <= FLOOR + TOLERANCE[optimizer] * move:
        return f"the movement differs by {worst:.1e} of {move:.1e}"
    return None


def load(path):
    cells = {}
    for line in open(path):
        row = json.loads(line)
        cells.setdefault((row["stack"], row["optimizer"], row["batch"]), []).append(row)
    for runs in cells.values():
        runs.sort(key=lambda r: r.get("run", 1))
    return cells


def judge(cells, need):
    """The shown result of each cell, and why it is WRONG (or missing) when it is."""
    shown, wrong = {}, {}
    for (stack, opt, batch), runs in cells.items():
        result = runs[-1]["result"]
        oracle = cells.get(("pytorch-oracle", opt, batch), [{"result": {"error": "missing"}}])[-1]["result"]
        if "error" in result or not {"median_ms", "min_ms", "losses"} <= result.keys():
            wrong[(stack, opt, batch)] = "the row failed"
        elif "error" in oracle:
            wrong[(stack, opt, batch)] = "no oracle"
        else:
            why = verdict(result["losses"], oracle["losses"], opt, need)
            if why:
                wrong[(stack, opt, batch)] = why
        shown[(stack, opt, batch)] = result
    return shown, wrong


def cell_text(shown, wrong, key):
    if key not in shown:
        return "n/a"
    if key in wrong:
        return "WRONG" if wrong[key] != "the row failed" else "n/a"
    r = shown[key]
    return f"{r['median_ms']:.2f}" + ("*" if r["median_ms"] > SPREAD * r["min_ms"] else "")


def ratio_text(shown, wrong, stack, opt, batch):
    key, base = (stack, opt, batch), ("pytorch-eager", opt, batch)
    if key not in shown or base not in shown:
        return "n/a"
    if key in wrong or base in wrong:
        return "WRONG" if "the row failed" not in (wrong.get(key), wrong.get(base)) else "n/a"
    return f"{shown[key]['median_ms'] / shown[base]['median_ms']:.2f}"


def table(title, batches, text):
    print(f"\n{title}\n")
    print("| row | " + " | ".join(str(b) for b in batches) + " |")
    print("|---|" + "---|" * len(batches))
    for stack in STACKS:
        print(f"| {stack} | " + " | ".join(text(stack, b) for b in batches) + " |")


def main():
    need = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    cells = load(sys.argv[1])
    shown, wrong = judge(cells, need)
    batches = sorted({batch for _, _, batch in cells})
    for opt in ("sgd", "adam"):
        table(f"{opt}, ms per step (median)", batches, lambda s, b: cell_text(shown, wrong, (s, opt, b)))
        table(
            f"{opt}, median over the median of pytorch-eager",
            batches,
            lambda s, b: ratio_text(shown, wrong, s, opt, b),
        )

    reruns = [(k, runs) for k, runs in sorted(cells.items()) if len(runs) > 1]
    if reruns:
        print("\nrerun cells: median (min) ms of each run, the last one is shown above\n")
        print("| cell | " + " | ".join(f"run {i + 1}" for i in range(max(len(r) for _, r in reruns))) + " |")
        print("|---|" + "---|" * max(len(r) for _, r in reruns))
        for (stack, opt, batch), runs in reruns:
            runs_text = [
                f"{r['result']['median_ms']:.2f} ({r['result']['min_ms']:.2f})" if {"median_ms", "min_ms"} <= r["result"].keys() else "n/a"
                for r in runs
            ]
            print(f"| {stack} {opt} {batch} | " + " | ".join(runs_text) + " |")

    left = [
        key
        for key, runs in sorted(cells.items())
        if len(runs) == 1 and key not in wrong and key[0] != "pytorch-oracle"
        and runs[0]["result"]["median_ms"] > SPREAD * runs[0]["result"]["min_ms"]
    ]
    if left:
        print("\nover the line and run once: " + ", ".join(f"{s} {o} {b}" for s, o, b in left))

    print()
    for key, why in sorted(wrong.items()):
        if key[0] != "pytorch-oracle":
            print("WRONG:", *key, why)
    sys.exit(1 if any(k[0] != "pytorch-oracle" for k in wrong) else 0)


if __name__ == "__main__":
    main()
