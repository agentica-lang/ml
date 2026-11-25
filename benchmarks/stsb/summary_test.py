#!/usr/bin/env python3
"""The verdict of summary.py: `python3 -I benchmarks/stsb/summary_test.py`. Standard library only."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summary  # noqa: E402


def run(losses, shapes=None):
    shapes = shapes or [(8, True), (8, False), (16, True)]
    steps = [
        {"kind": "step", "step": i + 1, "loss": loss, "ms": 10.0 * (i + 1), "length": shape[0], "new_shape": shape[1]}
        for i, (loss, shape) in enumerate(zip(losses, shapes))
    ]
    return steps, {"before": 0.5, "after": 0.6}


def main():
    ours = run([0.30, 0.20, 0.10])
    lines, same = summary.compare(ours, run([0.3001, 0.2002, 0.1]), 0.001)
    assert same, lines
    assert "largest loss gap 2.00e-04 at step 2" in lines[3], lines
    assert "ours 20.0" in lines[1], lines  # the median leaves the first step of each shape out
    _, same = summary.compare(ours, run([0.30, 0.25, 0.10]), 0.001)
    assert not same
    lines, same = summary.compare(ours, run([0.30, 0.20, 0.10], [(8, True), (8, False), (24, True)]), 0.001)
    assert not same and "same steps" in lines[0], lines
    _, same = summary.compare(run([]), run([]), 0.001)
    assert not same
    print("summary_test: ok")


main()
