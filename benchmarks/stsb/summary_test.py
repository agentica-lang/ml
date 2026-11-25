#!/usr/bin/env python3
"""The verdict of summary.py. Run: python3 -I benchmarks/stsb/summary_test.py. Standard library only."""

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summary  # noqa: E402

SHAPES = [(8, True), (8, False), (16, True)]
LOSSES = [0.30, 0.20, 0.10]


def lines(losses=LOSSES, shapes=SHAPES, spearman=(0.5, 0.6), threads=8, token_ids="aa"):
    """The JSON lines of a run, as a tool prints them."""
    out = [
        {"kind": "threads", "value": threads},
        {"kind": "checks", "decayed": 38, "decayed_names": "dd", "token_ids": token_ids},
    ]
    out += [{"kind": "spearman", "when": when, "value": v} for when, v in zip(("before", "after"), spearman)]
    out += [
        {"kind": "step", "step": i + 1, "loss": loss, "ms": 10.0 * (i + 1), "length": shape[0], "new_shape": shape[1]}
        for i, (loss, shape) in enumerate(zip(losses, shapes))
    ]
    return out


class Verdict(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def run_of(self, name, **changes):
        path = os.path.join(self.folder.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("a line that is not JSON\n" + "".join(json.dumps(row) + "\n" for row in lines(**changes)))
        return path

    def verdict(self, **changes):
        ours, peer = summary.read(self.run_of("ours")), summary.read(self.run_of("peer", **changes))
        return summary.compare(ours, peer, summary.TOLERANCE)

    def test_the_gap_of_a_right_run_passes_the_shipped_tolerance(self):
        report, same = self.verdict(losses=[0.30, 0.20 + 6e-8, 0.10])
        self.assertTrue(same, report)
        self.assertIn("largest loss gap 6.00e-08 at step 2", report[4])
        self.assertIn("ours 20.0", report[2])  # the median leaves the first step of each shape out

    def test_the_gap_of_a_run_without_weight_decay_fails_it(self):
        self.assertFalse(self.verdict(losses=[0.30, 0.20, 0.10 + 2.8e-7])[1])

    def test_a_loss_that_is_not_finite_fails_at_any_step(self):
        for bad in (math.nan, math.inf):
            for at in range(3):
                losses = [bad if i == at else loss for i, loss in enumerate(LOSSES)]
                self.assertFalse(self.verdict(losses=losses)[1], (bad, at))

    def test_spearman_is_required_and_compared(self):
        self.assertFalse(self.verdict(spearman=(0.5,))[1])
        self.assertFalse(self.verdict(spearman=(0.5, math.nan))[1])
        self.assertFalse(self.verdict(spearman=(0.5, 0.61))[1])

    def test_runs_that_cannot_be_compared_say_why(self):
        for changes, why in (
            ({"shapes": [(8, True), (8, False), (24, True)]}, "same steps"),
            ({"threads": 4}, "thread counts"),
            ({"token_ids": "bb"}, "token ids"),
            ({"losses": [], "shapes": []}, "same steps"),
        ):
            report, same = self.verdict(**changes)
            self.assertFalse(same)
            self.assertIn(why, report[0])

    def test_the_command_exits_1_when_the_runs_differ(self):
        ours = self.run_of("ours")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(summary.main(["summary.py", ours, self.run_of("peer")]), 0)
            self.assertEqual(summary.main(["summary.py", ours, self.run_of("bad", losses=[0.30, math.nan, 0.10])]), 1)
            far = self.run_of("far", losses=[0.30, 0.21, 0.10])
            self.assertEqual(summary.main(["summary.py", ours, far, "0.1"]), 0)
        self.assertIn("verdict: the runs DIFFER", out.getvalue())


if __name__ == "__main__":
    unittest.main()
