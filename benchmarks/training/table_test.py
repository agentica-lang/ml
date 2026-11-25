#!/usr/bin/env python3
# The loss-curve verdict of table.py: a curve that does not move, or is not finite, or is short, is WRONG, and one that
# follows the oracle is not. Run: python3 -I benchmarks/training/table_test.py
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import table  # noqa: E402

SGD_ORACLE = [2.3029225 - 0.00017 * i for i in range(10)]
ADAM_ORACLE = [2.3029222, 2.2967281, 2.2729499, 2.2492206, 2.2093322, 2.1662831, 2.1, 2.05, 1.99, 1.93]


class Verdict(unittest.TestCase):
    def test_a_curve_that_follows_the_oracle_is_right(self):
        noisy = [x + 1e-7 for x in SGD_ORACLE]
        self.assertIsNone(table.verdict(noisy, SGD_ORACLE, "sgd", 10))
        self.assertIsNone(table.verdict(ADAM_ORACLE, ADAM_ORACLE, "adam", 10))

    def test_an_optimizer_that_does_nothing_is_wrong(self):
        flat = [SGD_ORACLE[0]] * 10
        self.assertIsNotNone(table.verdict(flat, SGD_ORACLE, "sgd", 10))
        self.assertIsNotNone(table.verdict([ADAM_ORACLE[0]] * 10, ADAM_ORACLE, "adam", 10))

    def test_a_learning_rate_of_zero_in_the_smallest_run_is_wrong(self):
        tiny = [2.3025850 - 3.6e-7 * i for i in range(10)]
        self.assertIsNotNone(table.verdict([tiny[0]] * 10, tiny, "sgd", 10))

    def test_a_learning_rate_ten_times_too_small_is_wrong(self):
        slow = [SGD_ORACLE[0] + (x - SGD_ORACLE[0]) / 10 for x in SGD_ORACLE]
        self.assertIsNotNone(table.verdict(slow, SGD_ORACLE, "sgd", 10))

    def test_a_curve_that_is_not_finite_or_short_is_wrong(self):
        nan = SGD_ORACLE[:5] + [float("nan")] + SGD_ORACLE[6:]
        self.assertIsNotNone(table.verdict(nan, SGD_ORACLE, "sgd", 10))
        self.assertIsNotNone(table.verdict([float("inf")] * 10, SGD_ORACLE, "sgd", 10))
        self.assertIsNotNone(table.verdict(SGD_ORACLE[:4], SGD_ORACLE, "sgd", 10))

    def test_a_wrong_first_loss_is_wrong(self):
        shifted = [x + 0.01 for x in SGD_ORACLE]
        self.assertIsNotNone(table.verdict(shifted, SGD_ORACLE, "sgd", 10))


if __name__ == "__main__":
    unittest.main()
