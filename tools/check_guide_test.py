"""tools/check_guide.py: a page that shows its program passes, and a program that changed does not."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_guide

PROGRAM = "## doc\n\nimport A from \"a\"\n\nX = 1\n\ndef f() -> Int\n  X\nend\n"
PAGE = "text\n\n```agentica\nimport A from \"a\"\n\nX = 1\n```\n\nmore\n\n```agentica\ndef f() -> Int\n  X\nend\n```\n"


class Guide(unittest.TestCase):
    def test_fences_in_order_are_the_program(self):
        self.assertIsNone(check_guide.difference(PAGE, PROGRAM))

    def test_a_changed_program_is_found(self):
        found = check_guide.difference(PAGE, PROGRAM.replace("X = 1", "X = 2"))
        self.assertIn("X = 1", found)

    def test_a_missing_fence_is_found(self):
        self.assertIsNotNone(check_guide.difference(PAGE.split("more")[0], PROGRAM))


if __name__ == "__main__":
    unittest.main()
