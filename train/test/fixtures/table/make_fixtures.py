"""Writes `scores.csv` and `scores.json` (the columns it must read as) for `test/table_test.agc`, with the standard
library alone. A manual script, run from `train/`: `python3 -I test/fixtures/table/make_fixtures.py`.

`name` has a comma, quotes and a non-ASCII letter, which a CSV writer must quote, and `gap` a missing cell.
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent

COLUMNS = {
    "x1": [0.5, -1.25, 2.0, 3.5, 0.0, 1.0],
    "x2": [1.0, 0.25, -0.5, 2.0, 4.0, 8.0],
    "count": [1, 2, 3, 4, 5, 6],
    "label": [0, 1, 1, 0, 2, 2],
    "name": ["ann", "bob, jr", "café", "", "dee", 'eve "x"'],
    "gap": [0.5, 1.5, None, 3.5, 4.5, 5.5],
}


def main():
    names = list(COLUMNS)
    with open(HERE / "scores.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(names)
        for i in range(6):
            writer.writerow(["" if COLUMNS[n][i] is None else COLUMNS[n][i] for n in names])
    (HERE / "scores.json").write_text(json.dumps(COLUMNS, indent=1, ensure_ascii=True) + "\n")


main()
