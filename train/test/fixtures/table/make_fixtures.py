"""Writes `scores.csv`, `scores.parquet` and `scores.json` (the columns they must read as) for `test/table_test.agc`.

A manual script, not run by CI. The CSV needs the standard library, the Parquet file pyarrow (pinned: pyarrow==26.0.1,
pip install --only-binary :all: pyarrow==26.0.1). From `train/`:
  ~/.cache/agentica-packages/venv-pyarrow/bin/python test/fixtures/table/make_fixtures.py

`name` has a comma, quotes and a non-ASCII letter, which a CSV writer must quote, and `gap` a missing cell (a null in the
Parquet file). The Parquet file has also a `flag` column of booleans, `x2` as a 32-bit float, `count` as 64-bit and `label`
as 32-bit integers, which a CSV cannot tell apart.
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


FLAG = [True, False, True, True, False, False]


def parquet():
    import pyarrow as pa
    import pyarrow.parquet as pq

    kinds = {"x1": pa.float64(), "x2": pa.float32(), "count": pa.int64(), "label": pa.int32(), "name": pa.string(), "gap": pa.float64()}
    arrays = {n: pa.array(v, type=kinds[n]) for n, v in COLUMNS.items()}
    arrays = {**{n: a for n, a in arrays.items() if n in ("x1", "x2", "count", "label")}, "flag": pa.array(FLAG, pa.bool_()),
              "name": arrays["name"], "gap": arrays["gap"]}
    pq.write_table(pa.table(arrays), HERE / "scores.parquet", compression="snappy", use_dictionary=False)


def main():
    names = list(COLUMNS)
    with open(HERE / "scores.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(names)
        for i in range(6):
            writer.writerow(["" if COLUMNS[n][i] is None else COLUMNS[n][i] for n in names])
    (HERE / "scores.json").write_text(json.dumps({**COLUMNS, "flag": FLAG}, indent=1, ensure_ascii=True) + "\n")
    parquet()


main()
