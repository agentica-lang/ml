#!/usr/bin/env python3
"""Writes the `sentence` and `label` columns of a SST-2 split as CSV, in the order of the file, the first ROWS rows
when ROWS is given:

    python3 -I tools/sst2_csv.py SPLIT.parquet OUT.csv [ROWS]

The Agentica side reads it with `Table.read` of the train package, which takes a file of at most 1 MiB, so it reads
the 3.1 MB training split of Parquet only through this cut. It needs `pyarrow` (the venv of `benchmarks/stsb/run.sh
setup`).
"""

import csv
import sys

import pyarrow.parquet as pq


def main():
    source, out = sys.argv[1], sys.argv[2]
    rows = pq.read_table(source, columns=["sentence", "label"]).to_pylist()
    if len(sys.argv) > 3:
        rows = rows[: int(sys.argv[3])]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "label"])
        for row in rows:
            if row["sentence"] is None or row["label"] not in (0, 1):
                raise SystemExit(f"{source}: a row without a sentence or with a label outside 0 and 1")
            writer.writerow([row["sentence"], row["label"]])
    print(f"{out}: {len(rows)} rows")


main()
