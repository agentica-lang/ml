#!/usr/bin/env python3
"""Writes a split of `sentence-transformers/stsb` as JSON lines, for both sides of the benchmark to read the same rows:

    python -I stsb_jsonl.py SPLIT.parquet OUT.jsonl

Each line has `sentence1`, `sentence2` and `score` (a float in [0, 1]), in the order of the file. It needs `pyarrow`
(the venv of `run.sh setup`); the Agentica side has no parquet reader yet (agentica-lang/packages#223).
"""

import json
import sys

import pyarrow.parquet as pq


def main():
    source, out = sys.argv[1], sys.argv[2]
    table = pq.read_table(source, columns=["sentence1", "sentence2", "score"])
    rows = table.to_pylist()
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            score = float(row["score"])
            if not 0.0 <= score <= 1.0:
                raise SystemExit(f"{source}: a score of {score} is outside [0, 1]")
            line = {"sentence1": row["sentence1"], "sentence2": row["sentence2"], "score": score}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"{out}: {len(rows)} pairs")


main()
