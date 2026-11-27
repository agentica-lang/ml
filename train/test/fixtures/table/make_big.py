"""Writes `big.parquet`, a file of about 2.2 MiB for the chunked read of `test/table_test.agc`.

A manual script, not run by CI (pyarrow is pinned as in `make_fixtures.py`). From `train/`:
  ~/.cache/agentica-packages/venv-pyarrow/bin/python test/fixtures/table/make_big.py

`id` counts the rows from 0; `payload` is 1100 random letters and digits (seeded), which snappy cannot shrink, in a file
without compression.
"""

import random
import string
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROWS = 2000
WIDTH = 1100

rng = random.Random(1)
alphabet = string.ascii_letters + string.digits
payload = ["".join(rng.choices(alphabet, k=WIDTH)) for _ in range(ROWS)]
table = pa.table({"id": pa.array(range(ROWS), pa.int64()), "payload": pa.array(payload, pa.string())})
pq.write_table(table, Path(__file__).parent / "big.parquet", compression="none", use_dictionary=False)
