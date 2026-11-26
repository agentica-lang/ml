"""Writes `test/fixtures/pyjson_golden.json`: a list of texts that Python's `json.dumps(value, ensure_ascii=False)`
writes for about 300 seeded values (floats at the edges of `repr`'s formats and of random bit patterns, integers,
strings with control characters, quotes and non-ASCII text, nested objects with their keys out of order).
`test/pyjson_test.agc` parses each text and checks that `Pyjson.dumps` writes it back, character for character.

Manual, not run by the tests; the standard library only. Run it from `transformers/`, with Python's isolated mode:

    python -I test/pyjson_golden.py > test/fixtures/pyjson_golden.json
"""

import json
import math
import random
import struct

EDGE_FLOATS = [
    0.0, -0.0, 1.0, -1.0, 0.1, 0.5, 100.0, 1e15, 1e16, 1e17, 123456789012345.0, 1234567890123456.0,
    12345678901234567.0, 1e-4, 1e-5, 0.0001234, 0.00001234, 0.30000000000000004, 5e-324, 2.2250738585072014e-308,
    1.7976931348623157e308, 1e22, 1e23, 1e300, -1e-300, 2.5e-7, 9007199254740993.0, 4.35, 1e21, 1.5e-10,
]
STRINGS = [
    "", "plain", 'quote " and \\ backslash', "tab\t nl\n cr\r bs\b ff\f", "\x00\x01\x1f", "\x7f del", "é ñ 東京 😀",
    "  ", "</script>", "a/b", "'single'", "﻿", " ",
]
KEYS = ["z", "a", "m", "key with space", "é", "\n", "", "10", "b"]


def random_float(rng):
    while True:
        value = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
        if math.isfinite(value):
            return value


def random_value(rng, depth=0):
    kind = rng.randrange(8 if depth < 3 else 5)
    if kind == 0:
        return rng.choice(STRINGS)
    if kind == 1:
        return rng.randint(-(10**rng.randint(1, 25)), 10**rng.randint(1, 25))
    if kind == 2:
        return random_float(rng)
    if kind == 3:
        return rng.choice([True, False, None])
    if kind == 4:
        return "".join(chr(rng.choice([rng.randrange(32, 127), rng.randrange(0, 32), rng.randrange(160, 0xD000)])) for _ in range(rng.randint(0, 12)))
    if kind in (5, 6):
        keys = rng.sample(KEYS, rng.randint(0, 5))
        return {key: random_value(rng, depth + 1) for key in keys}
    return [random_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]


rng = random.Random(275)
values = list(EDGE_FLOATS) + [-value for value in EDGE_FLOATS[2:]] + [[value] for value in EDGE_FLOATS]
values += [random_float(rng) for _ in range(100)] + [rng.randint(-(2**70), 2**70) for _ in range(20)] + STRINGS
values += [random_value(rng) for _ in range(100)]
print(json.dumps([json.dumps(value, ensure_ascii=False) for value in values], ensure_ascii=False))
