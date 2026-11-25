"""Writes `tokenizer/unicode_data.agc`: the Unicode tables the tokenizers need
(Agentica has no Unicode properties or normalization of its own), from Python's
`unicodedata`. Standard library only. Frozen data: run it by hand when the
tables must change, then `agentica fmt` the output, and commit both.

    python3 -I tools/unicode_tables.py > tokenizer/unicode_data.agc
    agentica fmt tokenizer/unicode_data.agc

Ranges are flat lists `[lo, hi, lo, hi, ...]`, both ends included, sorted.
"""

import sys
import unicodedata

# The tables are Unicode 15.1's (Python 3.13); another version writes other
# tables, so this fails loudly instead of changing the checked-in file.
VERSION = "15.1.0"
assert unicodedata.unidata_version == VERSION, f"needs Unicode {VERSION}, this Python has {unicodedata.unidata_version}"

LAST = 0x10FFFF
SURROGATES = range(0xD800, 0xE000)


def ranges(test):
    out, start = [], None
    for cp in range(LAST + 1):
        hit = cp not in SURROGATES and test(cp)
        if hit and start is None:
            start = cp
        if not hit and start is not None:
            out += [start, cp - 1]
            start = None
    if start is not None:
        out += [start, LAST]
    return out


def category(*prefixes):
    return ranges(lambda cp: unicodedata.category(chr(cp)).startswith(prefixes))


def decomposition(cp):
    """The canonical full decomposition of `cp` (NFD of the one character)."""
    d = unicodedata.normalize("NFD", chr(cp))
    return [ord(c) for c in d] if d != chr(cp) else None


def lines(name, values, per=12):
    rows = [", ".join(str(v) for v in values[i:i + per]) for i in range(0, len(values), per)]
    body = ",\n  ".join(rows)
    return f"{name} = [\n  {body},\n]\n"


def main():
    # White_Space, as Oniguruma's \s and Rust's char::is_whitespace read it.
    space = [0x9, 0xD, 0x20, 0x20, 0x85, 0x85, 0xA0, 0xA0, 0x1680, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029,
             0x202F, 0x202F, 0x205F, 0x205F, 0x3000, 0x3000]

    # Hangul syllables decompose and compose by arithmetic, not by table.
    decomposed = {}
    for cp in range(LAST + 1):
        if cp in SURROGATES or 0xAC00 <= cp <= 0xD7A3:
            continue
        d = decomposition(cp)
        if d is not None:
            decomposed[cp] = d
    keys = sorted(decomposed)
    starts, flat = [], []
    for k in keys:
        starts.append(len(flat))
        flat += [len(decomposed[k])] + decomposed[k]

    # Primary composites: a pair that is a character's one-step canonical
    # decomposition (not a compatibility one) and that NFC joins back.
    pairs = {}
    for cp in range(LAST + 1):
        if cp in SURROGATES or 0xAC00 <= cp <= 0xD7A3:
            continue
        parts = unicodedata.decomposition(chr(cp)).split()
        if len(parts) == 2 and not parts[0].startswith("<"):
            a, b = int(parts[0], 16), int(parts[1], 16)
            if unicodedata.normalize("NFC", chr(a) + chr(b)) == chr(cp):
                pairs[a * 0x200000 + b] = cp
    pair_keys = sorted(pairs)

    # Combining classes, as ranges of one class.
    ccc_lo, ccc_hi, ccc_val = [], [], []
    start, cur = None, 0
    for cp in range(LAST + 2):
        c = unicodedata.combining(chr(cp)) if cp <= LAST and cp not in SURROGATES else 0
        if c != cur:
            if cur:
                ccc_lo.append(start)
                ccc_hi.append(cp - 1)
                ccc_val.append(cur)
            start, cur = cp, c

    lower = {}
    for cp in range(LAST + 1):
        if cp in SURROGATES:
            continue
        low = chr(cp).lower()
        if low != chr(cp):
            if len(low) != 1:
                assert cp == 0x130, hex(cp)
                continue
            lower[cp] = ord(low)
    lower_keys = sorted(lower)

    out = [
        f"## Unicode {unicodedata.unidata_version} tables for the tokenizers, written by `tools/unicode_tables.py`.\n",
        "## Frozen data: do not edit by hand.\n\n",
        "## Letters (L*), numbers (N*), nonspacing marks (Mn), punctuation (P*), other (C*: controls, format,\n",
        "## unassigned, private use) and White_Space, as ranges `[lo, hi, ...]`.\n",
        lines("LETTER", category("L")),
        "\n",
        lines("NUMBER", category("N")),
        "\n",
        lines("MARK_NONSPACING", category("Mn")),
        "\n## All marks (Mn, Mc and Me): `\\p{M}`, which the split of Qwen3.5 counts with the letters.\n",
        lines("MARK", category("M")),
        "\n",
        lines("PUNCTUATION", category("P")),
        "\n",
        lines("OTHER", category("C")),
        "\n",
        lines("SPACE", space),
        "\n## The canonical decomposition of a character that has one (Hangul syllables aside): the sorted characters,\n",
        "## where each one's entry starts in `DECOMPOSITION`, which holds the count and then the characters.\n",
        lines("DECOMPOSED", keys),
        "\n",
        lines("DECOMPOSED_AT", starts),
        "\n",
        lines("DECOMPOSITION", flat),
        "\n## The primary composites: for the sorted keys `first * 2097152 + second` the character that joins them.\n",
        lines("COMPOSE_PAIR", pair_keys),
        "\n",
        lines("COMPOSE_TO", [pairs[k] for k in pair_keys]),
        "\n## The combining classes that are not 0, as ranges `CLASS_FROM[i]...CLASS_TO[i]` of class `CLASS_OF[i]`.\n",
        lines("CLASS_FROM", ccc_lo),
        "\n",
        lines("CLASS_TO", ccc_hi),
        "\n",
        lines("CLASS_OF", ccc_val),
        "\n## The lowercase of a character, one character each (U+0130 lowercases to two and is the code's).\n",
        lines("LOWER_FROM", lower_keys),
        "\n",
        lines("LOWER_TO", [lower[k] for k in lower_keys]),
    ]
    sys.stdout.write("".join(out))


main()
