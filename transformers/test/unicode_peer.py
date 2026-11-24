"""The independent peer of `unicode_test.agc`: what Python's `unicodedata`
says of code point lists. Standard library only.

Usage: python3 -I unicode_peer.py JSON, where JSON is a list of code point
lists. Prints a JSON list with one object per input: `nfd`, `nfc` and `lower`
(each a code point list; `lower` one character at a time, as Rust does), and
`kinds`, a string with one letter per input character: L letter, N number,
M nonspacing mark, P punctuation, C other (control, format, unassigned,
private use), S white space, `.` anything else.
"""

import json
import sys
import unicodedata


def kind(cp):
    c = unicodedata.category(chr(cp))
    if c.startswith("L"):
        return "L"
    if c.startswith("N"):
        return "N"
    if c == "Mn":
        return "M"
    if c.startswith("P"):
        return "P"
    if cp in (0x9, 0xA, 0xB, 0xC, 0xD, 0x85) or c in ("Zs", "Zl", "Zp"):
        return "S"
    if c.startswith("C"):
        return "C"
    return "."


def main():
    out = []
    for cps in json.loads(sys.argv[1]):
        text = "".join(chr(c) for c in cps)
        out.append(
            {
                "nfd": [ord(c) for c in unicodedata.normalize("NFD", text)],
                "nfc": [ord(c) for c in unicodedata.normalize("NFC", text)],
                "lower": [ord(x) for c in text for x in c.lower()],
                "kinds": "".join(kind(c) for c in cps),
            }
        )
    json.dump(out, sys.stdout)


main()
