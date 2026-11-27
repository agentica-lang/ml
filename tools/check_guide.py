#!/usr/bin/env python3
"""Fails when the code of a guide page is not the program it shows.

Usage, from a package folder:

    python3 ../tools/check_guide.py ../site/content/train/finetune-minilm.md tools/finetune_stsb.agc

It joins the `agentica` fences of the page in order and compares them with the program from its first `import` line
to the end, line by line and ignoring blank lines. The doc comment before the imports is not part of the page.
"""

import re
import sys
from pathlib import Path

FENCE = re.compile(r"^```agentica\n(.*?)^```$", re.DOTALL | re.MULTILINE)


def lines(text):
    return [line.rstrip() for line in text.split("\n") if line.strip()]


def shown(page):
    return lines("\n".join(FENCE.findall(page)))


def program(source):
    text = source[source.index("\nimport ") + 1 :] if "\nimport " in source else source
    return lines(text)


def difference(page, source):
    """The first line where the page and the program part, or None when they are the same."""
    a, b = shown(page), program(source)
    for i in range(max(len(a), len(b))):
        left = a[i] if i < len(a) else "(the page ends)"
        right = b[i] if i < len(b) else "(the program ends)"
        if left != right:
            return f"line {i + 1} of the code: the page has {left!r}, the program {right!r}"
    return None


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: check_guide.py PAGE.md PROGRAM.agc")
    page, source = (Path(p).read_text(encoding="utf-8") for p in sys.argv[1:])
    found = difference(page, source)
    if found:
        sys.exit(f"{sys.argv[1]} does not show {sys.argv[2]}: {found}")
    print(f"{sys.argv[1]} shows {sys.argv[2]}")


if __name__ == "__main__":
    main()
