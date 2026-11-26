#!/usr/bin/env python3
"""Fails when the line coverage of a package is below FLOOR percent.

Usage, from the package folder after `agentica test --coverage`:

    python3 ../tools/coverage_check.py coverage/lcov.info --package tensor

It counts the lines of the package's own files, and skips the files of the
workspace packages it reads (`pkg/<name>@workspace/...`, covered by their own
job) and the files in EXCLUDED, which CI cannot run. A package below the floor
gets tests, not a lower floor or a longer list here.
"""

import argparse
import sys

FLOOR = 80.0

# Files that CI cannot run, by package, with the reason. Each is counted as not
# there. Code that only runs with a GPU or a real model is listed here;
# anything CI can run stays counted.
EXCLUDED = {
    "tensor": {
        "src/torch.agc": "the libtorch backend: CI neither builds nor installs libtorch, so its tests skip",
        "src/torch_dtypes.agc": "the dtype tables of the libtorch backend, used only by src/torch.agc",
        "src/native.agc": "the FFI boundary of native/torch.c, which needs libtorch to build",
        "test/vulkan_mlp.agc": "the helper program of the Vulkan tests: CI has no Vulkan device",
    },
    "vision": {
        "test/hf.agc": "the helper of the *_hf_test.agc tests, which need a real Hugging Face checkpoint",
    },
}

WORKSPACE_MARK = "@workspace/"
WORST_SHOWN = 10


def read_lcov(path):
    """Returns {file: (lines, hit)} from an LCOV report."""
    files = {}
    name = None
    with open(path, encoding="utf-8") as report:
        for line in report:
            line = line.rstrip("\n")
            if line.startswith("SF:"):
                name = line[3:]
                files[name] = [0, 0]
            elif line.startswith("LF:") and name is not None:
                files[name][0] = int(line[3:])
            elif line.startswith("LH:") and name is not None:
                files[name][1] = int(line[3:])
            elif line == "end_of_record":
                name = None
    return {name: (lines, hit) for name, (lines, hit) in files.items()}


def percent(hit, lines):
    return 100.0 * hit / lines if lines else 100.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lcov", help="coverage/lcov.info of `agentica test --coverage`")
    parser.add_argument("--package", required=True, help="the package folder name")
    args = parser.parse_args()

    try:
        files = read_lcov(args.lcov)
    except OSError as error:
        print(f"coverage: cannot read {args.lcov}: {error}", file=sys.stderr)
        return 2

    excluded = EXCLUDED.get(args.package, {})
    counted = {
        name: counts
        for name, counts in files.items()
        if WORKSPACE_MARK not in name and name not in excluded
    }
    skipped_deps = sum(1 for name in files if WORKSPACE_MARK in name)
    lines = sum(counts[0] for counts in counted.values())
    hit = sum(counts[1] for counts in counted.values())
    if lines == 0:
        print(f"coverage: {args.package}: no lines counted in {args.lcov}", file=sys.stderr)
        return 2

    total = percent(hit, lines)
    print(f"coverage: {args.package} {hit}/{lines} lines ({total:.1f}%), floor {FLOOR:.0f}%")
    print(f"  {skipped_deps} files of workspace packages not counted")
    for name, reason in excluded.items():
        if name not in files:
            print(f"coverage: {args.package}: excluded {name} is not in the report; update EXCLUDED", file=sys.stderr)
            return 2
        lines_x, hit_x = files[name]
        print(f"  excluded {name} {hit_x}/{lines_x}: {reason}")

    if total >= FLOOR:
        return 0

    under = sorted(
        ((percent(h, n), name, h, n) for name, (n, h) in counted.items() if n and percent(h, n) < FLOOR)
    )
    print(f"coverage: {args.package} is below {FLOOR:.0f}%. Files under the floor, worst first:")
    for pct, name, h, n in under[:WORST_SHOWN]:
        print(f"  {name} {h}/{n} ({pct:.1f}%), {n - h} lines not run")
    return 1


if __name__ == "__main__":
    sys.exit(main())
