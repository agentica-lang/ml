"""Splits one decoder step from two profile reports of asr/tools/bench_step.agc: one with PASSES 0 and one with PASSES N.

    python3 -I benchmarks/step_profile.py BASE.txt FULL.txt STEPS

The reports are the AGENTICA_VM_TRACE_FILE of the two runs. A section's rows of FULL minus those of BASE are the steps'
(the load, the log-mel, the encoder and the cross-attention cache are in both). STEPS is the number of decoder steps
FULL ran more than BASE: PASSES times the tokens and the prompt (5 passes of 23 tokens and the prompt is 120). Prints,
per step: the calls, the VM steps and the wall time with the profile on, which slows the VM, so the sections only tell
where the time goes. The `ml.*` sections also count the evaluations of the encoder and the cross-attention cache that
both reports hold (the subtraction removes them), and `ml.compile` of the first pass lands in the steps.
"""

import re
import sys

ROW = re.compile(r"profile (section|native) (.*)")
USAGE = "usage: python3 -I step_profile.py BASE.txt FULL.txt STEPS (STEPS a positive integer)"


def rows(path):
    out = {}
    with open(path) as report:
        for line in report:
            m = ROW.search(line)
            if not m:
                continue
            fields = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
            name = fields.get("name")
            if name is None:
                continue
            out[(m.group(1), name)] = {k: int(v) for k, v in fields.items() if re.fullmatch(r"\d+", v)}
    return out


def main(argv):
    if len(argv) != 4 or not re.fullmatch(r"\d+", argv[3]) or int(argv[3]) == 0:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        base, full = rows(argv[1]), rows(argv[2])
    except OSError as e:
        print(f"step_profile: {e}\n{USAGE}", file=sys.stderr)
        return 2
    n = int(argv[3])
    print(f"{'section':14} {'calls/step':>10} {'VM steps/step':>14} {'ms/step':>9}")
    for key in sorted(full, key=lambda k: -(full[k].get("us", 0) - base.get(k, {}).get("us", 0))):
        kind, name = key
        f, b = full[key], base.get(key, {})
        calls = f.get("calls", 0) - b.get("calls", 0)
        us = f.get("us", 0) - b.get("us", 0)
        steps = f.get("steps", 0) - b.get("steps", 0)
        if kind == "section" and not name.startswith("ml."):
            continue
        if us < 0 or calls < 0:
            print(f"step_profile: {name} went down from BASE to FULL: are they swapped?", file=sys.stderr)
            continue
        if calls == 0:
            continue
        label = name if kind == "section" else f"native {name}"
        print(f"{label:14} {calls / n:10.1f} {steps / n:14.0f} {us / n / 1000:9.2f}")
    return 0


sys.exit(main(sys.argv))
