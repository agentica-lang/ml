"""Prints the tables of benchmarks/README.md from the raw results of bench_whisper.sh.

Usage: python3 -I summarize.py RESULTS_DIR TRANSCRIBE_GOLDEN_JSON

For each row and clip: the load time, the first (warm-up) transcription, the median of the timed ones as milliseconds
and as a real-time factor (time over clip length), the peak memory of a load-only process and of the whole one, and
whether the token ids equal the ones Hugging Face's `generate` gave (the golden file).
"""

import json
import re
import statistics
import sys
from pathlib import Path

CLIPS = {"jfk": ("jfk", 176000 / 16000, "en"), "es-hola": ("es-hola", 42353 / 16000, "es")}
ROWS = [
    ("agentica-ggml-cpu", "ours, ggml CPU"),
    ("agentica-ggml-vulkan", "ours, ggml Vulkan"),
    ("agentica-libtorch", "ours, libtorch CPU"),
    ("wcpp-cpu-f32", "whisper.cpp CPU, F32"),
    ("wcpp-cpu-f32-nofa", "whisper.cpp CPU, F32, no flash attention"),
    ("wcpp-cpu-f16", "whisper.cpp CPU, F16"),
    ("wcpp-vulkan-f32", "whisper.cpp Vulkan, F32"),
    ("wcpp-vulkan-f32-nofa", "whisper.cpp Vulkan, F32, no flash attention"),
    ("wcpp-vulkan-f16", "whisper.cpp Vulkan, F16"),
]


def read(path):
    out = {"run": []}
    for line in path.read_text(encoding="utf8").splitlines():
        key, _, rest = line.partition("\t")
        if key == "run":
            out["run"].append(float(rest))
        elif key == "phase":
            out.setdefault("phase", []).append(rest.split("\t"))
        else:
            out[key] = rest
    return out


def per_run(rows, runs):
    """Our rows carry `NAME MS STEPS`. whisper.cpp's carry `NAME MS`, its totals over the runs: its stages, per run."""
    if all(len(row) == 3 for row in rows):
        return rows
    ms = {name: float(value) / runs for name, value in rows}
    decode = ms["decode"] + ms["batchd"] + ms["prompt"]
    return [
        ["log_mel", ms["mel"], 0],
        ["encoder", ms["encode"], 0],
        ["decode and prompt", decode, 0],
        ["sample", ms["sample"], 0],
    ]


def rss_mb(path):
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return int(m.group(1)) / 1024 if m else float("nan")


def main():
    results = Path(sys.argv[1])
    golden = json.loads(Path(sys.argv[2]).read_text(encoding="utf8"))["cases"]
    want = {}
    for c in golden:
        if not c["timestamps"] and c["task"] == "transcribe":
            want[c["clip"]] = " ".join(map(str, c["tokens"]))

    print("| row | clip | load ms | first ms | median ms | real-time factor | RSS after load MB | peak RSS MB | tokens |")
    print("|---|---|---|---|---|---|---|---|---|")
    phases = {}
    for row, label in ROWS:
        for clip, (_, seconds, _) in CLIPS.items():
            base = results / f"{row}.{clip}"
            if not Path(f"{base}.tsv").exists():
                continue
            r = read(Path(f"{base}.tsv"))
            median = statistics.median(r["run"])
            same = "same" if r.get("tokens") == want[clip] else "DIFFERENT"
            print(
                f"| {label} | {clip} | {float(r['load']):.0f} | {float(r['warmup']):.0f} | {median:.0f} "
                f"| {median / 1000 / seconds:.3f} | {rss_mb(Path(f'{base}.load.time')):.0f} | {rss_mb(Path(f'{base}.time')):.0f} | {same} |"
            )
            if "phase" in r:
                phases[(label, clip)] = per_run(r["phase"], len(r["run"]))
            if same != "same":
                print(f"<!-- {label} {clip} tokens {r.get('tokens')} text {r.get('text')!r} -->")

    print("\n| row | clip | stage | ms | VM steps |")
    print("|---|---|---|---|---|")
    for (label, clip), rows in phases.items():
        for name, ms, steps in rows:
            shown = f"{int(steps):,}" if int(steps) else "-"
            print(f"| {label} | {clip} | {name} | {float(ms):.0f} | {shown} |")


if __name__ == "__main__":
    main()
