#!/bin/sh
# Qwen3-0.6B generation, ours against llama.cpp, interleaved: every round runs each row once, so a change of load
# on the machine hits all of them. Never in CI.
#
#   benchmarks/bench_qwen.sh MODEL_DIR GGUF_DIR LLAMA_CPU_BUILD LLAMA_VULKAN_BUILD [ROUNDS]
#
# MODEL_DIR is the Hugging Face checkout of Qwen/Qwen3-0.6B; GGUF_DIR holds qwen3-0.6b-f32.gguf and
# qwen3-0.6b-f16.gguf (made with llama.cpp's convert_hf_to_gguf.py --outtype f32|f16, run as python3 -I); the two
# builds are llama.cpp's out-of-tree builds of llama-bench (CPU, and -DGGML_VULKAN=ON). `agentica` must be on
# the PATH (each row builds or finds its ggml engine with `transformers/tools/build-engine.sh`, and the Vulkan one
# needs the SDK on the PATH when it is not cached) and `/usr/bin/time` is GNU time. ROUNDS defaults to 5. Prints the
# raw lines, then the medians. The rows `ours-*-f16` load the big matrices as half weights (`WEIGHTS=f16`) and
# `ours-vulkan-f16-fp16` also lets Vulkan multiply in fp16 (`VULKAN_FP16=1`), which does not keep token-exactness
# against Hugging Face; every other row pins both off, whatever the caller exports.
set -eu

model=$1
ggufs=$2
cpu=$3
vulkan=$4
rounds=${5:-5}
here=$(cd "$(dirname "$0")" && pwd)
work=${TMPDIR:-/tmp}/bench_qwen.$$
mkdir -p "$work"
out=$work/rows.txt
: > "$out"

ours() { # name device [VAR=value...]: WEIGHTS=f32 and VULKAN_FP16=0 unless the row says otherwise
  name=$1
  device=$2
  shift 2
  exports=$(cd "$here/../transformers" && sh tools/build-engine.sh "$device") || { echo "bench_qwen.sh: no engine for $device" >&2; exit 1; }
  (cd "$here/../transformers" &&
    eval "$(printf '%s\n' "$exports" | grep '^export ')" &&
    env WEIGHTS=f32 VULKAN_FP16=0 "$@" /usr/bin/time -f "RSS %M" -o "$work/time" agentica --max-steps 100000000000 tools/bench_qwen.agc "$model" "$device" 2> "$work/err" |
    sed -n "s/^RESULT /$name /p") > "$work/line" || { cat "$work/err" >&2; exit 1; }
  [ -s "$work/line" ] || { echo "bench_qwen.sh: row $name printed no RESULT" >&2; cat "$work/err" >&2; exit 1; }
  echo "$(cat "$work/line") $(sed -n 's/^RSS //p' "$work/time")" >> "$out"
}

llama() { # name build gguf extra...
  name=$1
  build=$2
  gguf=$3
  shift 3
  /usr/bin/time -f "RSS %M" -o "$work/time" "$build/bin/llama-bench" -m "$gguf" -p 128 -n 64 -t 8 -r 1 -o csv "$@" > "$work/csv" 2> "$work/err" ||
    { cat "$work/err" >&2; exit 1; }
  python3 -I - "$name" "$work/csv" "$(sed -n 's/^RSS //p' "$work/time")" >> "$out" << 'PY'
import csv, sys
name, path, rss = sys.argv[1:4]
rates = {}
for row in csv.DictReader(open(path)):
    rates[row["n_prompt"] + "/" + row["n_gen"]] = float(row["avg_ts"])
print(name, "-", rates["128/0"], rates["0/64"], rss)
PY
}

round=0
while [ "$round" -lt "$rounds" ]; do
  round=$((round + 1))
  ours "ours-cpu" cpu WEIGHTS=f32
  ours "ours-cpu-f16" cpu WEIGHTS=f16
  llama llama-cpu-f32 "$cpu" "$ggufs/qwen3-0.6b-f32.gguf"
  llama llama-cpu-f16 "$cpu" "$ggufs/qwen3-0.6b-f16.gguf"
  ours "ours-vulkan" vulkan WEIGHTS=f32
  ours "ours-vulkan-f16" vulkan WEIGHTS=f16
  ours "ours-vulkan-f16-fp16" vulkan WEIGHTS=f16 VULKAN_FP16=1
  llama llama-vulkan-f32 "$vulkan" "$ggufs/qwen3-0.6b-f32.gguf" -ngl 99
  llama llama-vulkan-f16 "$vulkan" "$ggufs/qwen3-0.6b-f16.gguf" -ngl 99
  echo "round $round done, load average $(cut -d' ' -f1 /proc/loadavg)" >&2
done

echo "row load_ms prompt_tokens_per_s decode_tokens_per_s rss_kb"
cat "$out"
python3 -I - "$out" << 'PY'
import statistics, sys
rows = {}
for line in open(sys.argv[1]):
    name, load, pp, tg, rss = line.split()
    rows.setdefault(name, []).append((load, float(pp), float(tg), float(rss)))
print("median of", len(next(iter(rows.values()))), "rounds")
print("row prompt_tokens_per_s decode_tokens_per_s peak_rss_gb load_s")
for name, r in rows.items():
    load = [float(x[0]) for x in r if x[0] != "-"]
    print(name, round(statistics.median(x[1] for x in r), 1), round(statistics.median(x[2] for x in r), 2),
          round(statistics.median(x[3] for x in r) / 1048576, 2), round(statistics.median(load) / 1000, 2) if load else "-")
PY
