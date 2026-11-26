#!/bin/sh
# Qwen3-0.6B generation, ours against llama.cpp, interleaved: every round runs each row once, so a change of load
# on the machine hits all of them. Never in CI.
#
#   benchmarks/bench_qwen.sh MODEL_DIR GGUF_DIR LLAMA_CPU_BUILD LLAMA_VULKAN_BUILD [ROUNDS]
#
# `ROWS=vulkan` (several: `ROWS=q4k,f16`) runs only the rows whose name contains one of the words. MODEL_DIR is the
# Hugging Face checkout of Qwen/Qwen3-0.6B; GGUF_DIR holds qwen3-0.6b-f32.gguf and qwen3-0.6b-f16.gguf (made with
# llama.cpp's convert_hf_to_gguf.py --outtype f32|f16, run as python3 -I) and qwen3-0.6b-q4_k_m.gguf (llama-quantize of
# the F16 file to Q4_K_M; the rows `*-q4k` and `*-q4km` run it, and the first ones load it with `WEIGHTS=q4_k`); the two
# builds are llama.cpp's out-of-tree builds of llama-bench (CPU, and -DGGML_VULKAN=ON). `agentica` must be on
# the PATH (each row builds or finds its ggml engine with `transformers/tools/build-engine.sh`, and the Vulkan one
# needs the SDK on the PATH when it is not cached) and `/usr/bin/time` is GNU time. ROUNDS defaults to 5. Prints the
# raw lines, then the medians. The rows `ours-*-f16` load the big matrices as half weights (`WEIGHTS=f16`); Vulkan
# multiplies in fp16 (`VULKAN_FP16=1`, the library default), except in `ours-vulkan-exact`, which pins it off
# (`VULKAN_FP16=0`: exact `float32`, as the checks against Hugging Face run). Every row pins both variables,
# whatever the caller exports. The last column is the most that the GPU memory in use (VRAM and GTT, sampled every 0.1 s)
# rose above its value before the row, in MB: 0 on the CPU rows, and the weights that llama.cpp maps are in it.
set -eu

model=$1
ggufs=$2
cpu=$3
vulkan=$4
rounds=${5:-5}
q4km=$ggufs/qwen3-0.6b-q4_k_m.gguf
here=$(cd "$(dirname "$0")" && pwd)
work=${TMPDIR:-/tmp}/bench_qwen.$$
mkdir -p "$work"
out=$work/rows.txt
: > "$out"
cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sed 's/^/VRAM used before the run (bytes): /' >&2

want() { # name: true when ROWS is unset or one of its words is in the name
  [ -z "${ROWS:-}" ] && return 0
  for word in $(echo "$ROWS" | tr , ' '); do
    case $1 in *"$word"*) return 0 ;; esac
  done
  return 1
}

gpu_used() { # MB of VRAM and GTT in use, summed over the cards
  total=0
  for f in /sys/class/drm/card*/device/mem_info_vram_used /sys/class/drm/card*/device/mem_info_gtt_used; do
    [ -r "$f" ] && total=$((total + $(cat "$f")))
  done
  echo $((total / 1048576))
}

measured() { # name command...: runs it and appends the peak GPU memory above the start to $work/gpu
  base=$(gpu_used)
  : > "$work/peak"
  (
    peak=$base
    while :; do
      now=$(gpu_used)
      [ "$now" -gt "$peak" ] && peak=$now
      echo $((peak - base)) > "$work/peak"
      sleep 0.1
    done
  ) &
  sampler=$!
  status=0
  "$@" || status=$?
  kill "$sampler" 2> /dev/null || true
  wait "$sampler" 2> /dev/null || true
  return $status
}

run_ours() { # subject device name [VAR=value...]
  subject=$1
  device=$2
  name=$3
  shift 3
  exports=$(cd "$here/../transformers" && sh tools/build-engine.sh "$device") || { echo "bench_qwen.sh: no engine for $device" >&2; return 1; }
  (cd "$here/../transformers" &&
    eval "$(printf '%s\n' "$exports" | grep '^export ')" &&
    env WEIGHTS=f32 VULKAN_FP16=1 "$@" /usr/bin/time -f "RSS %M" -o "$work/time" agentica --max-steps 100000000000 tools/bench_qwen.agc "$subject" "$device" 2> "$work/err" |
    sed -n "s/^RESULT /$name /p") > "$work/line" || { cat "$work/err" >&2; return 1; }
  [ -s "$work/line" ] || { echo "bench_qwen.sh: row $name printed no RESULT" >&2; cat "$work/err" >&2; return 1; }
}

ours() { # name device subject [VAR=value...]: WEIGHTS=f32 and VULKAN_FP16=1 unless the row says otherwise
  name=$1
  device=$2
  subject=$3
  shift 3
  want "$name" || return 0
  measured run_ours "$subject" "$device" "$name" "$@" || exit 1
  echo "$(cat "$work/line") $(sed -n 's/^RSS //p' "$work/time") $(cat "$work/peak")" >> "$out"
}

run_llama() { # build gguf extra...
  build=$1
  gguf=$2
  shift 2
  /usr/bin/time -f "RSS %M" -o "$work/time" "$build/bin/llama-bench" -m "$gguf" -p 128 -n 64 -t 8 -r 1 -o csv "$@" > "$work/csv" 2> "$work/err" ||
    { cat "$work/err" >&2; return 1; }
}

llama() { # name build gguf extra...
  name=$1
  want "$name" || return 0
  build=$2
  gguf=$3
  shift 3
  measured run_llama "$build" "$gguf" "$@" || exit 1
  python3 -I - "$name" "$work/csv" "$(sed -n 's/^RSS //p' "$work/time")" "$(cat "$work/peak")" >> "$out" << 'PY'
import csv, sys
name, path, rss, gpu = sys.argv[1:5]
rates = {}
for row in csv.DictReader(open(path)):
    rates[row["n_prompt"] + "/" + row["n_gen"]] = float(row["avg_ts"])
print(name, "-", rates["128/0"], rates["0/64"], rss, gpu)
PY
}

round=0
while [ "$round" -lt "$rounds" ]; do
  round=$((round + 1))
  ours ours-cpu cpu "$model" WEIGHTS=f32
  ours ours-cpu-f16 cpu "$model" WEIGHTS=f16
  ours ours-cpu-q4k cpu "$q4km" WEIGHTS=q4_k
  llama llama-cpu-f32 "$cpu" "$ggufs/qwen3-0.6b-f32.gguf"
  llama llama-cpu-f16 "$cpu" "$ggufs/qwen3-0.6b-f16.gguf"
  llama llama-cpu-q4km "$cpu" "$q4km"
  ours ours-vulkan vulkan "$model" WEIGHTS=f32
  ours ours-vulkan-exact vulkan "$model" VULKAN_FP16=0
  ours ours-vulkan-f16 vulkan "$model" WEIGHTS=f16
  ours ours-vulkan-q4k vulkan "$q4km" WEIGHTS=q4_k
  llama llama-vulkan-f32 "$vulkan" "$ggufs/qwen3-0.6b-f32.gguf" -ngl 99
  llama llama-vulkan-f16 "$vulkan" "$ggufs/qwen3-0.6b-f16.gguf" -ngl 99
  llama llama-vulkan-q4km "$vulkan" "$q4km" -ngl 99
  echo "round $round done, load average $(cut -d' ' -f1 /proc/loadavg)" >&2
done

echo "row load_ms prompt_tokens_per_s decode_tokens_per_s rss_kb gpu_mb"
cat "$out"
python3 -I - "$out" << 'PY'
import statistics, sys
rows = {}
for line in open(sys.argv[1]):
    name, load, pp, tg, rss, gpu = line.split()
    rows.setdefault(name, []).append((load, float(pp), float(tg), float(rss), float(gpu)))
print("median of", len(next(iter(rows.values()))), "rounds")
print("row prompt_tokens_per_s decode_tokens_per_s peak_rss_gb load_s peak_gpu_mb")
for name, r in rows.items():
    load = [float(x[0]) for x in r if x[0] != "-"]
    print(name, round(statistics.median(x[1] for x in r), 1), round(statistics.median(x[2] for x in r), 2),
          round(statistics.median(x[3] for x in r) / 1048576, 2), round(statistics.median(load) / 1000, 2) if load else "-",
          round(statistics.median(x[4] for x in r)))
PY
