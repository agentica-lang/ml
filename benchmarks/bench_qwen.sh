#!/bin/sh
# Qwen3-0.6B generation, ours against llama.cpp, interleaved: every round runs each row once, so a change of load
# on the machine hits all of them, and the two rows of a pair swap places from one round to the next. Never in CI.
#
#   benchmarks/bench_qwen.sh MODEL_DIR GGUF_DIR LLAMA_CPU_BUILD LLAMA_VULKAN_BUILD [ROUNDS]
#
# `ROWS=vulkan` (several: `ROWS=q4k,f16`) runs only the rows whose name contains one of the words (a substring, so
# `f16` takes `ours-cpu-f16` and `llama-vulkan-f16`); no match is an error that lists the rows. MODEL_DIR is the
# Hugging Face checkout of Qwen/Qwen3-0.6B; GGUF_DIR holds qwen3-0.6b-f32.gguf and qwen3-0.6b-f16.gguf (made with
# llama.cpp's convert_hf_to_gguf.py --outtype f32|f16, run as python3 -I) and qwen3-0.6b-q4_k_m.gguf (llama-quantize of
# the F16 file to Q4_K_M; the rows `*-q4k` and `*-q4km` run it, and the first ones load it with `WEIGHTS=q4_k`); the two
# builds are llama.cpp's out-of-tree builds of llama-bench (CPU, and -DGGML_VULKAN=ON); a file COMMIT in a build folder
# holds the llama.cpp commit (the script fails without it) and is printed. `agentica` must be on the PATH (each row builds or finds its ggml engine
# with `transformers/tools/build-engine.sh`, and the Vulkan one needs the SDK on the PATH when it is not cached) and
# `/usr/bin/time` is GNU time. ROUNDS defaults to 5.
#
# The CPU rows of both sides run the same number of threads, `THREADS` (default: the physical cores): `TENSOR_THREADS` of
# ours and `-t` of llama-bench; ours prints the count it uses (`ggml_cpu_threads`) and llama-bench its `n_threads`. On
# Vulkan the graph runs on the GPU, and ours has no CPU thread pool there. llama-bench runs `-fa 1` and the `F16` cache, as ours does, and its generation row starts at a
# depth of 128 (`-d 128`), where ours is after the 128 tokens of its prompt: the prompt row is `-p 128 -n 0` from an
# empty cache. Its own `n_threads`, `flash_attn`, `type_k` and `build_commit` are printed after the rows, and the
# sha256 of every GGUF a row reads.
#
# Prints the raw lines (row, load ms, prompt tokens/s, tokens/s, peak RSS in KB, peak GPU MB, round), the medians, and
# for each pair of ours and llama.cpp the ratio of ours to llama.cpp in every round, with its minimum and maximum.
# The rows `ours-*-f16` load the big matrices as half weights (`WEIGHTS=f16`); Vulkan multiplies in fp16
# (`VULKAN_FP16=1`, the library default), except in `ours-vulkan-exact`, which pins it off (`VULKAN_FP16=0`: exact
# `float32`, as the checks against Hugging Face run). Every row pins both variables, whatever the caller exports. The
# GPU column is the most that the GPU memory in use (VRAM and GTT, sampled every 0.1 s) rose above its value before the
# row, in MB, for the Vulkan rows (llama.cpp's mapped weights are in it, not in the RSS); `-` on the CPU rows.
# The rows `ours-vulkan-q4k-n26`, `-n13` and `-n0` run the Q4_K_M file with `AGENTICA_ML_LAYERS_ON_DEVICE` 26, 13 and 0
# (the blocks from the 26th, the 13th or the first on the CPU with their KV cache, the table on the CPU, and the head and
# the final norm on the card for 26 and 13, on the CPU for 0), against llama.cpp's `-ngl 27`, `-ngl 14` and `-ngl 0` on
# the Vulkan build: llama.cpp puts the output layer on the card for `-ngl` 1 or more and the first blocks on the CPU
# (`i_gpu_start = n_layer + 1 - ngl`), so `-ngl N` holds N - 1 blocks on the card, as many as `n` = N - 1; ours puts the
# last blocks on the CPU, llama.cpp the first ones.
set -eu

model=$1
ggufs=$2
cpu=$3
vulkan=$4
rounds=${5:-5}
q4km=$ggufs/qwen3-0.6b-q4_k_m.gguf
threads=${THREADS:-$(lscpu -p=CORE,SOCKET 2> /dev/null | grep -v '^#' | sort -u | wc -l)}
[ "$threads" -gt 0 ] 2> /dev/null || threads=$(nproc)
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/bench_qwen.XXXXXX")
out=$work/rows.txt
meta=$work/meta.txt
: > "$out"
: > "$meta"
sampler=
gpu_warned=
round=0

stop_sampler() {
  [ -n "$sampler" ] && kill "$sampler" 2> /dev/null
  [ -n "$sampler" ] && wait "$sampler" 2> /dev/null
  sampler=
  return 0
}
cleanup() {
  stop_sampler
  rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

all_rows="ours-cpu ours-cpu-f16 ours-cpu-q4k llama-cpu-f32 llama-cpu-f16 llama-cpu-q4km ours-vulkan ours-vulkan-exact ours-vulkan-f16 ours-vulkan-q4k ours-vulkan-q4k-n26 ours-vulkan-q4k-n13 ours-vulkan-q4k-n0 llama-vulkan-f32 llama-vulkan-f16 llama-vulkan-q4km llama-vulkan-q4km-ngl27 llama-vulkan-q4km-ngl14 llama-vulkan-q4km-ngl0"

want() { # name: true when ROWS is unset or one of its comma-separated words is in the name
  [ -z "${ROWS:-}" ] && return 0
  set -f
  oldifs=$IFS
  IFS=,
  found=1
  for word in $ROWS; do
    [ -n "$word" ] && case $1 in *"$word"*) found=0 ;; esac
  done
  IFS=$oldifs
  set +f
  return $found
}

matched=
for name in $all_rows; do
  want "$name" && matched="$matched $name"
done
[ -n "$matched" ] || { echo "bench_qwen.sh: ROWS=$ROWS matches no row; the rows are: $all_rows" >&2; exit 2; }

needs_cpu=
needs_vulkan=
needs_f32=
needs_f16=
needs_q4km=
for name in $matched; do
  case $name in
    llama-cpu-*) needs_cpu=1 ;;
    llama-vulkan-*) needs_vulkan=1 ;;
  esac
  case $name in
    llama-*-f32) needs_f32=1 ;;
    llama-*-f16) needs_f16=1 ;;
    *-q4k* | *-q4km*) needs_q4km=1 ;;
  esac
done
for b in ${needs_cpu:+"$cpu"} ${needs_vulkan:+"$vulkan"}; do
  [ -s "$b/COMMIT" ] || { echo "bench_qwen.sh: $b has no COMMIT file (the llama.cpp commit the build came from)" >&2; exit 2; }
  echo "llama.cpp build $b: commit $(cat "$b/COMMIT"), cmake flags $(cat "$b/CMAKE_FLAGS" 2> /dev/null || echo unknown)" >&2
  grep -E '^GGML_(NATIVE|OPENMP|LLAMAFILE|BLAS|CPU_REPACK|VULKAN):' "$b/CMakeCache.txt" | tr '\n' ' ' | sed 's/^/  its CPU and GPU options: /' >&2
  echo >&2
done
echo "ours: $(grep -o 'GGML_[A-Z]*=[A-Z]*' "$here/../tensor/tools/build-engine.sh" | sort -u | tr '\n' ' ')(ggml $(git -C "$here/../third_party/ggml" describe --tags --always 2> /dev/null || echo unknown), tensor/tools/build-engine.sh)" >&2
echo "threads: $threads for the CPU rows of both sides (TENSOR_THREADS and llama-bench -t)" >&2
for f in ${needs_f32:+"$ggufs/qwen3-0.6b-f32.gguf"} ${needs_f16:+"$ggufs/qwen3-0.6b-f16.gguf"} ${needs_q4km:+"$q4km"}; do
  echo "sha256 $(sha256sum "$f" | cut -d' ' -f1) $f" | tee -a "$meta" >&2
done
cat /sys/class/drm/card*/device/mem_info_vram_used 2> /dev/null | sed 's/^/VRAM used before the run (bytes): /' >&2

gpu_used() { # MB of VRAM and GTT in use, summed over the cards; nothing when the files are missing
  total=0
  seen=
  for f in /sys/class/drm/card*/device/mem_info_vram_used /sys/class/drm/card*/device/mem_info_gtt_used; do
    [ -r "$f" ] || continue
    read -r bytes < "$f"
    total=$((total + bytes))
    seen=1
  done
  [ -n "$seen" ] && echo $((total / 1048576))
  return 0
}

measured() { # gpu command...: runs it; with gpu=1 the peak GPU MB above the start goes to $work/peak, else `-`
  gpu=$1
  shift
  echo - > "$work/peak"
  if [ "$gpu" = 1 ]; then
    base=$(gpu_used)
    if [ -z "$base" ]; then
      [ -n "$gpu_warned" ] || echo "bench_qwen.sh: no /sys/class/drm GPU memory files, the GPU column is -" >&2
      gpu_warned=1
    else
      (
        peak=$base
        while :; do
          now=$(gpu_used)
          [ -n "$now" ] && [ "$now" -gt "$peak" ] && peak=$now
          echo $((peak - base)) > "$work/peak"
          sleep 0.1
        done
      ) &
      sampler=$!
    fi
  fi
  status=0
  "$@" || status=$?
  stop_sampler
  return $status
}

peak_gpu() { # the peak file, `-` when it is empty
  peak=-
  read -r peak < "$work/peak" || true
  echo "${peak:--}"
}

run_ours() { # subject device name [VAR=value...]
  subject=$1
  device=$2
  name=$3
  shift 3
  exports=$(cd "$here/../transformers" && sh tools/build-engine.sh "$device") || { echo "bench_qwen.sh: no engine for $device" >&2; return 1; }
  (cd "$here/../transformers" &&
    eval "$(printf '%s\n' "$exports" | grep '^export ')" &&
    env WEIGHTS=f32 VULKAN_FP16=1 TENSOR_THREADS="$threads" "$@" /usr/bin/time -f "RSS %M" -o "$work/time" agentica --max-steps 100000000000 tools/bench_qwen.agc "$subject" "$device" > "$work/ours.out" 2> "$work/err") || { cat "$work/err" >&2; return 1; }
  sed -n "s/^RESULT /$name /p" "$work/ours.out" > "$work/line"
  sed -n 's/^THREADS //p' "$work/ours.out" > "$work/ours-threads"
  [ -s "$work/line" ] || { echo "bench_qwen.sh: row $name printed no RESULT" >&2; cat "$work/err" >&2; return 1; }
}

ours() { # name device subject [VAR=value...]: WEIGHTS=f32 and VULKAN_FP16=1 unless the row says otherwise
  name=$1
  device=$2
  subject=$3
  shift 3
  want "$name" || return 0
  gpu=0
  [ "$device" = vulkan ] && gpu=1
  measured "$gpu" run_ours "$subject" "$device" "$name" "$@" || exit 1
  read -r line < "$work/line"
  read -r cpu_threads < "$work/ours-threads" && echo "ours $name: cpu_threads=$cpu_threads (ggml_cpu_threads)" >> "$meta"
  read -r rss < "$work/time" || true
  echo "$line ${rss#RSS } $(peak_gpu) $round" >> "$out"
}

run_llama() { # build gguf extra...: the prompt row from an empty cache, then the generation row at a depth of 128
  build=$1
  gguf=$2
  shift 2
  common="-m $gguf -t $threads -fa 1 -ctk f16 -ctv f16 -r 1 -o csv"
  # shellcheck disable=SC2086
  /usr/bin/time -f "RSS %M" -o "$work/time-pp" "$build/bin/llama-bench" $common -p 128 -n 0 "$@" > "$work/csv-pp" 2> "$work/err" || { cat "$work/err" >&2; return 1; }
  # shellcheck disable=SC2086
  /usr/bin/time -f "RSS %M" -o "$work/time-tg" "$build/bin/llama-bench" $common -p 0 -n 64 -d 128 "$@" > "$work/csv-tg" 2> "$work/err" || { cat "$work/err" >&2; return 1; }
}

llama() { # name build gguf extra...
  name=$1
  want "$name" || return 0
  build=$2
  gguf=$3
  shift 3
  gpu=0
  [ "$build" = "$vulkan" ] && gpu=1
  measured "$gpu" run_llama "$build" "$gguf" "$@" || exit 1
  python3 -I - "$name" "$work/csv-pp" "$work/csv-tg" "$work/time-pp" "$work/time-tg" "$(peak_gpu)" "$round" "$meta" >> "$out" << 'PY'
import csv, sys
name, pp, tg, tpp, ttg, gpu, rnd, meta = sys.argv[1:9]
rate = {"pp": None, "tg": None}
for key, path in (("pp", pp), ("tg", tg)):
    rows = list(csv.DictReader(open(path)))
    rate[key] = float(rows[0]["avg_ts"])
    info = {k: rows[0].get(k, "?") for k in ("n_threads", "flash_attn", "type_k", "type_v", "build_commit", "n_depth")}
rss = max(int(open(p).read().split()[-1]) for p in (tpp, ttg))
print(name, "-", rate["pp"], rate["tg"], rss, gpu, rnd)
with open(meta, "a") as f:
    f.write("llama-bench %s: n_threads=%s flash_attn=%s type_k=%s type_v=%s build_commit=%s generation depth=%s\n" % (
        name, info["n_threads"], info["flash_attn"], info["type_k"], info["type_v"], info["build_commit"], info["n_depth"]))
PY
}

# One pair per model and device: ours first in the odd rounds, llama.cpp first in the even ones.
pair() { # ours_function llama_function
  if [ $((round % 2)) -eq 1 ]; then
    "$1"
    "$2"
  else
    "$2"
    "$1"
  fi
}

ours_cpu_f32() { ours ours-cpu cpu "$model" WEIGHTS=f32; }
llama_cpu_f32() { llama llama-cpu-f32 "$cpu" "$ggufs/qwen3-0.6b-f32.gguf"; }
ours_cpu_f16() { ours ours-cpu-f16 cpu "$model" WEIGHTS=f16; }
llama_cpu_f16() { llama llama-cpu-f16 "$cpu" "$ggufs/qwen3-0.6b-f16.gguf"; }
ours_cpu_q4k() { ours ours-cpu-q4k cpu "$q4km" WEIGHTS=q4_k; }
llama_cpu_q4km() { llama llama-cpu-q4km "$cpu" "$q4km"; }
ours_vulkan_f32() { ours ours-vulkan vulkan "$model" WEIGHTS=f32; }
llama_vulkan_f32() { llama llama-vulkan-f32 "$vulkan" "$ggufs/qwen3-0.6b-f32.gguf" -ngl 99; }
ours_vulkan_f16() { ours ours-vulkan-f16 vulkan "$model" WEIGHTS=f16; }
llama_vulkan_f16() { llama llama-vulkan-f16 "$vulkan" "$ggufs/qwen3-0.6b-f16.gguf" -ngl 99; }
ours_vulkan_q4k() { ours ours-vulkan-q4k vulkan "$q4km" WEIGHTS=q4_k; }
llama_vulkan_q4km() { llama llama-vulkan-q4km "$vulkan" "$q4km" -ngl 99; }
ours_vulkan_q4k_n26() { ours ours-vulkan-q4k-n26 vulkan "$q4km" WEIGHTS=q4_k AGENTICA_ML_LAYERS_ON_DEVICE=26; }
llama_vulkan_q4km_ngl27() { llama llama-vulkan-q4km-ngl27 "$vulkan" "$q4km" -ngl 27; }
ours_vulkan_q4k_n13() { ours ours-vulkan-q4k-n13 vulkan "$q4km" WEIGHTS=q4_k AGENTICA_ML_LAYERS_ON_DEVICE=13; }
llama_vulkan_q4km_ngl14() { llama llama-vulkan-q4km-ngl14 "$vulkan" "$q4km" -ngl 14; }
ours_vulkan_q4k_n0() { ours ours-vulkan-q4k-n0 vulkan "$q4km" WEIGHTS=q4_k AGENTICA_ML_LAYERS_ON_DEVICE=0; }
llama_vulkan_q4km_ngl0() { llama llama-vulkan-q4km-ngl0 "$vulkan" "$q4km" -ngl 0; }

while [ "$round" -lt "$rounds" ]; do
  round=$((round + 1))
  pair ours_cpu_f32 llama_cpu_f32
  pair ours_cpu_f16 llama_cpu_f16
  pair ours_cpu_q4k llama_cpu_q4km
  pair ours_vulkan_f32 llama_vulkan_f32
  ours ours-vulkan-exact vulkan "$model" VULKAN_FP16=0
  pair ours_vulkan_f16 llama_vulkan_f16
  pair ours_vulkan_q4k llama_vulkan_q4km
  pair ours_vulkan_q4k_n26 llama_vulkan_q4km_ngl27
  pair ours_vulkan_q4k_n13 llama_vulkan_q4km_ngl14
  pair ours_vulkan_q4k_n0 llama_vulkan_q4km_ngl0
  echo "round $round done, load average $(cut -d' ' -f1 /proc/loadavg)" >&2
done

echo "row load_ms prompt_tokens_per_s decode_tokens_per_s rss_kb gpu_mb round"
cat "$out"
sort -u "$meta"
python3 -I - "$out" << 'PY'
import statistics, sys
pairs = {"ours-cpu": "llama-cpu-f32", "ours-cpu-f16": "llama-cpu-f16", "ours-cpu-q4k": "llama-cpu-q4km",
         "ours-vulkan": "llama-vulkan-f32", "ours-vulkan-f16": "llama-vulkan-f16", "ours-vulkan-q4k": "llama-vulkan-q4km",
         "ours-vulkan-q4k-n26": "llama-vulkan-q4km-ngl27", "ours-vulkan-q4k-n13": "llama-vulkan-q4km-ngl14", "ours-vulkan-q4k-n0": "llama-vulkan-q4km-ngl0"}
rows = {}
for line in open(sys.argv[1]):
    name, load, pp, tg, rss, gpu, rnd = line.split()
    rows.setdefault(name, {})[int(rnd)] = (load, float(pp), float(tg), float(rss), gpu)
print("median of", len(next(iter(rows.values()))), "rounds")
print("row prompt_tokens_per_s decode_tokens_per_s peak_rss_gb load_s peak_gpu_mb")
for name, by_round in rows.items():
    r = list(by_round.values())
    load = [float(x[0]) for x in r if x[0] != "-"]
    gpu = [float(x[4]) for x in r if x[4] != "-"]
    print(name, round(statistics.median(x[1] for x in r), 1), round(statistics.median(x[2] for x in r), 2),
          round(statistics.median(x[3] for x in r) / 1048576, 2), round(statistics.median(load) / 1000, 2) if load else "-",
          round(statistics.median(gpu)) if gpu else "-")
print("ours / llama.cpp, per round (prompt, then tokens): minimum median maximum")
for ours, theirs in pairs.items():
    if ours not in rows or theirs not in rows:
        continue
    common = sorted(set(rows[ours]) & set(rows[theirs]))
    for label, i in (("prompt", 1), ("tokens", 2)):
        ratios = [rows[ours][k][i] / rows[theirs][k][i] for k in common]
        print(ours, "/", theirs, label, " ".join("%.2f" % x for x in ratios), "|",
              "%.2f %.2f %.2f" % (min(ratios), statistics.median(ratios), max(ratios)))
PY
