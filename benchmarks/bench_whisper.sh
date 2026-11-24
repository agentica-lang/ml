#!/bin/bash
# whisper-tiny inference: this repository's `asr/` (ggml CPU, ggml Vulkan, libtorch) against whisper.cpp (CPU and
# Vulkan), on the same machine, clips, weights and thread count. README.md has the method and the results.
#
#   benchmarks/bench_whisper.sh build  MODEL_DIR MEL_FILTERS_NPZ WHISPER_CPP_SRC WORK_DIR
#   benchmarks/bench_whisper.sh run    MODEL_DIR MEL_FILTERS_NPZ WHISPER_CPP_SRC WORK_DIR
#   benchmarks/bench_whisper.sh report WORK_DIR
#
# MODEL_DIR is a checkout of huggingface.co/openai/whisper-tiny (model.safetensors, config.json, vocab.json,
# added_tokens.json, tokenizer.json), MEL_FILTERS_NPZ is openai/whisper's whisper/assets/mel_filters.npz, WHISPER_CPP_SRC
# is a whisper.cpp checkout (never modified) and WORK_DIR is where everything else goes: the converted ggml models, the
# whisper.cpp builds, the harness and the raw results.
#
# `build` converts the weights (F32 and F16), builds whisper.cpp out of tree for the CPU and for Vulkan, builds the
# harness against each, and builds this repository's ggml engines. `run` times every row, a load-only pass for the
# memory, and checks the token ids. `report` prints the tables. Environment: THREADS (default: every core, as ours),
# RUNS (default 5), ONLY (a regex on row names, `run` only), VULKAN_SDK (default ~/.local/vulkan-sdk/usr: glslc and the
# headers) and JOBS (default 2).
# Builds and runs are heavy and the numbers need a quiet machine: wrap each step in a lock if others share it.
set -euo pipefail

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/.." && pwd)
threads=${THREADS:-$(nproc)}
runs=${RUNS:-5}
jobs=${JOBS:-2}
sdk=${VULKAN_SDK:-$HOME/.local/vulkan-sdk/usr}

step=${1:-}
case "$step" in
  build | run)
    [ $# -eq 5 ] || { sed -n 5,7p "$0" >&2; exit 2; }
    model_dir=$(CDPATH= cd -- "$2" && pwd)
    mel=$(realpath "$3")
    wsrc=$(CDPATH= cd -- "$4" && pwd)
    mkdir -p "$5"
    work=$(CDPATH= cd -- "$5" && pwd)
    ;;
  report)
    [ $# -eq 2 ] || { sed -n 5,7p "$0" >&2; exit 2; }
    work=$(CDPATH= cd -- "$2" && pwd)
    exec python3 -I "$here/summarize.py" "$work/results" "$root/asr/test/fixtures/transcribe_golden.json"
    ;;
  *)
    sed -n 5,7p "$0" >&2
    exit 2
    ;;
esac

vulkan_env() {
  export PATH="$sdk/bin:$PATH"
  export LD_LIBRARY_PATH="$sdk/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

if [ "$step" = build ]; then
  mkdir -p "$work/models"
  for t in f32 f16; do
    [ -f "$work/models/ggml-tiny-$t.bin" ] ||
      python3 -I "$here/convert_whisper_ggml.py" "$model_dir" "$mel" "$work/models/ggml-tiny-$t.bin" "$t"
  done

  for dev in cpu vulkan; do
    flags=
    if [ "$dev" = vulkan ]; then
      vulkan_env
      flags="-DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH=$sdk"
    fi
    # shellcheck disable=SC2086
    cmake -S "$wsrc" -B "$work/wcpp-$dev" -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF $flags > "$work/wcpp-$dev.configure.log"
    cmake --build "$work/wcpp-$dev" --target whisper-cli -j"$jobs" > "$work/wcpp-$dev.build.log"
    libwhisper=$(find -L "$work/wcpp-$dev" -name 'libwhisper.so' | head -n 1)
    [ -n "$libwhisper" ] || { echo "bench_whisper.sh: no libwhisper.so under $work/wcpp-$dev" >&2; exit 1; }
    lib=$(dirname "$libwhisper")
    g++ -O2 -std=c++17 -I"$wsrc/include" -I"$wsrc/ggml/include" "$here/whisper_cpp_bench.cpp" -o "$work/wcpp-$dev/whisper_cpp_bench" \
      -L"$lib" -lwhisper -Wl,-rpath,"$lib"
  done

  for dev in cpu vulkan; do
    (vulkan_env; JOBS=$jobs sh "$root/asr/tools/build-engine.sh" "$dev" > "$work/engine-$dev.log")
  done
  exit 0
fi

mkdir -p "$work/results"
# ONLY is a regular expression on the row names (agentica-ggml-cpu, agentica-ggml-vulkan-f16, wcpp-vulkan-f16, ...): it runs those rows only.
wanted() { [ -z "${ONLY:-}" ] || printf '%s\n' "$1" | grep -Eq "$ONLY"; }
jfk="$wsrc/samples/jfk.wav"
hola="$root/whisper/live/fixtures/es-hola.wav"

# row clip: the clip name is `jfk` (English) or `es-hola` (Spanish).
clip_wav() { [ "$1" = jfk ] && echo "$jfk" || echo "$hola"; }
clip_lang() { [ "$1" = jfk ] && echo en || echo es; }

timed() {  # OUT COMMAND...: the command's stdout in OUT.tsv, `time -v` in OUT.time
  out=$1
  shift
  /usr/bin/time -v -o "$out.time" "$@" > "$out.tsv"
}

# Each row runs the timed pass on both clips, then a pass that stops after the load (jfk only), for the memory of the load.
ours() {  # ROW BACKEND [VAR=value...]: the variables go to the tool (WEIGHTS=f16, VULKAN_FP16=1)
  wanted "$1" || return 0
  rm -f "$work/results/$1".*
  engine=$([ "$2" = vulkan ] && echo vulkan || echo cpu)
  for pass in jfk es-hola load; do
    clip=$pass
    n=$runs
    suffix=
    if [ "$pass" = load ]; then
      clip=jfk
      n=-1
      suffix=.load
    fi
    (
      cd "$root/asr"
      vulkan_env
      exports=$(sh tools/build-engine.sh "$engine")
      eval "$(printf '%s\n' "$exports" | grep '^export ')"
      if [ "$2" = torch ]; then
        torch_exports=$(cd "$root/tensor" && sh tools/build-torch.sh)
        eval "$torch_exports"
      fi
      timed "$work/results/$1.$clip$suffix" env WEIGHTS=f32 VULKAN_FP16=0 "${@:3}" agentica tools/bench_transcribe.agc "$model_dir" "$(clip_wav "$clip")" "$2" "$(clip_lang "$clip")" "$n"
    )
  done
}

cpp() {  # ROW DEVICE MODEL FLASH_ATTN
  wanted "$1" || return 0
  rm -f "$work/results/$1".*
  gpu=$([ "$2" = vulkan ] && echo 1 || echo 0)
  for pass in jfk es-hola load; do
    clip=$pass
    n=$runs
    suffix=
    if [ "$pass" = load ]; then
      clip=jfk
      n=-1
      suffix=.load
    fi
    timed "$work/results/$1.$clip$suffix" "$work/wcpp-$2/whisper_cpp_bench" "$work/models/ggml-tiny-$3.bin" "$(clip_wav "$clip")" \
      "$(clip_lang "$clip")" "$threads" "$gpu" "$4" "$n"
  done
}

ours agentica-ggml-cpu ggml
ours agentica-ggml-vulkan vulkan
ours agentica-ggml-cpu-f16 ggml WEIGHTS=f16
ours agentica-ggml-vulkan-f16 vulkan WEIGHTS=f16
ours agentica-ggml-vulkan-f16-fp16 vulkan WEIGHTS=f16 VULKAN_FP16=1
ours agentica-libtorch torch
cpp wcpp-cpu-f32 cpu f32 1
cpp wcpp-cpu-f32-nofa cpu f32 0
cpp wcpp-cpu-f16 cpu f16 1
cpp wcpp-vulkan-f32 vulkan f32 1
cpp wcpp-vulkan-f32-nofa vulkan f32 0
cpp wcpp-vulkan-f16 vulkan f16 1
echo "raw results in $work/results; benchmarks/bench_whisper.sh report $work"
