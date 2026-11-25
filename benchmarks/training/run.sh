#!/bin/sh
# Training benchmarks of benchmarks/training/README.md. Never in CI.
#
#   sh benchmarks/training/run.sh setup                  the PyTorch and JAX venvs: torch, jax and jaxlib pinned,
#                                                        their dependencies as installed that day (setup prints them)
#   sh benchmarks/training/run.sh check                  every row at batch 4 and 8, two steps: the rows run and agree
#   sh benchmarks/training/run.sh run [OUT]              the full grid into OUT (results.jsonl), one JSON line per cell
#   sh benchmarks/training/run.sh cell STACK OPT BATCH STEPS RUN   one cell again: its line, to add to OUT (RUN 2)
#   python3 -I benchmarks/training/table.py OUT          the tables and the loss-curve verdict
#
# The threads: THREADS must be the online CPUs (8 here) for PyTorch and libtorch, which take OMP_NUM_THREADS, which
# run.sh sets. ggml's CPU rows run on the physical cores (tensor/native/ggml.c; `TENSOR_THREADS=n` in the environment
# sets another count, for example the online CPUs), and JAX's CPU pool is XLA's own default (all the cores; its size
# is not checked here). The XLA rows need the plugin (`agentica tools/install_xla.agc` in tensor/); their threads are
# not pinned: the plugin uses its default CPU pool, whose size is not checked here either. `run` refuses to start
# when the load average is above 1.0, and every line records the load average at its start. The venvs are
# `venv-torch` and `venv-jax`, in ${AGENTICA_ML_CACHE:-~/.cache/agentica-ml}.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
here=$root/benchmarks/training
cache=${AGENTICA_ML_CACHE:-$HOME/.cache/agentica-ml}
threads=${THREADS:-8}
torch_venv=$cache/venv-torch
jax_venv=$cache/venv-jax
torch_py=$torch_venv/bin/python
jax_py=$jax_venv/bin/python

online=$(getconf _NPROCESSORS_ONLN)
[ "$threads" = "$online" ] || { echo "run.sh: THREADS is $threads, the machine has $online online CPUs" >&2; exit 1; }
export OMP_NUM_THREADS=$threads

setup() {
  [ -x "$torch_venv/bin/pip" ] || python3 -m venv "$torch_venv"
  "$torch_venv/bin/pip" install --only-binary :all: --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
  "$torch_venv/bin/pip" freeze
  [ -x "$jax_venv/bin/pip" ] || python3 -m venv "$jax_venv"
  "$jax_venv/bin/pip" install --only-binary :all: jax==0.12.0.1 jaxlib==0.12.0.1
  "$jax_venv/bin/pip" freeze
}

# agentica_row BACKEND MODE OPT BATCH STEPS: the engines of the backend, then one run of bench_train.agc
agentica_row() {
  case $1 in
    ggml-cpu) exports=$(sh "$root/train/tools/build-engine.sh") ;;
    ggml-vulkan) exports=$(sh "$root/train/tools/build-engine.sh" vulkan) ;;
    torch-cpu) exports=$(cd "$root/tensor" && sh tools/build-torch.sh) ;;
    xla-cpu) exports= ;;
  esac
  (cd "$root/train" && eval "$(printf '%s\n' "$exports" | grep '^export ')" &&
    agentica --max-steps 100000000000 tools/bench_train.agc "$1" "$2" "$3" "$4" "$5" | tail -n 1)
}

# cell STACK OPT BATCH STEPS RUN: one line of OUT
cell() {
  stack=$1
  opt=$2
  batch=$3
  steps=$4
  run=$5
  load=$(cut -d' ' -f1 /proc/loadavg)
  case $stack in
    ggml-cpu-fit) result=$(agentica_row ggml-cpu fit "$opt" "$batch" "$steps") ;;
    ggml-cpu-compiled) result=$(agentica_row ggml-cpu compiled "$opt" "$batch" "$steps") ;;
    ggml-vulkan-fit) result=$(agentica_row ggml-vulkan fit "$opt" "$batch" "$steps") ;;
    ggml-vulkan-compiled) result=$(agentica_row ggml-vulkan compiled "$opt" "$batch" "$steps") ;;
    torch-cpu-fit) result=$(agentica_row torch-cpu fit "$opt" "$batch" "$steps") ;;
    xla-cpu-fit) result=$(agentica_row xla-cpu fit "$opt" "$batch" "$steps") ;;
    xla-cpu-compiled) result=$(agentica_row xla-cpu compiled "$opt" "$batch" "$steps") ;;
    pytorch-eager) result=$("$torch_py" -I "$here/bench_torch.py" eager "$opt" "$batch" "$steps") ;;
    pytorch-compile) result=$("$torch_py" -I "$here/bench_torch.py" compile "$opt" "$batch" "$steps") ;;
    pytorch-oracle) result=$("$torch_py" -I "$here/bench_torch.py" oracle "$opt" "$batch" 10) ;;
    jax-jit) result=$("$jax_py" -I "$here/bench_jax.py" "$opt" "$batch" "$steps") ;;
  esac || result=
  case $result in
    "{"*) ;;
    *) result='{"error": "the row failed"}' ;;
  esac
  printf '{"stack": "%s", "optimizer": "%s", "batch": %s, "steps": %s, "run": %s, "loadavg": %s, "result": %s}\n' \
    "$stack" "$opt" "$batch" "$steps" "$run" "$load" "$result"
}

grid() { # BATCHES STEPS OUT: the lines go to OUT.tmp, which becomes OUT at the end
  : > "$3.tmp"
  for opt in sgd adam; do
    for batch in $1; do
      for stack in pytorch-oracle ggml-cpu-fit ggml-cpu-compiled ggml-vulkan-fit ggml-vulkan-compiled torch-cpu-fit \
        xla-cpu-fit xla-cpu-compiled pytorch-eager pytorch-compile jax-jit; do
        steps=$2
        [ -n "$steps" ] || { [ "$batch" -ge 4096 ] && steps=15 || steps=30; }
        cell "$stack" "$opt" "$batch" "$steps" 1 | tee -a "$3.tmp"
      done
    done
  done
  mv "$3.tmp" "$3"
}

versions() {
  echo "agentica $(agentica --version), $("$torch_py" -I -c 'import torch; print("torch", torch.__version__)'), $("$jax_py" -I -c 'import jax; print("jax", jax.__version__)')" >&2
}

case ${1:-} in
  setup) setup ;;
  check)
    out=$(mktemp)
    trap 'rm -f "$out" "$out.tmp"' EXIT
    grid "4 8" 2 "$out"
    python3 -I "$here/table.py" "$out" 6
    ;;
  run)
    load=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$load" 'BEGIN { exit !(l > 1.0) }' && { echo "run.sh: the load average is $load, the machine is not quiet" >&2; exit 1; }
    versions
    grid "64 256 1024 4096" "" "${2:-results.jsonl}"
    ;;
  cell) cell "$2" "$3" "$4" "$5" "$6" ;;
  *) echo "usage: run.sh setup|check|run [OUT]|cell STACK OPT BATCH STEPS RUN" >&2; exit 2 ;;
esac
