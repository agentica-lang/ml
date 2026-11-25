#!/bin/sh
# Training benchmarks of benchmarks/training/README.md. Never in CI.
#
#   sh benchmarks/training/run.sh setup             the JAX venv (jax 0.12.0.1, --only-binary :all:)
#   sh benchmarks/training/run.sh check             every row at batch 4 and 8, two steps: the rows run and agree
#   sh benchmarks/training/run.sh run [OUT]         the full grid into OUT (results.jsonl), one JSON line per cell
#   sh benchmarks/training/run.sh cell STACK OPT BATCH STEPS   one cell again: its line, to replace one of OUT
#   python3 benchmarks/training/table.py OUT        the tables and the loss-curve verdict
#
# Every row runs on the CPUs 0 to THREADS-1 (taskset: ggml and XLA size their pool from the affinity) with
# OMP_NUM_THREADS=THREADS (libtorch and PyTorch). THREADS is 8, the threads of the machine. `run` refuses to start
# when the load average is above 1.0. The venvs are `venv-torch` (torch 2.6.0+cpu) and `venv-jax`, in
# ${AGENTICA_ML_CACHE:-~/.cache/agentica-ml}.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
here=$root/benchmarks/training
cache=${AGENTICA_ML_CACHE:-$HOME/.cache/agentica-ml}
threads=${THREADS:-8}
last=$((threads - 1))
jax_venv=$cache/venv-jax
torch_py=$cache/venv-torch/bin/python
jax_py=$jax_venv/bin/python

export OMP_NUM_THREADS=$threads
pin="taskset -c 0-$last"

setup() {
  python3 -m venv "$jax_venv"
  "$jax_venv/bin/pip" install --only-binary :all: jax==0.12.0.1 jaxlib==0.12.0.1
  "$jax_venv/bin/pip" freeze
}

# agentica ROW BACKEND MODE OPT BATCH STEPS: the engines of the backend, then one run of bench_train.agc
agentica_row() {
  case $2 in
    ggml-cpu) exports=$(sh "$root/train/tools/build-engine.sh") ;;
    ggml-vulkan) exports=$(sh "$root/train/tools/build-engine.sh" vulkan) ;;
    torch-cpu) exports=$(cd "$root/tensor" && sh tools/build-torch.sh) ;;
  esac
  (cd "$root/train" && eval "$(printf '%s\n' "$exports" | grep '^export ')" &&
    $pin agentica --max-steps 100000000000 tools/bench_train.agc "$2" "$3" "$4" "$5" "$6" | tail -n 1)
}

# cell STACK OPT BATCH STEPS: one line of OUT
cell() {
  stack=$1
  opt=$2
  batch=$3
  steps=$4
  case $stack in
    ggml-cpu-fit) result=$(agentica_row "$stack" ggml-cpu fit "$opt" "$batch" "$steps") ;;
    ggml-cpu-compiled) result=$(agentica_row "$stack" ggml-cpu compiled "$opt" "$batch" "$steps") ;;
    ggml-vulkan-fit) result=$(agentica_row "$stack" ggml-vulkan fit "$opt" "$batch" "$steps") ;;
    ggml-vulkan-compiled) result=$(agentica_row "$stack" ggml-vulkan compiled "$opt" "$batch" "$steps") ;;
    torch-cpu-fit) result=$(agentica_row "$stack" torch-cpu fit "$opt" "$batch" "$steps") ;;
    pytorch-eager) result=$($pin "$torch_py" "$here/bench_torch.py" eager "$opt" "$batch" "$steps") ;;
    pytorch-compile) result=$($pin "$torch_py" "$here/bench_torch.py" compile "$opt" "$batch" "$steps") ;;
    pytorch-oracle) result=$($pin "$torch_py" "$here/bench_torch.py" oracle "$opt" "$batch" 10) ;;
    jax-jit) result=$($pin "$jax_py" "$here/bench_jax.py" "$opt" "$batch" "$steps") ;;
  esac || result=
  case $result in
    "{"*) ;;
    *) result='{"error": "the row failed"}' ;;
  esac
  printf '{"stack": "%s", "optimizer": "%s", "batch": %s, "steps": %s, "result": %s}\n' \
    "$stack" "$opt" "$batch" "$steps" "$result"
}

grid() { # BATCHES STEPS OUT
  : > "$3"
  for opt in sgd adam; do
    for batch in $1; do
      for stack in pytorch-oracle ggml-cpu-fit ggml-cpu-compiled ggml-vulkan-fit ggml-vulkan-compiled torch-cpu-fit \
        pytorch-eager pytorch-compile jax-jit; do
        steps=$2
        [ -n "$steps" ] || { [ "$batch" -ge 4096 ] && steps=15 || steps=30; }
        cell "$stack" "$opt" "$batch" "$steps" | tee -a "$3"
      done
    done
  done
}

case ${1:-} in
  setup) setup ;;
  cell) cell "$2" "$3" "$4" "$5" ;;
  check) grid "4 8" 2 "${TMPDIR:-/tmp}/training-check.jsonl" && python3 "$here/table.py" "${TMPDIR:-/tmp}/training-check.jsonl" ;;
  run)
    load=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$load" 'BEGIN { exit !(l > 1.0) }' && { echo "run.sh: the load average is $load, the machine is not quiet" >&2; exit 1; }
    grid "64 256 1024 4096" "" "${2:-results.jsonl}"
    ;;
  *) echo "usage: run.sh setup|check|run [OUT]|cell STACK OPT BATCH STEPS" >&2; exit 2 ;;
esac
