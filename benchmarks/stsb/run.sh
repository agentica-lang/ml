#!/bin/sh
# The STS-B fine-tune of benchmarks/README.md (ml#182): a sentence-transformers DistilBERT trained on scored sentence
# pairs, on the CPU, in Agentica ml and in PyTorch, with the same rules. Never in CI.
#
#   sh benchmarks/stsb/run.sh setup              the venv: torch, transformers, pyarrow and their dependencies, pinned
#   sh benchmarks/stsb/run.sh fetch              the checkpoint and the two parquet files, at pinned revisions
#   sh benchmarks/stsb/run.sh data               the parquet files as JSON lines, which both sides read
#   sh benchmarks/stsb/run.sh smoke              both sides on the 3 pairs of fixture.jsonl, and their summary
#   sh benchmarks/stsb/run.sh ours [OUT]         the run in Agentica on ggml CPU, its lines into OUT (ours.jsonl)
#   sh benchmarks/stsb/run.sh peer [OUT]         the run in PyTorch, its lines into OUT (peer.jsonl)
#   python3 -I benchmarks/stsb/summary.py ours.jsonl peer.jsonl
#
# PAIRS (1000), BATCH (16) and PAD_TO (1) set the run on both sides. The threads are the online CPUs on both sides: ggml
# takes every online CPU (tensor/native/ggml.c), and the peer sets `torch.set_num_threads` to the same count, with
# OMP_NUM_THREADS set here. Everything lives in ${AGENTICA_ML_CACHE:-~/.cache/agentica-ml}: the venv `venv-torch`
# (shared with benchmarks/training), `hf/msmarco-distilbert-cos-v5` and `hf/stsb`. Only `model.safetensors` is
# fetched, never the pickle.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
here=$root/benchmarks/stsb
cache=${AGENTICA_ML_CACHE:-$HOME/.cache/agentica-ml}
venv=$cache/venv-torch
py=$venv/bin/python
model=$cache/hf/msmarco-distilbert-cos-v5
data=$cache/hf/stsb
pairs=${PAIRS:-1000}
batch=${BATCH:-16}
pad_to=${PAD_TO:-1}
export OMP_NUM_THREADS=$(getconf _NPROCESSORS_ONLN)

model_repo=sentence-transformers/msmarco-distilbert-cos-v5
model_rev=c598d92e340e735b82e5b73d7db135209b57b4b0
data_repo=sentence-transformers/stsb
data_rev=ab7a5ac0e35aa22088bdcf23e7fd99b220e53308

# The versions were more than two weeks old when they were pinned. Every wheel is a binary one, and --no-deps keeps pip
# from adding what is not in this list.
setup() {
  [ -x "$venv/bin/pip" ] || python3 -m venv "$venv"
  "$venv/bin/pip" install --only-binary :all: --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
  "$venv/bin/pip" install --only-binary :all: --no-deps \
    transformers==4.57.6 tokenizers==0.22.1 huggingface_hub==0.36.0 safetensors==0.6.2 regex==2025.9.18 \
    numpy==2.2.6 PyYAML==6.0.2 packaging==25.0 requests==2.32.5 tqdm==4.67.1 fsspec==2025.9.0 \
    typing_extensions==4.15.0 hf-xet==1.1.10 urllib3==2.5.0 idna==3.10 certifi==2025.8.3 charset-normalizer==3.4.3 \
    pyarrow==26.0.0
  "$venv/bin/pip" check
}

# got URL FILE SHA256: the file, fetched when it is not there, and its hash checked either way
got() {
  mkdir -p "$(dirname -- "$2")"
  [ -f "$2" ] || curl -fsSL -o "$2" "$1"
  echo "$3  $2" | sha256sum -c - >/dev/null || { echo "run.sh: $2 is not the pinned file" >&2; exit 1; }
}

fetch() {
  base=https://huggingface.co/$model_repo/resolve/$model_rev
  got "$base/config.json" "$model/config.json" fc11aeb93f66e18a950c753b9b1d4dad0fa07a097dfa78b41ec264460dd84cb0
  got "$base/tokenizer.json" "$model/tokenizer.json" 081419d86d3f6843c68074f7eefb82c9810048b9ddcf831b400a532bf9797f95
  got "$base/model.safetensors" "$model/model.safetensors" \
    3083803ab54614d7c0627b19aa2d7070cd4c49116dc8efd6fae3016526a10015
  base=https://huggingface.co/datasets/$data_repo/resolve/$data_rev/data
  got "$base/train-00000-of-00001.parquet" "$data/train.parquet" \
    eae324ff1eac2d0ba769851736eb7232eda64f370a16eb20e74a2c5f8f5fafe0
  got "$base/validation-00000-of-00001.parquet" "$data/validation.parquet" \
    9c6e0e9881f1b398abe3e439a482f4686305c3784568c462f6bba58bdff03b0a
}

data() {
  "$py" -I "$here/stsb_jsonl.py" "$data/train.parquet" "$data/train.jsonl"
  "$py" -I "$here/stsb_jsonl.py" "$data/validation.parquet" "$data/validation.jsonl"
}

# ours TRAIN VALIDATION PAIRS BATCH PAD_TO: the Agentica side, with the ggml engine built
ours() {
  exports=$(sh "$root/transformers/tools/build-engine.sh")
  (cd "$root/transformers" && eval "$(printf '%s\n' "$exports" | grep '^export ')" &&
    agentica --max-steps 1000000000000 tools/distilbert_stsb.agc "$model" "$1" "$2" "$3" "$4" "$5")
}

peer() {
  "$py" -I "$here/distilbert_stsb.py" "$model" "$1" "$2" "$3" "$4" "$5"
}

quiet() {
  load=$(cut -d' ' -f1 /proc/loadavg)
  awk -v l="$load" 'BEGIN { exit !(l > 1.0) }' &&
    { echo "run.sh: the load average is $load, the machine is not quiet" >&2; exit 1; }
  return 0
}

case ${1:-} in
  setup) setup ;;
  fetch) fetch ;;
  data) data ;;
  smoke)
    out=$(mktemp -d)
    trap 'rm -rf "$out"' EXIT
    ours "$here/fixture.jsonl" "$here/fixture.jsonl" 3 2 1 | grep '^{' >"$out/ours.jsonl"
    peer "$here/fixture.jsonl" "$here/fixture.jsonl" 3 2 1 | grep '^{' >"$out/peer.jsonl"
    python3 -I "$here/summary.py" "$out/ours.jsonl" "$out/peer.jsonl"
    ;;
  ours)
    quiet
    ours "$data/train.jsonl" "$data/validation.jsonl" "$pairs" "$batch" "$pad_to" | grep '^{' >"${2:-ours.jsonl}"
    ;;
  peer)
    quiet
    peer "$data/train.jsonl" "$data/validation.jsonl" "$pairs" "$batch" "$pad_to" | grep '^{' >"${2:-peer.jsonl}"
    ;;
  *)
    echo "usage: run.sh setup | fetch | data | smoke | ours [OUT] | peer [OUT]" >&2
    exit 2
    ;;
esac
