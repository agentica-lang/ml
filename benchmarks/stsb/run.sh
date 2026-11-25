#!/bin/sh
# The STS-B fine-tune of benchmarks/README.md (ml#182): a sentence-transformers DistilBERT trained on scored sentence
# pairs, on the CPU, in Agentica ml and in PyTorch, with the same rules. Never in CI.
#
#   sh benchmarks/stsb/run.sh setup              the venv: torch, transformers and pyarrow at pinned versions
#   sh benchmarks/stsb/run.sh fetch              the checkpoint and the two parquet files, at pinned revisions
#   sh benchmarks/stsb/run.sh smoke              both sides on the 4 pairs of fixture.jsonl with a weight decay of 1:
#                                                they agree, and a peer without the decay does not
#   sh benchmarks/stsb/run.sh ours [OUT]         the run in Agentica on ggml CPU, its lines into OUT (ours.jsonl)
#   sh benchmarks/stsb/run.sh peer [OUT]         the run in PyTorch, its lines into OUT (peer.jsonl)
#   sh benchmarks/stsb/run.sh ours-lora [OUT]    the LoRA run in Agentica (ml#137): it saves the adapter it starts from
#   sh benchmarks/stsb/run.sh peer-lora [OUT]    the LoRA run with PEFT, from that adapter: run it after ours-lora
#   python3 -I benchmarks/stsb/summary.py ours.jsonl peer.jsonl
#
# PAIRS (1000), BATCH (16) and PAD_TO (1) set the run on both sides. The threads are the online CPUs on both sides: ggml
# takes every online CPU (`sysconf(_SC_NPROCESSORS_ONLN)` in tensor/native/ggml.c) and reports no count, so this script
# writes the `threads` line of our side: that count by construction, not a measurement; the peer sets
# `torch.set_num_threads` to the CPUs of its process and prints what PyTorch then runs with. A process kept to fewer
# CPUs than are online is refused. Everything lives in ${AGENTICA_ML_CACHE:-~/.cache/agentica-ml}:
# the venv `venv-torch` (shared with benchmarks/training, so a setup of either changes it for both),
# `hf/msmarco-distilbert-cos-v5` and `hf/stsb`. Only `model.safetensors` is fetched, never the pickle. Every run
# checks the hashes of the parquet files and writes their JSON lines again, so a stale copy is never read.
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
# Where `ours-lora` saves the adapter it starts from, which `peer-lora` loads.
adapter=$cache/lora/stsb-adapter
# The loss tolerance of the LoRA runs (see the README).
lora_tolerance=1.5e-7
online=$(getconf _NPROCESSORS_ONLN)
allowed=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
[ "$allowed" = "$online" ] ||
  { echo "run.sh: the process may use $allowed CPUs, the machine has $online online" >&2; exit 1; }
unset OMP_THREAD_LIMIT
export OMP_NUM_THREADS=$online

model_repo=sentence-transformers/msmarco-distilbert-cos-v5
model_rev=c598d92e340e735b82e5b73d7db135209b57b4b0
data_repo=sentence-transformers/stsb
data_rev=ab7a5ac0e35aa22088bdcf23e7fd99b220e53308

# The versions were more than two weeks old when they were pinned. Every wheel is a binary one. The second install
# has --no-deps, so pip adds nothing to its list; torch's own dependencies are whatever its wheel asks for, not
# pinned here, and the freeze records what the venv ended with.
setup() {
  [ -x "$venv/bin/pip" ] || python3 -m venv "$venv"
  "$venv/bin/pip" install --only-binary :all: --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
  "$venv/bin/pip" install --only-binary :all: --no-deps \
    transformers==4.57.6 tokenizers==0.22.1 huggingface_hub==0.36.0 safetensors==0.6.2 regex==2025.9.18 \
    numpy==2.2.6 PyYAML==6.0.2 packaging==25.0 requests==2.32.5 tqdm==4.67.1 fsspec==2025.9.0 \
    typing_extensions==4.15.0 hf-xet==1.1.10 urllib3==2.5.0 idna==3.10 certifi==2025.8.3 charset-normalizer==3.4.3 \
    pyarrow==26.0.0 peft==0.21.2 accelerate==1.15.0 psutil==7.2.2
  "$venv/bin/pip" check
  "$venv/bin/pip" freeze
}

# pinned FILE SHA256: whether FILE has that hash
pinned() {
  echo "$2  $1" | sha256sum -c - >/dev/null 2>&1
}

# got URL FILE SHA256: the file, fetched when it is not there; it gets its name only once its hash is right
got() {
  mkdir -p "$(dirname -- "$2")"
  if [ ! -f "$2" ]; then
    curl -fsSL --proto '=https' --proto-redir '=https' -o "$2.part" "$1"
    pinned "$2.part" "$3" || { echo "run.sh: $1 is not the pinned file (left in $2.part)" >&2; exit 1; }
    mv "$2.part" "$2"
  fi
  pinned "$2" "$3" || { echo "run.sh: $2 is not the pinned file" >&2; exit 1; }
}

train_sha=eae324ff1eac2d0ba769851736eb7232eda64f370a16eb20e74a2c5f8f5fafe0
validation_sha=9c6e0e9881f1b398abe3e439a482f4686305c3784568c462f6bba58bdff03b0a

fetch() {
  base=https://huggingface.co/$model_repo/resolve/$model_rev
  got "$base/config.json" "$model/config.json" fc11aeb93f66e18a950c753b9b1d4dad0fa07a097dfa78b41ec264460dd84cb0
  got "$base/tokenizer.json" "$model/tokenizer.json" 081419d86d3f6843c68074f7eefb82c9810048b9ddcf831b400a532bf9797f95
  got "$base/model.safetensors" "$model/model.safetensors" \
    3083803ab54614d7c0627b19aa2d7070cd4c49116dc8efd6fae3016526a10015
  base=https://huggingface.co/datasets/$data_repo/resolve/$data_rev/data
  got "$base/train-00000-of-00001.parquet" "$data/train.parquet" "$train_sha"
  got "$base/validation-00000-of-00001.parquet" "$data/validation.parquet" "$validation_sha"
}

# The JSON lines of the two pinned parquet files, written again for every run.
data() {
  pinned "$data/train.parquet" "$train_sha" && pinned "$data/validation.parquet" "$validation_sha" ||
    { echo "run.sh: the parquet files in $data are not the pinned ones: run.sh fetch" >&2; exit 1; }
  "$py" -I "$here/stsb_jsonl.py" "$data/train.parquet" "$data/train.jsonl" >&2
  "$py" -I "$here/stsb_jsonl.py" "$data/validation.parquet" "$data/validation.jsonl" >&2
}

# peak COMMAND...: the command, then a `memory` line with the peak resident size of its process (GNU time)
peak() {
  note=$(mktemp)
  status=0
  /usr/bin/time -o "$note" -f '{"kind": "memory", "peak_kb": %M}' "$@" || status=$?
  grep '^{' "$note" || true
  rm -f "$note"
  return $status
}

# ours TRAIN VALIDATION PAIRS BATCH PAD_TO [WEIGHT_DECAY [ADAPTER_DIR]]: the Agentica side, with the ggml engine built
ours() {
  exports=$(sh "$root/transformers/tools/build-engine.sh") || return 1
  echo "{\"kind\": \"threads\", \"value\": $online}"
  (cd "$root/transformers" && eval "$(printf '%s\n' "$exports" | grep '^export ')" &&
    peak agentica --max-steps 1000000000000 tools/distilbert_stsb.agc "$model" "$@")
}

peer() {
  HF_HUB_OFFLINE=1 peak "$py" -I "$here/distilbert_stsb.py" "$model" "$@"
}

# lines OUT SIDE ARGS...: the JSON lines of a run into OUT, its other lines to stderr; a run that fails leaves no OUT
lines() {
  to=$1
  shift
  rm -f "$to"
  if ! "$@" >"$to.raw"; then
    cat "$to.raw" >&2
    rm -f "$to.raw"
    echo "run.sh: the run failed ($1)" >&2
    exit 1
  fi
  grep -v '^{' "$to.raw" >&2 || true
  grep '^{' "$to.raw" >"$to.lines" || true
  rm -f "$to.raw"
  mv "$to.lines" "$to"
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
  smoke)
    out=$(mktemp -d)
    trap 'rm -rf "$out"' EXIT
    fixture=$here/fixture.jsonl
    lines "$out/ours.jsonl" ours "$fixture" "$fixture" 4 2 1 1.0
    lines "$out/peer.jsonl" peer "$fixture" "$fixture" 4 2 1 1.0
    lines "$out/plain.jsonl" peer "$fixture" "$fixture" 4 2 1 0.0
    python3 -I "$here/summary.py" "$out/ours.jsonl" "$out/peer.jsonl"
    python3 -I "$here/summary.py" "$out/ours.jsonl" "$out/plain.jsonl" >"$out/plain.txt" || true
    if ! { grep -q '^verdict: the runs DIFFER' "$out/plain.txt" && grep -q '^largest loss gap' "$out/plain.txt"; }; then
      cat "$out/plain.txt" >&2
      echo "run.sh: a peer without the weight decay must differ by its losses, and it does not" >&2
      exit 1
    fi
    grep '^largest' "$out/plain.txt" | sed 's/^/without the weight decay the peer DIFFERS, as it must: /'
    echo "LoRA:"
    lines "$out/ours-lora.jsonl" ours "$fixture" "$fixture" 4 2 1 1.0 "$out/adapter"
    lines "$out/peer-lora.jsonl" peer "$fixture" "$fixture" 4 2 1 1.0 "$out/adapter"
    python3 -I "$here/summary.py" "$out/ours-lora.jsonl" "$out/peer-lora.jsonl" "$lora_tolerance"
    ;;
  ours)
    quiet
    data
    lines "${2:-ours.jsonl}" ours "$data/train.jsonl" "$data/validation.jsonl" "$pairs" "$batch" "$pad_to"
    ;;
  peer)
    quiet
    data
    lines "${2:-peer.jsonl}" peer "$data/train.jsonl" "$data/validation.jsonl" "$pairs" "$batch" "$pad_to"
    ;;
  ours-lora)
    quiet
    data
    lines "${2:-ours-lora.jsonl}" ours "$data/train.jsonl" "$data/validation.jsonl" "$pairs" "$batch" "$pad_to" \
      0.01 "$adapter"
    ;;
  peer-lora)
    quiet
    data
    [ -f "$adapter/adapter_config.json" ] || { echo "run.sh: no adapter in $adapter: run ours-lora first" >&2; exit 1; }
    lines "${2:-peer-lora.jsonl}" peer "$data/train.jsonl" "$data/validation.jsonl" "$pairs" "$batch" "$pad_to" \
      0.01 "$adapter"
    ;;
  *)
    echo "usage: run.sh setup | fetch | smoke | ours [OUT] | peer [OUT] | ours-lora [OUT] | peer-lora [OUT]" >&2
    exit 2
    ;;
esac
