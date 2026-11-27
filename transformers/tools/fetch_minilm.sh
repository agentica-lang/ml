#!/bin/sh
# Fetches sentence-transformers/all-MiniLM-L6-v2 (6 layers, width 384) at a pinned revision, for tools/minilm_finetune.agc,
# the BERT-family fine-tune check. Never in CI. Only `model.safetensors` is fetched, never the pickle. A file gets its name
# only once its sha256 is right, and a file that is there is checked again.
#
#   sh tools/fetch_minilm.sh [DIR]       DIR defaults to ${AGENTICA_ML_CACHE:-~/.cache/agentica-ml}/hf/all-minilm-l6-v2
set -eu

cache=${AGENTICA_ML_CACHE:-$HOME/.cache/agentica-ml}
dir=${1:-$cache/hf/all-minilm-l6-v2}
repo=sentence-transformers/all-MiniLM-L6-v2
rev=1110a243fdf4706b3f48f1d95db1a4f5529b4d41

pinned() {
  echo "$2  $1" | sha256sum -c - >/dev/null 2>&1
}

# got NAME SHA256
got() {
  file=$dir/$1
  mkdir -p "$(dirname -- "$file")"
  if [ ! -f "$file" ]; then
    curl -fsSL --proto '=https' --proto-redir '=https' -o "$file.part" "https://huggingface.co/$repo/resolve/$rev/$1"
    pinned "$file.part" "$2" || { echo "fetch_minilm.sh: $1 is not the pinned file (left in $file.part)" >&2; exit 1; }
    mv "$file.part" "$file"
  fi
  pinned "$file" "$2" || { echo "fetch_minilm.sh: $file is not the pinned file" >&2; exit 1; }
}

got config.json 953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41
got tokenizer.json be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037
got model.safetensors 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
echo "$dir"
