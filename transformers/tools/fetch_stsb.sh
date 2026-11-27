#!/bin/sh
# Fetches what tools/finetune_stsb.agc reads into DIR, at pinned revisions, and checks the SHA-256 of each file with
# `sha256sum`: DIR/all-minilm-l6-v2 (the checkpoint of tools/fetch_minilm.sh) and DIR/stsb (the train and validation
# splits of sentence-transformers/stsb, as Parquet). It needs the network and `curl`. A file that is there with the right
# hash is not fetched again, and a file gets its name only once its hash is right.
#
#   sh tools/fetch_stsb.sh DIR
set -eu

[ $# -eq 1 ] || { echo "usage: fetch_stsb.sh DIR" >&2; exit 2; }
dir=$1
data=https://huggingface.co/datasets/sentence-transformers/stsb/resolve/ab7a5ac0e35aa22088bdcf23e7fd99b220e53308/data

# got URL FILE SHA256: FILE, fetched when it is not there
got() {
  mkdir -p "$(dirname -- "$2")"
  if [ ! -f "$2" ]; then
    curl -fsSL --proto '=https' --proto-redir '=https' -o "$2.part" "$1"
    echo "$3  $2.part" | sha256sum -c - >/dev/null 2>&1 ||
      { echo "fetch_stsb.sh: $1 is not the pinned file (left in $2.part)" >&2; exit 1; }
    mv "$2.part" "$2"
  fi
  echo "$3  $2" | sha256sum -c - >/dev/null 2>&1 || { echo "fetch_stsb.sh: $2 is not the pinned file" >&2; exit 1; }
}

sh "$(dirname -- "$0")/fetch_minilm.sh" "$dir/all-minilm-l6-v2" >/dev/null
got "$data/train-00000-of-00001.parquet" "$dir/stsb/train.parquet" \
  eae324ff1eac2d0ba769851736eb7232eda64f370a16eb20e74a2c5f8f5fafe0
got "$data/validation-00000-of-00001.parquet" "$dir/stsb/validation.parquet" \
  9c6e0e9881f1b398abe3e439a482f4686305c3784568c462f6bba58bdff03b0a
echo "fetch_stsb.sh: the files are in $dir"
