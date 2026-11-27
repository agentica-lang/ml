#!/bin/sh
# Fetches what tools/finetune_sst2.agc reads into DIR, at pinned revisions, and checks the SHA-256 of each file (as
# benchmarks/stsb/run.sh does): DIR/distilbert-base-uncased (config, tokenizer and weights of the checkpoint) and
# DIR/sst2 (the validation split of stanfordnlp/sst2 and its first TRAIN_ROWS training rows, as CSV files below the
# 1 MiB that Table.read takes: the parquet files are converted by tools/sst2_csv.py with PYTHON, which needs pyarrow,
# default python3). A file gets its name only once its hash is right.
#
#   sh tools/fetch_sst2.sh DIR
set -eu

[ $# -eq 1 ] || { echo "usage: fetch_sst2.sh DIR" >&2; exit 2; }
dir=$1
TRAIN_ROWS=10000
model=https://huggingface.co/distilbert/distilbert-base-uncased/resolve/12040accade4e8a0f71eabdb258fecc2e7e948be
data=https://huggingface.co/datasets/stanfordnlp/sst2/resolve/8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb/data

# got URL FILE SHA256: FILE, fetched when it is not there
got() {
  mkdir -p "$(dirname -- "$2")"
  if [ ! -f "$2" ]; then
    curl -fsSL --proto '=https' --proto-redir '=https' -o "$2.part" "$1"
    echo "$3  $2.part" | sha256sum -c - >/dev/null 2>&1 ||
      { echo "fetch_sst2.sh: $1 is not the pinned file (left in $2.part)" >&2; exit 1; }
    mv "$2.part" "$2"
  fi
  echo "$3  $2" | sha256sum -c - >/dev/null 2>&1 || { echo "fetch_sst2.sh: $2 is not the pinned file" >&2; exit 1; }
}

got "$model/config.json" "$dir/distilbert-base-uncased/config.json" \
  69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83
got "$model/tokenizer.json" "$dir/distilbert-base-uncased/tokenizer.json" \
  ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98
got "$model/model.safetensors" "$dir/distilbert-base-uncased/model.safetensors" \
  5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063
got "$data/train-00000-of-00001.parquet" "$dir/sst2/train.parquet" \
  c7921283b75a42e685f50edecb96798607ea0fcbfd0739ee8975f22c12d55f09
got "$data/validation-00000-of-00001.parquet" "$dir/sst2/validation.parquet" \
  fb00fe008f6828f86ba2beda8415a4cf5da0c884f21c5f238c87131b5aa19529
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"${PYTHON:-python3}" -I "$here/sst2_csv.py" "$dir/sst2/train.parquet" "$dir/sst2/train.csv" "$TRAIN_ROWS"
"${PYTHON:-python3}" -I "$here/sst2_csv.py" "$dir/sst2/validation.parquet" "$dir/sst2/validation.csv"
echo "fetch_sst2.sh: the files are in $dir"
