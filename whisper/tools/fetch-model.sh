#!/bin/sh
# Downloads the pinned tiny model that validates real transcription.
#
# Weights never enter git: this script fetches them to a folder outside the
# tree (or to whisper/models/, which .gitignore keeps out) and checks the
# sha256. CI and local runs share this one pin; rotate it by changing the two
# variables below (and the whisper/README.md coverage note).
#
# Usage:
#   sh whisper/tools/fetch-model.sh [out-dir]   # default: whisper/models
#
# Prints the model path on stdout.
set -eu

# The commit of ggerganov/whisper.cpp that holds the file, so the download
# does not move under us; the sha256 of ggml-tiny.bin at that commit.
MODEL_COMMIT="5359861c739e955e79d9a303bcbc70fb988958b1"
MODEL_SHA256="be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"
MODEL_FILE="ggml-tiny.bin"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/$MODEL_COMMIT/$MODEL_FILE"

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
pkg=$(CDPATH= cd -- "$here/.." && pwd)
out="${1:-$pkg/models}"
mkdir -p "$out"
model="$out/$MODEL_FILE"

# The sha256 of a file, via sha256sum (Linux) or shasum -a 256 (macOS).
sha256_of() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

if [ -f "$model" ]; then
  got=$(sha256_of "$model")
  if [ "$got" = "$MODEL_SHA256" ]; then
    echo "$model"
    exit 0
  fi

  echo "fetch-model.sh: $model has sha256 $got, want $MODEL_SHA256; downloading again" >&2
  rm -f "$model"
fi

curl -sSfL "$MODEL_URL" -o "$model"
got=$(sha256_of "$model")
[ "$got" = "$MODEL_SHA256" ] || {
  echo "fetch-model.sh: $model has sha256 $got, want $MODEL_SHA256" >&2
  rm -f "$model"
  exit 1
}

echo "$model"
