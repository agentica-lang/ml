#!/bin/sh
# Downloads the pinned models that validate real transcription.
#
# Weights never enter git: this script fetches them to a folder outside the
# tree (or to whisper/models/, which .gitignore keeps out) and checks each
# sha256. CI and local runs share these pins; rotate one by changing its line
# below (and the whisper/README.md coverage note).
#
# Two pins, both multilingual:
#   tiny   ggml-tiny.bin    74 MB  the structure checks: the live test file and
#                                  the English end-to-end program
#   small  ggml-small.bin  466 MB  the Spanish case: it hears Spanish better
#                                  than tiny does, and is what
#                                  live/transcribe_spanish.agc loads
#
# Usage:
#   sh whisper/tools/fetch-model.sh [out-dir] [model...]   # default: whisper/models, all of them
#
# Prints the path of each model on stdout, one per line.
set -eu

# The commit of ggerganov/whisper.cpp that holds both files, so the downloads
# do not move under us, and the sha256 of each file at that commit.
MODEL_COMMIT="5359861c739e955e79d9a303bcbc70fb988958b1"
TINY_FILE="ggml-tiny.bin"
TINY_SHA256="be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"
SMALL_FILE="ggml-small.bin"
SMALL_SHA256="1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
pkg=$(CDPATH= cd -- "$here/.." && pwd)
out="${1:-$pkg/models}"
if [ $# -gt 0 ]; then
  shift
fi

if [ $# -eq 0 ]; then
  set -- tiny small
fi

# The sha256 of a file, via sha256sum (Linux) or shasum -a 256 (macOS).
sha256_of() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# Fetches one pinned file into $out and prints its path.
fetch() {
  file=$1
  want=$2
  model="$out/$file"

  if [ -f "$model" ]; then
    got=$(sha256_of "$model")
    if [ "$got" = "$want" ]; then
      echo "$model"
      return 0
    fi

    echo "fetch-model.sh: $model has sha256 $got, want $want; downloading again" >&2
    rm -f "$model"
  fi

  curl -sSfL "https://huggingface.co/ggerganov/whisper.cpp/resolve/$MODEL_COMMIT/$file" -o "$model"
  got=$(sha256_of "$model")
  [ "$got" = "$want" ] || {
    echo "fetch-model.sh: $model has sha256 $got, want $want" >&2
    rm -f "$model"
    return 1
  }

  echo "$model"
}

mkdir -p "$out"

for name in "$@"; do
  case "$name" in
    tiny) fetch "$TINY_FILE" "$TINY_SHA256" ;;
    small) fetch "$SMALL_FILE" "$SMALL_SHA256" ;;
    *)
      echo "fetch-model.sh: unknown model '$name' (tiny, small)" >&2
      exit 1
      ;;
  esac
done
