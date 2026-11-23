#!/bin/sh
# Runs cchecker (https://github.com/agentica-lang/cchecker) over the native C of the packages: the project rules
# for locks and ownership, gcc -fanalyzer and clang-tidy. Run it from anywhere; it needs git, gcc 10+, clang, clang-tidy,
# the pinned `agentica` on PATH, and the submodules (`git submodule update --init third_party/ggml third_party/whisper.cpp`).
#
#   sh tools/cchecker.sh                      new findings fail it (exit 1); the settings are cchecker.toml
#   sh tools/cchecker.sh --update-baseline    accepts today's findings in cchecker-baseline.json (the diff is the review)
#
# Other arguments go to cchecker (`--json`, `--gcc gcc-14`, `--clang clang-19`, `--clang-tidy clang-tidy-19`).
# The database is built here, from the include paths of tools/build-engine.sh of each package: the headers of the
# submodules and of the Agentica native API are all it needs, no engine is built. torch.c (C++ over libtorch) is not
# checked: cchecker reads C only.
set -eu

# A new release is these two together (`git ls-remote https://github.com/agentica-lang/cchecker 'X^{}' X`).
CCHECKER_VERSION=0.8.0
CCHECKER_COMMIT=6f94efd9307ff126025cd34f025e6e4804f3ba37

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

dir=build/cchecker/$CCHECKER_VERSION
if [ ! -f "$dir/cchecker.agc" ]; then
  rm -rf "$dir" "$dir.tmp"
  mkdir -p build/cchecker
  for try in 1 2 3; do
    rm -rf "$dir.tmp"
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$CCHECKER_VERSION" \
      https://github.com/agentica-lang/cchecker "$dir.tmp" && break
    [ "$try" -lt 3 ] || { echo "cchecker.sh: could not clone cchecker" >&2; exit 1; }
    sleep 15
  done
  [ "$(git -C "$dir.tmp" rev-parse HEAD)" = "$CCHECKER_COMMIT" ] || {
    echo "cchecker.sh: cchecker $CCHECKER_VERSION is not commit $CCHECKER_COMMIT" >&2
    rm -rf "$dir.tmp"
    exit 1
  }
  mv "$dir.tmp" "$dir"
fi

native_include=${AGENTICA_INCLUDE:-$(dirname -- "$(readlink -f -- "$(command -v agentica)")")/../include}
[ -f "$native_include/agentica_native.h" ] || {
  echo "cchecker.sh: no agentica_native.h in $native_include (set AGENTICA_INCLUDE)" >&2
  exit 1
}
for engine in third_party/ggml/include third_party/whisper.cpp/include; do
  [ -d "$engine" ] || {
    echo "cchecker.sh: no $engine (git submodule update --init third_party/ggml third_party/whisper.cpp)" >&2
    exit 1
  }
done

# One entry per file (a new native .c file is added to the list below): the compiler, the flags, the package's native/ folder and the engine headers it links.
db=build/cchecker/compile_commands.json
sep='['
{
  while read -r file includes; do
    args=
    for i in $includes; do args="$args, \"-I$i\""; done
    printf '%s\n  {"directory": "%s", "file": "%s", "arguments": ["cc", "-std=c11", "-fvisibility=hidden", "-O2", "-Wall", "-fPIC", "-I%s"%s, "-c", "%s", "-o", "/dev/null"]}' \
      "$sep" "$root" "$file" "$native_include" "$args" "$file"
    sep=,
  done <<LIST
tensor/native/ggml.c tensor/native third_party/ggml/include
tensor/native/ggml_graph.c tensor/native third_party/ggml/include
whisper/native/whisper.c whisper/native third_party/whisper.cpp/include third_party/whisper.cpp/ggml/include
LIST
  printf '\n]\n'
} > "$db"

exec agentica "$dir/cchecker.agc" -p "$db" -c cchecker.toml --baseline cchecker-baseline.json "$@"
