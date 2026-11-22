#!/bin/sh
# Builds the pinned whisper.cpp engine that the native `whisper` module links.
#
# The engine enters only as the pinned submodule at third_party/whisper.cpp
# (never copied into native/): this script compiles it CPU-only, static and
# with position-independent code, so the thin C layer in native/whisper.c can
# link it into the module. Fetched copies of this package have no third_party/
# (only native/ travels), so they use the prebuilt libraries instead; this
# script is for checkouts (local work and CI).
#
# Usage, from anywhere:
#   sh whisper/tools/build-engine.sh
#
# It configures third_party/whisper.cpp/build-cpu once and rebuilds the
# `whisper` target (idempotent: rerun after a submodule update). It prints the
# two directories its libraries land in; export them before running agentica,
# since the module's build table only names `-l` libraries:
#
#   export C_INCLUDE_PATH="$ENGINE/include:$ENGINE/ggml/include"
#   export LIBRARY_PATH="$ENGINE/build-cpu/src:$ENGINE/build-cpu/ggml/src"
#
# Needs cmake and a C/C++ compiler. CPU only: no CUDA, Metal, Vulkan, OpenMP
# or server/tests/examples.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
pkg=$(CDPATH= cd -- "$here/.." && pwd)
root=$(CDPATH= cd -- "$pkg/.." && pwd)
engine="$root/third_party/whisper.cpp"
build="$engine/build-cpu"

[ -f "$engine/CMakeLists.txt" ] || {
  echo "build-engine.sh: no engine at third_party/whisper.cpp (git submodule update --init)" >&2
  exit 1
}

cmake -S "$engine" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DWHISPER_BUILD_TESTS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF \
  -DWHISPER_BUILD_SERVER=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF > /dev/null

cmake --build "$build" --target whisper -j"$(nproc 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 4)"

echo "engine libraries: $build/src $build/ggml/src"
echo "export C_INCLUDE_PATH=\"$engine/include:$engine/ggml/include\""
echo "export LIBRARY_PATH=\"$build/src:$build/ggml/src\""
