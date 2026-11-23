#!/bin/sh
# Builds the pinned ggml submodule (`third_party/ggml`) twice: a CPU build with
# the AVX2 instruction set pinned, and the same build with ggml's Vulkan
# backend on. ggml is the pinned submodule and nothing else: no source is
# copied into the package, and the build trees land in `build/` (gitignored).
#
# Usage, from this folder:
#
#   sh tools/build-ggml.sh              # CPU + Vulkan builds
#   sh tools/build-ggml.sh cpu          # the CPU build only
#   sh tools/build-ggml.sh vulkan       # the Vulkan build only (needs the
#                                       # Vulkan headers, glslc and
#                                       # SPIRV-Headers; see below)
#   JOBS=2 sh tools/build-ggml.sh       # parallel jobs (default: every CPU)
#
# Each build prints the cmake command it configured, stores it in
# `build/<name>/cmake-flags.txt`, and cmake keeps the same flags in
# `build/<name>/CMakeCache.txt`. After both builds the script prints the
# exports the module needs, which the build table of `agentica.toml` cannot
# name (`sources`, `-l` names and nothing else):
#
#   export C_INCLUDE_PATH="third_party/ggml/include"
#   export LIBRARY_PATH="build/vulkan/src:build/vulkan/src/ggml-vulkan:..."
#
# The module links the Vulkan tree, because that tree holds every backend this
# spike measures (ggml always compiles the CPU backend; `GGML_VULKAN=ON` adds
# the Vulkan one). The CPU-only tree is built to show that the same source at
# the same flags builds without Vulkan, and its `libggml-cpu.a` is what the
# Vulkan tree's CPU backend is built from.
#
# The Vulkan build needs what ggml's Vulkan backend needs: the Vulkan headers
# and loader (`libvulkan-dev`), `glslc` (`glslc`, or the Vulkan SDK), and
# `SPIRV-Headers` (`spirv-headers`). They are system dependencies, looked for
# in `$VULKAN_SDK` first and then wherever the distribution puts them. Missing
# ones are listed and the script exits 1: no Vulkan build, no Vulkan numbers.
set -eu

action=${1:-all}
case "$action" in
  all | cpu | vulkan) ;;
  *)
    echo "build-ggml.sh: unknown argument '$action' (cpu, vulkan or nothing for both)" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
pkg=$(CDPATH= cd -- "$here/.." && pwd)
root=$(CDPATH= cd -- "$pkg/../.." && pwd)
engine="$root/third_party/ggml"
build="$pkg/build"
jobs=${JOBS:-$(nproc 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 4)}

[ -f "$engine/CMakeLists.txt" ] || {
  echo "build-ggml.sh: no ggml at third_party/ggml (git submodule update --init)" >&2
  exit 1
}

# The flags both builds share. GGML_NATIVE=OFF pins the instruction set to
# AVX2/FMA/F16C/BMI2 (GGML_AVX512 stays off), so this machine's build is the
# same one another x86-64 AVX2 machine gets; GGML_OPENMP=OFF keeps the thread
# count in ggml's own threadpool, which is what `backend_cpu` sets.
common="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF"

configure() {
  name=$1
  flags=$2
  echo "== cmake -S third_party/ggml -B build/$name $common $flags"
  # shellcheck disable=SC2086
  cmake -S "$engine" -B "$build/$name" $common $flags > "$build/$name.configure.log" 2>&1 || {
    echo "build-ggml.sh: the $name configure failed; its log:" >&2
    tail -20 "$build/$name.configure.log" >&2
    exit 1
  }
  printf 'cmake -S third_party/ggml -B build/%s %s %s\n' "$name" "$common" "$flags" > "$build/$name/cmake-flags.txt"
}

build_targets() {
  name=$1
  shift
  # shellcheck disable=SC2086
  cmake --build "$build/$name" --target "$@" -j"$jobs"
}

mkdir -p "$build"

if [ "$action" = all ] || [ "$action" = cpu ]; then
  configure cpu "-DGGML_VULKAN=OFF"
  build_targets cpu ggml ggml-base ggml-cpu
fi

vulkan_lib_dir=
if [ "$action" = all ] || [ "$action" = vulkan ]; then
  # A Vulkan SDK or a distribution's development packages, in that order.
  sdk_prefix=""
  for candidate in "${VULKAN_SDK:-}" "$HOME/.local/vulkan-sdk/usr" /usr; do
    if [ -n "$candidate" ] && [ -f "$candidate/include/vulkan/vulkan.h" ] && [ -d "$candidate/share/cmake/SPIRV-Headers" ]; then
      sdk_prefix=$candidate
      break
    fi
  done

  glslc=$(command -v glslc 2> /dev/null || true)
  if [ -z "$glslc" ] && [ -n "$sdk_prefix" ] && [ -x "$sdk_prefix/bin/glslc" ]; then
    glslc="$sdk_prefix/bin/glslc"
  fi

  if [ -z "$sdk_prefix" ] || [ -z "$glslc" ]; then
    echo "build-ggml.sh: ggml's Vulkan backend needs the Vulkan headers, glslc and SPIRV-Headers" >&2
    echo "  (looked for include/vulkan/vulkan.h and share/cmake/SPIRV-Headers in \$VULKAN_SDK," >&2
    echo "   $HOME/.local/vulkan-sdk/usr and /usr, and for glslc on PATH and in \$VULKAN_SDK/bin)" >&2
    echo "  a distribution has them in libvulkan-dev, glslc and spirv-headers" >&2
    exit 1
  fi

  for candidate in "$sdk_prefix/lib" "$sdk_prefix/lib/x86_64-linux-gnu" /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib; do
    if [ -f "$candidate/libvulkan.so" ]; then
      vulkan_lib_dir=$candidate
      break
    fi
  done
  [ -n "$vulkan_lib_dir" ] || {
    echo "build-ggml.sh: no libvulkan.so to link against (libvulkan-dev) under $sdk_prefix/lib or /usr/lib" >&2
    exit 1
  }

  # FindVulkan looks for the headers and libvulkan.so under the prefix, and
  # glslc under its bin/ (glslc itself needs the SDK's libshaderc at run time).
  VULKAN_SDK=$sdk_prefix
  PATH="$sdk_prefix/bin:$PATH"
  LD_LIBRARY_PATH="$sdk_prefix/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export VULKAN_SDK PATH LD_LIBRARY_PATH

  configure vulkan "-DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH=$sdk_prefix"
  build_targets vulkan ggml ggml-base ggml-cpu ggml-vulkan
fi

echo
echo "ggml headers: $engine/include"
echo "export C_INCLUDE_PATH=\"$engine/include\${C_INCLUDE_PATH:+:\$C_INCLUDE_PATH}\""
echo "export LIBRARY_PATH=\"$build/vulkan/src:$build/vulkan/src/ggml-vulkan:$build/cpu/src${vulkan_lib_dir:+:$vulkan_lib_dir}\${LIBRARY_PATH:+:\$LIBRARY_PATH}\""
