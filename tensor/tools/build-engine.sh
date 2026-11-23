#!/bin/sh
# Builds the pinned ggml submodule (`third_party/ggml`) that the native `ggml`
# module links, and prints the environment it needs.
#
# Usage, from anywhere, in the shell that then runs agentica:
#
#   eval "$(sh tools/build-engine.sh | grep '^export ')"          # CPU only (CI)
#   eval "$(sh tools/build-engine.sh vulkan | grep '^export ')"   # CPU + Vulkan
#   JOBS=2 sh tools/build-engine.sh vulkan                        # parallel jobs
#   ENGINE_REBUILD=1 sh tools/build-engine.sh vulkan              # ignore the cache
#
# ggml is the pinned submodule and nothing else: no source is copied into the
# package. The build trees land in `third_party/ggml/build-tensor-cpu` and
# `build-tensor-vulkan` (ignored by ggml's own .gitignore), but what the
# module links lives outside the checkout, in a cache shared by every
# worktree:
#
#   ${AGENTICA_ML_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/agentica-ml}/ggml/<key>/
#
# The key is the ggml commit, the action, a hash of the cmake flags and a hash
# of the `cc` and `c++` versions. The entry holds ggml's static archives (the
# Vulkan one with its shaders already compiled in) and `libtensor_ggml.a`, so
# it needs nothing from a build tree. A hit builds nothing: the same pinned
# ggml is never compiled twice. A miss builds in the tree (through ccache when
# it is on PATH) and then publishes the entry whole (a temp dir, then `mv`),
# under a lock per key, so two scripts that need the same entry build it once.
# `ENGINE_REBUILD=1` wipes the tree and rebuilds the entry.
#
# The module's build table names one library, `tensor_ggml`. This script
# writes it into the cache entry as a GNU ld linker script that lists ggml's
# static archives: the CPU entry lists ggml, ggml-cpu and ggml-base; the Vulkan
# entry adds ggml-vulkan and the Vulkan loader. So one `agentica.toml` links
# either entry, CI needs no Vulkan, and the C module picks Vulkan up through
# ggml's backend registry at run time (no `#ifdef`). The two entries have
# different `LIBRARY_PATH`s, so the package cache keeps one module per entry.
#
# The Vulkan build needs the Vulkan headers and loader (`libvulkan-dev`),
# `glslc` and `SPIRV-Headers`, looked for in `$VULKAN_SDK`, then
# `$HOME/.local/vulkan-sdk/usr`, then `/usr`. Missing ones are listed and the
# script exits 1.
#
# It prints:
#
#   export C_INCLUDE_PATH="third_party/ggml/include:..."      ggml's headers (gcc)
#   export CPLUS_INCLUDE_PATH="third_party/ggml/include:..."  the same for g++, which
#                                                             tools/build-torch.sh makes CC
#   export LIBRARY_PATH="<cache>/ggml/<key>:..."              libtensor_ggml.a
set -eu

action=${1:-cpu}
case "$action" in
  cpu | vulkan) ;;
  *)
    echo "build-engine.sh: unknown argument '$action' (cpu or vulkan)" >&2
    exit 2
    ;;
esac

# The linker script below is GNU ld's, and the flags pin x86-64 AVX2.
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) ;;
  *)
    echo "build-engine.sh: Linux x86_64 only, got $(uname -s)/$(uname -m)" >&2
    exit 1
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../.." && pwd)
engine="$root/third_party/ggml"
build="$engine/build-tensor-$action"
jobs=${JOBS:-$(nproc 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 4)}

[ -f "$engine/CMakeLists.txt" ] || {
  echo "build-engine.sh: no ggml at third_party/ggml (git submodule update --init third_party/ggml)" >&2
  exit 1
}

# GGML_NATIVE=OFF pins the instruction set to AVX2/FMA/F16C, so every x86-64
# AVX2 machine gets the same build; GGML_OPENMP=OFF keeps ggml's own threads.
flags="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF"
targets="ggml ggml-base ggml-cpu"
vulkan_lib=

if [ "$action" = vulkan ]; then
  sdk=
  for candidate in "${VULKAN_SDK:-}" "$HOME/.local/vulkan-sdk/usr" /usr; do
    if [ -n "$candidate" ] && [ -f "$candidate/include/vulkan/vulkan.h" ] && [ -d "$candidate/share/cmake/SPIRV-Headers" ]; then
      sdk=$candidate
      break
    fi
  done

  glslc=$(command -v glslc 2> /dev/null || true)
  if [ -z "$glslc" ] && [ -n "$sdk" ] && [ -x "$sdk/bin/glslc" ]; then
    glslc="$sdk/bin/glslc"
  fi

  for candidate in "$sdk/lib" "$sdk/lib/x86_64-linux-gnu" /usr/lib/x86_64-linux-gnu /usr/lib; do
    if [ -n "$sdk" ] && [ -f "$candidate/libvulkan.so" ]; then
      vulkan_lib="$candidate/libvulkan.so"
      break
    fi
  done

  if [ -z "$sdk" ] || [ -z "$glslc" ] || [ -z "$vulkan_lib" ]; then
    echo "build-engine.sh: ggml's Vulkan backend needs the Vulkan headers and loader, glslc and SPIRV-Headers" >&2
    echo "  (looked for include/vulkan/vulkan.h, share/cmake/SPIRV-Headers and lib/libvulkan.so in \$VULKAN_SDK," >&2
    echo "   $HOME/.local/vulkan-sdk/usr and /usr, and for glslc on PATH and in their bin/)" >&2
    echo "  a distribution has them in libvulkan-dev, glslc and spirv-headers" >&2
    exit 1
  fi

  # glslc needs the SDK's libshaderc at run time.
  VULKAN_SDK=$sdk
  PATH="$sdk/bin:$PATH"
  LD_LIBRARY_PATH="$sdk/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export VULKAN_SDK PATH LD_LIBRARY_PATH
  flags="$flags -DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH=$sdk"
  targets="$targets ggml-vulkan"
else
  flags="$flags -DGGML_VULKAN=OFF"
fi

rev=$(git -C "$engine" rev-parse HEAD 2> /dev/null || true)
[ -n "$rev" ] || {
  echo "build-engine.sh: cannot read the ggml commit of third_party/ggml" >&2
  exit 1
}
hash() { sha256sum | cut -c1-8; }
flags_hash=$(printf 'v1 %s' "$flags" | hash)
cc_hash=$({ cc --version | head -n 1; c++ --version | head -n 1; } 2> /dev/null | hash)
key="$(printf '%s' "$rev" | cut -c1-12)-$action-$flags_hash-$cc_hash"
cache="${AGENTICA_ML_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/agentica-ml}/ggml"
entry="$cache/$key"

print_exports() {
  echo "ggml $action build: $entry"
  echo "export C_INCLUDE_PATH=\"$engine/include\${C_INCLUDE_PATH:+:\$C_INCLUDE_PATH}\""
  echo "export CPLUS_INCLUDE_PATH=\"$engine/include\${CPLUS_INCLUDE_PATH:+:\$CPLUS_INCLUDE_PATH}\""
  echo "export LIBRARY_PATH=\"$entry\${LIBRARY_PATH:+:\$LIBRARY_PATH}\""
}

# One lock per key, held to the end: a second script waits, then hits.
mkdir -p "$cache"
exec 9> "$cache/$key.lock"
flock 9

if [ -z "${ENGINE_REBUILD:-}" ] && [ -f "$entry/libtensor_ggml.a" ]; then
  echo "ggml $action: cache hit $key" >&2
  print_exports
  exit 0
fi

if [ -n "${ENGINE_REBUILD:-}" ]; then
  rm -rf "$build"
fi
if command -v ccache > /dev/null 2>&1; then
  flags="$flags -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi

mkdir -p "$build"
# shellcheck disable=SC2086
cmake -S "$engine" -B "$build" $flags > "$build/configure.log" 2>&1 || {
  echo "build-engine.sh: the $action configure failed; its log:" >&2
  tail -20 "$build/configure.log" >&2
  exit 1
}
# shellcheck disable=SC2086
cmake --build "$build" --target $targets -j"$jobs" >&2

tmp=$(mktemp -d "$cache/.tmp-XXXXXX")
chmod 755 "$tmp"
trap 'rm -rf "$tmp"' EXIT

archive() {
  found=$(find "$build/src" -name "lib$1.a" | head -n 1)
  [ -n "$found" ] || {
    echo "build-engine.sh: no lib$1.a under $build/src" >&2
    exit 1
  }
  cp "$found" "$tmp/lib$1.a"
  printf '%s' "$entry/lib$1.a"
}

libs="$(archive ggml) $(archive ggml-cpu) $(archive ggml-base)"
if [ "$action" = vulkan ]; then
  libs="$libs $(archive ggml-vulkan) $vulkan_lib"
fi
printf '/* ggml for the tensor/ native module: tools/build-engine.sh %s */\nGROUP ( %s )\n' "$action" "$libs" > "$tmp/libtensor_ggml.a"

rm -rf "$entry.old"
[ ! -d "$entry" ] || mv "$entry" "$entry.old"
mv "$tmp" "$entry"
rm -rf "$entry.old"
print_exports
