#!/bin/sh
# Builds the pinned whisper.cpp engine that the native `whisper` module links,
# and prints the environment it needs.
#
# Usage, from anywhere, in the shell that then runs agentica:
#
#   eval "$(sh whisper/tools/build-engine.sh | grep '^export ')"          # CPU only (CI)
#   eval "$(sh whisper/tools/build-engine.sh vulkan | grep '^export ')"   # CPU + Vulkan
#   JOBS=2 sh whisper/tools/build-engine.sh vulkan                        # parallel jobs
#
# The engine enters only as the pinned submodule at third_party/whisper.cpp
# (never copied into native/): this script compiles it static and with
# position-independent code, into `build-whisper-cpu` or
# `build-whisper-vulkan` of the submodule (ignored by its own .gitignore).
#
# The module's build table names one library, `whisper_engine`. This script
# writes it into the build tree as a GNU ld linker script that lists the
# engine's static archives: the CPU tree lists whisper, ggml, ggml-cpu and
# ggml-base; the Vulkan tree adds ggml-vulkan and the Vulkan loader. So one
# `agentica.toml` links either tree, CI needs no Vulkan, and the engine picks
# Vulkan up through ggml's backend registry at run time (no `#ifdef`). The two
# trees have different `LIBRARY_PATH`s, so the package cache keeps one module
# per tree.
#
# The Vulkan build needs the Vulkan headers and loader (`libvulkan-dev`),
# `glslc` and `SPIRV-Headers`, looked for in `$VULKAN_SDK`, then
# `$HOME/.local/vulkan-sdk/usr`, then `/usr`. Missing ones are listed and the
# script exits 1.
#
# It prints:
#
#   export C_INCLUDE_PATH="third_party/whisper.cpp/include:..."           whisper.h and ggml's headers
#   export LIBRARY_PATH="third_party/whisper.cpp/build-whisper-cpu:..."   libwhisper_engine.a
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
engine="$root/third_party/whisper.cpp"
build="$engine/build-whisper-$action"
jobs=${JOBS:-$(nproc 2> /dev/null || echo 4)}

[ -f "$engine/CMakeLists.txt" ] || {
  echo "build-engine.sh: no engine at third_party/whisper.cpp (git submodule update --init third_party/whisper.cpp)" >&2
  exit 1
}

# GGML_NATIVE=OFF pins the instruction set to AVX2/FMA/F16C, so every x86-64
# AVX2 machine (a container host included) gets the same build;
# GGML_OPENMP=OFF keeps ggml's own threads.
flags="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_SERVER=OFF"
targets="whisper ggml ggml-base ggml-cpu"
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

mkdir -p "$build"
# shellcheck disable=SC2086
cmake -S "$engine" -B "$build" $flags > "$build/configure.log" 2>&1 || {
  echo "build-engine.sh: the $action configure failed; its log:" >&2
  tail -20 "$build/configure.log" >&2
  exit 1
}
# shellcheck disable=SC2086
cmake --build "$build" --target $targets -j"$jobs" >&2

archive() {
  found=$(find "$build/src" "$build/ggml/src" -name "lib$1.a" | head -n 1)
  [ -n "$found" ] || {
    echo "build-engine.sh: no lib$1.a under $build" >&2
    exit 1
  }
  printf '%s' "$found"
}

libs="$(archive whisper) $(archive ggml) $(archive ggml-cpu) $(archive ggml-base)"
if [ "$action" = vulkan ]; then
  libs="$libs $(archive ggml-vulkan) $vulkan_lib"
fi
printf '/* whisper.cpp for the whisper/ native module: tools/build-engine.sh %s */\nGROUP ( %s )\n' "$action" "$libs" > "$build/libwhisper_engine.a"

echo "whisper.cpp $action build: $build"
echo "export C_INCLUDE_PATH=\"$engine/include:$engine/ggml/include\${C_INCLUDE_PATH:+:\$C_INCLUDE_PATH}\""
echo "export LIBRARY_PATH=\"$build\${LIBRARY_PATH:+:\$LIBRARY_PATH}\""
