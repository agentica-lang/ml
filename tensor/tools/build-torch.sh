#!/bin/sh
# Finds libtorch, the system dependency the `torch` native module builds and
# links against, and prints the environment it needs.
#
# libtorch is never copied into this repository and never downloaded by it: a
# release of libtorch (CPU, the shared library build) is a system dependency,
# like a system libsqlite3. This script only looks for it, checks it is
# usable, and prints the variables the build table cannot name
# (`agentica.toml` names `.c` sources, `-l` libraries and nothing else).
#
# Usage, from the package folder, in the shell that then runs agentica:
#
#   eval "$(sh tools/build-torch.sh)"   # or: . tools/build-torch.sh
#
# It prints:
#
#   export TORCH_DIR="/opt/libtorch"                      where libtorch is
#   export CC="g++ -Wl,-rpath,/opt/libtorch/lib"          the C++ driver, and the rpath
#   export CPLUS_INCLUDE_PATH="/opt/libtorch/include:..." its headers
#   export LIBRARY_PATH="/opt/libtorch/lib:..."           its libraries (-L)
#   export LD_LIBRARY_PATH="/opt/libtorch/lib:..."        where to find them at run time
#
# `CC` carries the rpath because the build table has no flag of its own: a
# `libtorch.so` linked without it needs `LD_LIBRARY_PATH` at run time.
#
# Without libtorch this script exits 1 with what it looked for, and says so
# loudly: the module stays unbuilt (it is `optional` in `agentica.toml`), and
# the package still checks and tests, with the conformance suite asserting the
# absent state (README, "Local setup" and "CI").
set -eu

# Every place a CPU libtorch release is usually unpacked, in order. TORCH_DIR
# wins when it is set and usable.
for candidate in "${TORCH_DIR:-}" \
  /opt/libtorch \
  /usr/local/libtorch \
  /usr/lib/libtorch \
  "$HOME/opt/libtorch" \
  "$HOME/libtorch"; do
  if [ -n "$candidate" ] && [ -f "$candidate/include/torch/csrc/api/include/torch/torch.h" ] && [ -d "$candidate/lib" ]; then
    torch_dir=$candidate
    break
  fi
done

if [ -z "${torch_dir:-}" ]; then
  echo "build-torch.sh: no libtorch found; set TORCH_DIR to a CPU libtorch release" >&2
  echo "  (looked at TORCH_DIR, /opt/libtorch, /usr/local/libtorch, /usr/lib/libtorch," >&2
  echo "   $HOME/opt/libtorch and $HOME/libtorch, wanting include/torch/csrc/api/include/torch/torch.h)" >&2
  echo "  a CPU release: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip" >&2
  echo "  without it the native 'torch' module stays unbuilt and the conformance tests self-skip" >&2
  exit 1
fi

if [ ! -f "$torch_dir/lib/libtorch.so" ]; then
  echo "build-torch.sh: $torch_dir has no lib/libtorch.so (a shared, not static, release)" >&2
  exit 1
fi

echo "export TORCH_DIR=\"$torch_dir\""
echo "export CC=\"g++ -Wl,-rpath,$torch_dir/lib\""
echo "export CPLUS_INCLUDE_PATH=\"$torch_dir/include:$torch_dir/include/torch/csrc/api/include\${CPLUS_INCLUDE_PATH:+:\$CPLUS_INCLUDE_PATH}\""
echo "export LIBRARY_PATH=\"$torch_dir/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}\""
echo "export LD_LIBRARY_PATH=\"$torch_dir/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
