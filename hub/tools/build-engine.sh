#!/bin/sh
# Builds the ggml engine of `tensor/` (its tools/build-engine.sh, same
# arguments and exports), so CI tests the half weights on `Backend.Ggml` here too.
set -eu
exec sh "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/../../tensor/tools/build-engine.sh" "$@"
