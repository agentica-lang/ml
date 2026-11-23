#!/bin/sh
# Wall-clock timers for the `ml.*` sections of benchmarks/README.md ("Where a decoder step goes"), which the profile
# cannot give: it slows the VM, so its microseconds only tell where the time goes. `apply` swaps every
# `System.Profile.measure` of tensor/trace.agc, tensor/src/lower.agc and asr/asr.agc for `Probe.measure`
# (tensor/src/probe.agc: `Time.monotonic_ns` around the block, added to an ATS), makes asr/tools/bench_step.agc print
# the totals as `probe NAME NS` lines, and makes `ggml_run` print `tgrun nodes= alloc= compute= finish=` ms to stderr
# when TG_TIME is set. It edits the working tree: run `revert` before committing.
#
#   sh benchmarks/step_timers.sh apply
#   # PASSES 0 and PASSES 5 (the 5 passes are 120 decoder steps: the prompt and 23 tokens each), no profile:
#   (cd asr && agentica tools/bench_step.agc MODEL_DIR WAV ggml en 0 | rg '^probe' > /tmp/base.txt)
#   (cd asr && agentica tools/bench_step.agc MODEL_DIR WAV ggml en 5 | rg '^probe' > /tmp/full.txt)
#   # a section's ms per step is (full - base) / 120 / 1e6
#   sh benchmarks/step_timers.sh revert
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
files="tensor/trace.agc tensor/src/lower.agc asr/asr.agc"

case "${1:-}" in
  apply)
    cat > tensor/src/probe.agc << 'AGC'
## Timers for benchmarks/step_timers.sh; not part of the package.
STATS: ATS[String, Int] = ATS.new()

def measure[T](label: String, f: Fn[T]) -> T
  t = Time.monotonic_ns()
  r = f.call()
  STATS.add(label, Time.monotonic_ns() - t)
  STATS.add("#{label}#", 1)
  r
end

def report() -> Void
  m = STATS.to_map
  for k in m.keys.sort
    puts("probe\t#{k}\t#{m[k]}")
  end
end
AGC
    # shellcheck disable=SC2086
    sed -i 's/System\.Profile\.measure(/Probe.measure(/' $files
    sed -i '0,/^import Types from "src\/types"/s//import Probe from "src\/probe"\nimport Types from "src\/types"/' tensor/trace.agc
    sed -i '0,/^import Types from "types"/s//import Probe from "probe"\nimport Types from "types"/' tensor/src/lower.agc
    sed -i '0,/^import Trace from "pkg\/tensor\/trace"/s//import Probe from "pkg\/tensor\/src\/probe"\nimport Trace from "pkg\/tensor\/trace"/' asr/asr.agc
    sed -i '0,/^import Tensor from "pkg\/tensor\/tensor"/s//import Probe from "pkg\/tensor\/src\/probe"\nimport Tensor from "pkg\/tensor\/tensor"/' asr/tools/bench_step.agc
    sed -i 's/^  if count > 0$/  Probe.report()\n  if count > 0/' asr/tools/bench_step.agc
    python3 -I - << 'PY'
p = "tensor/native/ggml_graph.c"
s = open(p).read()
s = "#define _POSIX_C_SOURCE 200809L\n" + s
s = s.replace("#include <inttypes.h>", "#include <time.h>\n#include <inttypes.h>", 1)
s = s.replace("  tg_lock_device();\n  if (p->fill_count > 0) ggml_backend_tensor_set(p->fill_store",
              "  struct timespec t0, t1, t2, t3;\n  clock_gettime(CLOCK_MONOTONIC, &t0);\n  tg_lock_device();\n  if (p->fill_count > 0) ggml_backend_tensor_set(p->fill_store")
s = s.replace("  if (why == NULL && !ggml_gallocr_alloc_graph(d->galloc, p->graph)) why = \"allocating the graph failed\";\n",
              "  if (why == NULL && !ggml_gallocr_alloc_graph(d->galloc, p->graph)) why = \"allocating the graph failed\";\n  clock_gettime(CLOCK_MONOTONIC, &t1);\n")
s = s.replace("  if (why == NULL) {\n    ggml_backend_tensor_get(p->checks,", "  clock_gettime(CLOCK_MONOTONIC, &t2);\n  if (why == NULL) {\n    ggml_backend_tensor_get(p->checks,")
s = s.replace("  release(p);\n  tg_unlock_device();\n  free(values);\n  if (why != NULL) return tg_error(env, why);\n  return ggml_result_Void_ok",
              "  clock_gettime(CLOCK_MONOTONIC, &t3);\n  if (getenv(\"TG_TIME\")) fprintf(stderr, \"tgrun nodes=%lld alloc=%.2f compute=%.2f finish=%.2f\\n\", (long long)p->count, (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6, (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_nsec - t1.tv_nsec) / 1e6, (t3.tv_sec - t2.tv_sec) * 1e3 + (t3.tv_nsec - t2.tv_nsec) / 1e6);\n  release(p);\n  tg_unlock_device();\n  free(values);\n  if (why != NULL) return tg_error(env, why);\n  return ggml_result_Void_ok")
open(p, "w").write(s)
PY
    ;;
  revert)
    # shellcheck disable=SC2086
    sed -i 's/Probe\.measure(/System.Profile.measure(/; /^import Probe from/d' $files asr/tools/bench_step.agc
    sed -i '/^  Probe\.report()$/d' asr/tools/bench_step.agc
    rm -f tensor/src/probe.agc
    git checkout -- tensor/native/ggml_graph.c
    ;;
  *)
    echo "usage: sh benchmarks/step_timers.sh apply|revert" >&2
    exit 2
    ;;
esac
