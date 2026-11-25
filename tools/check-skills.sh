#!/bin/sh
# Checks the code of the agent skills of this repository (.agents/skills/ml-*): every fenced `agentica` block of their
# Markdown files is a whole program. Each one is written to tools/skills/gen/, then
#   - `agentica check` runs over all of them, and `agentica fmt` must leave them as they are, and
#   - a block that a fenced `output` block follows runs on the reference backend, and its output must equal that block.
# The packages come from the folders of this repository (tools/skills/agentica.toml). Needs the pinned `agentica` on
# PATH and, for `agentica install`, git and network (or a full download cache).
#
#   sh tools/check-skills.sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root/tools/skills"

# The reference backend needs no native module: do not try to build them, which prints their compiler errors.
export AGENTICA_NATIVE_PREBUILT_ONLY=ggml,torch,xla
agentica install >/dev/null
rm -rf gen
mkdir gen

# One program per block: gen/<skill>-<file>-<n>.agc, and gen/<...>.out for the output block that follows it.
for md in "$root"/.agents/skills/ml-*/SKILL.md "$root"/.agents/skills/ml-*/references/*.md; do
  [ -f "$md" ] || continue
  skill=$(basename "$(dirname "$md")")
  [ "$skill" != references ] || skill=$(basename "$(dirname "$(dirname "$md")")")
  name=$(basename "$md" .md)
  awk -v prefix="gen/$skill-$name" '
    (state == "" || state == "after") && $0 == "```agentica" { n++; file = sprintf("%s-%02d.agc", prefix, n); state = "code"; next }
    state == "code" && $0 == "```" { close(file); state = "after"; next }
    state == "code" { print > file; next }
    state == "after" && $0 == "" { next }
    state == "after" && $0 == "```output" { out = file; sub(/\.agc$/, ".out", out); state = "out"; next }
    state == "after" { state = "" }
    state == "out" && $0 == "```" { close(out); state = ""; next }
    state == "out" { print > out; next }
  ' "$md"
done

programs=$(find gen -name '*.agc' | wc -l)
[ "$programs" -gt 0 ] || {
  echo "check-skills: no agentica block in .agents/skills/ml-*" >&2
  exit 1
}

# A block that the extraction skipped (indented, or with more on its fence line) would go unchecked.
fences=$(cat "$root"/.agents/skills/ml-*/SKILL.md "$root"/.agents/skills/ml-*/references/*.md |
  grep -c '^[[:space:]]*```agentica' || true)
[ "$fences" -eq "$programs" ] || {
  echo "check-skills: $fences agentica fences in the skills, but $programs programs extracted: a block that starts with a fence and agentica must have nothing else on its line" >&2
  exit 1
}

agentica check gen

# The formatter's layout is the one the examples show: print what it would change.
rm -rf gen.fmt
cp -R gen gen.fmt
agentica fmt gen.fmt >/dev/null
diff -ru gen gen.fmt >&2 || {
  echo "check-skills: the blocks above are not formatted (agentica fmt)" >&2
  exit 1
}
rm -rf gen.fmt

ran=0
for out in gen/*.out; do
  [ -f "$out" ] || continue
  program="${out%.out}.agc"
  agentica "$program" >"$out.actual" 2>"$out.err" || {
    cat "$out.err" >&2
    echo "check-skills: $program exited with an error" >&2
    exit 1
  }
  diff -u "$out" "$out.actual" >&2 || {
    echo "check-skills: $program prints something else than its output block" >&2
    exit 1
  }
  ran=$((ran + 1))
done

[ "$ran" -gt 0 ] || {
  echo "check-skills: no program has an output block, so nothing ran" >&2
  exit 1
}

echo "check-skills: $programs programs checked, $ran run on the reference backend"
