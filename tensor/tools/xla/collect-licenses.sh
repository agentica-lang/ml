#!/bin/sh
# Collects the license files of what the XLA PJRT CPU plugin links, after the
# plugin was built in an openxla/xla checkout.
#
#   sh collect-licenses.sh <xla checkout> <out dir>
#
# It writes XLA's own LICENSE to <out dir>/LICENSE and, under
# <out dir>/licenses/: one folder per external Bazel repository in the
# dependencies of the plugin, with the license files of its top two levels;
# xla-third_party/, the license files of XLA's third_party/ tree; PACKAGES.txt,
# the packages the plugin depends on; and MISSING.txt, the repositories where
# no license file was found. It exits 1 when XLA's or LLVM's license is absent.
set -eu

xla=$(CDPATH= cd -- "$1" && pwd)
mkdir -p "$2/licenses"
out=$(CDPATH= cd -- "$2" && pwd)
lic=$out/licenses
target=//xla/pjrt/c:pjrt_c_api_cpu_plugin.so

cd "$xla"
cp LICENSE "$out/LICENSE"
base=$(bazel info output_base)

# cquery sees the configured build, so it leaves out the tools that only run
# during the build. bazel query is the fallback: it lists more than is linked.
if ! bazel cquery --notool_deps "deps($target)" --output=label > "$lic/labels.tmp"; then
  echo "collect-licenses: cquery failed, falling back to bazel query" >&2
  bazel query --keep_going "deps($target)" --output=label > "$lic/labels.tmp" || true
fi
sed -e 's/ (.*//' -e 's/:.*//' "$lic/labels.tmp" | sort -u > "$lic/PACKAGES.txt"
rm "$lic/labels.tmp"
[ -s "$lic/PACKAGES.txt" ] || { echo "collect-licenses: no dependencies listed" >&2; exit 1; }

find_licenses() {
  find -L "$1" -maxdepth "$2" -type f \
    \( -iname 'LICEN[SC]E*' -o -iname 'COPYING*' -o -iname 'NOTICE*' -o -iname 'COPYRIGHT*' \) \
    ! -name '*.bzl' ! -name '*.py' ! -name '*.bazel' 2>/dev/null | sort
}

copy_licenses() { # <source dir> <depth> <folder under licenses/>; fails when none is found
  found=1
  for f in $(find_licenses "$1" "$2"); do
    rel=${f#"$1"/}
    mkdir -p "$lic/$3/$(dirname "$rel")"
    cp "$f" "$lic/$3/$rel"
    found=0
  done
  return $found
}

{
  echo "# External Bazel repositories in deps($target) with no license file in"
  echo "# their top two levels. Generated configuration repositories hold no"
  echo "# third-party code; for the others the license has to be found by hand."
} > "$lic/MISSING.txt"
for repo in $(sed -n -E 's#^@@?([^/]+)//.*#\1#p' "$lic/PACKAGES.txt" | sort -u); do
  copy_licenses "$base/external/$repo" 2 "$repo" || echo "$repo" >> "$lic/MISSING.txt"
done
copy_licenses "$xla/third_party" 4 xla-third_party || true

grep -rlq 'LLVM Exceptions' "$lic" || { echo "collect-licenses: LLVM's license was not found" >&2; exit 1; }
echo "collect-licenses: $(find "$lic" -type f | wc -l) files, $(grep -vc '^#' "$lic/MISSING.txt") repositories without a license file"
