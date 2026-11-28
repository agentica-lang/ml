"""What a CI run needs to run (the plan job of .github/workflows/ci.yml).

    git diff --name-only HEAD^1 HEAD | python3 tools/plan.py pull_request

reads the changed files on stdin and prints key=value lines for $GITHUB_OUTPUT: `packages`, a JSON list, and one
`true` or `false` per extra job. A push, or a change outside every package folder, runs everything; a pull request
runs the packages it changes and those that depend on them, as the agentica.toml files declare.
"""

import json
import sys
import tomllib
from pathlib import Path

# The packages whose tests each extra job runs.
EXTRA_JOBS = {
    "xla": {"tensor", "train"},
    "torch": {"tensor"},
    "ggml_arm64": {"tensor", "transformers"},
    # The software Vulkan job runs the pairing case of transformers/ (test/vulkan_pair_test.agc).
    "vulkan": {"transformers"},
}

REPOSITORY = "agentica-lang/ml"


def packages(root):
    return sorted(p.parent.name for p in Path(root).glob("*/agentica.toml"))


def dependencies(root):
    """Package -> the packages of this repository it depends on, in any group."""
    names = set(packages(root))
    deps = {}
    for name in names:
        manifest = tomllib.loads((Path(root) / name / "agentica.toml").read_text())
        found = set()
        for spec in manifest.get("dependencies", {}).values():
            if isinstance(spec, dict) and spec.get("git", "").removesuffix(".git").endswith(REPOSITORY):
                found.add(spec.get("subdir", ""))
        deps[name] = found & names
    return deps


def closure(deps, changed):
    """The changed packages and every package that depends on one of them, directly or through others."""
    selected = set(changed)
    grew = True
    while grew:
        grew = False
        for name, needs in deps.items():
            if name not in selected and needs & selected:
                selected.add(name)
                grew = True
    return selected


def select(root, event, files):
    everything = packages(root)
    chosen = set(everything)
    if event == "pull_request":
        # What the workflow's paths-ignore skips does not count.
        touched = {f.split("/", 1)[0] for f in files if f and not f.endswith(".md") and not f.startswith("site/")}
        # No file at all (a failed diff) is not "nothing changed": run everything.
        if touched and touched <= set(everything):
            chosen = closure(dependencies(root), touched)
    out = {"packages": sorted(chosen)}
    for job, covered in EXTRA_JOBS.items():
        out[job] = "true" if chosen & covered else "false"
    return out


def main():
    root = Path(__file__).resolve().parent.parent
    out = select(root, sys.argv[1], sys.stdin.read().split("\n"))
    out["packages"] = json.dumps(out["packages"], separators=(",", ":"))
    for key, value in out.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
