---
name: ml-release
description: >-
  Cut a release of the agentica-lang/ml packages: docs pruning PR, release PR
  (CHANGELOG, install snippet), per-package tags <pkg>-X.Y.Z, GitHub Releases
  and post-release checks. Use when asked to release, tag or publish an ml
  package version. Not for feature or docs-only PRs, nor for releases of the
  agentica compiler or of agentica-lang/packages.
---

# ml release

ml is one repository with one folder per package. Each package has one tag,
`<pkg>-X.Y.Z` (for example `tensor-0.1.0`), with no `v` prefix, and all the
members are tagged at the same version. The tags go on the merge commit of the
release PR.

Tags, tag pushes and GitHub Releases are outward-facing. Do none of them
until the owner says yes for those exact tags.

Rules for every step:

- Use `GH_TOKEN="$GITHUB_PAT_AGENTICA"` for `gh`. Check that it is set; never
  print it. Push over SSH: `git@github.com:agentica-lang/ml.git`.
- Merge with `gh pr merge --merge` only. Never force push. Never set the git
  identity.

## 1. Preconditions

Stop and report if one fails:

- The milestone of the release has no open issues:
  `gh issue list -R agentica-lang/ml -m <version> --state open`. If some are
  open, the owner must agree to move them first.
- CI is green on `main`:
  `gh run list -R agentica-lang/ml --workflow ci.yml --branch main -L 1`.
- The packages that ask for each other release together, at one version:
  every member (`tensor`, `nn`, `hub`, `train`, `transformers`, `asr`) is
  tagged `<pkg>-X.Y.Z` on the same merge commit, because the members ask for
  each other by that tag (step 3, first bullet), not by commit. A package left
  out leaves the tags of the others asking for a tag that does not exist, so a
  patch to one member means retagging all members at the new version (the cost
  of the tag scheme). `whisper`
  asks for no member and may move alone, but its `native-gen` tag must be the
  one of `tensor` (`rg native-gen */agentica.toml`).
- The owner gave an explicit OK for the tags, with the packages and versions.

## 2. Docs pruning PR (merged first)

Run the `documentation-pruning` skill over `README.md`, `roadmap.md`, each
package's `README.md` and `DESIGN.md`, `benchmarks/README.md` and
`site/content/`. Remove what is stale or restates the code. Fix numbers that
no longer match their source. Make the Done and Not yet lists of `roadmap.md`
match the milestone: closed issues go to Done, open ones stay in Not yet.

Open the PR from a branch off `origin/main`, wait for its Pages check, and
merge it. Section 3 starts after that merge.

## 3. Release PR

Branch `release/<version>` off the merged `origin/main`. Change only:

1. `CHANGELOG.md` at the root. Create it at the first release. Add one section
   per version, with a `### <pkg>` block per released package and short lines
   that say what users see. Keep the older sections as they are:

   ```md
   ## [0.1.0] - 2026-12-01

   ### tensor

   - Adds `Tensor.from_file` for float weights.
   ```

2. Version fields, if a package has one (`rg '^version' */agentica.toml`).
   The `agentica = "X.Y.Z"` line is the Agentica version: never change it.

3. The member manifests and the install snippet:
   - Each dependency of a member on another member (`rg 'ml", tag = ' */agentica.toml`)
     becomes `tag = "<pkg>-X.Y.Z"` of this release. Inside the workspace the
     tag is ignored (the folder is read); outside it, an app that declares the
     tags of one release gets one source for each package, with no `override`.
     Run `agentica update` in each member folder to rewrite its lock, and
     `agentica check` in each.
   - The `README.md` install snippet: the tags of the released packages.
     `site/content/_index.md` has the same snippet.

4. The tags and versions that the two agent skills (`.agents/skills/ml-inference`
   and `ml-training`) show. List them with
   `rg -e '-0\.[0-9]+\.[0-9]+' .agents/skills` (the install blocks, and the
   `git clone --branch tensor-0.4.0` of `references/backends.md`) and bump
   each `<pkg>-X.Y.Z` of a released package. The install blocks have no
   `override = true` and no `native-gen`: if one appears, the manifests of
   step 3 disagree. Also list `rg 'agentica = "' .agents/skills tools/skills`:
   those are the Agentica version, and change only when the packages move to a
   new Agentica. In `tools/skills/agentica.toml` the members are `path`
   folders with `override = true` (a `path` is not the tag the members ask for);
   `whisper` is pinned by tag, moved after the tags (step 4). `make check-skills`
   must pass.

Title the PR `Release <version>`; the body lists each package and its tag.
This PR changes manifests and locks, so CI runs, with the Pages check and the
Skills check when a skill changes. Merge when they are green.

## 4. Tags and GitHub Releases (owner's OK)

After the merge, for each released package:

```bash
git fetch origin
sha=$(gh pr view <N> -R agentica-lang/ml --json mergeCommit -q .mergeCommit.oid)
git ls-remote --tags git@github.com:agentica-lang/ml.git tensor-0.1.0   # prints nothing
git tag -a tensor-0.1.0 -m "tensor 0.1.0" "$sha"
git push git@github.com:agentica-lang/ml.git tensor-0.1.0
```

Then one GitHub Release per tag. Its notes are that package's `### <pkg>`
block, copied from the CHANGELOG section into a notes file in the scratchpad:

```bash
gh release create tensor-0.1.0 -R agentica-lang/ml --verify-tag \
  --title "tensor 0.1.0" --notes-file <notes-file>
```

The notes are never `Unreleased` or empty. Never create a release on a tag push.

When `whisper` is tagged, its line in `tools/skills/agentica.toml` moves to
`tag = "whisper-X.Y.Z"` (the tag does not exist before the tags are pushed, so
this cannot be in the release PR): a small PR after the tags, with
`agentica update` there and `make check-skills`.

## 5. Post-release checks

- A fresh project installs the packages by their tags. In an empty folder
  (never at the scratchpad root), write an `agentica.toml` with the snippet of
  `README.md` (`tensor` and `nn` by `tag = "<pkg>-X.Y.Z", subdir = "<pkg>"`,
  no `override`; the `agentica` version is the package's
  `AGENTICA_VERSION`). Run `agentica update`, `agentica install`, then
  `agentica check` on a one-file program that imports both. A
  `PKG-SOURCE-CONFLICT` here means a member manifest names another tag than
  the release (step 3, first bullet): stop, do not add an `override`.
- The install blocks of the skills work exactly as written, with no edit and
  no `override`: in an empty folder, copy the `toml` block of
  `ml-inference/SKILL.md` (and the `train` line of `ml-training/SKILL.md`)
  into an `agentica.toml`, run `agentica update` and `agentica install`, then
  `agentica check` on the first program of the skill.
- The Pages run on `main` for the merge commit is green. If it did not run,
  start it with `gh workflow run pages.yml --ref main -R agentica-lang/ml`.
- Each GitHub Release exists and is not a draft: `gh release view <tag>`.
- Close the milestone once these pass: find its number in
  `gh api repos/agentica-lang/ml/milestones`, then
  `gh api -X PATCH repos/agentica-lang/ml/milestones/<n> -f state=closed`.

- Report the PR, the merge commit, the tags, the release URLs and anything
  skipped.

## 6. Stop rules

- A check fails: stop and report. Do not tag.
- A tag already exists on the remote: stop. Never move it. Deleting a tag
  needs the owner's OK.
- A GitHub Release already exists for a tag: stop and ask the owner.
