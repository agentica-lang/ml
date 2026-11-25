.PHONY: help site site-serve site-tools check-skills cchecker cchecker-update-baseline iree-venv iree-goldens

help:
	@echo "site       build the documentation site into site/public"
	@echo "site-serve serve the site on localhost for a preview"
	@echo "check-skills   check the code of the agent skills (.agents/skills/ml-*)"
	@echo "cchecker   check the native C (locks, ownership, gcc -fanalyzer, clang-tidy) against cchecker-baseline.json"
	@echo "cchecker-update-baseline   accept today's findings of cchecker"
	@echo "iree-venv  make the venv with the pinned IREE compiler and runtime (.venv-iree/, local only)"
	@echo "iree-goldens   regenerate the StableHLO snapshots and their IREE goldens of train/ (local only)"

# The documentation site (Hugo): the look, the layouts and the highlighter come from the docs theme, the submodule
# site/themes/docs-theme; the pages are the README.md and DESIGN.md files of the packages (mounted in site/hugo.toml),
# roadmap.md and site/content. The theme's styles need Dart Sass.
HUGO ?= hugo
THEME := site/themes/docs-theme
DART_SASS ?= $(HOME)/.local/opt/dart-sass
site site-serve: export PATH := $(DART_SASS):$(PATH)

# The theme and Dart Sass must be there before Hugo runs.
site-tools:
	@test -e $(THEME)/hugo.toml || { echo "The site needs its theme: git submodule update --init $(THEME)" >&2; exit 1; }
	@command -v sass >/dev/null || { echo "The site needs Dart Sass, on PATH or in $(DART_SASS)" >&2; exit 1; }

# The highlighter colors the Agentica code blocks after Hugo built the pages.
$(THEME)/highlight/node_modules: $(THEME)/highlight/package.json $(THEME)/highlight/package-lock.json
	npm ci --prefix $(THEME)/highlight --ignore-scripts --no-audit --no-fund
	touch $@

# --panicOnWarning: a link to a page that does not exist fails the build. The highlighter fails when the site has no
# agentica block, or when no token got a color.
site: site-tools $(THEME)/highlight/node_modules
	$(HUGO) --source site --cleanDestinationDir --panicOnWarning $(HUGO_FLAGS)
	node $(THEME)/highlight/highlight.mjs site/public

# Agentica blocks stay plain here: the highlighter runs in `make site`.
site-serve: site-tools
	$(HUGO) server --source site

# cchecker over the native C of tensor/, whisper/ and asr/ (tools/cchecker.sh; the README says how). It needs gcc 10+,
# clang, clang-tidy, the submodules and the pinned `agentica` on PATH. CCHECKER_FLAGS: `--gcc gcc-14 --clang clang-19
# --clang-tidy clang-tidy-19` runs the versions of the baseline.
CCHECKER_FLAGS ?=

cchecker:
	sh tools/cchecker.sh $(CCHECKER_FLAGS)

cchecker-update-baseline:
	sh tools/cchecker.sh --update-baseline $(CCHECKER_FLAGS)

# The StableHLO goldens (tensor/DESIGN.md, section 12). IREE is local tooling, never a CI dependency: CI checks the
# committed snapshots and that the reference backend matches the committed goldens. The two packages are the official
# PyPI wheels of a release more than two weeks old, in a venv that git ignores.
IREE_VERSION := 3.13.0
IREE_VENV := .venv-iree
IREE_STAMP := $(IREE_VENV)/.installed-$(IREE_VERSION)
STABLEHLO_FIXTURES := train/test/fixtures/stablehlo
STABLEHLO_PLANS := build/iree

iree-venv: $(IREE_STAMP)

$(IREE_STAMP):
	rm -rf $(IREE_VENV)
	python3 -m venv $(IREE_VENV)
	$(IREE_VENV)/bin/pip install --only-binary :all: iree-base-compiler==$(IREE_VERSION) iree-base-runtime==$(IREE_VERSION)
	touch $@

# Writes the snapshots, then runs them on IREE and writes the goldens beside them.
iree-goldens: $(IREE_STAMP)
	cd train && agentica tools/stablehlo_emit.agc ../$(STABLEHLO_FIXTURES) ../$(STABLEHLO_PLANS)
	PATH="$(CURDIR)/$(IREE_VENV)/bin:$$PATH" $(IREE_VENV)/bin/python tools/iree_goldens.py $(STABLEHLO_PLANS) $(STABLEHLO_FIXTURES)

# The code of the agent skills of .agents/skills/ml-*: every `agentica` block is checked, formatted and, with an
# `output` block, run on the reference backend (tools/check-skills.sh). It needs the pinned `agentica` on PATH.
check-skills:
	sh tools/check-skills.sh
