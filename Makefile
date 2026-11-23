.PHONY: help site site-serve site-tools cchecker cchecker-update-baseline

help:
	@echo "site       build the documentation site into site/public"
	@echo "site-serve serve the site on localhost for a preview"
	@echo "cchecker   check the native C (locks, ownership, gcc -fanalyzer, clang-tidy) against cchecker-baseline.json"
	@echo "cchecker-update-baseline   accept today's findings of cchecker"

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
