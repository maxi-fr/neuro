# Project: Model Predictive Control for Neurostimulation

## Architecture

### knowledge-base/

A self-contained LLM-maintained wiki workspace with its **own** `CLAUDE.md` (symlinked to `AGENTS.md`). The knowledge-base lives in this repo but is conceptually separate from the Python project.

### src/

Python implementation of the project.

### scripts/

Scripts to run simulations.

## Project specific instructions

* uv is used for dependency management
* use the ty LSP
* Make sure no formatting, linting, type or test errors are present. Sometimes it might be allowed to selectively ingore rules if it makes the code cleaner
* You are working on a windows machine use only PowerShell commands.

```bash
uc add <package>
uv sync                                   # install deps
uv run pytest                             # tests (config in pyproject.toml: -v, --cov)
uv run ruff check . --fix --unsafe-fixes  # lint (rule set is "ALL" with curated ignores)
uv run ruff format .
uv run ty check                           # type checking with ty
uv run pre-commit run --all-files
uv run marimo check --fix --unsafe-fixes <notebook-name> # Run marimo specific lint issues
```

Pre-commit hooks run ruff (with `--fix --unsafe-fixes`), ruff-format, ty, pytest, `uv-lock`, and markdownlint on every commit — expect commits to mutate `requirements.txt` and reformat files. Note: since the pre-commit hooks runs all the checks and tests there is no point in running them manually as well.

## General instructions

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

* State your assumptions explicitly. If uncertain, ask.
* If multiple interpretations exist, present them - don't pick silently.
* If a simpler approach exists, say so. Push back when warranted.
* If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

* No features beyond what was asked.
* No abstractions for single-use code.
* No "flexibility" or "configurability" that wasn't requested.
* No error handling for impossible scenarios.
* If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

* Don't "improve" adjacent code, comments, or formatting.
* Don't refactor things that aren't broken.
* Match existing style, even if you'd do it differently.
* If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

* Remove imports/variables/functions that YOUR changes made unused.
* Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

* "Add validation" → "Write tests for invalid inputs, then make them pass"
* "Fix the bug" → "Write a test that reproduces it, then make it pass"
* "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
