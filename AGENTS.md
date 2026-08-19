# AGENT instructions

## Project Architecture

* src/        # Implementations inside packages.
* scripts/    # CLI entry points.
* notebooks/  # Marimo notebooks

## Project specific instructions

* uv is used for dependency management (prefer `uv add [package]` over editing the pyproject.toml) and to run scripts: uv run ...
* use the ty LSP
* Make sure no formatting, linting, type or test errors are present. Sometimes it might be allowed to selectively ingore rules if it makes the code cleaner

**Standard Workflow:**
Since `pre-commit` is configured to run all checks (ruff check with --fix --unsafe-fixes and format, ty, pytest, marimo checks, markdownlint), rely on it to verify your work.

## General instructions

### 1. Clarify Before Coding

Do not guess my intent. Before implementing:

* Explicitly look for missing constraints, edge cases, or unspoken assumptions in my prompt.
* If multiple interpretations exist, present them - don't pick silently.
* If a simpler approach exists, say so. Push back.
* If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

* No features beyond what was asked.
* No abstractions for single-use code.
* No "flexibility" or "configurability" that wasn't requested.
* No error handling for impossible scenarios.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

* Don't "improve" adjacent code, comments, or formatting.
* Don't refactor things that aren't broken.
* Match existing style, even if you'd do it differently.
* Always update doc-strings of functions you change
* Avoid long function doc-strings with explanations or big comments
* Do not add module level doc-strings

However, integrate cleanly. Don't force square pegs into round holes. Do not contort new code to fit outdated, poorly written, or convoluted structures just to minimize the lines changed. Leave the immediate code better than you found it.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

* "Add validation" → "Write tests for invalid inputs, then make them pass"
* "Fix the bug" → "Write a test that reproduces it, then make it pass"
* "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, break it down into sub-goals.

### Domain docs

Single-context (`CONTEXT.md` + `docs/adr/` at repo root). See `docs/agents/domain.md`.
