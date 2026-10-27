# AGENT instructions

## Commands

`uv` runs everything. Add dependencies with `uv add <package>` instead of editing `pyproject.toml`.

## Clarify, then go

Don't guess my intent. Name missing constraints, give the competing readings rather than picking one, push back if something simpler exists, ask when unclear. That comes before coding, not during. Once the approach is settled, execute; if ambiguity surfaces mid-task, stop and ask.

Make the goal verifiable first, a failing test that reproduces the bug or tests green either side of a refactor.

## Changes

No unrelated churn. Don't refactor code the task doesn't touch. Do restructure what you do touch: split the function you're editing if it needs it, change the surrounding structure rather than contorting new code into it, and say so.

Write the minimum that solves the problem. No features beyond the ask, no abstractions for single-use code, no configurability nobody requested, no error handling for states that can't happen.
No unnecessary backward compatibility

## Conventions

Update the docstring of any function you change. One line is the default; add numpy sections only for adding information about parameters, above all an array's shape, ``(n_windows, steps, n_channels)``. No module-level docstrings.

Capitalize `CONTEXT.md` glossary terms (Frame, Observable, Control Horizon) in the code. Use the `neuro.types` aliases, not raw `npt.NDArray[...]`. `__init__.py` stays empty; import from the module.

## Suppressions

Ruff runs `select = ["ALL"]`, so ill-fitting rules get suppressed, though fixing the code beats silencing the check. One rule code, one line, with a reason: `# noqa: ARG002 -- geometry is already in samples`.

## Domain docs

Single-context (`CONTEXT.md` + `docs/adr/` at repo root). See `docs/agents/domain.md`.
