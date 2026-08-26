# AGENT instructions

## Commands

`uv` runs everything. Add dependencies with `uv add <package>` instead of editing `pyproject.toml`. Torch is the exception, needing `uv pip install torch --torch-backend=auto`.

Iterate with targeted checks:

```bash
uv run pytest tests/test_foo.py -x
uv run ty check src/neuro/foo.py
uv run ruff check --fix
```

Then run the gate once before calling the work done:

```bash
uv run pre-commit run --all-files
```

## Clarify, then run

Don't guess my intent. Name missing constraints, give the competing readings rather than picking one, push back if something simpler exists, ask when unclear. That comes before coding, not during. Once the approach is settled, execute to green without checking back in; if ambiguity surfaces mid-task, do the parts that don't depend on it and state your assumption.

Make the goal verifiable first, a failing test that reproduces the bug or tests green either side of a refactor.

## Changes

No unrelated churn. Don't refactor code the task doesn't touch. Do restructure what you do touch: split the function you're editing if it needs it, change the surrounding structure rather than contorting new code into it, and say so.

Write the minimum that solves the problem. No features beyond the ask, no abstractions for single-use code, no configurability nobody requested, no error handling for states that can't happen.
No unnecessary backward compatibility

## Conventions

Update the docstring of any function you change. One line is the default; add numpy sections only for adding information about parameters, above all an array's shape, ``(n_windows, steps, n_channels)``. No module-level docstrings.

Capitalize `CONTEXT.md` glossary terms (Frame, Observable, Control Horizon). Use the `neuro.types` aliases, not raw `npt.NDArray[...]`. `__init__.py` stays empty; import from the module.

## Suppressions

Ruff runs `select = ["ALL"]`, so ill-fitting rules get suppressed, though fixing the code beats silencing the check. One rule code, one line, with a reason: `# noqa: ARG002 -- geometry is already in samples`.

## Domain docs

Single-context (`CONTEXT.md` + `docs/adr/` at repo root). See `docs/agents/domain.md`.
