# neuro

Model Predictive Control for Neurostimulation

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management, [ruff](https://github.com/astral-sh/ruff) for linting and formatting, and pre-commit hooks to automate code quality checks.

### Setup

1. Install `uv`:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Use separate repository copies on Windows and WSL. Keep the WSL copy under
   your Linux user directory. Each copy uses its own default `.venv`.
   Sync dependencies in that copy:

    ```bash
    uv sync
    ```

    This installs JAX for CPU execution on Windows, Linux, or macOS. For NVIDIA GPU
    training on Linux or WSL2, install the CUDA 12 extra instead:

    ```bash
    uv sync --extra cuda12
    uv run --extra cuda12 python -c "import jax; print(jax.devices())"
    ```

    Native Windows supports CPU JAX; GPU execution requires WSL2. CUDA 12 supports
    the GTX 1650; CUDA 13 requires newer GPU architectures. See the
    [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

    To avoid reserving most of the GTX 1650's 4 GB VRAM at startup, set
    `XLA_PYTHON_CLIENT_PREALLOCATE=false` before launching JAX in WSL. Set
    `training.device` to `auto`, `cpu`, or `cuda`; an explicit `cuda` request fails
    when JAX cannot find a GPU. Pass `--extra cuda12` to `uv run` for GPU commands
    so uv retains the optional CUDA packages.

    The original implementation is preserved on `legacy_pytorch_training`.
    Use a separate checkout and environment when switching to that branch.

3. Set up pre-commit hooks (this will run linting, formatting, and tests on every commit):

    ```bash
    uv run pre-commit install
    ```

### Running Tests

To run tests manually using `pytest` (otherwise pre-commit will run them):

```bash
uv run pytest
```

### Linting and Formatting

To check for linting errors:

```bash
uv run ruff check .
```

To format code:

```bash
uv run ruff format .
```
