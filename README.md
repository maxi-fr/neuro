# closed-loop-neurostimulation

Model Predictive Control for Neurostimulation

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management, [ruff](https://github.com/astral-sh/ruff) for linting and formatting, and pre-commit hooks to automate code quality checks.

### Setup

1. Install `uv`:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Sync dependencies:

    ```bash
    uv sync
    ```

    Pytorch and dependencies using it need to be installed seperately:

    ```bash
    uv pip install torch --torch-backend=auto
    uv pip install marker-pdf
    ```

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
