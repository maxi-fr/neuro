from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import jax
import pytest

from neuro.config import TrainingConfig
from neuro.predictor.train import _training_device


def test_project_installs_cpu_jax_without_pytorch() -> None:
    """CPU installs are portable, with CUDA dependencies confined to the Linux extra."""
    project = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    assert any(item.startswith("jax>=") for item in dependencies)
    assert any(item.startswith("equinox>=") for item in dependencies)
    assert not any("cuda" in item or "torch" in item for item in dependencies)
    extras = project["project"]["optional-dependencies"]
    assert set(extras) == {"cuda12"}
    assert "sys_platform == 'linux'" in extras["cuda12"][0]


def test_python_entry_points_do_not_import_pytorch() -> None:
    """Runtime, notebooks, scripts, and tests run without the legacy framework."""
    for directory in ("src", "scripts", "notebooks", "tests"):
        for path in Path(directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    assert all(alias.name.split(".")[0] != "torch" for alias in node.names), path
                elif isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] != "torch", path


def test_training_cpu_setting_selects_a_cpu() -> None:
    """Explicit CPU training selects CPU even when another backend is the default."""
    device = _training_device(TrainingConfig(device="cpu", eval_horizon_s=0.1))
    assert device.platform == "cpu"


def test_training_auto_setting_uses_default_backend() -> None:
    """Automatic training follows JAX's default backend."""
    assert _training_device(TrainingConfig(eval_horizon_s=0.1)) == jax.devices()[0]


def test_training_cuda_setting_requires_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A requested CUDA backend fails clearly when no GPU backend is installed."""

    def unavailable(backend: str | None = None) -> list[jax.Device]:
        """Represent a CPU-only JAX installation."""
        if backend == "gpu":
            msg = "Unknown backend: gpu"
            raise RuntimeError(msg)
        return []

    monkeypatch.setattr(jax, "devices", unavailable)
    with pytest.raises(RuntimeError, match="cuda12"):
        _training_device(TrainingConfig(device="cuda", eval_horizon_s=0.1))
