"""Verify the CasADi NN-predictor port against the checkpoint's float64 reference."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import pytest
import torch
from _predictor_reference import mlp_forward, mlp_prime, mlp_rollout

from neuro.checkpoint import MLPCheckpoint, load_mlp
from neuro.nn_predictor_casadi import NNSymbolicModel, mlp_forward_ca
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import Activation, FloatArray


def _random_layers(
    rng: np.random.Generator, in_size: int, out_size: int, hidden_size: int, depth: int
) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(weight, bias)`` pairs for an MLP with ``depth`` hidden layers of ``hidden_size``."""
    sizes = [in_size, *([hidden_size] * depth), out_size]
    return tuple(
        (rng.standard_normal((n_out, n_in)) / np.sqrt(n_in), rng.standard_normal(n_out))
        for n_in, n_out in itertools.pairwise(sizes)
    )


_SEED = 42

_CASES = [
    pytest.param(2, 2, 3, 5, 2, 2, 2, "relu", id="normal"),
    pytest.param(2, 2, 3, 5, 2, 2, 2, "tanh", id="tanh"),
    pytest.param(2, 2, 3, 5, 2, 2, 2, "softplus", id="softplus"),
    pytest.param(2, 2, 3, 5, 0, 2, 2, "relu", id="depth0"),
    pytest.param(1, 1, 3, 5, 2, 2, 2, "relu", id="window1"),
]


def _np2(result: object) -> FloatArray:
    return np.array(ca.DM(result))


def _checkpoint_horizon_rollout(
    ckpt: MLPCheckpoint, y_ctx: FloatArray, u_ctx: FloatArray, u_future: FloatArray
) -> FloatArray:
    """Reference EEG rollout over one ``horizon`` via the checkpoint's float64 weights.

    Mirrors what the CasADi ``f_step``/``f_out`` chain must reproduce: absorb the raw EEG/control
    context into a model-space state, free-run on raw future controls, and decode back to raw
    EEG. ``y_ctx`` is ``(n_channels, ctx)``; returns shape ``(n_channels, horizon)``.
    """

    return mlp_rollout(ckpt, mlp_prime(ckpt, y_ctx.T, u_ctx), u_future[: ckpt.horizon]).T


def _casadi_horizon_rollout(
    model: NNSymbolicModel, z_ctx: FloatArray, u_ctx: FloatArray, u_future: FloatArray
) -> FloatArray:
    """Chain ``f_step``/``f_out`` over one ``horizon``, returning raw EEG ``(n_channels, horizon)``.

    The two sides carry the *same* trajectory in two state conventions. ``f_step`` shifts the newest
    control in before predicting, so its state pairs a y-window ending at step ``t`` with a u-window
    ending at ``t - 1`` -- what ``absorb`` builds in the MPC. ``prime`` instead ends both windows at
    ``t - 1``. Lagging the u-window and prepending ``u_ctx``'s last entry converts one to the other.
    """
    ckpt = model.checkpoint
    x = np.concatenate([z_ctx[-ckpt.n_y :].flatten(), u_ctx[-ckpt.n_u - 1 : -1].flatten()])
    u_seq = np.concatenate([u_ctx[-1:], u_future[: ckpt.horizon - 1]])
    preds = []
    for u in u_seq:
        x = _np2(model.f_step(x, u)).flatten()
        preds.append(_np2(model.f_out(x)).flatten())
    return np.stack(preds, axis=1)


def _build_checkpoint(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: Activation = "relu",
) -> tuple[Path, tuple[tuple[FloatArray, FloatArray], ...], dict[str, FloatArray]]:
    """Build and save a tiny checkpoint via ``MLPCheckpoint.save``, returning its path, layers, and scalers."""
    rng = np.random.default_rng(_SEED)
    layers = _random_layers(rng, n_y * n_channels + n_u * n_controls, n_channels, hidden_size, depth)

    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }

    checkpoint = tmp_path / "art"
    MLPCheckpoint(
        layers=layers,
        activation=activation,
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=hidden_size,
        depth=depth,
        dt=0.01,
        downsample=1,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    ).save(checkpoint)

    return checkpoint, layers, scalers


@pytest.mark.parametrize(
    ("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls", "activation"), _CASES
)
def test_mlp_forward_matches_checkpoint(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: Activation,
) -> None:
    """The CasADi MLP forward pass matches the checkpoint's float64 forward bit-for-bit."""

    checkpoint, _layers, _ = _build_checkpoint(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )
    ckpt = load_mlp(checkpoint)

    in_size = n_y * n_channels + n_u * n_controls
    x_sym = ca.SX.sym("x", in_size, 1)
    fn = ca.Function("f", [x_sym], [mlp_forward_ca(x_sym, ckpt.layers, activation)])

    rng = np.random.default_rng(_SEED + 1)
    for _ in range(5):
        x = rng.standard_normal(in_size)
        got = _np2(fn(x)).flatten()
        want = mlp_forward(np.concatenate([x[: n_y * n_channels], x[n_y * n_channels :]]), ckpt.layers, ckpt.activation)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls", "activation"), _CASES
)
def test_single_step_matches_manual_scan_iteration(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: Activation,
) -> None:
    """f_step/f_out match one hand-written iteration of the AR loop."""

    checkpoint, _layers, scalers = _build_checkpoint(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )
    model = NNSymbolicModel.from_checkpoint(checkpoint)
    ckpt = load_mlp(checkpoint)

    rng = np.random.default_rng(_SEED + 2)
    y_w = rng.standard_normal((n_y, n_channels))
    u_w_raw = rng.standard_normal((n_u, n_controls))
    u_curr = rng.standard_normal(n_controls)

    x0 = np.concatenate([y_w.flatten(), u_w_raw.flatten()])
    x_next = _np2(model.f_step(x0, u_curr)).flatten()
    y_next_ca = _np2(model.f_out(x_next)).flatten()

    new_u_w = np.concatenate([u_w_raw[1:], u_curr[None, :]], axis=0)
    new_u_w_scaled = (new_u_w - scalers["u_mean"]) / scalers["u_scale"]
    y_next_model = mlp_forward(np.concatenate([y_w.flatten(), new_u_w_scaled.flatten()]), ckpt.layers, ckpt.activation)
    new_y_w_want = np.concatenate([y_w[1:], y_next_model[None, :]], axis=0)
    x_next_want = np.concatenate([new_y_w_want.flatten(), new_u_w.flatten()])
    y_next_raw_want = y_next_model * scalers["y_scale"] + scalers["y_mean"]

    np.testing.assert_allclose(x_next, x_next_want, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(y_next_ca, y_next_raw_want, rtol=1e-10, atol=1e-12)


def test_single_step_with_global_scalar_scalers(tmp_path: Path) -> None:
    """f_step matches the float64 reference when scalers are global scalars (shape (1,)).

    ``global_scaling=True`` trains a single shared mean/scale rather than a per-channel
    vector; the CasADi port must broadcast it across all channels just as numpy does, not
    tile it by ``n_y``/``n_u`` only.
    """

    n_y, n_u, horizon, n_channels, n_controls = 3, 2, 4, 5, 2
    rng = np.random.default_rng(_SEED + 8)
    layers = _random_layers(rng, n_y * n_channels + n_u * n_controls, n_channels, 5, 2)
    scalers = {  # global (scalar) statistics, as produced by global_scaling=True
        "u_mean": np.array([0.3]),
        "u_scale": np.array([1.7]),
        "y_mean": np.array([-0.9]),
        "y_scale": np.array([2.5]),
    }
    checkpoint = tmp_path / "art"
    MLPCheckpoint(
        layers=layers,
        activation="relu",
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=5,
        depth=2,
        dt=0.01,
        downsample=1,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    ).save(checkpoint)
    model = NNSymbolicModel.from_checkpoint(checkpoint)
    ckpt = load_mlp(checkpoint)

    rng = np.random.default_rng(_SEED + 9)
    y_w = rng.standard_normal((n_y, n_channels))
    u_w_raw = rng.standard_normal((n_u, n_controls))
    u_curr = rng.standard_normal(n_controls)

    x0 = np.concatenate([y_w.flatten(), u_w_raw.flatten()])
    x_next = _np2(model.f_step(x0, u_curr)).flatten()
    y_next_ca = _np2(model.f_out(x_next)).flatten()

    new_u_w = np.concatenate([u_w_raw[1:], u_curr[None, :]], axis=0)
    u_w_scaled = (new_u_w - scalers["u_mean"]) / scalers["u_scale"]
    y_next_raw_want = (
        mlp_forward(np.concatenate([y_w.flatten(), u_w_scaled.flatten()]), ckpt.layers, ckpt.activation)
        * scalers["y_scale"]
        + scalers["y_mean"]
    )
    np.testing.assert_allclose(y_next_ca, y_next_raw_want, rtol=1e-10, atol=1e-12)


def test_output_slices_most_recent_row_not_oldest(tmp_path: Path) -> None:
    """f_out must return the LAST row of the y-window, not the first.

    Uses distinguishable per-row values so a transposed/reversed slice would be caught,
    unlike random data where it could accidentally still pass.
    """
    n_y, n_u, n_channels, n_controls = 3, 2, 2, 2
    checkpoint, _layers, _scalers = _build_checkpoint(tmp_path, n_y, n_u, 3, 5, 2, n_channels, n_controls)
    model = NNSymbolicModel.from_checkpoint(checkpoint)
    ckpt = load_mlp(checkpoint)

    y_w = np.arange(n_y * n_channels, dtype=np.float64).reshape(n_y, n_channels)
    u_w = np.zeros((n_u, n_controls))
    x = np.concatenate([y_w.flatten(), u_w.flatten()])

    got = _np2(model.f_out(x)).flatten()
    np.testing.assert_allclose(got, ckpt.y_std.inverse_transform(y_w[-1]), rtol=1e-10, atol=1e-12)


def _module_checkpoint(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: Activation = "relu",
) -> Path:
    """Save a random torch MLP's checkpoint (float32-stored weights), returning its stem."""
    rng = np.random.default_rng(_SEED)
    layers = _random_layers(rng, n_y * n_channels + n_u * n_controls, n_channels, hidden_size, depth)
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }
    model = AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=hidden_size,
        depth=depth,
        activation=activation,
        dt=0.01,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, layers, strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    path = tmp_path / "module_ckpt"
    model.save(path)
    return path


@pytest.mark.parametrize(
    ("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls", "activation"), _CASES
)
def test_adapter_rollout_matches_the_checkpoint_float64_rollout(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: Activation,
) -> None:
    """The adapter rebuilt from a torch-written checkpoint equals the checkpoint's float64 rollout.

    The module stores float32 weights; the torch-free reader casts them to float64, so the CasADi
    chain and the reference are the same arithmetic to round-off, across activations and horizons.
    """
    path = _module_checkpoint(tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation)
    model = NNSymbolicModel.from_checkpoint(path)
    ckpt = load_mlp(path)

    rng = np.random.default_rng(_SEED + 3)
    ctx = max(n_y, n_u) + 2
    y_ctx = rng.standard_normal((n_channels, ctx))
    u_ctx = rng.standard_normal((ctx, n_controls))
    u_future = rng.standard_normal((horizon, n_controls))

    y_pred_np = _checkpoint_horizon_rollout(ckpt, y_ctx, u_ctx, u_future)
    y_pred_ca_arr = _casadi_horizon_rollout(model, ckpt.y_std.transform(y_ctx.T), u_ctx, u_future)

    np.testing.assert_allclose(y_pred_ca_arr, y_pred_np, rtol=1e-10, atol=1e-12)


def test_module_rollout_matches_the_checkpoint_float64_rollout(tmp_path: Path) -> None:
    """The float32 module loaded from the checkpoint matches the checkpoint's float64 rollout."""

    path = _module_checkpoint(tmp_path, 2, 2, 3, 5, 2, 2, 2, "relu")
    module = AutoregressiveMLP.load(path)
    ckpt = load_mlp(path)

    rng = np.random.default_rng(_SEED + 3)
    y_hist = rng.standard_normal((5, 2))
    u_hist = rng.standard_normal((5, 2))
    u_future = rng.standard_normal((ckpt.horizon, 2))

    want = mlp_rollout(ckpt, mlp_prime(ckpt, y_hist, u_hist), u_future)
    got = module.rollout(module.prime(y_hist, u_hist), u_future)

    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls", "activation"), _CASES
)
def test_multistep_rollout_matches_checkpoint_rollout(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: Activation,
) -> None:
    """Chaining f_step/f_out over the horizon matches the checkpoint's float64 rollout."""
    checkpoint, _layers, _scalers = _build_checkpoint(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )
    model = NNSymbolicModel.from_checkpoint(checkpoint)
    ckpt = load_mlp(checkpoint)

    rng = np.random.default_rng(_SEED + 3)
    ctx = max(n_y, n_u) + 2
    y_ctx = rng.standard_normal((n_channels, ctx))
    u_ctx = rng.standard_normal((ctx, n_controls))
    u_future = rng.standard_normal((horizon, n_controls))

    y_pred_np = _checkpoint_horizon_rollout(ckpt, y_ctx, u_ctx, u_future)

    y_pred_ca_arr = _casadi_horizon_rollout(model, ckpt.y_std.transform(y_ctx.T), u_ctx, u_future)

    np.testing.assert_allclose(y_pred_ca_arr, y_pred_np, rtol=1e-10, atol=1e-12)


def test_casadi_softplus_stays_finite_past_the_exp_overflow() -> None:
    """The CasADi softplus is the stable ``logaddexp`` form: finite and exact well past ``z = 709``."""
    z = np.array([-800.0, -20.0, 0.0, 20.0, 800.0, 5000.0])
    identity = ((np.ones((1, 1)), np.zeros(1)), (np.ones((1, 1)), np.zeros(1)))

    z_sym = ca.MX.sym("z", 1, 1)
    softplus = ca.Function("softplus", [z_sym], [mlp_forward_ca(z_sym, identity, "softplus")])
    got = np.array([_np2(softplus(zi)).item() for zi in z])

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, np.logaddexp(z, 0.0), rtol=1e-12, atol=1e-12)


def test_mlp_checkpoint_round_trip_preserves_exact_weights_and_meta(tmp_path: Path) -> None:
    """``MLPCheckpoint.save``/``load_mlp`` round-trips weights/biases and metadata bit-exactly."""
    n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation = 2, 2, 3, 5, 2, 2, 2, "relu"
    checkpoint, want_layers, scalers = _build_checkpoint(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )

    got = load_mlp(checkpoint)

    assert len(got.layers) == len(want_layers)
    for (got_w, got_b), (want_w, want_b) in zip(got.layers, want_layers, strict=True):
        np.testing.assert_array_equal(got_w, want_w)
        np.testing.assert_array_equal(got_b, want_b)

    assert got.n_y == n_y
    assert got.n_u == n_u
    assert got.horizon == horizon
    assert got.n_channels == n_channels
    assert got.n_controls == n_controls
    assert got.dt == pytest.approx(0.01)
    assert got.downsample == 1
    np.testing.assert_array_equal(got.u_std.center, scalers["u_mean"])
    np.testing.assert_array_equal(got.u_std.scale, scalers["u_scale"])
    np.testing.assert_array_equal(got.y_std.center, scalers["y_mean"])
    np.testing.assert_array_equal(got.y_std.scale, scalers["y_scale"])
