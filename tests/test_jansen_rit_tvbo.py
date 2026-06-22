import jax
import numpy as np

from neuro.jansen_rit import JansenRitParams, simulate_network
from neuro.jansen_rit_tvbo import build_tvbo_network, simulate_tvbo

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

_DT = 1e-4
_SEED = 42


def _toy_params(n_nodes: int = 3, *, sigma: float = 0.0) -> JansenRitParams:
    rng = np.random.default_rng(7)
    w = rng.uniform(0.0, 1.0, (n_nodes, n_nodes))
    np.fill_diagonal(w, 0.0)
    # Simple delay pattern
    d = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=np.int64)[:n_nodes, :n_nodes]
    a_gain = np.array([3.25, 3.4, 3.6])[:n_nodes]
    return JansenRitParams(
        A=a_gain,
        sigma=sigma,
        w_weights=w,
        delay_steps=d,
        K=0.6,
    )


def test_tvbo_simulate_matches_numba() -> None:
    n_steps = 20
    base = _toy_params(3, sigma=0.0)

    t_tvbo, x_tvbo = simulate_tvbo(base, duration=n_steps * _DT, dt=_DT, seed=_SEED)
    t_ref, x_ref = simulate_network(params=base, duration=n_steps * _DT, dt=_DT, seed=_SEED)

    # Note: the solvers might align their outputs differently (e.g. including t=0).
    # Let's crop or compare the overlapping times.
    min_len = min(x_tvbo.shape[2], x_ref.shape[2])

    # Handrolled includes t=0 at index 0. tvboptim might start at dt or include 0?
    if t_tvbo[0] == 0.0:
        np.testing.assert_allclose(t_tvbo[:min_len], t_ref[:min_len], rtol=1e-7)
        np.testing.assert_allclose(x_tvbo[:, :, :min_len], x_ref[:, :, :min_len], rtol=1e-5, atol=1e-6)
    else:
        # If tvboptim starts at dt, shift by 1 for reference
        np.testing.assert_allclose(t_tvbo[:min_len], t_ref[1 : min_len + 1], rtol=1e-7)
        np.testing.assert_allclose(x_tvbo[:, :, :min_len], x_ref[:, :, 1 : min_len + 1], rtol=1e-5, atol=1e-6)


def test_single_node_matches() -> None:
    n_steps = 20
    base = _toy_params(1, sigma=0.0)

    t_tvbo, x_tvbo = simulate_tvbo(base, duration=n_steps * _DT, dt=_DT, seed=_SEED)
    t_ref, x_ref = simulate_network(params=base, duration=n_steps * _DT, dt=_DT, seed=_SEED)

    min_len = min(x_tvbo.shape[2], x_ref.shape[2])

    if t_tvbo[0] == 0.0:
        np.testing.assert_allclose(t_tvbo[:min_len], t_ref[:min_len], rtol=1e-7)
        np.testing.assert_allclose(x_tvbo[:, :, :min_len], x_ref[:, :, :min_len], rtol=1e-10, atol=1e-12)
    else:
        np.testing.assert_allclose(t_tvbo[:min_len], t_ref[1 : min_len + 1], rtol=1e-7)
        np.testing.assert_allclose(x_tvbo[:, :, :min_len], x_ref[:, :, 1 : min_len + 1], rtol=1e-10, atol=1e-12)
