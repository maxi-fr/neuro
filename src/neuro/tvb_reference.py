"""Stage 7 -- TVB reference implementation for validating the hand-rolled backbone.

Rebuilds the *uncontrolled* Yu et al. 2024 network inside The Virtual Brain and uses
it as an independent oracle for Stages 1, 2, and 5 (single-node dynamics, the
coupled+delayed network, and the EEG forward operator ``L``). It does **not** model
tES or the slow ``Z`` dynamics -- those live only in the hand-rolled engine.

Parameter reconciliation (TVB works in **milliseconds**, the hand-rolled engine in
**seconds**), verified against ``models.JansenRit`` defaults:

* ``a = 0.1``, ``b = 0.05`` (1/ms)  == paper ``a = 100``, ``b = 50`` (1/s) -- TVB default
* ``nu_max = 0.0025`` (1/ms)         == paper ``e0 = 2.5`` (1/s) -- TVB default
* ``v0 = 6``, ``r = 0.56``, ``J = 135``, ``B = 22`` -- paper values
* ``mu = 0.09``                      == paper ``mean_input = 90`` (1/s)
* coupling ``K sum_j w_ij S(y_j)``   == ``SigmoidalJansenRit(cmin=0, cmax=2 nu_max,
  midpoint=v0, r=0.56, a=K)``: with those constants TVB's ``pre()`` is exactly the
  paper sigmoid ``S(y_j)`` and ``a`` is the global coupling ``K``.
* noise on the paper's ``x5'`` (= TVB state index 4) via ``HeunStochastic`` with
  ``Additive`` noise; ``nsig`` is an ``(6,)`` array nonzero only at index 4. The
  intensity is calibrated empirically (``nsig`` defaults to ``1e-4``) to reproduce
  the hand-rolled recruitment (EZ + PZ seize, left-hemisphere dominant); the
  std-to-nsig algebra overshoots the bifurcation, so it is tuned, not derived.

Structural data is taken from the same :class:`~neuro.connectome.Connectome` the
hand-rolled engine uses (weights, tract lengths, speed), so the two engines run on
bit-identical connectivity. The EEG monitor is built from the same projection,
sensor, and region-mapping files; TVB collapses its surface lead field to regions by
**sum**, matching :func:`neuro.connectome._build_eeg_gain`, so the monitor's region
gain equals ``connectome.gain`` up to the sagittal-mirror row permutation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neuro.connectome import (
    _PROJECTION_FILE,
    _REGION_MAPPING_FILE,
    _SENSORS_FILE,
    _mirror_partner_permutation,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.connectivity import Connectivity
    from tvb.datatypes.projections import ProjectionSurfaceEEG
    from tvb.datatypes.region_mapping import RegionMapping
    from tvb.datatypes.sensors import SensorsEEG
    from tvb.simulator import coupling, integrators, monitors, noise, simulator
    from tvb.simulator.models.jansen_rit import JansenRit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuro.connectome import Connectome
    from neuro.types import FloatArray


# Yu et al. 2024 parameters in TVB's millisecond units (see the module docstring).
_A_BG, _A_PZ, _A_EZ = 3.25, 3.4, 3.6
_B, _A_RATE, _B_RATE = 22.0, 0.1, 0.05
_NU_MAX, _V0, _R, _J, _MU = 0.0025, 6.0, 0.56, 135.0, 0.09
_X5_INDEX = 4  # paper's x5' equation == TVB state-variable index 4 (y4)
_DEFAULT_EZ = ("lHC", "lPHC", "lAMYG")
_DEFAULT_PZ = ("lTCI", "lTCV")


@dataclass(frozen=True)
class ReferenceResult:
    """Output of a TVB reference run.

    Attributes
    ----------
    t
        Sample times in **seconds**, shape ``(n_samples,)``.
    region_y
        Per-region output ``y = y1 - y2`` (paper ``x2 - x3``), shape
        ``(n_regions, n_samples)``.
    eeg
        62-channel scalp EEG ``L @ region_y`` in ``connectome.channel_labels`` order
        (the TVB monitor gain, mirror-corrected to own-hemisphere channels), or
        ``None`` when the simulator has no EEG monitor. Shape ``(62, n_samples)``.
    """

    t: FloatArray
    region_y: FloatArray
    eeg: FloatArray | None


def _spatialized_gain(  # noqa: PLR0913
    connectome: Connectome,
    ez: Sequence[str],
    pz: Sequence[str],
    a_ez: float,
    a_pz: float,
    a_bg: float,
) -> FloatArray:
    """Build the per-region excitatory-gain column ``A`` of shape ``(n_regions, 1)``."""
    n = connectome.weights.shape[0]
    gain = np.full((n, 1), a_bg, dtype=np.float64)
    for name in ez:
        gain[connectome.region_index[name], 0] = a_ez
    for name in pz:
        gain[connectome.region_index[name], 0] = a_pz
    return gain


def build_reference_simulator(  # noqa: PLR0913
    connectome: Connectome,
    *,
    dt_ms: float = 0.1,
    K: float = 0.54,
    nsig: float = 1e-4,
    ez: Sequence[str] = _DEFAULT_EZ,
    pz: Sequence[str] = _DEFAULT_PZ,
    a_ez: float = _A_EZ,
    a_pz: float = _A_PZ,
    a_bg: float = _A_BG,
    seed: int = 69,
    with_eeg: bool = True,
) -> simulator.Simulator:
    """Build a configured TVB Jansen-Rit reference simulator for the EZ/PZ network.

    Parameters
    ----------
    connectome
        Structural data (weights, tract lengths, speed) shared with the hand-rolled
        engine; pre-scale ``weights`` (e.g. ````) before passing if desired.
    dt_ms
        Integration step in milliseconds (0.1 ms == the hand-rolled ``dt = 1e-4`` s).
    K
        Global coupling, mapped to ``SigmoidalJansenRit.a``.
    nsig
        Additive-noise intensity on the ``x5'`` equation; ``0.0`` is deterministic.
    ez, pz
        Region labels for the epileptogenic and propagation zones.
    a_ez, a_pz, a_bg
        Excitatory gains for EZ, PZ, and background regions.
    seed
        Noise stream seed.
    with_eeg
        If ``True``, attach an EEG monitor (its region gain validates ``L``).

    Returns
    -------
    simulator.Simulator
        A configured simulator ready for :func:`run_reference`.
    """
    conn = Connectivity.from_file()
    conn.weights = np.asarray(connectome.weights, dtype=np.float64)
    conn.tract_lengths = np.asarray(connectome.tract_lengths, dtype=np.float64)
    conn.speed = np.array([connectome.speed], dtype=np.float64)
    conn.configure()

    model = JansenRit(
        A=_spatialized_gain(connectome, ez, pz, a_ez, a_pz, a_bg),
        B=np.array([_B]),
        a=np.array([_A_RATE]),
        b=np.array([_B_RATE]),
        v0=np.array([_V0]),
        nu_max=np.array([_NU_MAX]),
        r=np.array([_R]),
        J=np.array([_J]),
        mu=np.array([_MU]),
    )

    coupler = coupling.SigmoidalJansenRit(
        cmin=np.array([0.0]),
        cmax=np.array([2.0 * _NU_MAX]),
        midpoint=np.array([_V0]),
        r=np.array([_R]),
        a=np.array([K]),
    )

    nsig_vec = np.zeros(6, dtype=np.float64)
    nsig_vec[_X5_INDEX] = nsig
    integrator = integrators.HeunStochastic(
        dt=dt_ms,
        noise=noise.Additive(nsig=nsig_vec, noise_seed=seed),
    )

    mons: list[monitors.Monitor] = [monitors.TemporalAverage(period=1.0)]
    if with_eeg:
        mons.append(
            monitors.EEG(
                projection=ProjectionSurfaceEEG.from_file(_PROJECTION_FILE, matlab_data_name="ProjectionMatrix"),
                sensors=SensorsEEG.from_file(_SENSORS_FILE),
                region_mapping=RegionMapping.from_file(_REGION_MAPPING_FILE),
                period=1.0,
            )
        )

    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupler,
        integrator=integrator,
        monitors=tuple(mons),
    )
    sim.configure()
    return sim


def _eeg_monitor(sim: simulator.Simulator) -> monitors.EEG | None:
    """Return the simulator's EEG monitor, or ``None`` if it has none."""
    for mon in sim.monitors:
        if isinstance(mon, monitors.EEG):
            return mon
    return None


def reference_eeg_gain(sim: simulator.Simulator) -> FloatArray:
    """Return the EEG monitor's region-level gain ``(62, n_regions)``.

    This is TVB's own forward operator (surface lead field collapsed to regions by
    sum). It equals :attr:`neuro.connectome.Connectome.gain` up to the sagittal-mirror
    row permutation -- the direct ``L`` validation. The gain is populated by
    :meth:`Simulator.configure`, so this needs no simulation run.

    Raises
    ------
    ValueError
        If the simulator was built with ``with_eeg=False``.
    """
    mon = _eeg_monitor(sim)
    if mon is None:
        msg = "Simulator has no EEG monitor; build it with with_eeg=True."
        raise ValueError(msg)
    return np.asarray(mon.gain, dtype=np.float64)


def run_reference(sim: simulator.Simulator, duration_s: float) -> ReferenceResult:
    """Run the reference simulator and return region output and (optionally) EEG.

    Parameters
    ----------
    sim
        A simulator from :func:`build_reference_simulator`.
    duration_s
        Simulated duration in seconds.

    Returns
    -------
    ReferenceResult
        Times (s), per-region ``y1 - y2``, and the 62-channel EEG (own-hemisphere
        channel order) when an EEG monitor is present.
    """
    outputs = sim.run(simulation_length=duration_s * 1000.0)
    t_ms, region_data = outputs[0]  # TemporalAverage: (time, n_voi, n_nodes, modes)
    region_y = np.ascontiguousarray((region_data[:, 1, :, 0] - region_data[:, 2, :, 0]).T)
    t = np.asarray(t_ms, dtype=np.float64) / 1000.0

    eeg = None
    if _eeg_monitor(sim) is not None:
        # Mirror-correct the monitor's gain to own-hemisphere channels (same fix as
        # the hand-rolled L), then project: EEG = L @ Y, comparable to EEGMeasurement.
        raw_gain = reference_eeg_gain(sim)
        sensors = SensorsEEG.from_file(_SENSORS_FILE)
        partner = _mirror_partner_permutation(np.asarray(sensors.locations, dtype=np.float64))
        eeg = raw_gain[partner] @ region_y

    return ReferenceResult(t=t, region_y=region_y, eeg=eeg)
