from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neuro.types import FloatArray, StrArray

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.sensors import SensorsEEG

SENSORS_FILE = "eeg_unitvector_62.txt.bz2"

_SENSOR_TO_CONNECTOME = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
SCALP_RADIUS_MM = 90.0
# ``EX8`` is the return the ROAST leadfield is built against: the 10-05 site of the same name,
# taken from ROAST's capInfo.xlsx and mapped unit-vector -> connectome frame -> SCALP_RADIUS_MM,
# exactly as the 62 scalp sensors are. ``EX_NECK`` is the older off-head return the analytical
# configs were calibrated with; it has no ROAST counterpart.
EXTRACEPHALIC_ELECTRODES_MM = {
    "EX_NECK": (-60.0, 0.0, -180.0),
    "EX8": (-58.7458, -41.5440, -54.0650),
}


def sensor_positions_mm(
    sensors_file: str = SENSORS_FILE, radius: float = SCALP_RADIUS_MM
) -> tuple[StrArray, FloatArray]:
    """Scalp electrode positions in the connectome's coordinate frame, in millimetres.

    TVB's sensor file gives unit vectors in a (left, posterior, superior) frame; the region
    centres are millimetres in an (anterior, left, superior) frame. Distances between the two
    are only meaningful once the sensors are rotated into the connectome frame and scaled out
    to the scalp, which is what this does.

    Returns
    -------
    labels
        Channel names, shape ``(n_sensors,)``.
    positions
        Electrode positions, shape ``(n_sensors, 3)``, on the sphere of radius ``radius``.
    """
    sensors = SensorsEEG.from_file(sensors_file)
    labels = np.asarray(sensors.labels, dtype=np.str_)
    unit = np.asarray(sensors.locations, dtype=np.float64)
    return labels, (unit @ _SENSOR_TO_CONNECTOME.T) * radius


def centres_to_mni_ras(coords: FloatArray) -> FloatArray:
    """Convert connectome-frame coordinates to MNI RAS, in millimetres.

    The connectome frame is (anterior, left, superior); MNI RAS is (right, anterior, superior).
    This is the registration seam every external head model is reached through.
    """
    coords = np.asarray(coords, dtype=np.float64)
    return np.column_stack([-coords[:, 1], coords[:, 0], coords[:, 2]])
