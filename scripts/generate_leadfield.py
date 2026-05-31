"""Generate a neurolib AAL2 EEG lead-field matrix and cache it to disk.

neurolib only *generates* a lead-field (there is no load function): it builds a
BEM forward solution on the ``fsaverage`` template, projects a standard-1020 EEG
montage, and downsamples the per-dipole gain to AAL2 cortical regions.  This is
slow and network-dependent (it downloads ``fsaverage`` and EEGBCI data via mne),
so it is run once here and the result is saved as ``.npy`` for the plants to
``np.load`` (see :func:`neuro.plant._load_leadfield`).

The neurolib HCP datasets are parcellated into exactly these **80 cortical AAL2
regions** (LRLR ordering), so the lead-field shares the connectome's
parcellation and the saved matrix is directly usable as
``FHNPlant(leadfield_path=...)``.

``compute_downsampled_leadfield`` returns columns ordered by ascending raw AAL2
code and only for regions that received >=1 dipole, which is not guaranteed to
match the connectome's node order.  :func:`_align_to_connectome` therefore
re-orders the columns onto neurolib's fixed 80-region cortical ordering *by
region name* (the AAL2.xml names that neurolib's own pipeline matches against),
zero-filling any region the surface source space missed.  The saved matrix is
thus always ``(n_sensors, 80)`` and node-aligned with ``Cmat`` / ``Dmat``.

Requires the optional ``leadfield`` extra (mne + nibabel)::

    uv sync --extra leadfield

Usage
-----
    uv run python scripts/generate_leadfield.py
    uv run python scripts/generate_leadfield.py --out-dir data --subject fsaverage
"""

from __future__ import annotations

import argparse
import importlib
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

# AAL2 atlas files are not shipped with the neurolib wheel; fetch them from the
# repository's examples/data folder when missing.
_ATLAS_BASE_URL = "https://raw.githubusercontent.com/neurolib-dev/neurolib/master/examples/data/AAL2_atlas_data"
_DEFAULT_DATA_DIR = Path("data")
_ATLAS_DIR_NAME = "AAL2_atlas_data"


def _cortical_region_names() -> list[str]:
    """Cortical AAL2 region names in neurolib's connectome (LRLR) node order.

    Mirrors how ``LeadfieldGenerator`` names cortical regions
    (``aal2.aal2[r + 1]`` for ``r`` in ``aal2.cortex``), so position ``i`` here
    corresponds to connectome node ``i`` in ``Cmat`` / ``Dmat``.
    """
    atlas_mod = importlib.import_module("neurolib.utils.atlases")
    aal2 = atlas_mod.AutomatedAnatomicalParcellation2()
    return [aal2.aal2[r + 1] for r in aal2.cortex]


def _code_to_name_lut(xml_path: Path) -> dict[int, str]:
    """Parse AAL2.xml into a ``{region_code: region_name}`` lookup table."""
    root = ET.parse(xml_path).getroot()  # noqa: S314  # trusted atlas file
    data = root.find("data")
    if data is None:
        msg = f"{xml_path} has no <data> element"
        raise ValueError(msg)
    lut: dict[int, str] = {}
    for label in data.findall("label"):
        index = label.find("index")
        name = label.find("name")
        if index is not None and index.text and name is not None and name.text:
            lut[int(index.text)] = name.text
    return lut


def _align_to_connectome(leadfield: np.ndarray, labels: np.ndarray, xml_path: Path) -> np.ndarray:
    """Re-order lead-field columns onto the fixed 80-region connectome ordering.

    ``compute_downsampled_leadfield`` returns columns sorted by ascending AAL2
    code (``labels``) and only for regions with >=1 dipole.  Map each column to
    its region name via the AAL2.xml LUT, then place it at that region's position
    in :func:`_cortical_region_names`; regions with no dipoles are zero-filled.
    """
    code_to_name = _code_to_name_lut(xml_path)
    name_to_col = {code_to_name[int(code)]: idx for idx, code in enumerate(labels) if int(code) in code_to_name}
    ordered_names = _cortical_region_names()
    aligned = np.zeros((leadfield.shape[0], len(ordered_names)), dtype=np.float64)
    missing: list[str] = []
    for pos, name in enumerate(ordered_names):
        col = name_to_col.get(name)
        if col is None:
            missing.append(name)
        else:
            aligned[:, pos] = leadfield[:, col]
    if missing:
        print(f"WARNING: {len(missing)} cortical region(s) had no dipoles and are zero-filled: {missing}")
    return aligned


def _ensure_atlas_files(atlas_dir: Path) -> tuple[Path, Path]:
    """Return paths to ``AAL2.nii`` / ``AAL2.xml``, downloading them if absent."""
    atlas_dir.mkdir(parents=True, exist_ok=True)
    nii_path = atlas_dir / "AAL2.nii"
    xml_path = atlas_dir / "AAL2.xml"
    for path in (nii_path, xml_path):
        if path.exists():
            continue
        url = f"{_ATLAS_BASE_URL}/{path.name}"
        print(f"Downloading {url} -> {path}")
        # Hardcoded https constant, not user input.
        urllib.request.urlretrieve(url, path)  # noqa: S310
    return nii_path, xml_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the lead-field generator."""
    parser = argparse.ArgumentParser(description="Generate and cache a neurolib AAL2 EEG lead-field.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Directory for the cached lead-field and atlas files (default: data).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="fsaverage",
        help="FreeSurfer subject used for the BEM forward model (default: fsaverage).",
    )
    parser.add_argument(
        "--atlas-nii",
        type=Path,
        default=None,
        help="Path to AAL2.nii (default: <out-dir>/AAL2_atlas_data/AAL2.nii, downloaded if missing).",
    )
    parser.add_argument(
        "--atlas-xml",
        type=Path,
        default=None,
        help="Path to AAL2.xml (default: <out-dir>/AAL2_atlas_data/AAL2.xml, downloaded if missing).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the neurolib lead-field pipeline and save the connectome-aligned matrix."""
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_dir = out_dir / _ATLAS_DIR_NAME
    nii_path = args.atlas_nii if args.atlas_nii is not None else atlas_dir / "AAL2.nii"
    xml_path = args.atlas_xml if args.atlas_xml is not None else atlas_dir / "AAL2.xml"
    if args.atlas_nii is None or args.atlas_xml is None:
        nii_path, xml_path = _ensure_atlas_files(atlas_dir)

    # Imported lazily via importlib: pulls in mne + nibabel (the optional
    # `leadfield` extra) only at runtime, keeping them off the type-check path.
    leadfield_module = importlib.import_module("neurolib.utils.leadfield")
    lead = leadfield_module.LeadfieldGenerator(subject=args.subject)
    lead.load_data(subject=args.subject)
    lead.load_transformation_file(trans_path=None, subject=args.subject)
    bem, plot_bem_kwargs = lead.build_BEM(visualization=False)
    src = lead.generate_surface_source_space(plot_bem_kwargs, visualization=False)
    raw = lead.EEG_coregistration(src, visualization=False)
    fwd = lead.calculate_general_forward_solution(raw, src, bem)
    leadfield, labels = lead.compute_downsampled_leadfield(
        fwd,
        str(nii_path),
        str(xml_path),
        atlas="aal2_cortical",
        cortex_parts="only_cortical_parts",
    )

    leadfield = np.asarray(leadfield, dtype=np.float64)
    labels = np.asarray(labels)

    # Re-order columns onto the fixed 80-region connectome ordering.
    aligned = _align_to_connectome(leadfield, labels, xml_path)

    lf_path = out_dir / "leadfield_aal2.npy"
    labels_path = out_dir / "leadfield_aal2_labels.npy"
    np.save(lf_path, aligned)
    np.save(labels_path, labels)

    n_cortical = len(_cortical_region_names())
    print(f"Lead-field shape: {aligned.shape} (sensors x {n_cortical} AAL2 cortical regions, connectome-aligned)")
    print(f"Regions with >=1 dipole: {labels.size} / {n_cortical}")
    print(f"Saved lead-field -> {lf_path}")
    print(f"Saved region labels -> {labels_path}")


if __name__ == "__main__":
    main()
