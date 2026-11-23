from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_ARTIFACTS = REPO_ROOT / "artifacts" / "thesis"
HASH_MANIFEST = Path(__file__).resolve().parent / "data" / "thesis_output_hashes.json"
INPUT_MANIFEST = Path(__file__).resolve().parent / "data" / "thesis_input_manifest.json"

REQUIRES_ARTIFACTS = pytest.mark.skipif(
    not THESIS_ARTIFACTS.exists(),
    reason="artifacts/thesis is not present in this checkout",
)

FIGURE_SCRIPTS = (
    "thesis/figures/05_mpc_framework/suppression_references.py",
    "thesis/figures/06_experiment_and_discussion/closed_loop_results.py",
    "thesis/figures/06_experiment_and_discussion/offline_predictor_benchmark.py",
    "thesis/figures/06_experiment_and_discussion/plot_observable_predictability.py",
    "thesis/figures/06_experiment_and_discussion/reserved_sparse_current_traces.py",
    "thesis/figures/06_experiment_and_discussion/seed_7104_recruitment.py",
    "thesis/figures/06_experiment_and_discussion/sparse_current_traces.py",
    "thesis/figures/08_appendix/predictor_search_appendix.py",
    "thesis/figures/08_appendix/sparse_calibration_tradeoffs.py",
    "thesis/figures/08_appendix/transcription_benchmark.py",
)

_PDF_CREATION_DATE = re.compile(rb"/CreationDate\s*\([^)]*\)")
_PDF_MOD_DATE = re.compile(rb"/ModDate\s*\([^)]*\)")
_PDF_TRAILER_ID = re.compile(rb"/ID\s*\[\s*<[0-9A-Fa-f]+>\s*<[0-9A-Fa-f]+>\s*\]")


def _normalized_sha256(path: Path) -> str:
    """Compute SHA-256 of a thesis output, stripping pdflatex timestamps and trailer IDs for PDFs."""
    raw = path.read_bytes()
    if path.suffix.lower() == ".pdf":
        raw = _PDF_CREATION_DATE.sub(b"/CreationDate()", raw)
        raw = _PDF_MOD_DATE.sub(b"/ModDate()", raw)
        raw = _PDF_TRAILER_ID.sub(b"/ID[<> <>]", raw)
    return hashlib.sha256(raw).hexdigest()


def _load_hash_manifest() -> dict[str, str]:
    """Load the frozen SHA-256 manifest for thesis tables and figures."""
    return json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))


@REQUIRES_ARTIFACTS
def test_thesis_chapter_artifact_paths_exist() -> None:
    missing: list[str] = []
    for chapter in sorted((REPO_ROOT / "thesis" / "chapters").glob("*.tex")):
        text = chapter.read_text(encoding="utf-8")
        for match in re.finditer(r"artifacts/thesis/[A-Za-z0-9_./-]+", text):
            rel = match.group(0).rstrip(".")
            if not (REPO_ROOT / rel).exists():
                missing.append(f"{chapter.name}: {rel}")
    assert not missing, f"Missing artifact paths referenced in thesis chapters: {missing}"


@REQUIRES_ARTIFACTS
def test_offline_benchmark_manifest_and_checkpoint_hashes() -> None:
    matched = THESIS_ARTIFACTS / "sweep_2026_11" / "waveform_matched_geometry"
    summary = json.loads((matched / "offline" / "summary.json").read_text(encoding="utf-8"))
    for key, record in summary["checkpoints"].items():
        checkpoint = REPO_ROOT / record["path"]
        assert checkpoint.exists(), f"Missing checkpoint for {key}: {checkpoint}"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest == record["sha256"], f"Checkpoint SHA-256 mismatch for {key}: {checkpoint}"

    manifest = json.loads((matched / "offline_data_manifest.json").read_text(encoding="utf-8"))
    for rel in manifest["validation_files"]:
        assert (REPO_ROOT / rel).exists(), f"Missing validation trajectory: {rel}"


@REQUIRES_ARTIFACTS
def test_seed_7104_config_snapshots_match_runs() -> None:
    pairs = (
        (
            THESIS_ARTIFACTS / "seed_7104_forecast_check" / "base_simulation.yaml",
            THESIS_ARTIFACTS
            / "sweep_2026_11"
            / "waveform_matched_geometry"
            / "final"
            / "waveform_mlp"
            / "evaluation"
            / "runs"
            / "waveform_mlp_s7104"
            / "config.yaml",
        ),
        (
            THESIS_ARTIFACTS / "seed_7104_forecast_check" / "cnn_handover" / "spectral_cnn.yaml",
            THESIS_ARTIFACTS / "sweep_2026_11" / "observable_benchmark" / "runs" / "obs_cnn_s7104" / "config.yaml",
        ),
    )
    for snapshot, archived in pairs:
        assert snapshot.read_bytes() == archived.read_bytes(), f"Config snapshot drift: {snapshot} != {archived}"


@REQUIRES_ARTIFACTS
def test_frozen_thesis_inputs_match_manifest() -> None:
    entries: dict[str, dict[str, int | str]] = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for rel, expected in entries.items():
        path = REPO_ROOT / rel
        if not path.exists():
            mismatches.append(f"MISSING {rel}")
            continue
        actual_size = path.stat().st_size
        if actual_size != expected["size"]:
            mismatches.append(f"SIZE MISMATCH {rel}: {actual_size} != {expected['size']}")
            continue
        if "sha256" in expected:
            actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_digest != expected["sha256"]:
                mismatches.append(f"HASH MISMATCH {rel}: {actual_digest} != {expected['sha256']}")
    assert not mismatches, "Thesis figure inputs differ from frozen manifest:\n" + "\n".join(mismatches)


@REQUIRES_ARTIFACTS
def test_frozen_thesis_outputs_match_manifest() -> None:
    mismatches: list[str] = []
    for rel, expected in _load_hash_manifest().items():
        path = REPO_ROOT / rel
        if not path.exists():
            mismatches.append(f"MISSING {rel}")
            continue
        actual = _normalized_sha256(path)
        if actual != expected:
            mismatches.append(f"HASH MISMATCH {rel}: {actual} != {expected}")
    assert not mismatches, "Thesis outputs differ from frozen manifest:\n" + "\n".join(mismatches)


@REQUIRES_ARTIFACTS
@pytest.mark.slow
def test_regenerate_all_thesis_figures_and_tables() -> None:
    for rel in FIGURE_SCRIPTS:
        script = REPO_ROOT / rel
        result = subprocess.run(  # noqa: S603 -- runs trusted repository figure scripts with current interpreter
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{rel} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    test_frozen_thesis_outputs_match_manifest()
