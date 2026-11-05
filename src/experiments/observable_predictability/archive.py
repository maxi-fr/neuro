from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


from experiments.observable_predictability.features import (
    CONTROL_HORIZONS_S,
    REPRESENTATIONS,
    ConnectivitySupportSelection,
    HealthyConnectivityReference,
    HealthySpectralReference,
    HealthyWaveformReference,
)
from experiments.observable_predictability.metrics import RolloutIdentity, validate_identities
from experiments.observable_predictability.scoring import (
    PotentialPredictability,
    SeizureRelevanceResult,
)
from neuro.config import StftGeometry

SCHEMA_VERSION = 3


def _canonical_json_sha256(payload: object) -> str:
    """Compute a deterministic SHA-256 hex digest of a JSON-serializable structure."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_manifest_hashes(manifest: Mapping[str, Any]) -> tuple[str, str]:
    """Compute ``(manifest_schema_hash, identity_set_digest)`` for cache provenance validation."""
    identities = validate_identities([RolloutIdentity.from_dict(item) for item in manifest.get("identities", [])])
    identity_digest = _canonical_json_sha256([ident.to_dict() for ident in identities])
    schema_payload = {
        "schema_version": manifest.get("schema_version"),
        "dt": manifest.get("dt"),
        "fs": manifest.get("fs"),
        "region_fs": manifest.get("region_fs"),
        "rollout_s": manifest.get("rollout_s"),
        "pre_branch_s": manifest.get("pre_branch_s"),
        "history_s": manifest.get("history_s"),
        "n_parents": manifest.get("n_parents"),
        "n_children": manifest.get("n_children"),
        "branches": manifest.get("branches"),
        "arms": manifest.get("arms"),
        "channel_labels": manifest.get("channel_labels"),
        "region_labels": manifest.get("region_labels"),
        "channel_sets": manifest.get("channel_sets"),
        "splits": manifest.get("splits"),
        "identity_digest": identity_digest,
    }
    return _canonical_json_sha256(schema_payload), identity_digest


@dataclass(frozen=True)
class ScoreCacheMetadata:
    """Strict provenance and configuration header persisted with cached evaluation scores."""

    schema_version: int
    manifest_hash: str
    identity_digest: str
    waveform_geometry: dict[str, Any]
    spectral_geometry: dict[str, Any]
    connectivity_geometry: dict[str, Any]
    selected_connectivity_window_s: float
    channel_set: str = "all62"
    horizons_s: tuple[float, ...] = CONTROL_HORIZONS_S

    def to_dict(self) -> dict[str, Any]:
        """Serialize cache metadata to a JSON-compatible dictionary."""
        return {
            "schema_version": int(self.schema_version),
            "manifest_hash": str(self.manifest_hash),
            "identity_digest": str(self.identity_digest),
            "waveform_geometry": dict(self.waveform_geometry),
            "spectral_geometry": dict(self.spectral_geometry),
            "connectivity_geometry": dict(self.connectivity_geometry),
            "selected_connectivity_window_s": float(self.selected_connectivity_window_s),
            "channel_set": str(self.channel_set),
            "horizons_s": [float(h) for h in self.horizons_s],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScoreCacheMetadata:
        """Deserialize cache metadata from a dictionary."""
        return cls(
            schema_version=int(data["schema_version"]),
            manifest_hash=str(data["manifest_hash"]),
            identity_digest=str(data["identity_digest"]),
            waveform_geometry=dict(data["waveform_geometry"]),
            spectral_geometry=dict(data["spectral_geometry"]),
            connectivity_geometry=dict(data["connectivity_geometry"]),
            selected_connectivity_window_s=float(data["selected_connectivity_window_s"]),
            channel_set=str(data.get("channel_set", "all62")),
            horizons_s=tuple(float(h) for h in data.get("horizons_s", CONTROL_HORIZONS_S)),
        )


def _validate_cache_equality_fields(c_meta: ScoreCacheMetadata, e_meta: ScoreCacheMetadata) -> None:
    """Validate exact equality fields on :class:`ScoreCacheMetadata`."""
    exact_fields = (
        "schema_version",
        "manifest_hash",
        "identity_digest",
        "channel_set",
        "waveform_geometry",
        "spectral_geometry",
        "connectivity_geometry",
    )
    for field_name in exact_fields:
        c_val = getattr(c_meta, field_name)
        e_val = getattr(e_meta, field_name)
        if c_val != e_val:
            msg = f"Cache {field_name} mismatch: cached={c_val!r}, expected={e_val!r}."
            raise ValueError(msg)
    if len(c_meta.horizons_s) != len(e_meta.horizons_s) or not np.allclose(c_meta.horizons_s, e_meta.horizons_s):
        msg = f"Cache horizons_s mismatch: cached={c_meta.horizons_s}, expected={e_meta.horizons_s}."
        raise ValueError(msg)


def validate_score_cache_metadata(
    cached: ScoreCacheMetadata | Mapping[str, Any],
    expected: ScoreCacheMetadata | Mapping[str, Any],
) -> ScoreCacheMetadata:
    """Validate that cached score metadata matches the active manifest and geometry configuration.

    Raises :class:`ValueError` immediately if the manifest hash, identity digest, schema version,
    representation geometries, selected connectivity support window is missing or mismatched.
    """
    c_meta = cached if isinstance(cached, ScoreCacheMetadata) else ScoreCacheMetadata.from_dict(cached)
    e_meta = expected if isinstance(expected, ScoreCacheMetadata) else ScoreCacheMetadata.from_dict(expected)
    _validate_cache_equality_fields(c_meta, e_meta)

    if not np.isfinite(c_meta.selected_connectivity_window_s) or c_meta.selected_connectivity_window_s <= 0.0:
        msg = f"Cache has invalid selected_connectivity_window_s={c_meta.selected_connectivity_window_s}."
        raise ValueError(msg)
    if (
        np.isfinite(e_meta.selected_connectivity_window_s)
        and e_meta.selected_connectivity_window_s > 0.0
        and not np.isclose(c_meta.selected_connectivity_window_s, e_meta.selected_connectivity_window_s)
    ):
        msg = (
            "Cache selected_connectivity_window_s mismatch: "
            f"cached={c_meta.selected_connectivity_window_s}, expected={e_meta.selected_connectivity_window_s}."
        )
        raise ValueError(msg)

    return c_meta


@dataclass(frozen=True)
class ObservableEvaluationBundle:
    """Two-criterion evaluation bundle across Waveform, Spectral, and Connectivity."""

    metadata: ScoreCacheMetadata
    manifest: dict[str, Any]
    waveform_reference: HealthyWaveformReference
    spectral_reference: HealthySpectralReference
    connectivity_reference: HealthyConnectivityReference
    connectivity_support: ConnectivitySupportSelection
    seizure_relevance: dict[str, SeizureRelevanceResult]
    potential_predictability: dict[str, PotentialPredictability] = field(default_factory=dict)


EVALUATION_SCORES_FILE = "evaluation_scores.npz"


def _serialize_evaluation_bundle(bundle: ObservableEvaluationBundle) -> dict[str, Any]:
    """Encode the bundle's arrays and metadata for the score archive."""
    arrays: dict[str, Any] = {
        "cache_metadata_json": np.array(json.dumps(bundle.metadata.to_dict())),
        "ref|waveform|mean": bundle.waveform_reference.mean,
        "ref|waveform|var": bundle.waveform_reference.var,
        "ref|waveform|std": bundle.waveform_reference.std,
        "ref|spectral|mean_log_power": bundle.spectral_reference.mean_log_power,
        "ref|spectral|channel_bin_mean": bundle.spectral_reference.channel_bin_mean,
        "ref|spectral|envelope_power": bundle.spectral_reference.envelope_power,
        "ref|spectral|std_log_power": bundle.spectral_reference.std_log_power,
        "ref|connectivity|q05": bundle.connectivity_reference.q05,
        "ref|connectivity|q95": bundle.connectivity_reference.q95,
        "ref|connectivity|eligible": bundle.connectivity_reference.eligible,
        "ref|connectivity|pair_valid_fraction": bundle.connectivity_reference.pair_valid_fraction,
    }

    for rep, result in bundle.potential_predictability.items():
        arrays[f"potential|{rep}|times"] = result.times
        arrays[f"potential|{rep}|signal_fraction"] = result.signal_fraction
        arrays[f"potential|{rep}|noise_floor"] = result.noise_floor
        arrays[f"potential|{rep}|valid_component_counts"] = result.valid_component_counts

    relevance_meta: dict[str, Any] = {}
    for rep, rel in bundle.seizure_relevance.items():
        arrays[f"rel|{rep}|times"] = rel.times
        for branch, p_arr in rel.penalties_by_branch.items():
            arrays[f"rel|{rep}|pen|{branch}"] = p_arr
            arrays[f"rel|{rep}|state|{branch}"] = rel.states_by_branch[branch]
        relevance_meta[rep] = {
            "branches": list(rel.penalties_by_branch),
            "identities_by_branch": {
                b: [ident.to_dict() for ident in ids] for b, ids in rel.identities_by_branch.items()
            },
        }

    cs = bundle.connectivity_support
    summary = {
        "manifest": bundle.manifest,
        "references": {
            "waveform_fs": float(bundle.waveform_reference.fs),
            "spectral_fs": float(bundle.spectral_reference.fs),
            "spectral_geometry": bundle.spectral_reference.geometry.model_dump(mode="json"),
            "connectivity_pairs": [[int(p[0]), int(p[1])] for p in bundle.connectivity_reference.pairs],
            "connectivity_window_s": float(bundle.connectivity_reference.window_s),
        },
        "connectivity_support": {
            "selected_window_s": float(cs.selected_window_s),
            "candidate_windows_s": [float(w) for w in cs.candidate_windows_s],
            "candidate_spreads": {f"{w:g}": float(v) for w, v in cs.candidate_spreads.items()},
            "candidate_normalized_spreads": {f"{w:g}": float(v) for w, v in cs.candidate_normalized_spreads.items()},
            "candidate_valid_fractions": {f"{w:g}": float(v) for w, v in cs.candidate_valid_fractions.items()},
            "support_to_horizon_ratio": {f"{h:g}": float(v) for h, v in cs.support_to_horizon_ratio.items()},
            "repeatability_target": float(cs.repeatability_target),
            "qualified": bool(cs.qualified),
        },
        "relevance_meta": relevance_meta,
    }
    arrays["bundle_summary_json"] = np.array(json.dumps(summary))
    return arrays


def save_evaluation_bundle(cache_path: Path | str, bundle: ObservableEvaluationBundle) -> None:
    """Atomically persist a complete scored bundle and its metadata header."""
    target = Path(cache_path)
    if target.is_dir():
        target = target / EVALUATION_SCORES_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    arrays = _serialize_evaluation_bundle(bundle)
    with NamedTemporaryFile(dir=target.parent, delete=False) as staging:
        staging_path = Path(staging.name)
    try:
        with staging_path.open("wb") as output:
            np.savez_compressed(output, **arrays)
        staging_path.replace(target)
    finally:
        staging_path.unlink(missing_ok=True)


def _load_bundle_references_and_support(
    data: Any,  # noqa: ANN401 -- numpy NpzFile archive mapping
    summary: Mapping[str, Any],
    metadata: ScoreCacheMetadata,
) -> tuple[
    HealthyWaveformReference,
    HealthySpectralReference,
    HealthyConnectivityReference,
    ConnectivitySupportSelection,
]:
    """Reconstruct reference distributions, connectivity support selection, and frozen dev scales."""
    ref_meta = summary["references"]
    waveform_reference = HealthyWaveformReference(
        mean=np.asarray(data["ref|waveform|mean"], dtype=np.float64),
        var=np.asarray(data["ref|waveform|var"], dtype=np.float64),
        std=np.asarray(data["ref|waveform|std"], dtype=np.float64),
        fs=float(ref_meta["waveform_fs"]),
    )
    spectral_reference = HealthySpectralReference(
        mean_log_power=np.asarray(data["ref|spectral|mean_log_power"], dtype=np.float64),
        channel_bin_mean=np.asarray(data["ref|spectral|channel_bin_mean"], dtype=np.float64),
        envelope_power=np.asarray(data["ref|spectral|envelope_power"], dtype=np.float64),
        std_log_power=np.asarray(data["ref|spectral|std_log_power"], dtype=np.float64),
        fs=float(ref_meta["spectral_fs"]),
        geometry=StftGeometry(**ref_meta["spectral_geometry"]),
    )
    connectivity_reference = HealthyConnectivityReference(
        q05=np.asarray(data["ref|connectivity|q05"], dtype=np.float64),
        q95=np.asarray(data["ref|connectivity|q95"], dtype=np.float64),
        pairs=tuple((int(p[0]), int(p[1])) for p in ref_meta["connectivity_pairs"]),
        window_s=float(ref_meta["connectivity_window_s"]),
        pair_valid_fraction=np.asarray(data["ref|connectivity|pair_valid_fraction"], dtype=np.float64),
        eligible=np.asarray(data["ref|connectivity|eligible"], dtype=bool),
    )

    cs_raw = summary["connectivity_support"]
    connectivity_support = ConnectivitySupportSelection(
        selected_window_s=float(cs_raw["selected_window_s"]),
        candidate_windows_s=tuple(float(w) for w in cs_raw["candidate_windows_s"]),
        candidate_spreads={float(w): float(v) for w, v in cs_raw["candidate_spreads"].items()},
        candidate_normalized_spreads={float(w): float(v) for w, v in cs_raw["candidate_normalized_spreads"].items()},
        candidate_valid_fractions={float(w): float(v) for w, v in cs_raw["candidate_valid_fractions"].items()},
        support_to_horizon_ratio={float(h): float(v) for h, v in cs_raw["support_to_horizon_ratio"].items()},
        repeatability_target=float(cs_raw["repeatability_target"]),
        qualified=bool(cs_raw["qualified"]),
    )

    if not np.isclose(
        metadata.selected_connectivity_window_s, connectivity_support.selected_window_s
    ) or not np.isclose(metadata.selected_connectivity_window_s, connectivity_reference.window_s):
        msg = "Cache metadata selected_connectivity_window_s disagrees with stored connectivity reference/support."
        raise ValueError(msg)
    return waveform_reference, spectral_reference, connectivity_reference, connectivity_support


def _load_bundle_criteria_results(
    data: np.lib.npyio.NpzFile,
    summary: Mapping[str, Any],
) -> dict[str, SeizureRelevanceResult]:
    """Read aligned penalty and Seizure State arrays from the archive."""
    seizure_relevance: dict[str, SeizureRelevanceResult] = {}
    for rep, rm in summary["relevance_meta"].items():
        pen_by_b = {str(b): np.asarray(data[f"rel|{rep}|pen|{b}"], dtype=np.float64) for b in rm["branches"]}
        st_by_b = {str(b): np.asarray(data[f"rel|{rep}|state|{b}"], dtype=np.float64) for b in rm["branches"]}
        ids_by_b = {
            str(b): tuple(RolloutIdentity.from_dict(item) for item in ids)
            for b, ids in rm["identities_by_branch"].items()
        }
        seizure_relevance[rep] = SeizureRelevanceResult(
            times=np.asarray(data[f"rel|{rep}|times"], dtype=np.float64),
            penalties_by_branch=pen_by_b,
            states_by_branch=st_by_b,
            identities_by_branch=ids_by_b,
        )

    return seizure_relevance


def load_evaluation_bundle(
    cache_path: Path | str,
    *,
    manifest: Mapping[str, Any] | None = None,
    expected_metadata: ScoreCacheMetadata | Mapping[str, Any] | None = None,
) -> ObservableEvaluationBundle:
    """Load a cached :class:`ObservableEvaluationBundle` and strictly validate its metadata header."""
    target = Path(cache_path)
    if target.is_dir():
        target = target / EVALUATION_SCORES_FILE
    if not target.exists():
        msg = f"Evaluation score cache not found: {target}"
        raise FileNotFoundError(msg)

    with np.load(target) as data:
        if "cache_metadata_json" not in data.files or "bundle_summary_json" not in data.files:
            msg = f"Evaluation score cache {target} is missing required metadata headers."
            raise ValueError(msg)
        metadata = ScoreCacheMetadata.from_dict(json.loads(str(data["cache_metadata_json"].item())))
        if metadata.schema_version != SCHEMA_VERSION:
            msg = f"Unsupported evaluation score schema {metadata.schema_version}. Rebuild the experiment."
            raise ValueError(msg)

        if manifest is not None:
            m_hash, id_digest = compute_manifest_hashes(manifest)
            manifest_expected = ScoreCacheMetadata(
                schema_version=int(manifest.get("schema_version", SCHEMA_VERSION)),
                manifest_hash=m_hash,
                identity_digest=id_digest,
                waveform_geometry=metadata.waveform_geometry,
                spectral_geometry=metadata.spectral_geometry,
                connectivity_geometry=metadata.connectivity_geometry,
                selected_connectivity_window_s=metadata.selected_connectivity_window_s,
                channel_set=metadata.channel_set,
                horizons_s=metadata.horizons_s,
            )
            validate_score_cache_metadata(metadata, manifest_expected)

        if expected_metadata is not None:
            validate_score_cache_metadata(metadata, expected_metadata)

        summary = json.loads(str(data["bundle_summary_json"].item()))
        (
            waveform_reference,
            spectral_reference,
            connectivity_reference,
            connectivity_support,
        ) = _load_bundle_references_and_support(data, summary, metadata)
        seizure_relevance = _load_bundle_criteria_results(data, summary)

        potential_predictability = {
            rep: PotentialPredictability(
                times=data[f"potential|{rep}|times"],
                signal_fraction=data[f"potential|{rep}|signal_fraction"],
                noise_floor=data[f"potential|{rep}|noise_floor"],
                valid_component_counts=data[f"potential|{rep}|valid_component_counts"],
            )
            for rep in REPRESENTATIONS
            if f"potential|{rep}|times" in data.files
        }
        return ObservableEvaluationBundle(
            metadata=metadata,
            manifest=dict(manifest) if manifest is not None else dict(summary["manifest"]),
            waveform_reference=waveform_reference,
            spectral_reference=spectral_reference,
            connectivity_reference=connectivity_reference,
            connectivity_support=connectivity_support,
            seizure_relevance=seizure_relevance,
            potential_predictability=potential_predictability,
        )
