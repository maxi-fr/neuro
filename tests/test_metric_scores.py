from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments.observable_predictability.archive import (
    EVALUATION_SCORES_FILE,
    ScoreCacheMetadata,
    load_evaluation_bundle,
    validate_score_cache_metadata,
)
from experiments.observable_predictability.ensemble import (
    ARM_ZERO,
    Arm,
    Branch,
    EnsembleConfig,
    EvaluationProtocol,
    build_plants,
    generate,
    load_or_score_ensemble,
)
from experiments.observable_predictability.features import (
    REPRESENTATIONS,
)
from experiments.observable_predictability.metrics import (
    RolloutIdentity,
)
from experiments.observable_predictability.scoring import (
    ReplayMoments,
    compute_potential_predictability,
)
from neuro.config import StftGeometry

if TYPE_CHECKING:
    from pathlib import Path


ARM_STIM = Arm("stim", (2.0, 0.0, -2.0))


def _assert_cache_metadata_tampering_rejected(meta: ScoreCacheMetadata) -> None:
    """Verify validate_score_cache_metadata rejects tampered hash, digest, and support window."""
    base = meta.to_dict()
    with pytest.raises(ValueError, match="manifest_hash mismatch"):
        validate_score_cache_metadata({**base, "manifest_hash": "0" * 64}, meta)
    with pytest.raises(ValueError, match="identity_digest mismatch"):
        validate_score_cache_metadata({**base, "identity_digest": "f" * 64}, meta)
    with pytest.raises(ValueError, match="selected_connectivity_window_s mismatch"):
        validate_score_cache_metadata({**base, "selected_connectivity_window_s": 9.99}, meta)


def test_potential_predictability_corrects_finite_replay_noise_and_sums_variances() -> None:
    """Check the corrected ratio, shared component validity, and native floor."""
    times = np.array([0.1, 0.2, 0.3])
    means = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, np.nan], [0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0], [1.0, 0.0, 10.0], [0.0, 0.0, 0.0]]]
    )
    variances = np.array(
        [[[2.0, 8.0, 0.0], [2.0, 0.0, np.nan], [0.0, 0.0, 0.0]], [[2.0, 8.0, 0.0], [2.0, 0.0, 50.0], [0.0, 0.0, 0.0]]]
    )
    blocks = [ReplayMoments(times, means[i : i + 1], variances[i : i + 1], 2) for i in range(2)]

    result = compute_potential_predictability(blocks)
    np.testing.assert_allclose(result.signal_fraction, [3 / 13, -1.0, np.nan])
    np.testing.assert_allclose(result.noise_floor, np.sqrt([10 / 3, 1.0, 0.0]))
    np.testing.assert_array_equal(result.valid_component_counts, [3, 2, 3])

    single_snapshot = compute_potential_predictability(blocks[:1])
    assert np.isnan(single_snapshot.signal_fraction).all()
    np.testing.assert_allclose(single_snapshot.noise_floor, result.noise_floor)


def test_score_ensemble_pipeline_and_strict_cache_metadata_validation(tmp_path: Path) -> None:
    """Verify end-to-end reference->dev->eval scoring and strict cache metadata rejection on mismatch."""
    plants = build_plants()
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=2,
        rollout_s=0.60,
        pre_branch_s=1.20,
        branches=(
            Branch("healthy", "healthy", 1.20, arms=(ARM_ZERO, ARM_STIM)),
            Branch("pre_ictal", "seizure", 1.20, arms=(ARM_ZERO, ARM_STIM)),
            Branch("ictal", "seizure", 1.50, arms=(ARM_ZERO, ARM_STIM)),
            Branch("post_ictal", "seizure", 1.80, arms=(ARM_ZERO, ARM_STIM)),
        ),
        reference_seeds=(101,),
        dev_seeds=(201,),
        chunk_s=0.60,
    )
    out_dir = tmp_path / "mini_eval_ensemble"
    generate(out_dir, cfg, plants)

    bundle = load_or_score_ensemble(out_dir, protocol=EvaluationProtocol(channel_set="lTCI"))
    cache_file = out_dir / EVALUATION_SCORES_FILE
    assert cache_file.exists()
    assert bundle.metadata.schema_version == 3
    assert bundle.metadata.channel_set == "lTCI"
    assert bundle.metadata.selected_connectivity_window_s == bundle.connectivity_support.selected_window_s
    assert set(bundle.seizure_relevance) == set(REPRESENTATIONS)

    reloaded = load_or_score_ensemble(out_dir, protocol=EvaluationProtocol(channel_set="lTCI"))
    assert reloaded.metadata == bundle.metadata
    for rep in REPRESENTATIONS:
        np.testing.assert_allclose(
            reloaded.potential_predictability[rep].signal_fraction,
            bundle.potential_predictability[rep].signal_fraction,
        )

    with pytest.raises(ValueError, match="channel_set mismatch"):
        load_or_score_ensemble(out_dir, protocol=EvaluationProtocol(channel_set="all62"))
    alt_geom = StftGeometry(n_segment=25, n_hop=3, band_hz=(3.0, 20.0))
    with pytest.raises(ValueError, match="spectral_geometry mismatch"):
        load_or_score_ensemble(out_dir, protocol=EvaluationProtocol(channel_set="lTCI", stft_geometry=alt_geom))

    _assert_cache_metadata_tampering_rejected(bundle.metadata)

    manifest_path = out_dir / "manifest.json"
    modified_manifest = deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    modified_manifest["arms"][1]["current"] = [1.5, 0.0, -1.5]
    manifest_path.write_text(json.dumps(modified_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_hash mismatch"):
        load_evaluation_bundle(cache_file, manifest=modified_manifest)
    with pytest.raises(ValueError, match="manifest_hash mismatch"):
        load_or_score_ensemble(out_dir, protocol=EvaluationProtocol(channel_set="lTCI"))

    rebuilt = load_or_score_ensemble(out_dir, protocol=EvaluationProtocol(channel_set="lTCI"), on_mismatch="rebuild")
    assert rebuilt.metadata.manifest_hash != bundle.metadata.manifest_hash
