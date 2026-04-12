from __future__ import annotations

import json
from pathlib import Path


def test_namespace_hyperparams_manifest_tracks_signal_processing_tunables() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "hyperparams" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    reviewed = {record["atom"]: record for record in payload["reviewed_atoms"]}
    assert set(reviewed) == {
        "peak_correction",
        "reject_outlier_intervals",
        "heart_rate_computation_median_smoothed",
    }
    assert [param["name"] for param in reviewed["peak_correction"]["tunable_params"]] == [
        "tol"
    ]
    assert [param["name"] for param in reviewed["reject_outlier_intervals"]["tunable_params"]] == [
        "mad_scale",
        "min_interval_s",
        "max_interval_s",
    ]
    assert [
        param["name"]
        for param in reviewed["heart_rate_computation_median_smoothed"]["tunable_params"]
    ] == ["smoothing_window"]
