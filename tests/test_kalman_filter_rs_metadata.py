from __future__ import annotations

import json
from pathlib import Path


def test_kalman_metadata_record_targets_snake_case_measurement_oracle() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata_path = (
        root
        / "src"
        / "sciona"
        / "atoms"
        / "state_estimation"
        / "kalman_filters"
        / "heuristic_metadata.json"
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    record = payload["records"][0]

    assert record["atom_fqdn"] == (
        "sciona.atoms.state_estimation.kalman_filters."
        "filter_rs.evaluate_measurement_oracle"
    )
    assert record["heuristic_outputs"][0]["heuristic"]["heuristic_id"] == (
        "residual_structure_after_transform"
    )
    assert record["maintainers"] == ["sciona-atoms"]
