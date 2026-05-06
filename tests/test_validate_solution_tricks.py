from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_solution_tricks.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_solution_tricks", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_registry(path: Path, tricks: list[dict]) -> None:
    path.write_text(json.dumps({"schema_version": "v1", "tricks": tricks}, indent=2))


def _valid_trick(**overrides):
    trick = {
        "trick_id": "trick.test.valid_metric_clipping",
        "name": "Valid metric clipping",
        "kind": "metric_hack",
        "status": "allowed_with_validation",
        "risk_level": "medium",
        "generalization_level": "general",
        "summary": "Clip predictions to documented metric bounds.",
        "applies_when": ["The metric defines valid numeric bounds."],
        "do_not_use_when": ["The bounds are inferred from public feedback only."],
        "validation_requirements": ["Show held-out validation before and after clipping."],
        "architect_hint": "Use only as metric post-processing after selecting a CDG.",
        "related_cdgs": ["solution.kaggle.classical_tabular_ensemble_topology"],
        "related_operations": ["metric_calibration"],
        "source_competitions": ["synthetic"],
        "source_references": ["tests/test_validate_solution_tricks.py"],
        "tags": ["metric", "clipping"],
        "audit": {
            "source_kind": "manual_analysis",
            "review_status": "draft",
            "notes": "Test fixture."
        },
    }
    trick.update(overrides)
    return trick


def _messages(violations) -> list[str]:
    return [v.message for v in violations]


def test_default_registry_validates_cleanly():
    validator = _load_validator()
    violations = validator.validate_registry(
        REPO_ROOT / "data" / "solution_tricks" / "registry.json",
        strict=True,
        repo_root=REPO_ROOT,
    )
    assert violations == []


def test_valid_fixture_registry_passes(tmp_path):
    validator = _load_validator()
    path = tmp_path / "registry.json"
    _write_registry(path, [_valid_trick()])

    assert validator.validate_registry(path, strict=True, repo_root=REPO_ROOT) == []


def test_duplicate_trick_ids_fail(tmp_path):
    validator = _load_validator()
    path = tmp_path / "registry.json"
    _write_registry(path, [_valid_trick(), _valid_trick()])

    messages = _messages(validator.validate_registry(path, repo_root=REPO_ROOT))
    assert any("Duplicate trick_id" in message for message in messages)


def test_duplicate_trick_names_fail(tmp_path):
    validator = _load_validator()
    path = tmp_path / "registry.json"
    _write_registry(
        path,
        [
            _valid_trick(trick_id="trick.test.first", name="Repeated Name"),
            _valid_trick(trick_id="trick.test.second", name=" repeated name "),
        ],
    )

    messages = _messages(validator.validate_registry(path, repo_root=REPO_ROOT))
    assert any("Duplicate trick name" in message for message in messages)


def test_invalid_enum_fails(tmp_path):
    validator = _load_validator()
    path = tmp_path / "registry.json"
    _write_registry(path, [_valid_trick(kind="not_a_kind")])

    messages = _messages(validator.validate_registry(path, repo_root=REPO_ROOT))
    assert any(".kind has invalid value" in message for message in messages)


def test_high_risk_requires_validation_requirements(tmp_path):
    validator = _load_validator()
    path = tmp_path / "registry.json"
    _write_registry(path, [_valid_trick(risk_level="high", validation_requirements=[])])

    messages = _messages(validator.validate_registry(path, repo_root=REPO_ROOT))
    assert any("validation_requirements must not be empty" in message for message in messages)
    assert any("high/disallowed risk requires validation_requirements" in message for message in messages)


def test_unknown_related_cdg_fails(tmp_path):
    validator = _load_validator()
    path = tmp_path / "registry.json"
    _write_registry(path, [_valid_trick(related_cdgs=["solution.kaggle.missing_template"])])

    messages = _messages(validator.validate_registry(path, repo_root=REPO_ROOT))
    assert any("references unknown CDG" in message for message in messages)


def test_solution_asset_id_collision_fails(tmp_path):
    validator = _load_validator()
    path = tmp_path / "registry.json"
    _write_registry(
        path,
        [_valid_trick(trick_id="solution.kaggle.classical_tabular_ensemble_topology")],
    )

    messages = _messages(validator.validate_registry(path, repo_root=REPO_ROOT))
    assert any("trick_id must match" in message for message in messages)
    assert any("collides with a solution CDG asset_id" in message for message in messages)
