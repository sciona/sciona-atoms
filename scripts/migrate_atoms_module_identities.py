"""Align selected provider metadata with live ``atoms.py`` callable FQDNs."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATHS = (
    ROOT / "src/sciona/atoms/causal_inference/feature_primitives/references.json",
    ROOT / "src/sciona/atoms/state_estimation/kalman_filters/references.json",
    ROOT / "src/sciona/atoms/state_estimation/kalman_filters/static_kf/references.json",
    ROOT / "src/sciona/atoms/state_estimation/particle_filters/references.json",
)


def _mapping(*, reverse: bool = False) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in REFERENCE_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in payload.get("atoms", {}):
            fqdn, marker, source = str(key).partition("@")
            module, _, symbol = fqdn.rpartition(".")
            if not marker or "/atoms.py" not in source:
                continue
            if reverse and module.endswith(".atoms"):
                result[fqdn] = f"{module[:-6]}.{symbol}"
            elif not reverse and not module.endswith(".atoms"):
                result[fqdn] = f"{module}.atoms.{symbol}"
    return result


def _replace(value: object, mapping: dict[str, str]) -> object:
    if isinstance(value, dict):
        return {_replace(key, mapping): _replace(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace(item, mapping) for item in value]
    if isinstance(value, str):
        for old, new in mapping.items():
            value = value.replace(old, new)
        return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    mapping = _mapping(reverse=args.reverse)
    paths = [*REFERENCE_PATHS, *sorted((ROOT / "data/review_bundles").glob("*.json"))]
    changed = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        normalized = _replace(payload, mapping)
        if normalized != payload:
            path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
            changed += 1
    for path in (
        ROOT / "tests/test_causal_inference_feature_primitives_references_metadata.py",
        ROOT / "tests/test_kalman_filter_contract_references_metadata.py",
        ROOT / "tests/test_particle_filter_contract_references_metadata.py",
        ROOT / "tests/test_static_kf_references_metadata.py",
        ROOT / "tests/test_state_estimation_static_kf_review_bundle.py",
    ):
        text = path.read_text(encoding="utf-8")
        normalized = _replace(text, mapping)
        if normalized != text:
            path.write_text(str(normalized), encoding="utf-8")
            changed += 1
    print(f"migrated_identities={len(mapping)} changed_files={changed}")


if __name__ == "__main__":
    main()
