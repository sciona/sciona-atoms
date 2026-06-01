from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_publishability import _check_cdg  # noqa: E402


def test_check_cdg_requires_complexity_metadata_for_atomic_nodes(tmp_path: Path) -> None:
    family_dir = tmp_path / "src" / "sciona" / "atoms" / "demo"
    family_dir.mkdir(parents=True)
    (family_dir / "cdg.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node_id": "demo_root",
                        "name": "Demo Root",
                        "status": "decomposed",
                        "children": ["bad_atom"],
                    },
                    {
                        "node_id": "bad_atom",
                        "name": "bad_atom",
                        "status": "atomic",
                        "inputs": [{"name": "x", "type_desc": "object"}],
                        "outputs": [{"name": "result", "type_desc": "object"}],
                    },
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    errors = _check_cdg(family_dir, ["sciona.atoms.demo.bad_atom"])

    assert "CDG: bad_atom missing required complexity field `time_complexity`" in errors
    assert "CDG: bad_atom missing required complexity field `space_complexity`" in errors
    assert "CDG: bad_atom missing required complexity field `complexity_reasoning`" in errors
    assert "CDG: bad_atom missing required complexity field `complexity_confidence`" in errors
    assert all("Demo Root" not in error for error in errors)


def test_check_cdg_accepts_valid_complexity_metadata(tmp_path: Path) -> None:
    family_dir = tmp_path / "src" / "sciona" / "atoms" / "demo"
    family_dir.mkdir(parents=True)
    (family_dir / "cdg.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node_id": "good_atom",
                        "name": "good_atom",
                        "status": "atomic",
                        "inputs": [{"name": "x", "type_desc": "object"}],
                        "outputs": [{"name": "result", "type_desc": "object"}],
                        "time_complexity": "O(n)",
                        "space_complexity": "O(1)",
                        "complexity_reasoning": "The atom scans n inputs once and stores constant auxiliary state.",
                        "complexity_confidence": 95,
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    assert _check_cdg(family_dir, ["sciona.atoms.demo.good_atom"]) == []
