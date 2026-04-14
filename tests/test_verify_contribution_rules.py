from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_verify_contribution_rules_reports_structural_failures(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    atoms_dir = repo / "src" / "sciona" / "atoms" / "demo"
    probes_dir = repo / "src" / "sciona" / "probes" / "demo"
    atoms_dir.mkdir(parents=True)
    probes_dir.mkdir(parents=True)
    (atoms_dir / "__init__.py").write_text("", encoding="utf-8")
    (probes_dir / "__init__.py").write_text("", encoding="utf-8")

    (atoms_dir / "atoms.py").write_text(
        """from __future__ import annotations

import icontract
from sciona.ghost.registry import register_atom


@icontract.require(lambda x: x is not None, "x cannot be None")
def bad_atom(x):
    return x
""",
        encoding="utf-8",
    )

    (atoms_dir / "witnesses.py").write_text(
        """from __future__ import annotations

def witness_bad_atom(*args, **kwargs):
    return None
""",
        encoding="utf-8",
    )

    (probes_dir / "bad_probe.py").write_text(
        """from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str


_MODULE = "sciona.atoms.demo.atoms"
BAD_TARGETS = (
    ProbeTarget(f"{_MODULE}.missing_symbol", _MODULE, "missing_symbol"),
)
""",
        encoding="utf-8",
    )

    (repo / "data" / "hyperparams").mkdir(parents=True)
    (repo / "data" / "hyperparams" / "manifest.json").write_text(
        json.dumps(
            {
                "reviewed_atoms": [
                    {
                        "atom": "bad_atom",
                        "path": "src/sciona/atoms/demo/atoms.py",
                        "status": "approved",
                        "reason": "fixture",
                        "tunable_params": [],
                        "blocked_params": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_contribution_rules.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(repo), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["summary"]["error"] >= 3
    messages = [finding["message"] for finding in payload["findings"]]
    assert any("missing a docstring" in message for message in messages)
    assert any("approved but has no tunable params" in message for message in messages)
    assert any("does not exist in" in message for message in messages)
