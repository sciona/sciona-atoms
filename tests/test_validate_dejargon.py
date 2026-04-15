from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_validate_dejargon_flags_dense_docstrings(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    atoms_dir = repo / "src" / "sciona" / "atoms" / "demo"
    atoms_dir.mkdir(parents=True)
    (atoms_dir / "__init__.py").write_text("", encoding="utf-8")
    (atoms_dir / "module.py").write_text(
        '''"""Signal Processing Documentation."""

def clean_signal(x: list[float]) -> list[float]:
    """Adaptive ECG SQI HRV PSD pipeline with QRS morphology priors and no explanation."""
    return x
''',
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_dejargon.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--root", str(repo), "--threshold", "0.30", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["count"] >= 1
    assert any(finding["name"] == "clean_signal" for finding in payload["findings"])
