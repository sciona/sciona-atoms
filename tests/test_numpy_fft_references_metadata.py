from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "numpy" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_ATOM_KEYS = {
    "sciona.atoms.numpy.fft.fft",
    "sciona.atoms.numpy.fft.fftfreq",
    "sciona.atoms.numpy.fft.fftn",
    "sciona.atoms.numpy.fft.fftshift",
    "sciona.atoms.numpy.fft.hfft",
    "sciona.atoms.numpy.fft.ifft",
    "sciona.atoms.numpy.fft.ifftn",
    "sciona.atoms.numpy.fft.irfft",
    "sciona.atoms.numpy.fft.rfft",
}


def test_numpy_fft_references_are_canonical_registered_and_manual() -> None:
    refs = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]

    assert set(refs["atoms"]) == EXPECTED_ATOM_KEYS
    assert registry["numpy_fft_api"]["url"] == "https://numpy.org/doc/stable/reference/routines.fft.html"

    for atom_key, atom_refs in refs["atoms"].items():
        assert atom_key.startswith("sciona.atoms.numpy.fft.")
        assert atom_refs["references"]
        for binding in atom_refs["references"]:
            assert binding["ref_id"] in registry
            metadata = binding["match_metadata"]
            assert metadata["match_type"] == "manual"
            assert metadata["confidence"] == "high"
            assert metadata["notes"]
            assert metadata["matched_nodes"]
