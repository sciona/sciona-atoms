"""CLI wrapper for provider-owned license metadata seeding."""

from __future__ import annotations

from sciona.atoms.license_metadata import main


if __name__ == "__main__":  # pragma: no cover - script entrypoint.
    raise SystemExit(main())
