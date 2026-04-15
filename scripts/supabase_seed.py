"""CLI wrapper for provider-owned Supabase core seeding."""

from __future__ import annotations

from sciona.atoms.supabase_seed import main


if __name__ == "__main__":  # pragma: no cover - script entrypoint.
    raise SystemExit(main())
