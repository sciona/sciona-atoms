#!/usr/bin/env python3
"""Provider-owned file-backed Supabase backfill CLI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sciona.atoms.supabase_backfill import DEFAULT_RUNNER_VERSION, run_backfill_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "all-file-backed",
            "io-specs",
            "parameters",
            "technical-descriptions",
            "references-registry",
            "references",
            "audit-rollups",
            "audit-evidence",
            "uncertainty",
            "verification-matches",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--atoms-root", type=Path, default=None)
    parser.add_argument("--audit-manifest", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--runner-version", default=DEFAULT_RUNNER_VERSION)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    summary = run_backfill_command(
        args.command,
        dry_run=args.dry_run,
        atoms_root=args.atoms_root,
        audit_manifest_path=args.audit_manifest,
        registry_path=args.registry_path,
        batch_size=args.batch_size,
        runner_version=args.runner_version,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
