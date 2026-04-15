#!/usr/bin/env python3
"""Validate atom-facing documentation for jargon density."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from contributionlib import collect_docstring_records, json_dump, repo_relative, score_jargon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.58,
        help="Flag records with jargon score at or above this threshold.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    parser.add_argument("--ci", action="store_true", help="Exit non-zero when findings exist.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(args.root).resolve()

    findings: list[dict[str, object]] = []
    for record in collect_docstring_records(repo_root):
        score = score_jargon(record.text)
        if score["score"] < args.threshold:
            continue
        findings.append(
            {
                "path": repo_relative(record.path, repo_root),
                "kind": record.kind,
                "name": record.name,
                "line": record.line,
                "score": round(score["score"], 4),
                "components": {
                    key: round(value, 4)
                    for key, value in score.items()
                    if key != "score"
                },
            }
        )

    payload = {
        "root": repo_root.as_posix(),
        "threshold": args.threshold,
        "count": len(findings),
        "findings": findings,
    }

    if args.json:
        print(json_dump(payload))
    else:
        if not findings:
            print("dejargonization: no findings")
        else:
            print(f"dejargonization: {len(findings)} finding(s)")
            for item in findings:
                location = f"{item['path']}:{item['line']}"
                print(f"- {location} [{item['kind']}] {item['name']} score={item['score']}")

    if args.ci and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
