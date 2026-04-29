#!/usr/bin/env python3
"""Validate serialized artifact files against Sciona's format and security rules.

Checks each file for:
- Allowed format (by extension and magic bytes)
- No blocked serialization patterns (pickle, joblib, torch)
- Format-specific validity (JSON parses, NPZ members are .npy, etc.)
- SHA-256 hash (printed for manifests)

Usage::

    # Validate specific files:
    python scripts/validate_artifacts.py model.onnx vocab.txt checkpoint.npz

    # Validate all files in a directory:
    python scripts/validate_artifacts.py path/to/assets/

    # Quiet mode (exit code only):
    python scripts/validate_artifacts.py --quiet path/to/assets/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

# ---- Format constants (mirrored from sciona_infra.assets.format_scanner) ----

ALLOWED_FORMATS = {
    "safetensors", "onnx", "json", "jsonl", "parquet",
    "npy", "npz", "txt", "vocab",
}

BLOCKED_EXTENSIONS = {".pkl", ".pickle", ".joblib"}

BLOCKED_MAGIC_BYTES = {
    b"\x80\x05": "pickle_v5",
    b"\x80\x04": "pickle_v4",
    b"\x80\x03": "pickle_v3",
    b"\x80\x02": "pickle_v2",
}

TORCH_MARKERS = (
    b"torch._utils",
    b"torch.storage",
    b"torch\n",
    b"PyTorch",
    b"PYTORCH",
)

EXTENSION_TO_FORMAT = {
    ".safetensors": "safetensors",
    ".onnx": "onnx",
    ".json": "json",
    ".jsonl": "jsonl",
    ".parquet": "parquet",
    ".npy": "npy",
    ".npz": "npz",
    ".txt": "txt",
    ".vocab": "vocab",
}

_PREFIX_SIZE = 1024 * 1024  # 1 MB
_JSON_MAX_SIZE = 256 * 1024 * 1024  # 256 MB


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path) -> tuple[bool, list[str], str]:
    """Validate a single file. Returns (passed, errors, sha256)."""
    errors: list[str] = []

    if not path.exists():
        return False, ["file does not exist"], ""
    if not path.is_file():
        return False, ["not a regular file"], ""

    suffix = path.suffix.lower()
    declared_format = EXTENSION_TO_FORMAT.get(suffix)

    # Extension check
    if suffix in BLOCKED_EXTENSIONS:
        errors.append(f"blocked extension: {suffix}")
    if declared_format is None:
        errors.append(f"unknown extension: {suffix or '<none>'} (allowed: {', '.join(sorted(EXTENSION_TO_FORMAT.keys()))})")

    # Read prefix for magic byte and marker checks
    with path.open("rb") as fh:
        prefix = fh.read(_PREFIX_SIZE)

    if not prefix:
        errors.append("file is empty")

    # Magic byte check
    for magic, name in BLOCKED_MAGIC_BYTES.items():
        if prefix.startswith(magic):
            errors.append(f"blocked magic bytes: {name}")
            break

    # Torch marker check
    for marker in TORCH_MARKERS:
        if marker in prefix:
            errors.append(f"blocked torch serialization marker: {marker!r}")
            break

    # Format-specific checks
    if declared_format:
        errors.extend(_format_checks(path, declared_format, prefix))

    # Compute hash
    file_hash = sha256_file(path) if not errors or declared_format else ""

    return not errors, errors, file_hash


def _format_checks(path: Path, fmt: str, prefix: bytes) -> list[str]:
    if fmt == "json":
        if path.stat().st_size > _JSON_MAX_SIZE:
            return ["JSON file exceeds 256 MB limit"]
        try:
            with path.open("r", encoding="utf-8") as fh:
                json.load(fh)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return [f"invalid JSON: {exc}"]
    elif fmt == "jsonl":
        try:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    if line.strip():
                        json.loads(line)
        except UnicodeDecodeError:
            return ["invalid UTF-8"]
        except json.JSONDecodeError as exc:
            return [f"invalid JSONL at line {lineno}: {exc.msg}"]
    elif fmt in ("txt", "vocab"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                while fh.read(_PREFIX_SIZE):
                    pass
        except UnicodeDecodeError:
            return ["not valid UTF-8"]
    elif fmt == "npy":
        if not prefix.startswith(b"\x93NUMPY"):
            return ["missing NumPy magic header (\\x93NUMPY)"]
    elif fmt == "npz":
        if not zipfile.is_zipfile(path):
            return ["not a valid ZIP archive"]
        errs: list[str] = []
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if member.endswith(("/", "\\")):
                    continue
                if not member.endswith(".npy"):
                    errs.append(f"NPZ member '{member}' is not a .npy file (pickle risk)")
                else:
                    with zf.open(member) as mf:
                        header = mf.read(6)
                    if header != b"\x93NUMPY":
                        errs.append(f"NPZ member '{member}' missing NumPy header")
        return errs
    elif fmt == "parquet":
        if not prefix.startswith(b"PAR1"):
            return ["missing Parquet magic header (PAR1)"]
        with path.open("rb") as fh:
            fh.seek(-4, 2)
            trailer = fh.read(4)
        if trailer != b"PAR1":
            return ["missing Parquet magic trailer (PAR1)"]
    elif fmt == "safetensors":
        if len(prefix) < 10:
            return ["file too small for Safetensors header"]
        header_len = int.from_bytes(prefix[:8], "little", signed=False)
        if header_len <= 0:
            return ["invalid Safetensors header length"]
        file_size = path.stat().st_size
        if 8 + header_len > file_size:
            return ["Safetensors header length exceeds file size"]
        if header_len <= len(prefix) - 8:
            header_bytes = prefix[8:8 + header_len]
        else:
            with path.open("rb") as fh:
                fh.seek(8)
                header_bytes = fh.read(header_len)
        try:
            json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return ["Safetensors header is not valid JSON"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate serialized artifacts against Sciona format and security rules."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to validate",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print failures")
    args = parser.parse_args(argv)

    files: list[Path] = []
    for p in args.paths:
        p = p.resolve()
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and not child.name.startswith("."):
                    files.append(child)
        elif p.is_file():
            files.append(p)
        else:
            print(f"WARNING: {p} does not exist", file=sys.stderr)

    if not files:
        print("No files to validate.", file=sys.stderr)
        return 1

    total = 0
    failed = 0

    for path in files:
        total += 1
        passed, errors, file_hash = validate_file(path)

        if passed:
            if not args.quiet:
                print(f"PASS  {path.name}  sha256:{file_hash}  ({path.stat().st_size:,} bytes)")
        else:
            failed += 1
            print(f"FAIL  {path.name}")
            for err in errors:
                print(f"        {err}")

    print()
    if failed:
        print(f"{failed}/{total} file(s) failed validation")
        return 1
    else:
        print(f"{total}/{total} file(s) passed validation")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
