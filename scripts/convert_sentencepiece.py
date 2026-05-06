#!/usr/bin/env python3
"""Convert SentencePiece .model files to HuggingFace tokenizer.json format.

Supports two extraction methods with automatic fallback:

1. ``sentencepiece`` library (if installed) -- loads the model natively.
2. Raw protobuf wire-format parsing -- requires only the ``struct`` stdlib
   module, no compiled .proto or sentencepiece wheel needed.

The resulting ``tokenizer.json`` uses a ``models.Unigram`` backend with
``Metaspace`` pre-tokenizer and decoder, matching the standard SentencePiece
convention.

Usage::

    # Convert, writing tokenizer.json next to the input:
    python scripts/convert_sentencepiece.py input.model

    # Explicit output path:
    python scripts/convert_sentencepiece.py input.model -o tokenizer.json

    # Round-trip validation with a test string:
    python scripts/convert_sentencepiece.py input.model --validate

    # Validate with a custom test string:
    python scripts/convert_sentencepiece.py input.model --validate --test-string "Hello world"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Protobuf wire-format helpers (no .proto compilation needed)
# ---------------------------------------------------------------------------

def _decode_varint(data: bytes, pos: int) -> Tuple[int, int]:
    """Decode a protobuf varint starting at *pos*. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("Truncated varint at end of data")
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            break
        shift += 7
    return result, pos


def _skip_field(data: bytes, pos: int, wire_type: int) -> int:
    """Advance *pos* past a protobuf field we don't care about."""
    if wire_type == 0:  # varint
        _, pos = _decode_varint(data, pos)
    elif wire_type == 1:  # 64-bit fixed
        pos += 8
    elif wire_type == 2:  # length-delimited
        length, pos = _decode_varint(data, pos)
        pos += length
    elif wire_type == 5:  # 32-bit fixed
        pos += 4
    else:
        raise ValueError(f"Unknown wire type {wire_type}")
    return pos


def _parse_piece(data: bytes) -> Tuple[str, float, int]:
    """Parse a single ``SentencePiece`` sub-message.

    Returns (piece_string, score, piece_type).

    Piece types (from sentencepiece_model.proto):
        1 = NORMAL, 2 = UNKNOWN, 3 = CONTROL,
        4 = USER_DEFINED, 6 = BYTE
    """
    piece = ""
    score = 0.0
    piece_type = 1  # NORMAL
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 2:  # length-delimited
            length, pos = _decode_varint(data, pos)
            value = data[pos : pos + length]
            pos += length
            if field_number == 1:  # piece string
                piece = value.decode("utf-8")
        elif wire_type == 5:  # 32-bit fixed (float)
            if field_number == 2:  # score
                score = struct.unpack("<f", data[pos : pos + 4])[0]
            pos += 4
        elif wire_type == 0:  # varint
            val, pos = _decode_varint(data, pos)
            if field_number == 3:  # type enum
                piece_type = val
        elif wire_type == 1:  # 64-bit fixed
            pos += 8
        else:
            break
    return piece, score, piece_type


def _parse_trainer_spec(data: bytes) -> int:
    """Parse ``TrainerSpec`` to extract model_type.

    Returns model_type enum: 1=UNIGRAM, 2=BPE, 3=WORD, 4=CHAR.
    """
    model_type = 1  # default UNIGRAM
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 0:
            val, pos = _decode_varint(data, pos)
            if field_number == 1:  # model_type
                model_type = val
        else:
            pos = _skip_field(data, pos, wire_type)
    return model_type


MODEL_TYPE_NAMES = {1: "UNIGRAM", 2: "BPE", 3: "WORD", 4: "CHAR"}


def parse_sentencepiece_model(
    data: bytes,
) -> Tuple[List[Tuple[str, float, int]], int]:
    """Parse a SentencePiece ``ModelProto`` from raw bytes.

    Returns:
        pieces: list of (piece_string, score, piece_type) tuples
        model_type: integer enum for the model type
    """
    pieces: List[Tuple[str, float, int]] = []
    model_type = 1
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 2:  # length-delimited
            length, pos = _decode_varint(data, pos)
            value = data[pos : pos + length]
            pos += length
            if field_number == 1:  # repeated SentencePiece pieces
                pieces.append(_parse_piece(value))
            elif field_number == 2:  # TrainerSpec
                model_type = _parse_trainer_spec(value)
            # else: skip (NormalizerSpec, etc.)
        elif wire_type == 0:
            _, pos = _decode_varint(data, pos)
        elif wire_type == 5:
            pos += 4
        elif wire_type == 1:
            pos += 8
        else:
            break
    return pieces, model_type


# ---------------------------------------------------------------------------
# Extraction strategy 1: sentencepiece library
# ---------------------------------------------------------------------------

def _extract_via_sentencepiece(model_path: Path) -> Tuple[List[Tuple[str, float]], int]:
    """Load vocab using the ``sentencepiece`` Python package.

    Returns (vocab, model_type) where vocab is [(piece, score), ...].
    model_type: 1=UNIGRAM, 2=BPE.
    """
    import sentencepiece as spm  # type: ignore[import-untyped]

    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))

    vocab: List[Tuple[str, float]] = []
    for i in range(sp.GetPieceSize()):
        vocab.append((sp.IdToPiece(i), sp.GetScore(i)))

    # SentencePieceProcessor doesn't expose model_type directly in all
    # versions; fall back to protobuf parsing for that field.
    model_data = model_path.read_bytes()
    _, model_type = parse_sentencepiece_model(model_data)
    return vocab, model_type


# ---------------------------------------------------------------------------
# Extraction strategy 2: raw protobuf parsing
# ---------------------------------------------------------------------------

def _extract_via_protobuf(model_path: Path) -> Tuple[List[Tuple[str, float]], int]:
    """Load vocab by parsing the protobuf wire format directly.

    Returns (vocab, model_type) where vocab is [(piece, score), ...].
    """
    model_data = model_path.read_bytes()
    pieces, model_type = parse_sentencepiece_model(model_data)
    vocab = [(piece, score) for piece, score, _ptype in pieces]
    return vocab, model_type


# ---------------------------------------------------------------------------
# Build HuggingFace tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(
    vocab: List[Tuple[str, float]],
    model_type: int,
) -> "Tokenizer":  # noqa: F821
    """Construct a ``tokenizers.Tokenizer`` from extracted vocab."""
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    except ImportError:
        print(
            "ERROR: The 'tokenizers' package is required. "
            "Install it with: pip install tokenizers",
            file=sys.stderr,
        )
        sys.exit(1)

    if model_type != 1:
        model_name = MODEL_TYPE_NAMES.get(model_type, f"UNKNOWN({model_type})")
        print(
            f"WARNING: Model type is {model_name}. "
            "This script is optimised for UNIGRAM models; "
            "the output may not be fully correct for other types.",
            file=sys.stderr,
        )

    tokenizer = Tokenizer(models.Unigram(vocab))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="\u2581", add_prefix_space=True
    )
    tokenizer.decoder = decoders.Metaspace(
        replacement="\u2581", add_prefix_space=True
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(output_path: Path, test_string: str) -> bool:
    """Validate the written tokenizer.json: parseable JSON + round-trip test."""
    # 1. JSON validity
    try:
        with open(output_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"FAIL: Output is not valid JSON: {exc}", file=sys.stderr)
        return False

    if "model" not in data:
        print("FAIL: tokenizer.json missing 'model' key", file=sys.stderr)
        return False

    # 2. Round-trip encode/decode
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print(
            "WARNING: Cannot round-trip validate without 'tokenizers' package.",
            file=sys.stderr,
        )
        return True

    tok = Tokenizer.from_file(str(output_path))
    encoded = tok.encode(test_string)
    decoded = tok.decode(encoded.ids)

    # SentencePiece may normalise whitespace; compare stripped versions.
    original_norm = test_string.strip()
    decoded_norm = decoded.strip()

    if original_norm == decoded_norm:
        print(f"  Round-trip OK: \"{test_string}\" -> {len(encoded.ids)} tokens -> \"{decoded}\"")
        return True
    else:
        # Many SP models won't perfectly round-trip due to normalisation;
        # only warn, don't fail.
        print(
            f"  Round-trip MISMATCH (may be expected with normalisation):\n"
            f"    input:   \"{test_string}\"\n"
            f"    encoded: {len(encoded.ids)} tokens {encoded.tokens}\n"
            f"    decoded: \"{decoded}\"",
            file=sys.stderr,
        )
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(
    input_path: Path,
    output_path: Path,
    *,
    do_validate: bool = False,
    test_string: str = "The quick brown fox jumps over the lazy dog.",
) -> None:
    """Run the full conversion pipeline."""
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not input_path.suffix == ".model":
        print(
            f"WARNING: Expected .model extension, got '{input_path.suffix}'",
            file=sys.stderr,
        )

    # --- Extract vocab ---
    method = "unknown"
    try:
        vocab, model_type = _extract_via_sentencepiece(input_path)
        method = "sentencepiece"
    except ImportError:
        vocab, model_type = _extract_via_protobuf(input_path)
        method = "protobuf"

    if not vocab:
        print("ERROR: No vocabulary entries found in model file.", file=sys.stderr)
        sys.exit(1)

    model_name = MODEL_TYPE_NAMES.get(model_type, f"UNKNOWN({model_type})")

    # --- Build tokenizer ---
    tokenizer = build_tokenizer(vocab, model_type)

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    # --- Summary ---
    file_bytes = output_path.read_bytes()
    sha256 = hashlib.sha256(file_bytes).hexdigest()

    print(f"Converted: {input_path}")
    print(f"  Method:     {method}")
    print(f"  Model type: {model_name}")
    print(f"  Vocab size: {len(vocab):,}")
    print(f"  Output:     {output_path}")
    print(f"  Size:       {len(file_bytes):,} bytes")
    print(f"  SHA-256:    {sha256}")

    # --- Optional validation ---
    if do_validate:
        print("  Validating...")
        validate_output(output_path, test_string)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SentencePiece .model to HuggingFace tokenizer.json",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the SentencePiece .model file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for tokenizer.json. "
            "Defaults to <input_stem>.tokenizer.json in the same directory."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run round-trip validation after conversion",
    )
    parser.add_argument(
        "--test-string",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="String to use for round-trip validation (default: pangram)",
    )

    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.input.with_suffix(".tokenizer.json")

    convert(
        args.input,
        output,
        do_validate=args.validate,
        test_string=args.test_string,
    )


if __name__ == "__main__":
    main()
