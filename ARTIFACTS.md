# Serialized Artifact Support

This guide covers how to write atoms that depend on serialized resources
such as model weights, dictionaries, embedding matrices, tokenizer
vocabularies, and taxonomies.

Sciona treats executable code (logic atoms) and serialized resources
(state artifacts) as separate registry primitives. A logic atom
contains no weights or data files. A state artifact contains no
execution logic. CDG bindings compose them together with immutable
hash pins.

## When to use state artifacts

Use state artifacts when your atom needs a serialized resource that:

- Is too large to inline in source code (dictionaries, model weights)
- Has independent provenance from the code (a CMU dictionary, a
  pre-trained model checkpoint)
- Could be swapped for another version without changing the code (an
  English tagger vs a Spanish tagger using the same architecture)

Do **not** use state artifacts for small configuration that belongs in
function parameters (thresholds, hyperparameters, feature lists).

## Allowed formats

Only pure-data formats are allowed. Formats that can execute arbitrary
code are blocked at publish time and rejected by the format scanner.

### Allowed

| Format | Extension | Loader | Notes |
|--------|-----------|--------|-------|
| Safetensors | `.safetensors` | `safetensors.safe_open` | Memory-mappable. No pickle fallback. |
| ONNX | `.onnx` | `onnxruntime.InferenceSession` | Custom ops disabled by default. |
| JSON | `.json` | `json.load` | Schema validation against declared metadata. Max 256 MB. |
| JSONL | `.jsonl` | `json.loads` | Line-delimited JSON. |
| Parquet | `.parquet` | `pyarrow.parquet.read_table` | Memory-mappable. Explicit schema check. |
| NumPy array | `.npy` | `numpy.load` | `allow_pickle=False` enforced. Dtype allowlist. |
| NumPy archive | `.npz` | `numpy.load` | `allow_pickle=False` enforced. All members must be `.npy`. |
| Plain text | `.txt` | `pathlib.Path.read_text` | UTF-8 only. |
| Vocabulary | `.vocab` | `pathlib.Path.read_text` | UTF-8 only. Alias for `.txt`. |

### Blocked

These formats are **rejected** regardless of file extension. The
scanner checks magic bytes, not just the extension, so renaming a
pickle file to `.json` will still fail.

| Format | Why blocked |
|--------|-------------|
| `.pkl` / `.pickle` | Python pickle executes arbitrary code on load. |
| `.joblib` | Uses pickle internally. |
| `torch.load()` artifacts | PyTorch default serialization uses pickle. |
| ONNX with custom ops | Unless the custom op library is a separately audited logic artifact. |
| Any format with pickle magic bytes (`\x80\x02` through `\x80\x05`) | Detected by magic byte scan. |

### Converting blocked formats

If your upstream resource ships as pickle:

- **PyTorch models**: Convert to Safetensors (`safetensors.torch.save_file`)
  or ONNX (`torch.onnx.export`).
- **NumPy pickle archives**: Re-save with `numpy.savez` (the resulting
  `.npz` will pass `allow_pickle=False`).
- **NLTK pickle taggers**: Use the newer JSON-based variants (e.g.,
  `averaged_perceptron_tagger_eng` ships as `.json` files).
- **Joblib models**: Extract the underlying arrays and save as
  Safetensors, `.npy`, or Parquet.

## Writing an artifact-backed atom

### Step 1: Separate logic from data

The atom function accepts explicit `Path` parameters for each resource.
It must **never** download, import, or hardcode resource paths.

```python
from pathlib import Path

@register_atom(witness_my_atom)
@icontract.require(lambda text: isinstance(text, str) and len(text) > 0)
@icontract.ensure(lambda result: isinstance(result, list))
def my_nlp_atom(
    text: str,
    model_path: Path,       # state port: model weights
    vocab_path: Path,        # state port: vocabulary file
) -> list[str]:
    """Process text using pre-loaded model and vocabulary."""
    from ._vendor import load_model, load_vocab, predict
    model = load_model(model_path)
    vocab = load_vocab(vocab_path)
    return predict(text, model, vocab)
```

### Step 2: Declare state ports in cdg.json

Each CDG node that depends on artifacts must declare `state_ports`:

```json
{
  "node_id": "my_nlp_atom",
  "name": "my_nlp_atom",
  "state_ports": [
    {
      "port_name": "model",
      "type_desc": "ONNX inference model",
      "accepted_formats": ["onnx"],
      "required": true
    },
    {
      "port_name": "vocab",
      "type_desc": "Token vocabulary",
      "accepted_formats": ["txt", "vocab"],
      "required": true
    }
  ]
}
```

The `accepted_formats` array must only contain values from the allowed
formats list above.

### Step 3: Vendor the inference path

Do **not** `pip install` the upstream package if it has problematic
dependencies (GPL, pickle loaders, runtime downloads). Instead, vendor
the minimal inference code into a `_vendor.py` module within your atom
family directory.

Vendoring rules:

- Include the original license and attribution at the top of the file
- Import only `numpy`, `json`, `re`, and stdlib — no framework imports
- Load resources from explicit `Path` arguments, not package-relative paths
- Use `numpy.load(..., allow_pickle=False)` for `.npz` files
- Do not call `nltk.download()`, `torch.hub.load()`, or any network API

### Step 4: Enforce determinism

Artifact-backed atoms must produce identical output for identical input
across runs:

- Set all random seeds for heuristic fallbacks
- Lock tokenizer normalization options
- Force `temperature=0.0` or equivalent for any sampling
- Round floating-point confidence values to a declared precision
- Emit Sciona-standard typed output, not raw library tags

### Step 5: Write tests with real assets

Tests should exercise the atom against real (but small) assets. Use
`pytest.mark.skipif` when the assets live in temporary or local-only
locations:

```python
ASSETS_AVAILABLE = all(p.exists() for p in [MODEL_PATH, VOCAB_PATH])
skip_no_assets = pytest.mark.skipif(not ASSETS_AVAILABLE, reason="Assets not available")

@skip_no_assets
def test_deterministic():
    result1 = my_nlp_atom("hello", model_path=MODEL_PATH, vocab_path=VOCAB_PATH)
    result2 = my_nlp_atom("hello", model_path=MODEL_PATH, vocab_path=VOCAB_PATH)
    assert result1 == result2
```

## Validating artifacts locally

Run the artifact validation script to check that your resource files
pass the format scanner and security checks:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python \
  scripts/validate_artifacts.py path/to/model.onnx path/to/vocab.txt

# Or validate an entire directory:
/Users/conrad/personal/sciona-matcher/.venv/bin/python \
  scripts/validate_artifacts.py path/to/assets/
```

The script checks each file for:

- Allowed format (by extension and magic bytes)
- No blocked serialization patterns (pickle, joblib, torch)
- Format-specific validity (JSON parses, NPZ members are `.npy`,
  Parquet has correct header/trailer, Safetensors header is valid JSON)
- SHA-256 hash (printed for use in state artifact manifests)

## How artifacts flow through the system

```
Contributor                 Registry (Supabase)           Runtime
-----------                 -------------------           -------
                                                          
atom code  ──publish──►     artifacts table               
                            artifact_versions             
                            artifact_state_ports          
                                                          
asset files ──upload──►     artifact_assets               
                            (sha256 verified)             
                                                          
                            artifact_dependencies    ──►  hydrate_asset()
                            (hash-pinned)                 ~/.sciona/assets/sha256/<hash>/
                                                          
                            state_artifact_metadata       CryptographicIntegrityError
                            (provenance, limits)          if hash mismatch
```

1. The logic atom is published to the registry with state port declarations.
2. Each resource file is uploaded as a state artifact with SHA-256 hash.
3. A CDG binding pins the logic atom to exact state artifact content hashes.
4. At execution time, the runtime hydrates assets to a local cache,
   verifying hashes before any loader touches the bytes.
5. If any hash does not match, execution fails with
   `CryptographicIntegrityError` — no silent behavior changes.

## Reference example

See `src/sciona/atoms/ml/g2p/` in `sciona-atoms-ml` for a complete
working example:

- `atoms.py`: `g2p_convert` with four explicit `Path` parameters
- `_vendor.py`: Vendored inference from g2p-en (Apache 2.0), no NLTK
  import, no network downloads, `allow_pickle=False` on `.npz`
- `cdg.json`: State ports for cmudict, POS tagger, OOV model, homographs
- `witnesses.py`: Ghost witness returning stub phoneme list
- `tests/test_g2p.py`: Integration tests with real assets behind skip markers

## Relationship to other contracts

- **AGENT_INGESTION.md**: Covers the full atom ingestion workflow.
  Artifact-backed atoms follow the same phases plus the state port and
  vendoring steps described here.
- **PUBLISHING.md**: Covers the five publishability pillars. State
  artifacts add provenance and format security audit evidence.
- **CONTRIBUTION.md**: The quality bar for atom PRs. Artifact-backed
  atoms must meet the same bar plus the rules in this document.
