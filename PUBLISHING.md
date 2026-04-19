# Atom Publishing Guide

This guide covers the steps required to make newly ingested atoms publishable
under the strict catalog policy. An atom is not publishable until it satisfies
all five pillars and passes the audit manifest merge.

This guide applies after the atom code, witnesses, CDG, and references have
been written and verified per [AGENT_INGESTION.md](AGENT_INGESTION.md) or
[INGESTION.md](INGESTION.md).

## Environment

```
Workspace root:   /Users/conrad/personal
Primary repo:     /Users/conrad/personal/sciona-atoms
Python:           /Users/conrad/personal/sciona-matcher/.venv/bin/python
```

Do not use system Python or conda.

## Five Publishability Pillars

Every publishable atom must have all five pillars populated in the audit
manifest:

1. **`atom_io_specs`** — input/output type specifications from `cdg.json`
2. **`atom_parameters`** — parameter metadata from atom signatures
3. **`atom_descriptions`** — dejargonized English descriptions
4. **`atom_audit_rollups`** — review status, risk, acceptability metadata
5. **`atom_references`** — provenance bindings from `references.json`

No fallback publication for unaudited atoms. If any atom cannot be justified
against source/reference semantics, document it in `REMEDIATION.md` instead.

## Step 1: Complete CDG Inputs and Outputs

Each atomic node in `cdg.json` must have concrete `inputs` and `outputs`
matching the real function signature. Without these, the IO backfill
pipeline cannot generate `atom_io_specs` rows.

For each atomic node, add:

```json
"inputs": [
  {"name": "x", "type_desc": "NDArray[np.float64]", "required": true},
  {"name": "m", "type_desc": "int", "required": false}
],
"outputs": [
  {"name": "result", "type_desc": "float"}
]
```

Rules:
- The `name` field on each atomic node must be the snake_case function name
  (e.g., `"igci_asymmetry_score"`), NOT a human-readable title. The IO
  backfill derives the atom FQDN from `node["name"]`, so human-readable
  names produce invalid FQDNs and skip IO row generation.
- Parameter names must match the real function parameters exactly.
- Use concrete `type_desc` values: `NDArray[np.float64]`, `str`, `int`,
  `bool`, `float`, `list[float] | None`.
- Required parameters: `"required": true`. Parameters with defaults:
  `"required": false`.
- Every node gets exactly one output named `result` with the correct type.

## Step 2: Create a Review Bundle

Create a review bundle JSON file at:

```
data/review_bundles/<family_batch>.review_bundle.json
```

Follow the schema used by existing bundles (e.g.,
`data/review_bundles/state_estimation.review_bundle.json`).

### Bundle-level fields

| Field | Value |
|-------|-------|
| `schema_version` | `"1.0"` |
| `bundle_id` | `"sciona.atoms.review_bundle.<domain>.<family>.v1"` |
| `provider_repo` | `"sciona-atoms"` (or the appropriate sibling repo) |
| `family_batch` | `"<domain>_<family>"` (underscore-separated) |
| `review_status` | `"reviewed"` |
| `review_semantic_verdict` | `"pass"` or `"pass_with_limits"` |
| `review_developer_semantic_verdict` | `"pass_with_limits"` (unless fully vetted) |
| `trust_readiness` | `"reviewed_with_limits"` or `"catalog_ready"` |
| `authoritative_sources` | List of `{kind, path}` objects (see below) |
| `limitations` | List of concrete limitation strings |
| `blocking_findings` | `[]` (or list any blockers) |
| `required_actions` | `[]` |
| `review_record_path` | Path to this bundle file |

### Authoritative sources

List all files that constitute evidence for the review:

```json
{"kind": "local_wrapper", "path": "src/sciona/atoms/<domain>/<family>/atoms.py"},
{"kind": "local_metadata", "path": "src/sciona/atoms/<domain>/<family>/references.json"},
{"kind": "local_metadata", "path": "src/sciona/atoms/<domain>/<family>/cdg.json"},
{"kind": "local_test", "path": "tests/test_<family>_review_bundle.py"},
{"kind": "local_test", "path": "tests/test_<family>_references_metadata.py"},
{"kind": "local_test", "path": "tests/test_<family>_behavior.py"}
```

### Per-atom rows

Each atom gets a row in the `rows` array. The `atom_key` and `atom_name`
must be the package-level FQDN (e.g.,
`sciona.atoms.causal_inference.feature_primitives.igci_asymmetry_score`).

Required per-row fields:

| Field | Typical value |
|-------|---------------|
| `atom_key` | FQDN |
| `atom_name` | Same as atom_key |
| `review_status` | `"reviewed"` |
| `review_semantic_verdict` | `"pass"` or `"pass_with_limits"` |
| `review_developer_semantic_verdict` | `"pass_with_limits"` |
| `trust_readiness` | `"catalog_ready"` |
| `source_paths` | List of relevant file paths |
| `review_record_path` | Path to bundle file |
| `overall_verdict` | `"pass"` |
| `structural_status` | `"pass"` |
| `semantic_status` | `"pass"` |
| `runtime_status` | `"pass_with_limits"` |
| `developer_semantics_status` | `"pass_with_limits"` |
| `risk_tier` | `"low"`, `"medium"`, or `"high"` |
| `risk_score` | Integer 0–100 (DB column is INTEGER) |
| `risk_dimensions` | `{"provenance": "low", "semantic_drift": "low", ...}` |
| `risk_reasons` | `[]` |
| `acceptability_score` | Integer 0–100 (DB column is INTEGER) |
| `acceptability_band` | One of: `"review_ready"`, `"acceptable_with_limits_candidate"`, `"limited_acceptability"`, `"broken_candidate"`, `"misleading_candidate"` |
| `parity_coverage_level` | One of: `"unknown"`, `"none"`, `"not_applicable"`, `"positive_path"`, `"positive_and_negative"`, `"parity_or_usage_equivalent"` |
| `parity_test_status` | `"pass"` |
| `parity_fixture_count` | Integer |
| `parity_case_count` | Integer |
| `review_limitations` | List of strings |
| `review_required_actions` | `[]` |
| `trust_blockers` | `[]` |
| `has_references` | `true` |
| `references_status` | `"pass"` |
| `limitations` | `[]` |
| `blocking_findings` | `[]` |
| `required_actions` | `[]` |

Use `"pass_with_limits"` and add concrete limitations when there are
important semantic caveats. Do not force `"pass"` to satisfy the ratchet.

**Common mistakes:**
- `risk_score` and `acceptability_score` must be integers (0–100), not floats.
  The DB schema defines both columns as INTEGER.
- `acceptability_band` must use a value from the DB taxonomy listed above.
  Invalid values get normalized to `"unknown"` during backfill, weakening
  the audit rollup.

## Step 3: Write Focused Tests

Create three test files for the family:

### Review bundle test

`tests/test_<family>_review_bundle.py`

Validates:
- Bundle file exists and parses as valid JSON
- Contains the expected atom names (all of them)
- `provider_repo` matches
- `review_status == "reviewed"`
- Bundle and rows have pass/pass_with_limits semantic verdicts
- `trust_readiness` is catalog-ready or reviewed-with-limits
- Each row has source paths that exist on disk
- Each row has `has_references == true` and `references_status == "pass"`

### References metadata test

`tests/test_<family>_references_metadata.py`

Validates:
- `references.json` exists and has entries for all atom FQDNs
- Each atom has non-empty `references` list
- Each `ref_id` exists in `data/references/registry.json`
- Each reference has `match_metadata` with `match_type`, `confidence`, and
  non-empty `notes`

To extract leaf function names from the fully-qualified reference keys
(which contain `@<filepath>:<line>`), use:

```python
leaf_names = {k.split("@")[0].rsplit(".", 1)[-1] for k in refs["atoms"]}
```

### Behavioral test

`tests/test_<family>_behavior.py`

Lightweight behavioral checks (not exhaustive parity):
- All atoms import successfully
- Representative numerical inputs return expected output types
- Edge cases (constant inputs, low unique counts) don't crash
- Output contracts hold (e.g., probabilities in [0, 1])

Follow the test style used in existing test files in the repo.

## Step 4: Merge Review Bundle into Audit Manifest

After creating the bundle and tests, run:

```bash
cd /Users/conrad/personal/sciona-atoms
PYTHONPATH=src /Users/conrad/personal/sciona-matcher/.venv/bin/python \
  scripts/apply_audit_review_bundles.py
```

Then confirm:
- `data/audit_manifest.json` contains all atom FQDNs from the new family
- Entries have argument details, return annotation, source paths, review
  status, reference status, and audit rollup metadata
- No unresolved/skipped atoms are reported

If the merge cannot resolve package-level FQDNs, do not rename to
`.atoms.<symbol>` as a workaround. The expected public names are
package-level. Fix the merge/import issue or report the blocker.

## Step 5: Run Tests

```bash
cd /Users/conrad/personal/sciona-atoms
/Users/conrad/personal/sciona-matcher/.venv/bin/python -m pytest -q \
  tests/test_<family>_review_bundle.py \
  tests/test_<family>_references_metadata.py \
  tests/test_<family>_behavior.py \
  tests/test_review_bundles.py \
  tests/test_audit_review_bundles.py
```

All tests must pass. Fix failures and re-run until clean.

## Step 6 (Optional): Supabase Replay Verification

If local Supabase is available, verify the atoms are fully publishable:

```bash
# From sciona-infra:
supabase db reset --local --yes

# From sciona-atoms:
export SCIONA_SUPABASE_URL=http://127.0.0.1:54321
export SCIONA_SUPABASE_SERVICE_KEY=<local service role key>
export SUPABASE_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres

PYTHONPATH=src /Users/conrad/personal/sciona-matcher/.venv/bin/python \
  scripts/supabase_seed.py --apply --ensure-owner
PYTHONPATH=src /Users/conrad/personal/sciona-matcher/.venv/bin/python \
  scripts/supabase_backfill.py all-file-backed
```

Then verify with SQL:

```sql
select a.fqdn, a.is_publishable,
  count(distinct r.ref_key) as refs,
  count(distinct p.parameter_id) as params,
  count(distinct d.description_id) as descriptions,
  count(distinct io.io_spec_id) as ios
from public.atoms a
left join public.atom_references r on r.atom_id = a.atom_id
left join public.atom_parameters p on p.atom_id = a.atom_id
left join public.atom_descriptions d on d.atom_id = a.atom_id
left join public.atom_io_specs io on io.atom_id = a.atom_id
where a.fqdn like 'sciona.atoms.<domain>.<family>%'
group by a.fqdn, a.is_publishable
order by a.fqdn;
```

Expected: all atoms present, `is_publishable = true`, each with at least
one reference, parameter rows, description rows, and IO spec rows.

## Publishability Checklist

```text
CDG:
  [ ] Every atomic node has concrete inputs and outputs
  [ ] Parameter names match real function signatures
  [ ] Output named "result" with correct type

Review Bundle:
  [ ] Bundle exists in data/review_bundles/
  [ ] All atoms listed with package-level FQDNs
  [ ] Semantic verdicts are justified (not forced)
  [ ] Limitations are concrete, not hand-waved
  [ ] has_references and references_status set on every row

Tests:
  [ ] Review bundle test passes
  [ ] References metadata test passes
  [ ] Behavioral test passes
  [ ] test_review_bundles.py passes
  [ ] test_audit_review_bundles.py passes

Manifest:
  [ ] apply_audit_review_bundles.py ran without errors
  [ ] audit_manifest.json contains all atom FQDNs
  [ ] No unresolved or skipped atoms
```
