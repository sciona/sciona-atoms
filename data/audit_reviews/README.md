# Audit Review Bundles

Provider repos can place one or more JSON bundle files under `data/audit_reviews/`.

Bundle schema:

```json
{
  "schema_version": "1.0",
  "provider_repo": "sciona-atoms",
  "bundle_name": "optional-human-readable-name",
  "atoms": [
    {
      "atom_name": "sciona.atoms.some.module.atom",
      "audit": {
        "review_status": "approved",
        "review_priority": "review_now",
        "structural_status": "pass",
        "semantic_status": "pass",
        "runtime_status": "pass",
        "developer_semantics_status": "pass"
      }
    }
  ]
}
```

Only `schema_version` and `atoms` are required. Each atom entry must include
`atom_name` and at least one mergeable audit/review field. Fields present in a
bundle override the corresponding manifest entry fields; omitted fields remain
unchanged. The merger sorts bundle files and atom names deterministically before
updating `data/audit_manifest.json`.
