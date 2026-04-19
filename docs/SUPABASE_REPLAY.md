# Local Supabase Replay

Use this runbook to rebuild the local Supabase catalog from the current sibling
atom-provider repos.

## Preconditions

- Local Supabase CLI is installed.
- The schema owner repo exists at `/Users/conrad/personal/sciona-infra`.
- The shared atoms repo exists at `/Users/conrad/personal/sciona-atoms`.
- Use Python from `/Users/conrad/personal/sciona-matcher/.venv/bin/python`.

## 1. Reset The Local Database

```bash
cd /Users/conrad/personal/sciona-infra
supabase db reset --local --yes
```

## 2. Export Local Connection Settings

Get the local service role key from Supabase:

```bash
cd /Users/conrad/personal/sciona-infra
supabase status -o env
```

Then export the values used by the seed and backfill scripts:

```bash
export SCIONA_SUPABASE_URL=http://127.0.0.1:54321
export SCIONA_SUPABASE_SERVICE_KEY=<local service role key>
export SUPABASE_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres
```

Use the local service role key, not a hosted-project key.

## 3. Seed Provider Atoms

```bash
cd /Users/conrad/personal/sciona-atoms
PYTHONPATH=src /Users/conrad/personal/sciona-matcher/.venv/bin/python \
  scripts/supabase_seed.py --apply --ensure-owner
```

Expected output includes the discovered provider and atom counts, for example:

```text
repos=8 atoms=<count> parsed_atoms=<count> dry_run=False
```

## 4. Populate File-Backed Metadata

```bash
cd /Users/conrad/personal/sciona-atoms
PYTHONPATH=src /Users/conrad/personal/sciona-matcher/.venv/bin/python \
  scripts/supabase_backfill.py all-file-backed
```

This populates the publication pillars: IO specs, parameters, descriptions,
audit rollups, references, evidence, uncertainty, and verification metadata.

## 5. Verify Publishability

Query the overall count:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python - <<'PY'
import psycopg

conn = psycopg.connect("postgresql://postgres:postgres@127.0.0.1:54322/postgres")
print(conn.execute(
    "select count(*) filter (where is_publishable), count(*) from public.atoms"
).fetchone())
PY
```

Query a specific family:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python - <<'PY'
import psycopg

prefix = "sciona.atoms.causal_inference.feature_primitives%"
conn = psycopg.connect("postgresql://postgres:postgres@127.0.0.1:54322/postgres")
rows = conn.execute(
    """
    select
      a.fqdn,
      a.is_publishable,
      count(distinct r.ref_key) as refs,
      count(distinct p.parameter_id) as params,
      count(distinct d.description_id) as descriptions,
      count(distinct io.io_spec_id) as ios,
      count(distinct ar.atom_id) as rollups
    from public.atoms a
    left join public.atom_references r on r.atom_id = a.atom_id
    left join public.atom_parameters p on p.atom_id = a.atom_id
    left join public.atom_descriptions d on d.atom_id = a.atom_id
    left join public.atom_io_specs io on io.atom_id = a.atom_id
    left join public.atom_audit_rollups ar on ar.atom_id = a.atom_id
    where a.fqdn like %s
    group by a.fqdn, a.is_publishable
    order by a.fqdn
    """,
    (prefix,),
).fetchall()
for row in rows:
    print(row)
PY
```

For a publishable family, every row should have `is_publishable = true` and
non-zero counts for references, parameters, descriptions, IO specs, and rollups.

## 6. Refresh Publishability Audit Docs

After a successful replay, refresh the publishability backlog docs owned by
`sciona-atoms`:

```bash
cd /Users/conrad/personal/sciona-atoms
/Users/conrad/personal/sciona-matcher/.venv/bin/python \
  scripts/refresh_publishability_review_docs.py
```

## Troubleshooting

- `PGRST301` / JWT decode errors: re-export the local service role key from
  `supabase status -o env`.
- Constraint failures in `atom_audit_rollups`: a review bundle or manifest field
  is outside the DB taxonomy. Fix the provider metadata, merge review bundles,
  reset, and replay again.
- Seed count changed unexpectedly: inspect sibling repo status and provider
  discovery before trusting the replay.
