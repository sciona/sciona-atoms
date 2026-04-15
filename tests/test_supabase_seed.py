from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "sciona" / "atoms" / "supabase_seed.py"


def load_seed_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("core_supabase_seed", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def test_derive_seed_inventory_scans_provider_roots_and_builds_fqdns(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"

    _write(
        repo / "src" / "sciona" / "atoms" / "demo" / "ops.py",
        """
        from sciona.ghost.registry import register_atom

        def witness_scale(x):
            return x

        @register_atom(witness_scale)
        def scale(x):
            \"\"\"Scale values.\"\"\"
            return x
        """,
    )
    _write(
        repo / "src" / "sciona" / "atoms" / "special" / "atoms.py",
        """
        from sciona.ghost.registry import register_atom

        def witness_explicit(x):
            return x

        @register_atom(witness_explicit, name="demo.explicit")
        def implementation(x):
            return x
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)

    assert inventory.summary()["provider_repos"] == 1
    assert inventory.repository_rows[0].repo_name == "sciona-atoms"
    assert inventory.repository_rows[0].namespace_root == "sciona.atoms"
    assert {row.fqdn for row in inventory.atom_rows} == {
        "sciona.atoms.demo.ops.scale",
        "sciona.atoms.demo.explicit",
    }


def test_atoms_py_stem_is_omitted_from_module_namespace(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "ageo-atoms"

    _write(
        repo / "ageoa" / "rust_robotics" / "atoms.py",
        """
        from ageoa.ghost.registry import register_atom

        def witness_solver(x):
            return x

        @register_atom(witness_solver)
        def n_joint_arm_solver(x):
            return x
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)

    assert [row.fqdn for row in inventory.atom_rows] == [
        "ageoa.rust_robotics.n_joint_arm_solver"
    ]
    assert inventory.atom_rows[0].namespace_path == "rust_robotics"
    assert inventory.atom_rows[0].source_module_path == "rust_robotics"


def test_build_atom_rows_uses_owner_and_repo_mapping(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"

    _write(
        repo / "src" / "sciona" / "atoms" / "demo" / "ops.py",
        """
        from sciona.ghost.registry import register_atom

        def witness_scale(x):
            return x

        @register_atom(witness_scale)
        def scale(x):
            return x
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)
    owner = module.build_owner_seed("seed-owner", email="seed-owner@localhost")
    rows = module.build_atom_rows(
        inventory,
        owner_id=owner.user_id,
        source_repo_ids={"sciona-atoms": "repo-uuid"},
    )

    assert rows == [
        {
            "fqdn": "sciona.atoms.demo.ops.scale",
            "namespace_root": "sciona.atoms",
            "namespace_path": "demo.ops",
            "owner_id": owner.user_id,
            "domain_tags": ["demo"],
            "description": "",
            "status": "approved",
            "visibility_tier": "general",
            "source_kind": "hand_written",
            "stateful_kind": "none",
            "is_stochastic": False,
            "is_ffi": False,
            "is_publishable": False,
            "source_repo_id": "repo-uuid",
            "source_package": "sciona.atoms",
            "source_module_path": "demo.ops",
            "source_symbol": "scale",
        }
    ]


def test_render_owner_seed_sql_is_deterministic() -> None:
    module = load_seed_module()
    owner = module.build_owner_seed("seed-owner", email="seed-owner@localhost")
    sql = module.render_owner_seed_sql(owner)

    assert owner.user_id in sql
    assert "INSERT INTO auth.users" in sql
    assert "INSERT INTO public.users" in sql
    assert "seed-owner@localhost" in sql


def test_seed_core_supabase_dry_run_reports_deferred_tables(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"

    _write(
        repo / "src" / "sciona" / "atoms" / "demo" / "ops.py",
        """
        from sciona.ghost.registry import register_atom

        def witness_scale(x):
            return x

        @register_atom(witness_scale)
        def scale(x):
            return x
        """,
    )

    summary = module.seed_core_supabase(
        client=object(),
        base_dir=workspace,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["repository_rows"] == 1
    assert summary["atom_rows"] == 1
    assert summary["version_rows"] == 1
    assert summary["deferred_tables"] == []


def test_derive_seed_inventory_includes_hyperparam_rows(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"

    _write(
        repo / "src" / "sciona" / "atoms" / "demo" / "ops.py",
        """
        from sciona.ghost.registry import register_atom

        def witness_scale(x):
            return x

        @register_atom(witness_scale)
        def scale(x):
            return x
        """,
    )
    _write(
        repo / "data" / "hyperparams" / "manifest.json",
        """
        {
          "reviewed_atoms": [
            {
              "atom": "scale",
              "path": "src/sciona/atoms/demo/ops.py",
              "status": "approved",
              "tunable_params": [
                {
                  "name": "alpha",
                  "default": 0.5,
                  "min": 0.0,
                  "max": 1.0,
                  "constraints": {"type": "closed_interval"},
                  "safe_to_optimize": true
                }
              ]
            }
          ]
        }
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)

    assert inventory.summary()["hyperparam_rows"] == 1
    assert inventory.hyperparam_rows[0].fqdn == "sciona.atoms.demo.ops.scale"
    assert inventory.hyperparam_rows[0].name == "alpha"
    assert inventory.hyperparam_rows[0].kind == "float"


def test_build_hyperparam_rows_resolves_atom_ids(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "ageo-atoms"

    _write(
        repo / "ageoa" / "demo.py",
        """
        from ageoa.ghost.registry import register_atom

        def witness_scale(x):
            return x

        @register_atom(witness_scale)
        def scale(x):
            return x
        """,
    )
    _write(
        repo / "data" / "hyperparams" / "manifest.json",
        """
        {
          "reviewed_atoms": [
            {
              "atom": "ageoa.demo.scale",
              "status": "approved",
              "tunable_params": [
                {
                  "name": "mode",
                  "kind": "categorical",
                  "default": "fast",
                  "choices": ["fast", "safe"],
                  "safe_to_optimize": true
                }
              ]
            }
          ]
        }
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)
    rows, summary = module.build_hyperparam_rows(
        inventory,
        atom_ids={"ageoa.demo.scale": "atom-1"},
    )

    assert summary == {
        "hyperparam_rows": 1,
        "hyperparam_atoms": 1,
        "hyperparam_skipped_no_atom": 0,
    }
    assert rows == [
        {
            "atom_id": "atom-1",
            "name": "mode",
            "kind": "categorical",
            "default_value": "fast",
            "min_value": None,
            "max_value": None,
            "step_value": None,
            "log_scale": False,
            "choices_json": ["fast", "safe"],
            "constraints_json": None,
            "semantic_role": "",
            "status": "approved",
        }
    ]


def test_build_version_rows_resolves_atom_ids(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"

    _write(
        repo / "src" / "sciona" / "atoms" / "demo" / "ops.py",
        """
        from sciona.ghost.registry import register_atom

        def witness_scale(x):
            return x

        @register_atom(witness_scale)
        def scale(x):
            return x
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)
    rows, summary = module.build_version_rows(
        inventory,
        atom_ids={"sciona.atoms.demo.ops.scale": "atom-1"},
    )

    assert summary == {
        "version_rows": 1,
        "version_atoms": 1,
        "version_skipped_no_atom": 0,
    }
    assert rows[0]["atom_id"] == "atom-1"
    assert rows[0]["is_latest"] is True
    assert rows[0]["derives_from"] is None
    assert rows[0]["semver"].startswith("0.0.0+local.")
    assert rows[0]["content_hash"]
    assert rows[0]["fingerprint"]
    assert rows[0]["s3_key"].startswith("local-seed/atoms/")


def test_derive_seed_inventory_includes_benchmark_rows(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"

    _write(
        repo / "src" / "sciona" / "atoms" / "demo" / "ops.py",
        """
        from sciona.ghost.registry import register_atom

        def witness_scale(x):
            return x

        @register_atom(witness_scale)
        def scale(x):
            return x
        """,
    )
    _write(
        repo / "data" / "benchmarks" / "benchmark_suites.json",
        """
        [
          {
            "suite_id": "demo.scale.v1",
            "title": "Demo Scale",
            "artifact_scope": "both",
            "contract_summary": "Scale values.",
            "domain_tags": ["demo"],
            "family_tags": ["scale"],
            "modality_tags": ["tabular"],
            "dataset_tag": "demo_dataset",
            "metrics": [
              {"metric_name": "mae", "direction": "lower_is_better", "unit": "ratio", "primary": true}
            ],
            "status": "draft"
          }
        ]
        """,
    )
    _write(
        repo / "data" / "benchmarks" / "benchmark_results.json",
        """
        [
          {
            "suite_id": "demo.scale.v1",
            "artifact_fqdn": "sciona.atoms.demo.ops.scale",
            "artifact_kind": "atom",
            "content_hash": "placeholder",
            "metric_name": "mae",
            "metric_value": 0.25,
            "measured_at": "2026-04-14T15:10:00Z",
            "status": "completed"
          }
        ]
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)

    assert inventory.summary()["benchmark_suite_rows"] == 1
    assert inventory.summary()["benchmark_result_rows"] == 1
    assert inventory.benchmark_suite_rows[0].benchmark_id == "demo.scale.v1"
    assert inventory.benchmark_suite_rows[0].status == "proposed"
    assert inventory.benchmark_result_rows[0].dataset_tag == "demo_dataset"


def test_build_atom_benchmark_rows_resolves_versions_and_defers_cdgs(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"

    _write(
        repo / "src" / "sciona" / "atoms" / "demo" / "ops.py",
        """
        from sciona.ghost.registry import register_atom

        def witness_scale(x):
            return x

        @register_atom(witness_scale)
        def scale(x):
            return x
        """,
    )
    _write(
        repo / "data" / "benchmarks" / "benchmark_suites.json",
        """
        [
          {
            "suite_id": "demo.scale.v1",
            "title": "Demo Scale",
            "artifact_scope": "both",
            "contract_summary": "Scale values.",
            "domain_tags": ["demo"],
            "family_tags": ["scale"],
            "modality_tags": ["tabular"],
            "dataset_tag": "demo_dataset",
            "metrics": [
              {"metric_name": "mae", "direction": "lower_is_better", "unit": "ratio", "primary": true}
            ],
            "status": "active"
          }
        ]
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)
    atom_version = inventory.version_rows[0]
    _write(
        repo / "data" / "benchmarks" / "benchmark_results.json",
        f"""
        [
          {{
            "suite_id": "demo.scale.v1",
            "artifact_fqdn": "{atom_version.fqdn}",
            "artifact_kind": "atom",
            "content_hash": "{atom_version.content_hash}",
            "metric_name": "mae",
            "metric_value": 0.25,
            "measured_at": "2026-04-14T15:10:00Z",
            "status": "completed"
          }},
          {{
            "suite_id": "demo.scale.v1",
            "artifact_fqdn": "cdg.skeleton.demo",
            "artifact_kind": "cdg",
            "content_hash": "cdg-hash",
            "metric_name": "mae",
            "metric_value": 0.15,
            "measured_at": "2026-04-14T15:10:00Z",
            "status": "completed"
          }}
        ]
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)
    rows, summary = module.build_atom_benchmark_rows(
        inventory,
        version_ids={(atom_version.fqdn, atom_version.content_hash): atom_version.version_id},
    )

    assert summary == {
        "benchmark_suite_rows": 1,
        "benchmark_result_rows": 2,
        "atom_benchmark_rows": 1,
        "benchmark_atom_versions": 1,
        "benchmark_result_cdg_deferred": 1,
        "benchmark_result_skipped_no_version": 0,
    }
    assert rows == [
        {
            "version_id": atom_version.version_id,
            "benchmark_name": "demo.scale.v1",
            "metric_name": "mae",
            "metric_value": 0.25,
            "dataset_tag": "demo_dataset",
            "measured_at": "2026-04-14T15:10:00Z",
        }
    ]


def test_derive_seed_inventory_rejects_unknown_benchmark_metric(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"

    _write(
        repo / "data" / "benchmarks" / "benchmark_suites.json",
        """
        [
          {
            "suite_id": "demo.scale.v1",
            "title": "Demo Scale",
            "artifact_scope": "atom",
            "contract_summary": "Scale values.",
            "domain_tags": ["demo"],
            "dataset_tag": "demo_dataset",
            "metrics": [
              {"metric_name": "mae", "direction": "lower_is_better", "unit": "ratio", "primary": true}
            ],
            "status": "active"
          }
        ]
        """,
    )
    _write(
        repo / "data" / "benchmarks" / "benchmark_results.json",
        """
        [
          {
            "suite_id": "demo.scale.v1",
            "artifact_fqdn": "sciona.atoms.demo.ops.scale",
            "artifact_kind": "atom",
            "content_hash": "placeholder",
            "metric_name": "rmse",
            "metric_value": 0.25,
            "measured_at": "2026-04-14T15:10:00Z",
            "status": "completed"
          }
        ]
        """,
    )

    try:
        module.derive_seed_inventory(base_dir=workspace)
    except ValueError as exc:
        assert "undeclared metric" in str(exc)
    else:
        raise AssertionError("expected ValueError for undeclared benchmark metric")


def test_build_artifact_benchmark_rows_resolves_cdg_versions(tmp_path: Path) -> None:
    module = load_seed_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"

    _write(
        repo / "data" / "benchmarks" / "benchmark_suites.json",
        """
        [
          {
            "suite_id": "demo.cdg.v1",
            "title": "Demo CDG",
            "artifact_scope": "both",
            "contract_summary": "Demo macro artifact benchmark.",
            "domain_tags": ["demo"],
            "dataset_tag": "demo_dataset",
            "metrics": [
              {"metric_name": "quality", "direction": "higher_is_better", "unit": "ratio", "primary": true}
            ],
            "status": "active"
          }
        ]
        """,
    )
    _write(
        repo / "data" / "benchmarks" / "benchmark_results.json",
        """
        [
          {
            "suite_id": "demo.cdg.v1",
            "artifact_fqdn": "cdg.skeleton.demo",
            "artifact_kind": "cdg",
            "content_hash": "cdg-hash",
            "metric_name": "quality",
            "metric_value": 0.9,
            "measured_at": "2026-04-14T15:10:00Z",
            "status": "completed"
          }
        ]
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)
    rows, summary = module.build_artifact_benchmark_rows(
        inventory,
        version_ids={("cdg.skeleton.demo", "cdg-hash"): "artifact-version-1"},
    )

    assert summary == {
        "artifact_benchmark_rows": 1,
        "benchmark_artifact_versions": 1,
        "benchmark_cdg_skipped_no_version": 0,
    }
    assert rows == [
        {
            "version_id": "artifact-version-1",
            "benchmark_name": "demo.cdg.v1",
            "metric_name": "quality",
            "metric_value": 0.9,
            "dataset_tag": "demo_dataset",
            "measured_at": "2026-04-14T15:10:00Z",
        }
    ]
