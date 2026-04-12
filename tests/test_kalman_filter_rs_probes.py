from __future__ import annotations

from importlib import import_module

from sciona.probes.state_estimation.kalman_filter_rs import probe_records


def test_probe_records_resolve_to_live_symbols() -> None:
    for record in probe_records():
        module = import_module(str(record["module_import_path"]))
        assert hasattr(module, str(record["wrapper_symbol"]))
        fqdn_parts = str(record["atom_fqdn"]).split(".")
        imported = import_module(".".join(fqdn_parts[:-1]))
        assert getattr(imported, fqdn_parts[-1]) is getattr(
            module,
            str(record["wrapper_symbol"]),
        )


def test_probe_records_publish_snake_case_wrapper_symbols() -> None:
    wrapper_symbols = {str(record["wrapper_symbol"]) for record in probe_records()}
    assert wrapper_symbols == {
        "initialize_kalman_state_model",
        "predict_latent_state_and_covariance",
        "predict_latent_state_steady_state",
        "evaluate_measurement_oracle",
        "update_posterior_state_and_covariance",
        "update_posterior_state_steady_state",
    }
