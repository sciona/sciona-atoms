from __future__ import annotations

from importlib import import_module

from sciona.probes.algorithmic.divide_and_conquer_sorting import probe_records


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


def test_probe_records_publish_expected_sorting_symbols() -> None:
    wrapper_symbols = {str(record["wrapper_symbol"]) for record in probe_records()}
    assert wrapper_symbols == {
        "merge_sort",
        "quicksort",
        "heapsort",
        "counting_sort",
        "radix_sort",
    }
