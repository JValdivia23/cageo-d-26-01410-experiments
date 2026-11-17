"""Checks for the bounded exploratory mean-only large-array comparison."""

from pathlib import Path
import csv
import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_large_3d_boxcar_mean as large


@pytest.mark.parametrize("width", (3, 31))
@pytest.mark.parametrize("method", large.METHODS)
def test_inplace_mean_matches_released_api(width, method):
    toolkit = large.load_toolkit()
    values, _, _ = large.generate_input((8, 9, 10))
    values[0, 0, 0], values[2, 3, 4] = 0.0, np.inf
    expected = toolkit.dct_mean(values, width, kernel_type="boxcar_discrete",
                               restore_input_nan=False, _stable_fallback=False,
                               _denominator_floor=large.FLOOR)
    original = values.copy()
    actual, density = large.normalized_mean(values, width, method, toolkit)
    assert np.shares_memory(values, actual)
    assert not np.shares_memory(density, actual)
    np.testing.assert_allclose(actual, expected, atol=large.ATOL, rtol=large.RTOL, equal_nan=True)
    assert np.any(actual != original)


@pytest.mark.parametrize("method", large.METHODS)
def test_missing_and_zero_controls(method):
    toolkit = large.load_toolkit()
    mean, density = large.normalized_mean(np.full((5, 6, 7), np.nan), 31, method, toolkit)
    assert np.isnan(mean).all() and np.count_nonzero(density) == 0
    mean, density = large.normalized_mean(np.zeros((5, 6, 7)), 31, method, toolkit)
    np.testing.assert_allclose(mean, 0, atol=1e-14)
    np.testing.assert_allclose(density, 1, atol=1e-14)


def test_identical_inputs_and_sample_positions():
    a, hash_a, missing_a = large.generate_input((8, 9, 10))
    b, hash_b, missing_b = large.generate_input((8, 9, 10))
    np.testing.assert_array_equal(a, b)
    assert hash_a == hash_b and missing_a == missing_b
    indices = large.sample_indices(a.shape)
    assert len(indices[0]) == 125 + 2048
    assert (0, 0, 0) in set(zip(*indices))
    assert (7, 8, 9) in set(zip(*indices))


def test_two_finite_filter_inputs_and_one_preparation(monkeypatch):
    prepare = large.prepare_filter
    calls, setups = [], []

    def tracked_prepare(*args):
        setups.append(args[0])
        prepared = prepare(*args)

        def tracked_filter(values):
            assert np.isfinite(values).all()
            calls.append(values.shape)
            return prepared(values)

        return tracked_filter

    monkeypatch.setattr(large, "prepare_filter", tracked_prepare)
    values, _, _ = large.generate_input((8, 9, 10))
    large.normalized_mean(values, 3, large.METHODS[0], large.load_toolkit())
    assert setups == [large.METHODS[0]] and calls == [(8, 9, 10)] * 2


def test_memory_preflight_refuses_to_start_worker(monkeypatch):
    monkeypatch.setattr(large, "memory_state", lambda: dict(available_bytes=0, pressure_level=2))

    def forbidden(*args, **kwargs):
        pytest.fail("Low-memory preflight must not start an allocation worker")

    monkeypatch.setattr(large.subprocess, "Popen", forbidden)
    result = large.measure(None, (512, 512, 512), large.METHODS[0], 3)
    assert result["status"] == "skipped_memory" and result["elapsed_s"] is None


def test_native_memory_page_size_is_not_assumed_from_rosetta(monkeypatch):
    output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 10.
Pages speculative: 2.
Pages inactive: 100.
File-backed pages: 80.
Swapins: 3.
Swapouts: 4.
Pages occupied by compressor: 5.
"""
    monkeypatch.setattr(large.sys, "platform", "darwin")
    monkeypatch.setattr(large.subprocess, "check_output",
                        lambda command, **kwargs: output if command == ["vm_stat"] else "2\n")
    result = large.memory_state()
    assert result["available_bytes"] == 92 * 16384
    assert result["native_page_size"] == 16384
    assert result["swapins"] == 3 and result["swapouts"] == 4


def test_sampled_comparison_rejects_missing_output():
    row = large.comparison("unit", "case", 1, "mean", [1.0, np.nan], [1.0, 2.0])
    assert row["status"] == "fail"
    assert row["n_compared"] == row["n_method_undefined"] == 1


def test_paging_measurement_is_retained_but_not_reported_as_a_success(tmp_path, monkeypatch):
    monkeypatch.setattr(large, "self_tests", lambda toolkit: None)
    calls = []

    def paging(*args):
        calls.append(args)
        return dict(status="paging_detected", elapsed_s=1.25, ram_before_mib=100,
                    peak_ram_mib=200, reason="system swap activity during timing")

    monkeypatch.setattr(large, "measure", paging)
    monkeypatch.setattr(sys, "argv", [str(Path(large.__file__)), "--run-id", "unit",
                                     "--output-root", str(tmp_path)])
    assert large.main() == 1
    assert len(calls) == 1
    directory = tmp_path / "3d_large_boxcar_mean" / "unit"
    raw = list(csv.DictReader((directory / "benchmark_runs.csv").open()))
    summary, = csv.DictReader((directory / "benchmark_summary.csv").open())
    assert raw[0]["status"] == "paging_detected" and float(raw[0]["elapsed_s"]) == 1.25
    assert summary["n_success"] == "0" and summary["n_failure"] == "1"
    assert summary["median_time_s"] == summary["max_peak_ram_mib"] == ""


RESULT = os.environ.get("LARGE_MEAN_RESULT_DIR")


def saved_rows(name):
    with (Path(RESULT) / name).open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert rows and all(None not in row and None not in row.values() for row in rows)
    return rows


@pytest.mark.skipif(not RESULT, reason="set LARGE_MEAN_RESULT_DIR to the selected large-array run")
def test_saved_large_mean_inputs_source_and_no_paging():
    cases = {}
    for row in saved_rows("settings.csv"):
        cases.setdefault(row["case_id"], {})[row["setting"]] = row["value"]
    hashes, workers = set(), []
    source_hash = hashlib.sha256(Path(large.__file__).read_bytes()).hexdigest()
    for values in cases.values():
        assert values["script_sha256"] == source_hash
        assert values["dct_toolkit_version"] == "1.0.0"
        assert values["dct_toolkit_commit"] == "9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c"
        assert int(values["input_bytes"]) == large.GIB
        assert json.loads(values["shape_zyx"]) == [512, 512, 512]
        for key, value in values.items():
            if not key.startswith("worker_"):
                continue
            worker = json.loads(value)
            workers.append(worker)
            hashes.add(worker["input_sha256"])
            assert worker["status"] == "success"
            assert worker["filter_calls"] == 2 and worker["input_overwritten"] is True
            assert worker["peak_source"] == "new lifetime high-water mark"
            for counter in ("swapins", "swapouts"):
                assert worker["native_vm_before"][counter] == worker["native_vm_after"][counter]
            assert abs(worker["actual_missing_fraction"] - .20) < .001
    assert len(workers) == 8 and len(hashes) == 1


@pytest.mark.skipif(not RESULT, reason="set LARGE_MEAN_RESULT_DIR to the selected large-array run")
def test_saved_large_mean_summary_recomputed_from_measurements():
    raw, summary = saved_rows("benchmark_runs.csv"), saved_rows("benchmark_summary.csv")
    assert len(raw) == 8 and len(summary) == 4
    assert {row["method"] for row in raw} == set(large.METHODS)
    for row in raw:
        assert row["operation"] == large.OPERATION and row["status"] == "success"
        assert float(row["elapsed_s"]) > 0
        assert float(row["peak_ram_mib"]) >= float(row["ram_before_mib"]) > 0
    for row in summary:
        group = [r for r in raw if r["case_id"] == row["case_id"] and r["method"] == row["method"]]
        times = [float(r["elapsed_s"]) for r in group]
        assert len(group) == int(row["n_success"]) == 2 and row["n_failure"] == "0"
        for field, q in (("median_time_s", .5), ("q25_time_s", .25), ("q75_time_s", .75)):
            assert float(row[field]) == pytest.approx(float(np.quantile(times, q)), rel=1e-12)
        assert float(row["max_peak_ram_mib"]) == max(float(r["peak_ram_mib"]) for r in group)


@pytest.mark.skipif(not RESULT, reason="set LARGE_MEAN_RESULT_DIR to the selected large-array run")
def test_saved_large_mean_agreement_is_explicitly_sampled():
    rows = saved_rows("comparison.csv")
    assert len(rows) == 8
    assert {row["statistic"] for row in rows} == {"mean", "raw_mask_response"}
    for row in rows:
        assert row["region"] == "sampled_edges_and_random" and row["status"] == "pass"
        assert int(row["n_considered"]) == int(row["n_compared"]) == 2173
        assert "not all-cell validation" in row["notes"]
        assert float(row["max_abs_difference"]) < 1e-12
