"""Explicit polar cell loops: donor sums, mask handling and saved timing audits."""

import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_2d_radar_gaussian_double_loop as loops


@pytest.fixture(scope="module")
def toolkit():
    return loops.gaussian.base.load_toolkit()


def donor_sum(values, width, az_res):
    """Independent 2D original-donor enumeration; no staged intermediates."""
    n_az, n_range = values.shape
    sigma_r = width / np.sqrt(12)
    radial_offsets = np.arange(-int(np.ceil(4*sigma_r)), int(np.ceil(4*sigma_r))+1)
    radial = np.exp(-.5*(radial_offsets/sigma_r)**2)
    radial /= radial.sum()
    result = np.zeros(values.shape)
    for az in range(n_az):
        for gate in range(n_range):
            for dr, wr in zip(radial_offsets, radial):
                donor = gate - int(dr)
                while donor < 0 or donor >= n_range:
                    donor = -donor-1 if donor < 0 else 2*n_range-1-donor
                sigma_a = width / ((donor+1)*np.deg2rad(az_res)*np.sqrt(12))
                offsets = np.arange(-int(np.ceil(4*sigma_a)), int(np.ceil(4*sigma_a))+1)
                angular = np.exp(-.5*(offsets/sigma_a)**2)
                angular /= angular.sum()
                for da, wa in zip(offsets, angular):
                    result[az, gate] += wr * wa * values[(az-int(da)) % n_az, donor]
    return result


@pytest.mark.parametrize("shape,width,az_res", [((5, 6), .7, 72.), ((6, 5), 3., 60.),
                                                ((3, 4), 5., 30.), ((1, 5), 2., 90.), ((5, 1), 3., 72.)])
def test_loops_match_original_donors_and_spectral_filter_including_repeated_boundaries(toolkit, shape, width, az_res):
    arrays = [np.ones(shape), np.random.default_rng(29101).normal(size=shape)]
    impulse = np.zeros(shape)
    impulse[0, -1] = 1.
    arrays.append(impulse)
    for values in arrays:
        untouched = values.tobytes()
        explicit = loops.prepare_filter(loops.LOOP, shape, width, az_res, toolkit, {})(values)
        spectral = loops.prepare_filter(loops.SPECTRAL, shape, width, az_res, toolkit, {})(values)
        np.testing.assert_allclose(explicit, donor_sum(values, width, az_res), atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(explicit, spectral, atol=3e-14, rtol=3e-14)
        assert values.tobytes() == untouched


@pytest.mark.parametrize("pattern", ("complete", "gap", "all_missing", "single_observation", "zero"))
def test_complete_moments_match_centered_reference_and_preserve_actual_cell_counts(toolkit, pattern):
    data = np.random.default_rng(29102).normal(size=(6, 7))
    if pattern == "gap":
        data[:3, 1:5] = np.nan
    elif pattern == "all_missing":
        data[:] = np.nan
    elif pattern == "single_observation":
        data[:] = np.nan
        data[2, 3] = 35.
    elif pattern == "zero":
        data[:] = 0.
    original = data.tobytes()
    reference = loops.gaussian.centered_reference(data, 3., 60.)
    audit = {}
    actual = loops.calculate(loops.LOOP, data, 3., 60., toolkit, audit)
    assert audit == loops.expected_audit(loops.LOOP, data.shape)
    assert audit["azimuth_cell_updates"] == audit["range_cell_updates"] == 3*data.size
    for key in loops.FIELDS:
        row = loops.compare("unit", pattern, loops.LOOP, key, "all", actual[key], reference[key], loops.gaussian.REFERENCE)
        assert row["status"] in ("pass", "no_comparable"), row
    assert data.tobytes() == original


def test_explicit_loop_never_calls_convolution_or_transforms(toolkit, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The explicit double loop must not call a convolution or transform")
    monkeypatch.setattr(loops.gaussian, "prepare_filter", forbidden)
    for name in ("convolve1d", "rfft", "irfft", "dct", "idct"):
        monkeypatch.setattr(loops.gaussian, name, forbidden)
    monkeypatch.setattr(np, "convolve", forbidden)
    actual = loops.calculate(loops.LOOP, np.arange(42.).reshape(6, 7), 3., 60., toolkit)
    assert np.isfinite(actual["mean"]).all()


def test_validation_has_separate_centered_and_direct_spectral_loop_comparisons(toolkit):
    data = np.random.default_rng(29103).normal(size=(6, 7))
    data[1:3, :2] = np.nan
    rows, matched, diagnostic, descriptor, compact = loops.validate_case(
        "unit", "tiny", data, np.arange(7)*100., 3., 60., toolkit)
    assert descriptor["verified"]
    assert len(rows) == 54 and len(matched) == 18 and len(compact) == 2
    assert all(row["status"] == "pass" for row in (*rows, *matched))
    assert {row["reference_method"] for row in rows} == {loops.gaussian.REFERENCE}
    assert {row["reference_method"] for row in matched} == {loops.LOOP}
    assert all(row["n_negative_beyond_reference_tolerance"] == 0 for row in diagnostic)
    for row in compact:
        for key, statistic in (("max_abs_mean_difference", "mean"), ("max_abs_variance_difference", "raw_variance")):
            assert row[key] == next(r["max_abs_difference"] for r in matched if r["region"] == row["region"] and r["statistic"] == statistic)


@pytest.mark.parametrize("method", loops.METHODS)
def test_worker_verifies_sources_nine_arrays_and_actual_cell_loop_workload(toolkit, tmp_path, monkeypatch, method):
    shape = (6, 7)
    monkeypatch.setattr(loops, "SHAPE", shape)
    for key in loops.gaussian.base.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    data = np.random.default_rng(29104).normal(size=shape)
    output = loops.calculate(method, data, 3., 60., toolkit)
    descriptor = dict(verified=True, source_sha256=loops.source_hashes(), input_sha256=loops.gaussian.base.array_sha256(data),
                      shape=list(shape), width=3., az_res_deg=60., output_sha256={method: loops.gaussian.base.fingerprints(output)})
    source, manifest = tmp_path/"input.npy", tmp_path/"manifest.json"
    np.save(source, data)
    manifest.write_text(json.dumps({"tiny": descriptor}))
    # This unit test shares pytest's process with earlier full-sweep figures,
    # unlike production workers, which start fresh with the recorded 1-GiB cap.
    args = SimpleNamespace(case="tiny", method=method, input=source, manifest=manifest, max_ram_mib=4096)
    payload = loops.worker(args)
    assert payload["status"] == "success", payload
    assert loops.check_worker(payload, descriptor, "tiny", method)["status"] == "success"
    assert 0 < payload["preparation_s"] <= payload["elapsed_s"]
    args.max_ram_mib = payload["ram_before_mib"] / 2
    limited = loops.worker(args)
    assert limited["status"] == "resource_limit"
    assert limited["reason"] == "Calculation exceeded worker RSS budget"
    if method == loops.LOOP:
        payload["audit"]["azimuth_cell_updates"] -= 1
    else:
        payload["n_outputs"] -= 1
    assert loops.check_worker(payload, descriptor, "tiny", method)["status"] == "invalid_worker"


@pytest.mark.parametrize("argv", (["--worker"], ["--method", loops.LOOP], ["--repeats", "0"],
                                   ["--timeout-s", "nan"], ["--max-ram-mib", "0"]))
def test_cli_rejects_incomplete_or_unsafe_options(argv):
    with pytest.raises(SystemExit):
        loops.parse_args(argv)


RESULT = os.environ.get("GAUSSIAN_DOUBLE_LOOP_RESULT_DIR")
SELECTED = Path(__file__).resolve().parents[1] / "results/round1/2d_radar_gaussian_double_loop/radar-gaussian-double-loop-20261111"
CURRENT = Path(__file__).resolve().parents[1] / "results/round1/v2.0.0/2d_radar_gaussian_double_loop/release-v2-20261111"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def scientific(value):
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"












@pytest.mark.skipif(not RESULT, reason="set GAUSSIAN_DOUBLE_LOOP_RESULT_DIR to the selected run")
def test_saved_full_sweep_correctness_and_worker_audits():
    root = Path(RESULT)
    def read(name):
        with (root/name).open() as stream:
            return list(csv.DictReader(stream))
    for name, length in (("comparison.csv", 108), ("spectral_loop_comparison.csv", 36), ("kernel_checks.csv", 2362)):
        rows = read(name)
        assert len(rows) == length and all(row["status"] == "pass" for row in rows)
    runs, summaries = read("benchmark_runs.csv"), read("benchmark_summary.csv")
    assert len(runs) == 30 and all(row["status"] == "success" for row in runs)
    settings = {}
    for row in read("settings.csv"):
        settings.setdefault(row["case_id"], {})[row["setting"]] = row["value"]
    for case, values in settings.items():
        assert json.loads(values["source_sha256"]) == loops.source_hashes()
        assert values["spectral_matches_previous_output_hashes"] == "True"
        descriptor = dict(shape=json.loads(values["shape"]), source_sha256=json.loads(values["source_sha256"]),
                          input_sha256=values["input_sha256"], output_sha256=json.loads(values["output_sha256"]))
        for method in loops.METHODS:
            selected = [row for row in runs if row["case_id"] == case and row["method"] == method]
            summary = next(row for row in summaries if row["case_id"] == case and row["method"] == method)
            assert len(selected) == 5
            assert float(summary["median_time_s"]) == np.median([float(row["elapsed_s"]) for row in selected])
            assert float(summary["max_peak_ram_mib"]) == max(float(row["peak_ram_mib"]) for row in selected)
            for repeat in range(1, 6):
                payload = json.loads(values[f"worker_{method}_{repeat}"])
                assert loops.check_worker(payload, descriptor, case, method)["status"] == "success"
        if case == "natural_W15":
            assert descriptor["output_sha256"][loops.SPECTRAL] == loops.previous_spectral_hashes()[case]
    assert len(read("agreement_summary.csv")) == 4
    for row in read("timing_summary.csv"):
        assert float(row["double_loop_over_spectral"]) == float(row[loops.LOOP]) / float(row[loops.SPECTRAL])
    for row in read("benchmark_phases.csv"):
        raw = next(r for r in runs if all(r[key] == row[key] for key in ("case_id", "method", "repetition")))
        assert float(row["total_s"]) == float(raw["elapsed_s"])
        assert 0 < float(row["preparation_s"]) <= float(row["total_s"])
