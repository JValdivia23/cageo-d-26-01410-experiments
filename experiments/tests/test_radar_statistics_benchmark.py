"""Matched radar benchmark controls and independently checked saved measurements.

Only tiny arrays are calculated by unit tests. Set RADAR_BENCHMARK_RESULT_DIR
to check the selected real-sweep validation and measurements; tests never start
a real-radar timing campaign or modify earlier results/figures.
"""

import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import benchmark_2d_radar_statistics as bench


@pytest.fixture(scope="module")
def toolkit():
    return bench.load_toolkit()


def forbidden(*args, **kwargs):
    pytest.fail("This unit check must not launch a real-radar timing campaign")


@pytest.mark.parametrize("shape,width,az_res", [
    ((7, 8), 3.7, 360 / 7), ((8, 9), 5., 45.), ((5, 4), 15., 72.),
    ((1, 6), 1.5, 10.), ((6, 1), 5., .5),
])
def test_full_weights_match_independent_cosine_sums_and_keep_tails(shape, width, az_res):
    original = bench.radar.SpatialPolarGaussian(shape, width, az_res)
    a, r = bench.independent_weights(shape, width, az_res)
    np.testing.assert_allclose(a, original.az_weights, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(r, original.range_weights, rtol=2e-12, atol=2e-14)
    assert a.shape == (2 * (shape[0] // 2) + 1, shape[1])
    assert r.size == 2 * shape[1] + 1
    np.testing.assert_allclose(a.sum(axis=0), 1, atol=2e-15)
    assert r.sum() == pytest.approx(1, abs=2e-15)
    # The benchmark has no sampled Gaussian, positivity correction or renormalization.
    if width == 3.7:
        assert (a < 0).any()


@pytest.mark.parametrize("shape,width,az_res", [((7, 8), 3.7, 360 / 7), ((8, 9), 5., 45.), ((5, 4), 15., 72.)])
def test_operator_impulses_constants_and_signed_data_match_release_and_dense_oracle(toolkit, shape, width, az_res):
    oracle = bench.radar.SpatialPolarGaussian(shape, width, az_res)
    arrays = [np.ones(shape), np.random.default_rng(5301).normal(size=shape)]
    for position in ((0, 0), (shape[0] - 1, shape[1] - 1)):
        impulse = np.zeros(shape)
        impulse[position] = 1
        arrays.append(impulse)
    for values in arrays:
        expected = oracle(values)
        np.testing.assert_allclose(expected, bench.radar.dense_small_operator(values, width, az_res), atol=2e-12, rtol=2e-12)
        released = toolkit.smooth_polar(values, width, az_res_deg=az_res,
                                        az_boundary="periodic", range_boundary="reflective", kernel_type="gaussian")
        np.testing.assert_allclose(released, expected, atol=2e-12, rtol=2e-12)
        for method in bench.METHODS:
            audit = {}
            linear = bench.prepare_filter(method, shape, width, az_res, toolkit, audit)
            np.testing.assert_allclose(linear(values), expected, atol=2e-12, rtol=2e-12)
            assert audit == dict(filter_preparations=1, full_filter_calls=1)


@pytest.mark.parametrize("pattern", ("incomplete", "all_missing", "zeros", "single_observation"))
def test_shared_three_filter_moments_undefined_masks_and_original_input(toolkit, pattern):
    shape, width, az_res = (8, 9), 5., .5
    data = np.random.default_rng(5302).normal(size=shape)
    data[1:4, 2:6] = np.nan
    data[0, 0] = 0
    if pattern == "all_missing":
        data[:] = np.nan
    elif pattern == "zeros":
        data[:] = 0
    elif pattern == "single_observation":
        data[:] = np.nan
        data[3, 4] = 2.0
    before = data.tobytes()
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0)
    oracle = bench.radar.SpatialPolarGaussian(shape, width, az_res)
    expected = bench.radar.direct_fields(oracle(valid.astype(float)), oracle(filled), oracle(filled**2),
                                         valid, width, az_res, bench.FLOOR)
    expected["coverage"] = np.clip(expected["raw_mask_response"], 0, 1)
    for method in bench.METHODS:
        audit = {}
        output = bench.calculate(method, data, width, az_res, toolkit, audit)
        assert audit == dict(filter_preparations=1, full_filter_calls=3)
        assert set(output) == set(bench.FIELDS)
        for key in bench.FIELDS:
            row = bench.compare_arrays("unit", pattern, method, key, "all", output[key], expected[key])
            assert row["status"] in ("pass", "no_comparable"), row
        np.testing.assert_array_equal(output["direct_acceptance"], np.isfinite(output["raw_mask_response"]) & (output["raw_mask_response"] > .001))
        np.testing.assert_array_equal(output["std"], np.sqrt(output["variance"]))
        assert data.tobytes() == before
        if pattern == "all_missing":
            assert np.isnan(output["mean"]).all() and np.isnan(output["std"]).all()
            assert not output["coverage"].any() and not output["support"].any()


def test_strict_raw_denominator_and_explicit_zero_fill_are_not_replaced_by_clipped_coverage(monkeypatch):
    data = np.array([[np.nan, 0., 2., np.inf, -1., 4.]])
    density = np.array([[-.1, .001, np.nextafter(.001, np.inf), np.nan, np.inf, 1.5]])
    calls = []

    def prepare(method, shape, width, az_res, toolkit, audit):
        audit["filter_preparations"] = 1

        def linear(values):
            calls.append(values.copy())
            audit["full_filter_calls"] = len(calls)
            return density.copy() if len(calls) == 1 else np.full(shape, 2 if len(calls) == 2 else 10.)

        return linear

    monkeypatch.setattr(bench, "prepare_filter", prepare)
    audit = {}
    output = bench.calculate(bench.METHODS[0], data, 5., .5, object(), audit)
    assert audit == dict(filter_preparations=1, full_filter_calls=3)
    np.testing.assert_array_equal(calls[0], np.isfinite(data).astype(float))
    np.testing.assert_array_equal(calls[1], [[0, 0, 2, 0, -1, 4]])
    np.testing.assert_array_equal(calls[2], calls[1]**2)
    np.testing.assert_array_equal(output["direct_acceptance"], [[False, False, True, False, False, True]])
    assert output["mean"][0, -1] == pytest.approx(2 / 1.5)
    assert output["coverage"][0, -1] == 1


@pytest.mark.parametrize("method", bench.METHODS[1:])
def test_independent_baselines_never_call_toolkit_or_original_spectral_helpers(monkeypatch, method):
    class ForbiddenToolkit:
        def __getattr__(self, name):
            forbidden()

    monkeypatch.setattr(bench.radar, "toolkit_fields", forbidden)
    output = bench.calculate(method, np.ones((8, 9)), 5., .5, ForbiddenToolkit())
    np.testing.assert_allclose(output["mean"], 1., atol=2e-15)


def test_reported_library_baselines_use_scipy_routines():
    import scipy.fft
    import scipy.ndimage

    assert bench.rfft is scipy.fft.rfft and bench.irfft is scipy.fft.irfft
    assert bench.convolve1d is scipy.ndimage.convolve1d


@pytest.mark.parametrize("shape,width,az_res", [((0, 8), 5, .5), ((4, 5, 6), 5, .5), ((4, 5), 0, .5),
                                              ((4, 5), np.nan, .5), ((4, 5), 5, 0), ((4, 5), 5, np.inf)])
def test_invalid_geometry_rejected(shape, width, az_res):
    with pytest.raises(ValueError):
        bench.independent_responses(shape, width, az_res)


def test_controlled_removal_is_the_original_middle_quarter_and_keeps_zeros():
    data = np.arange(80, dtype=float).reshape(8, 10)
    data[0, 3] = np.nan
    patterns = bench.input_patterns(data)
    assert patterns["natural"] is data
    expected = data.copy()
    expected[3:5, :] = np.nan
    np.testing.assert_array_equal(patterns["controlled_extra_gap"], expected)
    assert patterns["natural"][0, 0] == 0


def valid_payload():
    hashes = {key: "field-sha" for key in bench.FIELDS}
    descriptor = dict(input_sha256="input-sha", benchmark_script_sha256="source-sha",
                      output_sha256={method: hashes for method in bench.METHODS})
    payload = dict(status="success", case_id="natural_W5", method=bench.METHODS[0],
                   input_sha256="input-sha", benchmark_script_sha256="source-sha",
                   audit=dict(filter_preparations=1, full_filter_calls=3), n_outputs=len(bench.FIELDS),
                   output_fingerprints_verified=True, output_sha256=hashes,
                   elapsed_s=.1, ram_before_mib=100., peak_ram_mib=120.,
                   preloaded_libraries=["numpy", "scipy.fft", "scipy.ndimage", "psutil", "dct_toolkit"])
    return payload, descriptor


@pytest.mark.parametrize("key,value", [
    ("input_sha256", "wrong"), ("benchmark_script_sha256", "wrong"),
    ("audit", dict(filter_preparations=2, full_filter_calls=3)),
    ("audit", dict(filter_preparations=1, full_filter_calls=2)), ("n_outputs", 8),
    ("output_fingerprints_verified", False), ("output_sha256", {}), ("preloaded_libraries", []),
    ("elapsed_s", 0.), ("elapsed_s", float("nan")), ("peak_ram_mib", 99.),
])
def test_invalid_worker_cannot_enter_successful_timing_summaries(key, value):
    payload, descriptor = valid_payload()
    payload[key] = value
    checked = bench.check_worker(payload, descriptor, "natural_W5", bench.METHODS[0])
    assert checked["status"] == "invalid_worker"
    row = dict(run_id="unit", case_id="natural_W5", method=bench.METHODS[0], **{
        key: checked.get(key) for key in ("status", "elapsed_s", "ram_before_mib", "peak_ram_mib")})
    summary, = bench.summarize([row])
    assert summary["n_success"] == 0 and summary["n_failure"] == 1
    assert summary["median_time_s"] is summary["max_peak_ram_mib"] is None


def test_good_worker_and_summary_quantiles_use_only_matching_successes():
    payload, descriptor = valid_payload()
    assert bench.check_worker(payload, descriptor, "natural_W5", bench.METHODS[0])["status"] == "success"
    rows = [dict(run_id="unit", case_id="natural_W5", method=bench.METHODS[0], status="success",
                 elapsed_s=t, peak_ram_mib=120+i) for i, t in enumerate((1., 2., 3., 4., 5.))]
    rows.append(dict(run_id="unit", case_id="natural_W5", method=bench.METHODS[0], status="error", elapsed_s=1000.))
    summary, = bench.summarize(rows)
    assert summary["n_success"] == 5 and summary["n_failure"] == 1
    assert (summary["median_time_s"], summary["q25_time_s"], summary["q75_time_s"]) == (3., 2., 4.)
    assert summary["max_peak_ram_mib"] == 124


def test_real_worker_contract_on_tiny_unit_fixture_not_a_radar_measurement(tmp_path, monkeypatch, toolkit):
    data = np.random.default_rng(5303).normal(size=(8, 9))
    data[2:5, 3:6] = np.nan
    fields = bench.calculate(bench.METHODS[0], data, 5., .5, toolkit)
    descriptor = dict(verified=True, shape=list(data.shape), width=5., az_res_deg=.5,
                      benchmark_script_sha256=bench.radar.sha256_file(Path(bench.__file__)),
                      input_sha256=bench.array_sha256(data), output_sha256={bench.METHODS[0]: bench.fingerprints(fields)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"unit": descriptor}))
    path = tmp_path / "data.npy"
    np.save(path, data)
    monkeypatch.setattr(bench, "SHAPE", data.shape)
    for key in bench.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    args = SimpleNamespace(case="unit", method=bench.METHODS[0], input=path, manifest=manifest, max_ram_mib=4096)
    result = bench.worker(args)
    assert result["status"] == "success", result
    assert result["audit"] == dict(filter_preparations=1, full_filter_calls=3)
    assert result["output_sha256"] == descriptor["output_sha256"][bench.METHODS[0]]
    assert result["peak_ram_mib"] >= result["ram_before_mib"] > 0
    monkeypatch.setattr(bench, "calculate", forbidden)
    np.save(path, np.zeros_like(data))
    corrupt = bench.worker(args)
    assert corrupt["status"] == "error" and "not the numerically verified" in corrupt["reason"]


def test_comparison_requires_exact_undefined_masks_and_original_tolerances():
    for key in bench.radar.TOLERANCES:
        assert bench.TOLERANCES[key] == bench.radar.TOLERANCES[key]
    row = bench.compare_arrays("unit", "unit", bench.METHODS[0], "mean", "all", np.array([1., np.nan]), np.array([1., 2.]))
    assert row["status"] == "fail"
    row = bench.compare_arrays("unit", "unit", bench.METHODS[0], "mean", "all", np.array([1.]), np.array([2.]))
    assert row["status"] == "fail"


def test_failed_numerical_validation_blocks_all_workers(tmp_path, monkeypatch, toolkit):
    data = np.ones((8, 9))
    monkeypatch.setattr(bench, "SHAPE", data.shape)
    monkeypatch.setattr(bench, "load_verified_inputs", lambda path: (
        bench.input_patterns(data), np.arange(9), .5, {"input_sha256": ("file-sha", "")}))
    monkeypatch.setattr(bench, "load_toolkit", lambda: toolkit)
    monkeypatch.setattr(bench, "validate_case", lambda *args: ([dict(status="fail")], dict(verified=False)))
    monkeypatch.setattr(bench, "measure_worker", forbidden)
    captured = []
    monkeypatch.setattr(bench, "write_settings", lambda *args: captured.append(args))
    for key in bench.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    assert bench.main(["--run-id", "failed", "--output-root", str(tmp_path), "--scratch-root", str(tmp_path)]) == 1
    directory = tmp_path / "2d_radar_statistics_benchmark" / "failed"
    assert (directory / "comparison.csv").exists()
    assert not (directory / "benchmark_runs.csv").exists()
    assert len(captured[0][2]) == 4
    assert all("failure" in case["timing_status"] for case in captured[0][2].values())


@pytest.mark.parametrize("args", (["--repeats", "0"], ["--timeout-s", "nan"], ["--max-ram-mib", "0"],
                                   ["--method", bench.METHODS[0]], ["--worker"], ["--case", "natural_W5"]))
def test_cli_rejects_unsafe_or_mixed_worker_options(args):
    with pytest.raises(SystemExit) as error:
        bench.parse_args(args)
    assert error.value.code == 2


RESULT = os.environ.get("RADAR_BENCHMARK_RESULT_DIR")


def saved_rows(name):
    with (Path(RESULT) / name).open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert rows and all(None not in row and None not in row.values() for row in rows)
    return rows


@pytest.mark.skipif(not RESULT, reason="set RADAR_BENCHMARK_RESULT_DIR to selected matched radar run")
def test_saved_radar_full_array_comparisons_original_tolerances_and_gate():
    rows = saved_rows("comparison.csv")
    assert len(rows) == 580
    assert all(row["status"] in ("pass", "no_comparable") for row in rows)
    assert {row["method"] for row in rows} == {*bench.METHODS, bench.API, "inverse_fft_weight_construction"}
    for row in rows:
        assert (float(row["atol"]), float(row["rtol"])) == bench.TOLERANCES[row["statistic"]]
        assert row["n_method_undefined"] == row["n_reference_undefined"]
        if row["region"] == "all":
            assert int(row["n_considered"]) == 849600
        assert row["n_method_gap_filled"] == row["n_reference_gap_filled"] == "0"


@pytest.mark.skipif(not RESULT, reason="set RADAR_BENCHMARK_RESULT_DIR to selected matched radar run")
def test_saved_radar_source_inputs_three_filters_and_all_worker_outputs():
    cases = {}
    for row in saved_rows("settings.csv"):
        cases.setdefault(row["case_id"], {})[row["setting"]] = row["value"]
    assert set(cases) == {f"{pattern}_W{width:g}" for pattern in bench.PATTERNS for width in bench.WIDTHS}
    source_hash = bench.radar.sha256_file(Path(bench.__file__))
    pids, hashes = [], set()
    for case_id, values in cases.items():
        assert values["script_sha256"] == values["benchmark_script_sha256"] == source_hash
        assert values["radar_reference_script_sha256"] == bench.radar.sha256_file(Path(bench.radar.__file__))
        assert values["radar_file_sha256"] == bench.radar.DOCUMENTED_SHA256
        assert values["dct_toolkit_commit"] == "9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c"
        assert values["dct_toolkit_version"] == "1.0.0"
        assert values["filter_preparations"] == "1" and values["full_filter_calls"] == "3"
        assert values["completion"] == values["restoration"] == "False"
        assert values["verified"] == "True" and values["timing_status"] == "measured and verified"
        assert values["repetitions"] == "5" and values["warmups"] == "0"
        assert json.loads(values["shape"]) == [720, 1180]
        for key in bench.THREAD_VARIABLES:
            assert values[key] == "1"
        expected = json.loads(values["output_sha256"])
        workers = [json.loads(value) for key, value in values.items() if key.startswith("worker_") and key.rsplit("_", 1)[-1].isdigit()]
        assert len(workers) == 15
        for item in workers:
            assert item["status"] == "success" and item["output_fingerprints_verified"] is True
            assert item["input_sha256"] == values["input_sha256"]
            assert item["benchmark_script_sha256"] == source_hash
            assert item["audit"] == dict(filter_preparations=1, full_filter_calls=3)
            assert item["output_sha256"] == expected[item["method"]]
            assert item["n_outputs"] == 9 and item["case_id"] == case_id
            assert item["preloaded_libraries"] == ["numpy", "scipy.fft", "scipy.ndimage", "psutil", "dct_toolkit"]
            hashes.add(item["input_sha256"])
            pids.append(item["worker_pid"])
    assert len(pids) == len(set(pids)) == 60
    assert len(hashes) == 2


@pytest.mark.skipif(not RESULT, reason="set RADAR_BENCHMARK_RESULT_DIR to selected matched radar run")
def test_saved_radar_timing_summary_recomputed_from_all_five_repetitions():
    raw, summaries = saved_rows("benchmark_runs.csv"), saved_rows("benchmark_summary.csv")
    assert len(raw) == 60 and len(summaries) == 12
    assert all(row["status"] == "success" and row["operation"] == bench.OPERATION for row in raw)
    for summary in summaries:
        group = [r for r in raw if (r["case_id"], r["method"]) == (summary["case_id"], summary["method"])]
        assert {r["repetition"] for r in group} == {"1", "2", "3", "4", "5"}
        assert summary["n_success"] == "5" and summary["n_failure"] == "0"
        times = [float(r["elapsed_s"]) for r in group]
        assert min(times) > 0
        for key, q in (("median_time_s", .5), ("q25_time_s", .25), ("q75_time_s", .75)):
            assert float(summary[key]) == pytest.approx(float(np.quantile(times, q)), rel=1e-12)
        assert float(summary["max_peak_ram_mib"]) == max(float(r["peak_ram_mib"]) for r in group)
        assert all(float(r["peak_ram_mib"]) >= float(r["ram_before_mib"]) > 0 for r in group)


# Current manuscript/table checks live in test_radar_range_loops.py. This file
# retains all numerical/source/worker checks for the earlier frozen FFT run.
