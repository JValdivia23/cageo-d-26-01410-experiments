"""Lead-review checks independent of each experiment's internal test routines."""

from pathlib import Path
from itertools import product
import csv
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _review_common import load_toolkit
import generate_1d_boxcar_validation as one
import generate_3d_wind_boxcar_validation as three
import generate_2d_radar_gaussian_validation as radar


def reflected_index(index, length):
    index %= 2 * length
    return index if index < length else 2 * length - 1 - index


def manual_boxcar_1d(values, width):
    fields = {name: np.full(values.shape, np.nan) for name in ("mean", "variance", "std")}
    fields["support"] = np.zeros(values.shape)
    for position in range(values.size):
        neighbors = np.array([
            values[reflected_index(position + offset, values.size)]
            for offset in range(-(width // 2), width // 2 + 1)
        ])
        neighbors = neighbors[np.isfinite(neighbors)]
        fields["support"][position] = neighbors.size
        if neighbors.size / width <= one.DENOMINATOR_FLOOR:
            continue
        mean = sum(neighbors) / neighbors.size
        variance = sum((value - mean) ** 2 for value in neighbors) / neighbors.size
        for name, value in (("mean", mean), ("variance", variance), ("std", variance**0.5)):
            fields[name][position] = value
    return fields


@pytest.mark.parametrize("width", (1, 3, 7, 31))
def test_1d_against_manual_reflected_neighbors(width):
    rng = np.random.default_rng(60713)
    values = rng.normal(size=23)
    values[[0, 4, 5, 6, 20]] = np.nan
    values[9] = np.inf
    expected = manual_boxcar_1d(values, width)
    for name, method in one.methods(load_toolkit()).items():
        actual = method(values, width)
        for statistic in expected:
            atol = 2e-7 if statistic == "std" else 2e-12
            np.testing.assert_allclose(actual[statistic], expected[statistic],
                                       atol=atol, rtol=2e-12, equal_nan=True,
                                       err_msg=f"{name} {statistic} width={width}")


def test_1d_missing_outputs_cannot_pass_a_comparison():
    row = one.comparison_row("review", "case", "signal", "broken", "mean", "all",
                             np.ones(2, dtype=bool), np.array([1.0, np.nan]), np.ones(2))
    assert row["status"] == "fail"
    assert row["n_compared"] == row["n_method_undefined"] == 1


def test_1d_near_zero_std_is_checked_through_variance_too():
    values = np.r_[np.full(100, -2.0), np.full(100, 6.0)]
    actual = one.toolkit_moments(values, 15, load_toolkit())
    expected = manual_boxcar_1d(values, 15)
    np.testing.assert_allclose(actual["variance"], expected["variance"], atol=2e-12)
    zero_variance = expected["variance"] == 0
    assert np.max(actual["std"][zero_variance]) < 1e-6


def manual_boxcar_3d(observed, width):
    valid = np.logical_and.reduce([np.isfinite(a) for a in observed])
    shape = valid.shape
    result = {name: {stat: np.full(shape, np.nan) for stat in three.STATISTICS}
              for name in three.COMPONENTS}
    offsets = list(product(*(range(-(w // 2), w // 2 + 1) for w in width)))
    for position in np.ndindex(shape):
        neighbors = [tuple(reflected_index(i + d, n) for i, d, n in zip(position, delta, shape))
                     for delta in offsets]
        retained = [p for p in neighbors if valid[p]]
        for name, values in zip(three.COMPONENTS, observed):
            result[name]["support"][position] = len(retained)
            if len(retained) / len(offsets) <= three.DENOMINATOR_FLOOR:
                continue
            sample = [values[p] for p in retained]
            mean = sum(sample) / len(sample)
            variance = sum((v - mean) ** 2 for v in sample) / len(sample)
            result[name]["mean"][position] = mean
            result[name]["variance"][position] = variance
            result[name]["std"][position] = variance**0.5
    return result


@pytest.mark.parametrize("width", ((3, 3, 3), (5, 7, 9)))
def test_3d_all_methods_against_manual_neighbors(width):
    rng = np.random.default_rng(17061)
    observed = tuple(rng.normal(size=(4, 5, 6)) for _ in range(3))
    observed[0][0, 0, 0] = np.nan
    observed[1][1:3, 1:3, 1:3] = np.nan
    observed[2][2, 3, 4] = np.inf
    expected = manual_boxcar_3d(observed, width)
    for method in three.METHODS:
        actual = three.calculate(method, observed, width, three.prepare_backend(method))
        for component in three.COMPONENTS:
            for statistic in three.STATISTICS:
                np.testing.assert_allclose(actual[component][statistic], expected[component][statistic],
                                           atol=3e-12, rtol=3e-12, equal_nan=True,
                                           err_msg=f"{method} {component} {statistic} width={width}")


@pytest.mark.parametrize("method", three.METHODS)
def test_3d_every_benchmark_method_prepares_once_and_filters_seven_fields(method, monkeypatch):
    observed, _, _ = three.generate_input(three.CASES["smoke"])
    prepare = three.prepare_filter
    setups, calls = [], []

    def tracked_prepare(*args):
        setups.append(args[0])
        linear_filter = prepare(*args)

        def tracked_filter(values):
            calls.append(values.copy())
            return linear_filter(values)

        return tracked_filter

    monkeypatch.setattr(three, "prepare_filter", tracked_prepare)
    audit = {}
    output = three.calculate(method, observed, (3, 7, 7), three.prepare_backend(method), audit=audit)
    assert setups == [method]
    assert len(calls) == three.FULL_FILTER_CALLS == 7
    assert audit == {"full_nd_filter_calls": 7, "filter_preparations": 1}
    valid = np.logical_and.reduce([np.isfinite(a) for a in observed])
    np.testing.assert_array_equal(calls[0], valid.astype(float))
    for index, component in enumerate(observed):
        safe = np.where(valid, component, 0.0)
        np.testing.assert_array_equal(calls[1 + 2 * index], safe)
        np.testing.assert_array_equal(calls[2 + 2 * index], safe**2)
    assert sum(len(stats) for stats in output.values()) == 12
    assert output["u"]["support"] is output["v"]["support"] is output["w"]["support"]


def test_3d_dct_prepares_transfers_once_without_statistics_api_calls(monkeypatch):
    toolkit = load_toolkit()
    original = toolkit.get_dct_transfer_function
    transfers = []

    def tracked_transfer(*args):
        transfers.append(args)
        return original(*args)

    def forbidden(*args, **kwargs):
        pytest.fail("Statistics API must not be called inside the matched benchmark")

    monkeypatch.setattr(toolkit, "get_dct_transfer_function", tracked_transfer)
    for name in ("dct_mean", "dct_variance", "dct_std", "dct_count", "smooth_cartesian"):
        monkeypatch.setattr(toolkit, name, forbidden)
    observed, _, _ = three.generate_input(three.CASES["smoke"])
    three.calculate(three.SPECTRAL, observed, (3, 7, 7), toolkit)
    assert transfers == [(6, "boxcar_discrete", 3.0), (16, "boxcar_discrete", 7.0),
                         (16, "boxcar_discrete", 7.0)]


@pytest.mark.parametrize("shape,width", (((4, 5, 6), (3, 3, 3)), ((4, 5, 6), (5, 7, 9)),
                                       ((1, 5, 6), (1, 3, 3))))
def test_3d_prepared_dct_matches_the_released_filter(shape, width):
    toolkit = load_toolkit()
    values = np.random.default_rng(8472).normal(size=shape)
    prepared = three.prepare_filter(three.SPECTRAL, shape, width, toolkit, 1)
    np.testing.assert_allclose(prepared(values), toolkit.smooth_cartesian(
        values, width, kernel_type="boxcar_discrete"), atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("case_id", ("smoke", "all_missing", "single_observation", "block_control"))
def test_3d_shared_dct_agrees_with_released_statistics_api(case_id):
    toolkit = load_toolkit()
    case = three.CASES[case_id]
    observed, _, _ = three.generate_input(case)
    shared = three.calculate(three.SPECTRAL, observed, case.width, toolkit)
    released = three.calculate_toolkit_api(observed, case.width, toolkit)
    rows = three.compare_outputs("review", case_id, three.API_VALIDATION, released, shared, observed)
    assert all(row["status"] in ("pass", "no_comparable") for row in rows)


def test_3d_saved_summary_uses_successes_and_preserves_failures():
    rows = [dict(run_id="r", case_id="c", method="m", status="success", elapsed_s=t,
                 peak_ram_mib=100 + t) for t in (1.0, 2.0, 6.0)]
    rows.append(dict(run_id="r", case_id="c", method="m", status="timeout",
                     elapsed_s=None, peak_ram_mib=None))
    summary, = three.summarize_benchmarks(rows)
    assert summary["median_time_s"] == 2.0
    assert summary["n_success"] == 3 and summary["n_failure"] == 1
    assert summary["max_peak_ram_mib"] == 106.0


@pytest.mark.parametrize("field,value", (("input_sha256", "wrong"), ("n_outputs", 11),
                                        ("full_nd_filter_calls", 8), ("filter_preparations", 2)))
def test_3d_rejects_workers_that_do_not_match_the_verified_workload(tmp_path, monkeypatch, field, value):
    measured = []

    def fake_worker(args, case_id, method):
        measured.append(method)
        _, _, settings = three.generate_input(three.CASES[case_id])
        payload = dict(status="success", elapsed_s=1.0, ram_before_mib=100.0,
                       peak_ram_mib=110.0, reason="", input_sha256=settings["input_sha256"],
                       n_outputs=12, full_nd_filter_calls=7, filter_preparations=1)
        payload[field] = value
        return payload

    monkeypatch.setattr(three, "measure_worker", fake_worker)
    monkeypatch.setattr(sys, "argv", [str(Path(three.__file__)), "--mode", "benchmark",
                                     "--cases", "smoke", "--methods", three.SPECTRAL,
                                     "--repeats", "1", "--run-id", "unit",
                                     "--output-root", str(tmp_path)])
    assert three.main() == 1
    assert measured == [three.SPECTRAL]
    directory = tmp_path / "3d_wind_boxcar" / "unit"
    rows = list(csv.DictReader((directory / "benchmark_runs.csv").open()))
    assert len(rows) == 1
    assert rows[0]["status"] == "error" and rows[0]["elapsed_s"] == ""
    assert "matched seven-filter preparation" in rows[0]["reason"]


@pytest.mark.parametrize("shape", ((5, 6), (6, 5), (7, 7), (8, 9)))
@pytest.mark.parametrize("width", (0.3, 1.0, 5.0, 15.0))
def test_radar_spatial_weights_against_independent_transform_calculation(shape, width):
    from scipy.fft import dct, idct
    values = np.random.default_rng(12308).normal(size=shape)
    az_res = 360.0 / shape[0]
    angular_width = width / ((np.arange(shape[1]) + 1) * np.deg2rad(az_res))
    angular_frequency = 2 * np.pi * np.fft.rfftfreq(shape[0])
    h_az = np.exp(-0.5 * (angular_frequency[:, None] * angular_width[None, :] / np.sqrt(12)) ** 2)
    stage1 = np.fft.irfft(np.fft.rfft(values, axis=0) * h_az, n=shape[0], axis=0)
    h_range = np.exp(-0.5 * (np.pi * np.arange(shape[1]) / shape[1] * width / np.sqrt(12)) ** 2)
    expected = idct(dct(stage1, axis=1, type=2, norm="ortho") * h_range,
                    axis=1, type=2, norm="ortho")
    actual = radar.SpatialPolarGaussian(shape, width, az_res)(values)
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)


def test_radar_infinite_estimates_cannot_pass_as_matching_missing_values():
    row = radar.comparison_row("review", "selftest", "mean", "all",
                               np.ones((1, 1), bool), np.array([[np.inf]]), np.array([[np.inf]]))
    assert row["status"] == "fail"


def test_radar_has_no_ordinary_gaussian_or_spatial_performance_baseline():
    assert not hasattr(radar, "sampled_gaussian_fields")
    assert not hasattr(radar, "gaussian_approximation_rows")

    class Toolkit:
        def dct_mean(self, data, width, **kwargs):
            return np.full_like(data, width)

    runs, summaries = radar.benchmark(Toolkit(), "review", np.ones((4, 5)), 90, 2, 0)
    assert len(runs) == 4 and len(summaries) == 2
    assert {row["method"] for row in runs} == {radar.SPECTRAL}
    assert all(row["status"] == "success" for row in runs)


@pytest.mark.parametrize("strict", (False, True))
def test_radar_retains_stress_failures_without_relaxing_tolerances(tmp_path, monkeypatch, strict):
    values = np.arange(80.0).reshape(10, 8)
    values[2:4, 2] = np.nan
    monkeypatch.setattr(radar, "read_radar", lambda path, quick: (
        values, 95 + 75 * np.arange(8), 36.0, {},
    ))
    original = radar.comparison_row

    def injected_failure(*args, **kwargs):
        row = original(*args, **kwargs)
        if "_floor1e-05" in row["case_id"] and row["statistic"] == "mean":
            row["status"] = "fail"
        return row

    monkeypatch.setattr(radar, "comparison_row", injected_failure)
    argv = [str(Path(radar.__file__)), "--skip-timing", "--run-id", "unit",
            "--output-root", str(tmp_path)]
    if strict:
        argv.append("--strict-sensitivity")
    monkeypatch.setattr(sys, "argv", argv)
    assert radar.main() == int(strict)
    directory = tmp_path / "2d_radar_gaussian" / "unit"
    assert not (directory / "gaussian_approximation.csv").exists()
    primary = list(csv.DictReader((directory / "comparison.csv").open()))
    stress = list(csv.DictReader((directory / "threshold_sensitivity.csv").open()))
    assert all(row["status"] != "fail" for row in primary)
    assert any(row["status"] == "fail" for row in stress)
    for row in stress:
        if row["statistic"] == "mean":
            assert float(row["atol"]) == radar.TOLERANCES["mean"][0]
