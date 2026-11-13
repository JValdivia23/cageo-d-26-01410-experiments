"""Discrete Gaussian equivalence, independent centered moments and timing gates."""

import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_2d_radar_discrete_gaussian as experiment


@pytest.fixture(scope="module")
def toolkit():
    return experiment.base.load_toolkit()


def explicit_reference(data, width, az_res):
    """Neighbor-by-neighbor positive weights, independent boundary mapping."""
    n_az, n_range = data.shape
    radial_sigma = width / np.sqrt(12)
    r_radius = int(np.ceil(4 * radial_sigma))
    r_offsets = np.arange(-r_radius, r_radius + 1)
    radial = np.exp(-r_offsets**2 / (2 * radial_sigma**2))
    radial /= radial.sum()
    mean, variance, density = [np.full(data.shape, np.nan) for _ in range(3)]
    for az in range(n_az):
        for gate in range(n_range):
            observations, weights = [], []
            for dr, wr in zip(r_offsets, radial):
                donor_gate = gate - int(dr)
                while donor_gate < 0 or donor_gate >= n_range:
                    donor_gate = -donor_gate - 1 if donor_gate < 0 else 2 * n_range - 1 - donor_gate
                sigma = width / ((donor_gate + 1) * np.deg2rad(az_res) * np.sqrt(12))
                radius = int(np.ceil(4 * sigma))
                offsets = np.arange(-radius, radius + 1)
                angular = np.exp(-offsets**2 / (2 * sigma**2))
                angular /= angular.sum()
                for da, wa in zip(offsets, angular):
                    value = data[(az - int(da)) % n_az, donor_gate]
                    if np.isfinite(value):
                        observations.append(value)
                        weights.append(wr * wa)
            weights, observations = np.asarray(weights), np.asarray(observations)
            mass = weights.sum()
            density[az, gate] = mass
            if mass > experiment.FLOOR:
                mu = np.sum(weights * observations) / mass
                mean[az, gate] = mu
                variance[az, gate] = np.sum(weights * (observations - mu)**2) / mass
    return mean, variance, density


@pytest.mark.parametrize("sigma", (.05, .3, 1., 8.3))
@pytest.mark.parametrize("period", (1, 5, 6, 14))
def test_transfer_is_exact_cosine_sum_including_repeated_wraps(sigma, period):
    kernel = experiment.gaussian_kernel(sigma)
    offsets = np.arange(-(kernel.size // 2), kernel.size // 2 + 1)
    frequency = 2 * np.pi * np.arange(period // 2 + 1) / period
    expected = np.cos(frequency[:, None] * offsets) @ kernel
    np.testing.assert_allclose(experiment.discrete_response(kernel, period), expected, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(kernel, experiment.astropy_kernel(sigma), atol=1e-15, rtol=1e-14)
    assert np.all(kernel >= 0) and abs(kernel.sum() - 1) < 1e-14


@pytest.mark.parametrize("shape,width,az_res", [((5, 6), .7, 72.), ((6, 5), 3., 60.),
                                                ((3, 4), 5., 30.), ((1, 5), 2., 90.), ((5, 1), 3., 72.)])
def test_full_centered_reference_and_both_methods_match_explicit_neighbors(toolkit, shape, width, az_res):
    data = np.random.default_rng(19101).normal(size=shape)
    data[0, 0] = np.nan
    data[-1, -1] = 0.0
    unchanged = data.tobytes()
    mean, variance, density = explicit_reference(data, width, az_res)
    expected = experiment.centered_reference(data, width, az_res)
    np.testing.assert_allclose(expected["mean"], mean, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(expected["variance"], variance, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(expected["raw_mask_response"], density, atol=3e-14, rtol=3e-14)
    assert np.all(expected["variance"][np.isfinite(variance)] >= 0)
    for method in experiment.METHODS[1:]:
        audit = {}
        actual = experiment.calculate(method, data, width, az_res, toolkit, audit)
        assert audit == experiment.expected_audit(method, shape)
        for name in experiment.FIELDS:
            np.testing.assert_allclose(actual[name], expected[name], atol=2e-12, rtol=2e-12)
    assert data.tobytes() == unchanged


@pytest.mark.parametrize("pattern", ("complete", "all_missing", "single_observation", "finite_zero", "gap"))
def test_masks_zeros_and_raw_variance_are_checked_before_clamping(toolkit, pattern):
    data = np.full((6, 7), 35.)
    if pattern == "all_missing":
        data[:] = np.nan
    elif pattern == "single_observation":
        data[:] = np.nan
        data[2, 3] = 35.
    elif pattern == "finite_zero":
        data[:] = 0.
    elif pattern == "gap":
        data[:3, 1:5] = np.nan
    reference = experiment.centered_reference(data, 3., 60.)
    for method in experiment.METHODS[1:]:
        actual = experiment.calculate(method, data, 3., 60., toolkit)
        np.testing.assert_array_equal(actual["direct_acceptance"], reference["direct_acceptance"])
        diagnostic = experiment.variance_diagnostics("unit", pattern, method, "all", actual, reference,
                                                      np.ones(data.shape, bool))
        assert diagnostic["n_negative_beyond_reference_tolerance"] == 0
        if pattern == "all_missing":
            assert np.isnan(actual["mean"]).all() and np.isnan(actual["std"]).all()
            assert diagnostic["min_raw_variance"] is None
        else:
            assert np.nanmax(np.abs(actual["raw_variance"])) < experiment.base.TOLERANCES["variance"][0]
            checked = experiment.compare("unit", pattern, method, "std", "all", actual["std"], reference["std"])
            assert checked["status"] == "pass"


def test_near_zero_std_uses_variance_bound_without_relaxing_nonzero_checks():
    compare = experiment.compare
    assert compare("unit", "zero", "discrete_spectral", "std", "all",
                   np.array([1e-5]), np.array([0.]))["status"] == "pass"
    assert compare("unit", "zero", "discrete_spectral", "std", "all",
                   np.array([.01]), np.array([0.]))["status"] == "fail"
    assert compare("unit", "nonzero", "discrete_spectral", "std", "all",
                   np.array([.1 + 3e-6]), np.array([.1]))["status"] == "fail"
    assert compare("unit", "variance", "discrete_spectral", "raw_variance", "all",
                   np.array([-3e-6]), np.array([0.]))["status"] == "fail"
    assert compare("unit", "released", "analytical_spectral", "std", "all",
                   np.array([1e-5]), np.array([0.]), "released_analytical_api")["status"] == "fail"


def test_centered_reference_does_not_call_filters_transforms_or_raw_moments(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Centered reference must not reuse filters, transforms or raw-moment subtraction")
    for name in ("dct", "idct", "rfft", "irfft", "convolve1d", "prepare_filter", "calculate"):
        monkeypatch.setattr(experiment, name, forbidden)
    monkeypatch.setattr(experiment.base, "moment_fields", forbidden)
    result = experiment.centered_reference(np.array([[1., np.nan], [3., 5.]]), 3., 90.)
    assert np.all(result["variance"] >= 0)


def test_analytical_and_sampled_filters_are_not_assumed_identical(toolkit):
    data = np.zeros((8, 9))
    data[3, 4] = 1.
    analytical = experiment.prepare_filter(experiment.METHODS[0], data.shape, .7, 45., toolkit, {})(data)
    discrete = experiment.prepare_filter("discrete_spectral", data.shape, .7, 45., toolkit, {})(data)
    spatial = experiment.prepare_filter("discrete_spatial", data.shape, .7, 45., toolkit, {})(data)
    np.testing.assert_allclose(discrete, spatial, atol=1e-14)
    assert np.all(spatial >= 0) and np.min(analytical) < -1e-5
    assert not np.allclose(analytical, discrete, atol=1e-5)


def test_validation_labels_different_operators_without_relaxing_correctness(toolkit):
    data = np.random.default_rng(19102).normal(size=(6, 7))
    data[1:3, :2] = np.nan
    rows, diagnostics, differences, descriptor, summary = experiment.validate_case(
        "unit", "tiny", data, np.arange(7) * 100., 3., 60., toolkit)
    assert descriptor["verified"]
    assert all(r["status"] == "pass" for r in rows)
    assert all(r["status"] == "different_operator" for r in differences)
    assert {r["reference_method"] for r in rows} == {experiment.REFERENCE, "released_analytical_api"}
    assert len(summary) == 2
    assert all(r["n_negative_beyond_reference_tolerance"] == 0 for r in diagnostics if r["method"] != experiment.METHODS[0])


def test_empty_summary_does_not_fabricate_zero_errors(toolkit):
    rows, diagnostics, differences, descriptor, summary = experiment.validate_case(
        "unit", "empty", np.full((3, 4), np.nan), np.arange(4) * 100., 3., 120., toolkit)
    assert descriptor["verified"]
    assert all(r["max_abs_variance_difference"] is None for r in summary)
    assert all(r["max_abs_mean_difference"] is None for r in summary)


@pytest.mark.parametrize("method", experiment.METHODS)
def test_worker_checks_the_actual_workload_and_nine_output_hashes(toolkit, tmp_path, monkeypatch, method):
    shape = (6, 7)
    monkeypatch.setattr(experiment, "SHAPE", shape)
    for key in experiment.base.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    data = np.random.default_rng(19103).normal(size=shape)
    output = experiment.calculate(method, data, 3., 60., toolkit)
    descriptor = dict(verified=True, source_sha256=experiment.source_hashes(), input_sha256=experiment.base.array_sha256(data),
                      shape=list(shape), width=3., az_res_deg=60., output_sha256={method: experiment.base.fingerprints(output)})
    source, manifest = tmp_path/"input.npy", tmp_path/"manifest.json"
    np.save(source, data)
    manifest.write_text(json.dumps(dict(tiny=descriptor)))
    args = SimpleNamespace(case="tiny", method=method, input=source, manifest=manifest, max_ram_mib=1024)
    payload = experiment.worker(args)
    assert payload["status"] == "success", payload
    assert experiment.check_worker(payload, descriptor, "tiny", method)["status"] == "success"
    assert payload["audit"] == experiment.expected_audit(method, shape)
    assert 0 < payload["preparation_s"] <= payload["elapsed_s"]
    payload["n_outputs"] = 1
    assert experiment.check_worker(payload, descriptor, "tiny", method)["status"] == "invalid_worker"


@pytest.mark.parametrize("argv", (["--worker"], ["--method", "discrete_spectral"], ["--repeats", "0"],
                                   ["--timeout-s", "nan"], ["--max-ram-mib", "0"]))
def test_cli_rejects_incomplete_or_unsafe_worker_options(argv):
    with pytest.raises(SystemExit):
        experiment.parse_args(argv)


RESULT = os.environ.get("DISCRETE_GAUSSIAN_RESULT_DIR")


@pytest.mark.skipif(not RESULT, reason="set DISCRETE_GAUSSIAN_RESULT_DIR to the selected run")
def test_large_analytical_negatives_survive_independent_spatial_centered_sum(toolkit):
    path = Path(os.environ.get("REAL_RADAR_PATH", str(experiment.base.radar.DEFAULT_RADAR))).expanduser()
    if not path.is_file():
        pytest.skip("Exact processed radar input is not available locally")
    data, _, az_res, metadata = experiment.base.radar.read_radar(path, quick=False)
    assert metadata["input_sha256"][0] == experiment.base.radar.DOCUMENTED_SHA256
    assert experiment.base.array_sha256(data) == experiment.base.DECODED_SHA256
    valid = np.isfinite(data)
    for width in experiment.WIDTHS:
        fields = experiment.calculate("analytical_spectral", data, width, az_res, toolkit)
        az, gate = np.unravel_index(np.nanargmin(fields["raw_variance"]), data.shape)
        # Independent cosine-sum construction, then explicit original-donor
        # weights at one target. The variance uses centered squared residuals,
        # not a difference of large raw moments, and still remains negative.
        oracle = experiment.base.radar.SpatialPolarGaussian(data.shape, width, az_res)
        radial_offsets = np.arange(-data.shape[1], data.shape[1] + 1)
        donors = experiment.reflect_indices(gate - radial_offsets, data.shape[1])
        radial = np.zeros(data.shape[1])
        np.add.at(radial, donors, oracle.range_weights)
        angular_offsets = np.arange(-(data.shape[0] // 2), data.shape[0] // 2 + 1)
        angular = np.zeros(data.shape)
        np.add.at(angular, (az - angular_offsets) % data.shape[0], oracle.az_weights)
        weights = (angular * radial[None, :])[valid]
        values = data[valid]
        mass = weights.sum()
        mean = np.sum(weights * values) / mass
        centered_variance = np.sum(weights * (values - mean)**2) / mass
        assert mass > experiment.FLOOR and centered_variance < -1.
        np.testing.assert_allclose(centered_variance, fields["raw_variance"][az, gate], atol=2e-6, rtol=2e-9)


@pytest.mark.skipif(not RESULT, reason="set DISCRETE_GAUSSIAN_RESULT_DIR to the selected run")
def test_saved_full_array_correctness_raw_variance_and_compact_tables():
    root = Path(RESULT)
    def read(name):
        with (root/name).open() as stream:
            return list(csv.DictReader(stream))
    comparisons = read("comparison.csv")
    assert all(r["status"] in ("pass", "no_comparable") for r in comparisons)
    assert all(r["status"] == "pass" for r in read("kernel_checks.csv"))
    assert all(r["status"] == "different_operator" for r in read("kernel_comparison.csv"))
    diagnostics = read("variance_diagnostics.csv")
    assert all(int(r["n_negative_beyond_reference_tolerance"]) == 0 for r in diagnostics if r["method"] != experiment.METHODS[0])
    assert len(read("agreement_summary.csv")) == 4
    assert len(read("timing_summary.csv")) == 2
    timings = read("benchmark_runs.csv")
    assert len(timings) == 30 and all(r["status"] == "success" for r in timings)
    settings = {}
    for row in read("settings.csv"):
        settings.setdefault(row["case_id"], {})[row["setting"]] = row["value"]
    for case, values in settings.items():
        assert values["dct_toolkit_commit"] == experiment.base.TOOLKIT_COMMIT
        assert values["input_sha256"] == experiment.base.DECODED_SHA256
        assert values["kernel_radius"] == "ceil(4*sigma)"
        assert values["source_sha256"] == json.dumps(experiment.source_hashes(), sort_keys=True)
        for method in experiment.METHODS:
            subset = [r for r in timings if r["case_id"] == case and r["method"] == method]
            summary = next(r for r in read("benchmark_summary.csv") if r["case_id"] == case and r["method"] == method)
            assert len(subset) == 5
            assert float(summary["median_time_s"]) == np.median([float(r["elapsed_s"]) for r in subset])
            for repeat in range(1, 6):
                payload = json.loads(values[f"worker_{method}_{repeat}"])
                descriptor = dict(shape=json.loads(values["shape"]), source_sha256=json.loads(values["source_sha256"]),
                                  input_sha256=values["input_sha256"], output_sha256=json.loads(values["output_sha256"]))
                assert experiment.check_worker(payload, descriptor, case, method)["status"] == "success"
