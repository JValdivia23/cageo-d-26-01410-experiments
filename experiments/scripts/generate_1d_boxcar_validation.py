#!/usr/bin/env python3
"""Validate reflective 1D boxcar moments, or explicitly export the approved figure.

The NumPy reference uses explicit symmetric windows and centered, population
variance. SciPy/FFT use normalized raw moments. The toolkit calls its released
statistics APIs with fallback disabled and an explicit strict denominator floor.
All estimates use only originally valid observations, even at missing inputs.
Numerical agreement is not a test of statistical recovery or denoising accuracy.
The default validation is unchanged. --figure-only verifies and draws just two
saved cases without running the validation campaign or writing result CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve

from _review_common import (
    COMPARISON_FIELDS,
    create_run_directory,
    load_toolkit,
    write_csv,
    write_settings,
)


N = 200
WIDTH = 15
DENOMINATOR_FLOOR = 1e-3
MASTER_SEED = 20261027
# Fixed before the validation runs. The looser std absolute tolerance accounts
# for sqrt near zero, while variance is independently checked much more tightly.
TOLERANCES = {
    "mean": (5e-12, 2e-12),
    "variance": (5e-11, 2e-11),
    "std": (float(np.sqrt(5e-11)), 2e-11),
    "support": (5e-13, 2e-13),
}
UNITS = {
    "mean": "signal_unit",
    "variance": "signal_unit^2",
    "std": "signal_unit",
    "support": "nominal_cells",
}
SIGNALS = {
    "trend_sine": (0, "1.2*x + 2.5*sin(1.5*x)", 1.0),
    "endpoint_curvature": (1, "0.20*(x-5)^2 + 2*cos(pi*x/10)", 1.0),
    "abrupt_step": (2, "-2 if x < 5 else 6", 0.0),
}
MASKS = ("complete", "random_nan", "middle_gap", "boundary_gap", "all_missing")
REFERENCE = "numpy_symmetric_windows"
COVERAGE_FIELDS = (
    "run_id", "case_id", "method", "region", "n_considered",
    "n_original_observed", "n_missing_inputs", "n_direct_defined",
    "n_direct_defined_at_missing_inputs", "n_undefined", "n_fallback_eligible",
    "n_gap_filled", "denominator_min", "denominator_max",
    "nominal_support_min", "nominal_support_max", "min_raw_moment_variance",
    "n_negative_raw_moment_variance", "notes",
)


def normalized_moments(
    numerator: np.ndarray,
    square_numerator: np.ndarray,
    denominator: np.ndarray,
    width: int,
) -> dict[str, np.ndarray]:
    """Direct ratios only: undefined at d <= floor; no additive epsilon/fill."""
    accepted = np.isfinite(denominator) & (denominator > DENOMINATOR_FLOOR)
    mean = np.full(denominator.shape, np.nan)
    second = np.full(denominator.shape, np.nan)
    np.divide(numerator, denominator, out=mean, where=accepted)
    np.divide(square_numerator, denominator, out=second, where=accepted)
    raw_variance = second - mean**2
    variance = np.maximum(raw_variance, 0.0)
    return {
        "mean": mean,
        "variance": variance,
        "std": np.sqrt(variance),
        "support": np.clip(denominator, 0.0, 1.0) * width,
        "denominator": denominator,
        "raw_moment_variance": raw_variance,
    }


def numpy_moments(data: np.ndarray, width: int) -> dict[str, np.ndarray]:
    """Independent spatial reference, including repeated boundary positions."""
    half = width // 2
    windows = sliding_window_view(np.pad(data, half, mode="symmetric"), width)
    valid = np.isfinite(windows)
    safe = np.where(valid, windows, 0.0)  # Never multiply NaN by zero.
    counts = valid.sum(axis=-1)
    denominator = counts.astype(float) / width
    result = normalized_moments(
        safe.sum(axis=-1) / width,
        (safe**2).sum(axis=-1) / width,
        denominator,
        width,
    )
    # Centered ddof=0 calculation is independent of E[x^2]-E[x]^2 cancellation.
    residual = np.where(valid, safe - result["mean"][:, None], 0.0)
    variance = np.full(data.shape, np.nan)
    np.divide(
        (residual**2).sum(axis=-1), counts, out=variance,
        where=denominator > DENOMINATOR_FLOOR,
    )
    result["variance"] = variance
    result["std"] = np.sqrt(variance)
    return result


def filtered_moments(
    data: np.ndarray, width: int, filter_function: Callable[[np.ndarray], np.ndarray]
) -> dict[str, np.ndarray]:
    valid = np.isfinite(data)
    safe = np.where(valid, data, 0.0)
    return normalized_moments(
        filter_function(safe), filter_function(safe**2),
        filter_function(valid.astype(float)), width,
    )


def scipy_moments(data: np.ndarray, width: int) -> dict[str, np.ndarray]:
    # SciPy reflect repeats the edge sample, as does NumPy symmetric (not reflect).
    return filtered_moments(
        data, width,
        lambda values: uniform_filter(values, size=width, mode="reflect", origin=0),
    )


def fft_moments(data: np.ndarray, width: int) -> dict[str, np.ndarray]:
    weights = np.ones(width, dtype=float) / width

    def filter_function(values: np.ndarray) -> np.ndarray:
        # Linear FFT convolution of the symmetric extension, then valid crop.
        # This is not a raw periodic FFT of the unextended input.
        return fftconvolve(np.pad(values, width // 2, mode="symmetric"), weights, mode="valid")

    return filtered_moments(data, width, filter_function)


def toolkit_moments(data: np.ndarray, width: int, toolkit: Any) -> dict[str, np.ndarray]:
    valid = np.isfinite(data)
    cleaned = np.where(valid, data, np.nan)
    options = dict(
        coordinates="cartesian", kernel_type="boxcar_discrete",
        restore_input_nan=False, _stable_fallback=False,
        _denominator_floor=DENOMINATOR_FLOOR,
    )
    mean = toolkit.dct_mean(cleaned, width, **options)
    variance = toolkit.dct_variance(cleaned, width, **options)
    std = toolkit.dct_std(cleaned, width, **options)
    denominator = toolkit.smooth_cartesian(valid.astype(float), width, kernel_type="boxcar_discrete")
    safe = np.where(valid, data, 0.0)
    second = np.full(data.shape, np.nan)
    np.divide(
        toolkit.smooth_cartesian(safe**2, width, kernel_type="boxcar_discrete"),
        denominator, out=second,
        where=np.isfinite(denominator) & (denominator > DENOMINATOR_FLOOR),
    )
    return {
        "mean": mean, "variance": variance, "std": std,
        "support": toolkit.dct_count(
            valid, width, kernel_type="boxcar_discrete", restore_input_nan=False,
        ),
        "denominator": denominator,
        "raw_moment_variance": second - mean**2,
    }


def methods(toolkit: Any) -> dict[str, Callable[[np.ndarray, int], dict[str, np.ndarray]]]:
    return {
        REFERENCE: numpy_moments,
        "toolkit_dct_direct": lambda data, width: toolkit_moments(data, width, toolkit),
        "scipy_uniform_reflect": scipy_moments,
        "scipy_fft_symmetric_linear": fft_moments,
    }


def small_tests(toolkit: Any) -> int:
    """Fast, deterministic regression checks; failures abort before full runs."""
    reference = numpy_moments(np.arange(5.0), 3)
    np.testing.assert_allclose(reference["mean"], [1/3, 1, 2, 3, 11/3], atol=1e-14)
    np.testing.assert_allclose(reference["variance"], [2/9, 2/3, 2/3, 2/3, 2/9], atol=1e-14)
    np.testing.assert_array_equal(reference["support"], np.full(5, 3.0))
    count = 3

    floor = np.array([0.0, DENOMINATOR_FLOOR, np.nextafter(DENOMINATOR_FLOOR, np.inf)])
    strict = normalized_moments(floor, floor, floor, 3)
    np.testing.assert_array_equal(np.isfinite(strict["mean"]), [False, False, True])
    count += 1
    singleton = numpy_moments(np.array([np.nan, 2.0, np.nan, np.nan, np.nan]), 3)
    np.testing.assert_allclose(singleton["mean"], [2, 2, 2, np.nan, np.nan], equal_nan=True)
    np.testing.assert_array_equal(singleton["support"], [1, 1, 1, 0, 0])
    count += 2

    probes = [
        np.arange(9.0), np.full(9, 2.0),
        np.array([-2.0] * 4 + [6.0] * 5),
        np.array([0, np.nan, np.inf, -np.inf, 4, 2, np.nan, 0, 1.0]),
        np.array([np.nan] * 7 + [0.0, 1.0]), np.full(9, np.nan),
    ]
    for width in (1, 3, 5):
        for data in probes:
            expected = numpy_moments(data, width)
            for name, calculate in methods(toolkit).items():
                actual = calculate(data, width)
                for statistic, (atol, rtol) in TOLERANCES.items():
                    np.testing.assert_allclose(
                        actual[statistic], expected[statistic], atol=atol, rtol=rtol,
                        equal_nan=True, err_msg=f"{name}, width={width}, {statistic}",
                    )
                    count += 1
                if np.all(~np.isfinite(data)):
                    np.testing.assert_array_equal(actual["support"], np.zeros(data.size))
                    count += 1
    selected = np.ones(2, dtype=bool)
    for actual, expected, status in (
        ([np.nan, np.nan], [np.nan, np.nan], "no_comparable"),
        ([0.0, np.nan], [0.0, 0.0], "fail"),
        ([0.0, 1e-6], [0.0, 0.0], "fail"),
    ):
        row = comparison_row(
            "test", "test", "test", "test", "variance", "all", selected,
            np.asarray(actual), np.asarray(expected),
        )
        assert row["status"] == status, row
        count += 1
    return count


def make_cases(mode: str, master_seed: int):
    x = np.linspace(0.0, 10.0, N)
    truths = {
        "trend_sine": 1.2 * x + 2.5 * np.sin(1.5 * x),
        "endpoint_curvature": 0.20 * (x - 5.0)**2 + 2.0 * np.cos(np.pi * x / 10.0),
        "abrupt_step": np.where(x < 5.0, -2.0, 6.0),
    }
    for signal, (signal_id, formula, noise_std) in SIGNALS.items():
        for replicate in range(5 if mode == "full" else 1):
            # Stable named identities avoid dependence on loop ordering or RNG calls
            # in another signal. Noise and missingness are independently spawned.
            sequence = np.random.SeedSequence([master_seed, signal_id, replicate])
            noise_sequence, mask_sequence = sequence.spawn(2)
            complete = truths[signal] + np.random.default_rng(noise_sequence).normal(0, noise_std, N)
            random_missing = np.random.default_rng(mask_sequence).random(N) < 0.20
            for pattern in MASKS:
                missing = np.zeros(N, dtype=bool)
                if pattern == "random_nan":
                    missing = random_missing.copy()
                elif pattern == "middle_gap":
                    missing[80:120] = True
                elif pattern == "boundary_gap":
                    missing[:40] = True
                elif pattern == "all_missing":
                    missing[:] = True
                data = np.where(missing, np.nan, complete)
                case_id = f"{signal}_seed{replicate}_{pattern}_n{N}_m{WIDTH}"
                settings = {
                    "mode": (mode, ""), "signal": (signal, ""),
                    "signal_formula": (formula, ""),
                    "signal_formula_source": (
                        "generate_1d_boundary_data.py inspected, not imported; only trend_sine inherited", "",
                    ),
                    "signal_units": ("arbitrary signal_unit", ""),
                    "array_size": (N, "cells"), "dtype": ("float64", ""),
                    "x_min": (0.0, "x_unit"),
                    "x_max": (10.0, "x_unit"), "x_spacing": (10.0 / (N - 1), "x_unit"),
                    "grid_endpoints": ("both included", ""),
                    "kernel": ("centered odd discrete boxcar", ""),
                    "requested_width": (WIDTH, "cells"), "discrete_width": (WIDTH, "cells"),
                    "kernel_weight": (1.0 / WIDTH, "1"),
                    "boundary": ("half-sample symmetric: NumPy symmetric / SciPy reflect", ""),
                    "fft_boundary": ("symmetric pad by M//2; linear fftconvolve; valid crop", ""),
                    "edge_region": ("first and last M//2=7 cells (14 total)", ""),
                    "gap_region": ("originally missing input cells; may overlap edge region", ""),
                    "noise_std": (noise_std, "signal_unit"),
                    "noise_distribution": ("independent Normal(0,noise_std); step intentionally noiseless", ""),
                    "master_seed": (master_seed, ""), "signal_seed_id": (signal_id, ""),
                    "replicate": (replicate, ""), "rng": ("NumPy default_rng / PCG64", ""),
                    "seed_sequence_entropy": (f"[{master_seed},{signal_id},{replicate}]", ""),
                    "noise_spawn_key": (str(noise_sequence.spawn_key), ""),
                    "mask_spawn_key": (str(mask_sequence.spawn_key), ""),
                    "noise_reuse": ("same realization across the five mask patterns", ""),
                    "missingness_pattern": (pattern, ""),
                    "random_missing_probability": (0.20 if pattern == "random_nan" else "not applicable", "1"),
                    "gap_interval": ({"middle_gap": "[80,120)", "boundary_gap": "[0,40)"}.get(pattern, "not applicable"), "index"),
                    "n_original_observed": (int((~missing).sum()), "cells"),
                    "n_missing_inputs": (int(missing.sum()), "cells"),
                    "actual_missing_fraction": (float(missing.mean()), "1"),
                    "valid_mask": ("np.isfinite(input); invalids explicitly replaced by zero for filtering", ""),
                    "denominator_rule": ("finite raw mask response strictly > floor; no additive epsilon", ""),
                    "denominator_floor": (DENOMINATOR_FLOOR, "1"),
                    "gap_filling": ("disabled; eligibility reported, never executed", ""),
                    "fallback_eligibility": ("denominator not accepted and any original finite observation in 1D domain", ""),
                    "restore_input_nan": (False, ""),
                    "toolkit_controls": ("_stable_fallback=False; _denominator_floor=1e-3", ""),
                    "variance_convention": ("population ddof=0; negative raw-moment variance clamped at zero", ""),
                    "numpy_variance_reference": ("explicit centered squared residuals over valid reflected window positions", ""),
                    "support_definition": ("clip(raw mask response,0,1)*M; not statistical ESS or distinct observations", ""),
                    "comparison_target": ("numerical equivalence on identical incomplete data; not statistical recovery accuracy", ""),
                    "std_tolerance_reason": ("atol=sqrt(variance_atol) near zero; variance separately checked", ""),
                    "input_sha256": (hashlib.sha256(data.tobytes()).hexdigest(), ""),
                    "mask_sha256": (hashlib.sha256(missing.tobytes()).hexdigest(), ""),
                }
                for statistic, (atol, rtol) in TOLERANCES.items():
                    settings[f"{statistic}_atol"] = (atol, UNITS[statistic])
                    settings[f"{statistic}_rtol"] = (rtol, "1")
                yield case_id, signal, data, settings


def regions(data: np.ndarray) -> dict[str, np.ndarray]:
    edge = np.zeros(data.size, dtype=bool)
    half = WIDTH // 2
    edge[:half] = True
    edge[-half:] = True
    return {"all": np.ones(data.size, dtype=bool), "edge": edge, "gap": ~np.isfinite(data)}


def comparison_row(
    run_id: str, case_id: str, signal: str, method: str, statistic: str,
    region: str, selected: np.ndarray, actual: np.ndarray, expected: np.ndarray,
) -> dict[str, Any]:
    a, b = actual[selected], expected[selected]
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    joint = finite_a & finite_b
    difference = a[joint] - b[joint]
    atol, rtol = TOLERANCES[statistic]
    same_defined = np.array_equal(finite_a, finite_b)
    no_infinities = not (np.isinf(a).any() or np.isinf(b).any())
    agreement = np.all(np.abs(difference) <= atol + rtol * np.abs(b[joint]))
    status = "fail" if not (same_defined and no_infinities and agreement) else (
        "pass" if joint.any() else "no_comparable"
    )
    return {
        "run_id": run_id, "case_id": case_id, "method": method,
        "reference_method": REFERENCE, "component": signal, "statistic": statistic,
        "region": region, "estimate_type": "direct_normalized_moments" if statistic != "support" else "nominal_weighted_support",
        "kernel_match": "exact_discrete_boxcar", "n_considered": int(selected.sum()),
        "n_compared": int(joint.sum()), "n_method_undefined": int((~finite_a).sum()),
        "n_reference_undefined": int((~finite_b).sum()),
        "n_method_gap_filled": 0, "n_reference_gap_filled": 0,
        "max_abs_difference": float(np.max(np.abs(difference))) if joint.any() else "",
        "rmse": float(np.sqrt(np.mean(difference**2))) if joint.any() else "",
        "atol": atol, "rtol": rtol, "status": status, "units": UNITS[statistic],
        "notes": (
            "No fallback; original-mask restoration disabled. Ratios at missing inputs are not observations. "
            "Defined masks must match; matching all-NaN moments yield no_comparable. "
            "NumPy variance uses centered ddof=0 residuals; variance checked independently of std. "
            "Numerical agreement is not statistical recovery accuracy."
        ),
    }


def extrema(values: np.ndarray) -> tuple[float | str, float | str]:
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else ("", "")


def coverage_row(
    run_id: str, case_id: str, method: str, region: str,
    selected: np.ndarray, data: np.ndarray, result: dict[str, np.ndarray],
) -> dict[str, Any]:
    original = np.isfinite(data)
    direct = np.isfinite(result["mean"]) & np.isfinite(result["variance"]) & np.isfinite(result["std"])
    denominator = result["denominator"]
    accepted = np.isfinite(denominator) & (denominator > DENOMINATOR_FLOOR)
    fallback_eligible = ~accepted & bool(original.any())
    d_min, d_max = extrema(denominator[selected])
    s_min, s_max = extrema(result["support"][selected])
    raw_variance = result["raw_moment_variance"][selected]
    v_min, _ = extrema(raw_variance)
    return dict(zip(COVERAGE_FIELDS, (
        run_id, case_id, method, region, int(selected.sum()),
        int((selected & original).sum()), int((selected & ~original).sum()),
        int((selected & direct).sum()), int((selected & direct & ~original).sum()),
        int((selected & ~direct).sum()), int((selected & fallback_eligible).sum()),
        0, d_min, d_max, s_min, s_max, v_min, int((raw_variance < 0.0).sum()),
        "Eligibility is diagnostic only; no fallback executed. All-missing domain is ineligible. "
        "Raw-moment cancellation is recorded before clamping; NumPy comparison variance uses centered residuals.",
    )))


FIGURE_RUN_ID = "lead-review-1d-20261027"
FIGURE_RUN_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "results" / "round1" / "1d_boxcar" / FIGURE_RUN_ID
)
FIGURE_CASE_IDS = (
    "trend_sine_seed0_middle_gap_n200_m15",
    "trend_sine_seed0_boundary_gap_n200_m15",
)
FIGURE_STEM = "boxcar_missingness"
FIGURE_SIZE = (6.5, 6.2)
FIGURE_Y_LIMITS = ((-2.0, 18.0), (-0.08, 2.4), (-0.06, 1.06))


def _figure_csv_rows(path: Path, fields) -> list[dict[str, str]]:
    """Read the frozen evidence, without creating a validation run directory."""
    import csv

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError(f"Unexpected saved CSV columns: {path}")
        return [row for row in reader if row["case_id"] in FIGURE_CASE_IDS]


def prepare_missingness_figure(saved_run: Path = FIGURE_RUN_DIRECTORY):
    """Reproduce only the approved cases, checking identity and fixed tolerances.

    Saved numerical CSVs remain read-only. A changed seed, mask, case setting,
    defined count or comparison policy is an error, not a new figure experiment.
    """
    from _review_common import SETTINGS_FIELDS, TOOLKIT_COMMIT, _RELEASE_HASHES

    settings_rows = _figure_csv_rows(saved_run / "settings.csv", SETTINGS_FIELDS)
    coverage_rows = _figure_csv_rows(saved_run / "coverage.csv", COVERAGE_FIELDS)
    comparison_rows = _figure_csv_rows(saved_run / "comparison.csv", COMPARISON_FIELDS)
    if any(row["run_id"] != FIGURE_RUN_ID
           for row in settings_rows + coverage_rows + comparison_rows):
        raise ValueError("Figure evidence must come from the approved saved run.")
    saved_settings = {case_id: {} for case_id in FIGURE_CASE_IDS}
    for row in settings_rows:
        values = saved_settings[row["case_id"]]
        if row["setting"] in values:
            raise ValueError("Duplicate saved figure-case setting.")
        values[row["setting"]] = row["value"]

    selected = []
    for case_id, signal, data, settings in make_cases("full", MASTER_SEED):
        if case_id in FIGURE_CASE_IDS:
            frozen = saved_settings[case_id]
            for name, (value, _) in settings.items():
                if frozen.get(name) != str(value):
                    raise ValueError(f"Saved figure case mismatch: {case_id}, {name}")
            if (frozen.get("dct_toolkit_commit") != TOOLKIT_COMMIT
                    or frozen.get("dct_toolkit_version") != "1.0.0"):
                raise ValueError("Saved figure case does not use the approved toolkit release.")
            selected.append(dict(case_id=case_id, signal=signal, data=data, settings=settings))
        # Stop within replicate 0: no other signal/replicate is calculated.
        if len(selected) == len(FIGURE_CASE_IDS):
            break
    if tuple(case["case_id"] for case in selected) != FIGURE_CASE_IDS:
        raise ValueError("The two approved figure cases were not reproduced.")

    toolkit = load_toolkit()  # Checks all nine released Python source hashes.
    metadata = {
        "purpose": "Direct reflective boxcar mean, standard deviation and local data coverage near missing intervals",
        "figure_kind": "deterministic export of two approved saved cases; no new experiment or benchmark",
        "selected_run": FIGURE_RUN_ID,
        "selected_run_directory": str(saved_run.resolve()),
        "selected_cases": list(FIGURE_CASE_IDS),
        "master_seed": MASTER_SEED, "replicate": 0, "array_size": N,
        "kernel": "centered odd discrete boxcar", "requested_width": WIDTH,
        "discrete_width": WIDTH, "boundary": "half-sample reflective (NumPy symmetric)",
        "denominator_floor": DENOMINATOR_FLOOR,
        "denominator_rule": "finite raw d strictly > floor; no additive epsilon",
        "completion": False, "restoration": False,
        "coverage_definition": "rho=clip(raw denominator,0,1); not ratio-gated",
        "equivalent_count": "N_eq=15*rho; not statistical ESS or distinct observations",
        "secondary_count_axis": "omitted to avoid crowding; rho is primary",
        "reference": "independent NumPy symmetric windows; centered population variance (ddof=0)",
        "dct_toolkit_commit": TOOLKIT_COMMIT, "dct_toolkit_version": toolkit.__version__,
        "toolkit_source_verification": "all nine released Python files checked by load_toolkit",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "common_helper_sha256": hashlib.sha256(Path(__file__).with_name("_review_common.py").read_bytes()).hexdigest(),
        "x_limits": [0, 10], "row_y_limits": list(FIGURE_Y_LIMITS),
        "panel_order": "columns: middle_gap, boundary_gap; rows: mean/observations, std, rho",
        "display": "gray observed points; blue solid DCT; orange dashed NumPy; one shared legend",
        "missing_input_shading": "light gray across all rows; cell edges at adjacent coordinate midpoints, clipped to [0,10]",
        "undefined_moments": "NaN segments left blank; no interpolation, bridging or arbitrary fill",
        "truth_display": "none; numerical agreement, not recovery or denoising accuracy",
    }
    for name, digest in _RELEASE_HASHES.items():
        metadata[f"toolkit_source_sha256.{name}"] = digest
    for filename in ("settings.csv", "coverage.csv", "comparison.csv"):
        metadata[f"saved_{filename}_sha256"] = hashlib.sha256((saved_run / filename).read_bytes()).hexdigest()

    count_fields = (
        "n_considered", "n_original_observed", "n_missing_inputs", "n_direct_defined",
        "n_direct_defined_at_missing_inputs", "n_undefined", "n_fallback_eligible", "n_gap_filled",
    )
    comparison_fields = (
        "reference_method", "component", "estimate_type", "kernel_match", "n_considered",
        "n_compared", "n_method_undefined", "n_reference_undefined", "n_method_gap_filled",
        "n_reference_gap_filled", "status", "units",
    )
    for case in selected:
        case_id, data = case["case_id"], case["data"]
        case["dct"] = toolkit_moments(data, WIDTH, toolkit)
        case["numpy"] = numpy_moments(data, WIDTH)
        for method, result in (("toolkit_dct_direct", case["dct"]), (REFERENCE, case["numpy"])):
            accepted = np.isfinite(result["denominator"]) & (result["denominator"] > DENOMINATOR_FLOOR)
            for statistic in ("mean", "variance", "std"):
                if not np.array_equal(np.isnan(result[statistic]), ~accepted):
                    raise ValueError(f"Undefined figure moments changed: {case_id}, {method}, {statistic}")
            atol, rtol = TOLERANCES["support"]
            if (not np.isfinite(result["denominator"]).all()
                    or not np.allclose(result["support"], WIDTH * np.clip(result["denominator"], 0, 1),
                                       atol=atol, rtol=rtol)):
                raise ValueError(f"Figure coverage/count mapping changed: {case_id}, {method}")
            for region, mask in regions(data).items():
                rows = [row for row in coverage_rows if row["case_id"] == case_id
                        and row["method"] == method and row["region"] == region]
                actual = coverage_row(FIGURE_RUN_ID, case_id, method, region, mask, data, result)
                if len(rows) != 1 or any(int(rows[0][name]) != actual[name] for name in count_fields):
                    raise ValueError(f"Saved figure coverage mismatch: {case_id}, {method}, {region}")
        for region, mask in regions(data).items():
            for statistic, (atol, rtol) in TOLERANCES.items():
                row = comparison_row(
                    FIGURE_RUN_ID, case_id, case["signal"], "toolkit_dct_direct", statistic,
                    region, mask, case["dct"][statistic], case["numpy"][statistic],
                )
                rows = [old for old in comparison_rows if old["case_id"] == case_id
                        and old["method"] == "toolkit_dct_direct" and old["region"] == region
                        and old["statistic"] == statistic]
                if (row["status"] != "pass" or len(rows) != 1
                         or any(rows[0][name] != str(row[name]) for name in comparison_fields)
                         or float(rows[0]["atol"]) != atol or float(rows[0]["rtol"]) != rtol):
                    raise ValueError(f"Saved/fixed-tolerance figure comparison failed: {case_id}, {region}, {statistic}")
                for metric in ("max_abs_difference", "rmse"):
                    if not np.isclose(float(rows[0][metric]), row[metric], atol=atol, rtol=rtol):
                        raise ValueError(f"Saved figure discrepancy changed: {case_id}, {region}, {statistic}, {metric}")
                metadata[f"{case_id}.{region}.{statistic}.max_abs_difference"] = row["max_abs_difference"]
                metadata[f"{case_id}.{region}.{statistic}.rmse"] = row["rmse"]
        frozen = saved_settings[case_id]
        for name in ("input_sha256", "mask_sha256", "seed_sequence_entropy", "noise_spawn_key",
                     "mask_spawn_key", "gap_interval", "signal_formula", "noise_std"):
            metadata[f"{case_id}.{name}"] = frozen[name]
        metadata[f"{case_id}.saved_script_sha256"] = frozen["script_sha256"]
        summary = coverage_row(
            FIGURE_RUN_ID, case_id, "toolkit_dct_direct", "all",
            np.ones(N, dtype=bool), data, case["dct"],
        )
        for name in count_fields:
            metadata[f"{case_id}.{name}"] = summary[name]
    for statistic, (atol, rtol) in TOLERANCES.items():
        metadata[f"{statistic}_atol"] = atol
        metadata[f"{statistic}_rtol"] = rtol
    metadata["n_regional_comparisons_verified"] = len(selected) * len(regions(selected[0]["data"])) * len(TOLERANCES)
    metadata["checks"] = (
        "PASS: saved run/case/settings/input/mask identity; all nine released source hashes; "
        "24 saved regional comparisons and defined counts; unchanged mean/variance/std/support tolerances; "
        "rho=clip(raw denominator,0,1) and saved support=N_eq=15*rho; "
        "matching DCT/NumPy defined masks; rejected moments remain NaN; no completion"
    )
    return selected, metadata


def create_missingness_figure(cases):
    """Draw the checked arrays without changing undefined moments or the mask."""
    from _review_figures import configure_plotting

    configure_plotting(use_agg=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    fig, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE, sharex=True, sharey="row")
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.10, top=0.875, hspace=0.40, wspace=0.16)
    x = np.linspace(0.0, 10.0, N)
    edges = np.r_[0.0, (x[:-1] + x[1:]) / 2, 10.0]
    titles = (("(a) Mean: middle gap", "(b) Mean: boundary gap"),
              ("(c) Standard deviation", "(d) Standard deviation"),
              ("(e) Local data coverage", "(f) Local data coverage"))
    for column, case in enumerate(cases):
        missing = np.flatnonzero(~np.isfinite(case["data"]))
        for row, ax in enumerate(axes[:, column]):
            ax.axvspan(edges[missing[0]], edges[missing[-1] + 1], color="#E9E9E9",
                       linewidth=0, zorder=0, gid="original_missing_interval")
            ax.set(xlim=(0, 10), ylim=FIGURE_Y_LIMITS[row], xticks=np.arange(0, 11, 2))
            ax.set_title(titles[row][column], loc="left", fontweight="bold", pad=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=3, pad=3)
        observed, = axes[0, column].plot(x, case["data"], linestyle="none", marker="o",
                                        markersize=2.6, color="#777777", alpha=0.65, gid="observed")
        for row, statistic in enumerate(("mean", "std")):
            dct, = axes[row, column].plot(x, case["dct"][statistic], color="#0072B2",
                                         linewidth=1.8, gid=f"dct_{statistic}")
            spatial, = axes[row, column].plot(x, case["numpy"][statistic], color="#D55E00",
                                             linewidth=1.2, linestyle=(0, (4, 3)), gid=f"numpy_{statistic}")
        axes[2, column].plot(x, np.clip(case["dct"]["denominator"], 0, 1),
                              color="#0072B2", linewidth=1.8, gid="dct_rho")
        axes[2, column].plot(x, np.clip(case["numpy"]["denominator"], 0, 1),
                              color="#D55E00", linewidth=1.2, linestyle=(0, (4, 3)), gid="numpy_rho")
        axes[2, column].set_xlabel(r"$t$ (arbitrary coordinate units)")
    for row, (label, ticks) in enumerate((("Signal (a.u.)", [0, 5, 10, 15]),
                                         ("Std (a.u.)", [0, 1, 2]),
                                         (r"Coverage, $\rho$", [0, 0.5, 1]))):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_yticks(ticks)
    fig.legend([observed, dct, spatial, Patch(facecolor="#E9E9E9", edgecolor="none")],
               ["Observed", "DCT", "NumPy spatial", "Missing input"],
                loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=4,
                columnspacing=1.1, handlelength=1.8, handletextpad=0.45)
    # The manuscript linewidth slightly reduces this 6.5-in canvas. Keep even
    # its smallest labels above 10 pt after that final-size scaling.
    for text in fig.findobj(Text):
        if text.get_fontsize() < 10.1:
            text.set_fontsize(10.1)
    return fig


def export_missingness_figure(output: Path | None = None) -> dict[str, Path]:
    """Lazy, explicit figure export; never create or overwrite numerical CSVs."""
    from _review_figures import DEFAULT_FIGURE_ROOT, export_figure

    output = (output if output is not None else DEFAULT_FIGURE_ROOT / "1d_boxcar").expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1] / "results"):
        raise ValueError("Figure output must not be inside saved experiment results.")
    cases, metadata = prepare_missingness_figure()
    fig = create_missingness_figure(cases)
    import matplotlib.pyplot as plt

    try:
        return export_figure(fig, output, FIGURE_STEM, metadata)
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full", help="one or five independent seed replicates per signal")
    parser.add_argument("--run-id", help="new run subfolder name (existing runs are never overwritten)")
    parser.add_argument("--output-root", type=Path, help="round root; creates <root>/1d_boxcar/<run-id>/")
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--self-test-only", action="store_true", help="run small checks without creating output files")
    parser.add_argument("--figure-only", action="store_true", help="verify and export only the two approved saved cases; no validation CSVs")
    parser.add_argument("--figure-output", type=Path, help="figure case directory (default: manuscript/figures/round1/1d_boxcar)")
    args = parser.parse_args(argv)
    if args.master_seed < 0:
        parser.error("--master-seed must be nonnegative")
    if args.figure_output is not None and not args.figure_only:
        parser.error("--figure-output requires --figure-only")
    if args.figure_only:
        if (args.master_seed != MASTER_SEED or args.mode != "full" or args.self_test_only
                or args.run_id is not None or args.output_root is not None):
            parser.error("--figure-only uses the approved fixed cases/seed; do not combine it with validation options")
        for kind, path in export_missingness_figure(args.figure_output).items():
            print(f"{kind}: {path}")
        return 0
    toolkit = load_toolkit()
    n_tests = small_tests(toolkit)
    print(f"Passed {n_tests} small numerical checks.")
    if args.self_test_only:
        return 0
    run_id, output = create_run_directory("1d_boxcar", args.run_id, args.output_root)
    comparisons, coverage, case_settings = [], [], {}
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for case_id, signal, data, settings in make_cases(args.mode, args.master_seed):
        settings.update({
            "script": (str(Path(__file__).resolve()), ""),
            "script_sha256": (script_hash, ""), "small_checks_passed": (n_tests, "checks"),
        })
        case_settings[case_id] = settings
        outputs = {name: calculate(data, WIDTH) for name, calculate in methods(toolkit).items()}
        for name, result in outputs.items():
            for region, selected in regions(data).items():
                coverage.append(coverage_row(run_id, case_id, name, region, selected, data, result))
                if name == REFERENCE:
                    continue
                for statistic in TOLERANCES:
                    comparisons.append(comparison_row(
                        run_id, case_id, signal, name, statistic, region,
                        selected, result[statistic], outputs[REFERENCE][statistic],
                    ))
    write_csv(output / "comparison.csv", COMPARISON_FIELDS, comparisons)
    write_csv(output / "coverage.csv", COVERAGE_FIELDS, coverage)
    write_settings(output / "settings.csv", run_id, case_settings)
    failed = [row for row in comparisons if row["status"] == "fail"]
    for statistic in TOLERANCES:
        errors = [row["max_abs_difference"] for row in comparisons if row["statistic"] == statistic and row["n_compared"]]
        print(f"{statistic}: max absolute difference = {max(errors, default=float('nan')):.6g}")
    print(f"{len(case_settings)} cases; {len(comparisons)} comparisons; {len(failed)} failures.")
    print(f"CSV results: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
