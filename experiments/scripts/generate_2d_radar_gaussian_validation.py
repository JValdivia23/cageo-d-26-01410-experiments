#!/usr/bin/env python3
"""Single-sweep, direct-ratio Gaussian radar validation; opt-in figure export.

The real input is --radar-path, then REAL_RADAR_PATH, then the file documented
in experiments/data/README.md. Missing input is an error, never a request for
sample/synthetic radar data. --quick retains all 720 rays but only the first
64 gates AFTER extracting/validating sweep 12; it is not full-sweep evidence
and never runs timings. --self-test-only explicitly tests small artificial
arrays without reading radar; these are implementation tests, not radar data.
Outputs are CSV by default; --save-support-arrays additionally saves the raw
denominator arrays and original validity masks in raw_support.npz.

Example (set the verified v1.0.0 source/dependency environment first)::

    python experiments/scripts/generate_2d_radar_gaussian_validation.py \
        --quick --skip-timing --run-id radar-quick
    python experiments/scripts/generate_2d_radar_gaussian_validation.py \
        --skip-timing --run-id radar-correctness
    python experiments/scripts/generate_2d_radar_gaussian_validation.py \
        --repeats 3 --run-id radar-full

The spatial reference uses no toolkit responses, impulse filtering, FFTs or
DCTs. It derives ALL finite-grid convolution weights by direct cosine sums of
the analytic Gaussian response, and applies scipy.ndimage.convolve1d. Range
uses half-sample symmetric extension (mode='reflect'), NOT mode='mirror'.
Finite analytic responses can induce signed weights; no tail truncation,
positivity correction, or renormalization is applied. All normalizations
follow the COMPLETE azimuth-first/range-second operator. Completion/fallback
is disabled, including in denominator-floor sensitivity and mean-only timing.
This is numerical operator agreement, not validation against unknown radar
truth, a default-completion test, or a predecessor-method comparison.

comparison.csv contains the primary checks at the released default floor.
threshold_sensitivity.csv retains every nondefault-floor check, including
fixed-tolerance failures when a small denominator amplifies roundoff. Such
failures are not removed or given looser tolerances. --strict-sensitivity also
makes them produce a nonzero exit; normal timing requires the default checks
and mask-response/acceptance checks to pass.

There is no ordinary sampled/truncated SciPy Gaussian comparison: its weights
do not match this finite analytic response. The exact spatial calculation is
retained for numerical validation only, not as a practical speed baseline.
Optional mean-only timings measure the spectral method alone; no spatial
speed ratio is calculated. Earlier comparison outputs are preserved separately.

--figure-only reproduces ONLY the approved natural-mask W15 full-sweep case,
checking the frozen settings/diagnostics before drawing six native-grid panels.
It does not run self-tests, spatial references, sensitivity tests or timings,
and never creates a numerical run directory. --figure-output selects only the
figure directory; REAL_RADAR_PATH may locate the documented, hash-checked input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.ndimage import convolve1d

from _review_common import (
    BENCHMARK_FIELDS,
    COMPARISON_FIELDS,
    SUMMARY_FIELDS,
    create_run_directory,
    load_toolkit,
    write_csv,
    write_settings,
)


DEFAULT_RADAR = Path(
    "~/onedrive/code/research/smoothing/"
    "cfrad_QCed_20220223_023027.051_to_20220223_023410.833_COW1high_SUR.nc"
)
DOCUMENTED_SHA256 = "f1e45c26da3002c31cfea1e62ece5a40c18a668d374193205daf793f41a4fae0"
FLOORS = (1e-5, 1e-4, 1e-3, 1e-2)
WIDTHS = (5.0, 15.0)
DEFAULT_FLOOR = 1e-3
SPECTRAL = "toolkit_v1_polar_direct"
SPATIAL = "spatial_full_analytic_weights"
OPERATION = "mean_only_direct_no_fallback_kernel_construction_included"
# Fixed before comparisons. Std has a distinct absolute tolerance because the
# square root is ill-conditioned at a clamped zero variance. No tolerance is
# fitted to a radar run. Raw moment difference is also compared, before clamp.
TOLERANCES = {
    "mean": (2e-8, 2e-9),
    "second_moment": (2e-6, 2e-9),
    "raw_variance": (2e-6, 2e-9),
    "variance": (2e-6, 2e-9),
    "std": (2e-6, 2e-9),
    "support": (2e-9, 2e-11),
    "raw_mask_response": (2e-12, 2e-11),
    "direct_acceptance": (0.0, 0.0),
    "fallback_eligibility": (0.0, 0.0),
    "operator": (2e-12, 2e-12),
}
UNITS = {
    "mean": "dBZ", "second_moment": "dBZ^2", "raw_variance": "dBZ^2",
    "variance": "dBZ^2", "std": "dBZ", "support": "nominal_grid_support",
    "raw_mask_response": "1", "direct_acceptance": "1",
    "fallback_eligibility": "1", "operator": "input_units",
}
DIAGNOSTIC_FIELDS = (
    "run_id", "case_id", "method", "region", "n_considered", "n_input_valid",
    "n_direct_accepted", "direct_acceptance_fraction", "n_denominator_rejected",
    "n_fallback_eligible", "fallback_eligible_fraction", "n_gap_filled",
    "n_direct_at_original_missing", "raw_support_min", "raw_support_q25",
    "raw_support_median", "raw_support_q75", "raw_support_max",
    "n_raw_support_negative", "n_raw_support_above_one", "n_negative_raw_variance",
)
RADIAL_FIELDS = (
    "run_id", "case_id", "method", "gate_index_zero_based", "range_m",
    "azimuth_width_bins", "nominal_support_factor", "n_input_valid",
    "raw_support_min", "raw_support_median", "raw_support_max",
    "n_direct_accepted", "n_denominator_rejected", "n_fallback_eligible",
    "n_gap_filled", "n_direct_at_original_missing",
)


def settings_values(**values: Any) -> dict[str, tuple[Any, str]]:
    """Use the shared (value, unit) settings contract, without changing it."""
    return {key: (value, "") for key, value in values.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def azimuth_widths(n_range: int, width: float, az_res_deg: float) -> np.ndarray:
    """Legacy index radius, explicitly NOT a physical-range conversion."""
    return width / (np.arange(1, n_range + 1, dtype=float) * np.deg2rad(az_res_deg))


class SpatialPolarGaussian:
    """Independent full finite-grid spatial realization of the analytic response.

    Periodic length N: h[l] = (1 + 2 sum H[k] cos(2*pi*k*l/N)
    + H[N/2]*cos(pi*l), when N even) / N. The antipodal weight is
    split over the two endpoints of the odd centered convolution vector.

    Reflective length R: an even extension has period 2R. Use the same direct
    inverse cosine sum on that period with H[k]=exp(-(pi*k*sigma/R)^2/2),
    k=0,...,R-1. Set the unrepresented Nyquist H[R] to zero: an input's
    half-sample even extension has zero Nyquist coefficient, so its choice
    cannot change the reflective operator. ALL offsets -R,...,+R are retained,
    splitting the antipodal endpoints. This gives C.T @ diag(H) @ C, where C
    is the orthonormal DCT-II matrix, without using a DCT to build/filter weights.
    """

    def __init__(self, shape: tuple[int, int], width: float, az_res_deg: float):
        n_az, n_range = shape
        self.shape = shape
        self.widths = azimuth_widths(n_range, width, az_res_deg)
        half = n_az // 2
        offsets = np.arange(-half, half + 1, dtype=float)
        positive_modes = np.arange(1, (n_az - 1) // 2 + 1, dtype=float)
        response = np.exp(
            -0.5 * (2 * np.pi * positive_modes[:, None] / n_az
                    * self.widths[None, :] / np.sqrt(12)) ** 2
        )
        cosine = np.cos(2 * np.pi * offsets[:, None] * positive_modes[None, :] / n_az)
        self.az_weights = (1 + 2 * cosine @ response) / n_az
        if n_az % 2 == 0:
            nyquist = np.exp(-0.5 * (np.pi * self.widths / np.sqrt(12)) ** 2)
            self.az_weights += np.cos(np.pi * offsets[:, None]) * nyquist / n_az
            self.az_weights[[0, -1], :] *= 0.5

        offsets = np.arange(-n_range, n_range + 1, dtype=float)
        modes = np.arange(1, n_range, dtype=float)
        response = np.exp(-0.5 * (np.pi * modes / n_range * width / np.sqrt(12)) ** 2)
        cosine = np.cos(np.pi * offsets[:, None] * modes[None, :] / n_range)
        self.range_weights = (1 + 2 * cosine @ response) / (2 * n_range)
        self.range_weights[[0, -1]] *= 0.5

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if values.shape != self.shape:
            raise ValueError("Spatial reference shape differs from its frozen geometry")
        # Exact linear zero-input shortcut avoids billions of useless sums in
        # the all-missing control; it never replaces a nonzero radar workload.
        if not np.any(values):
            return np.zeros(self.shape, dtype=float)
        stage1 = np.empty(self.shape, dtype=float)
        for gate in range(self.shape[1]):
            stage1[:, gate] = convolve1d(
                values[:, gate], self.az_weights[:, gate], mode="wrap", origin=0,
            )
        return convolve1d(stage1, self.range_weights, axis=1, mode="reflect", origin=0)

    def audit(self) -> dict[str, tuple[Any, str]]:
        return settings_values(
            azimuth_weight_vector_length=self.az_weights.shape[0],
            range_weight_vector_length=self.range_weights.size,
            azimuth_min_weight=float(self.az_weights.min()),
            range_min_weight=float(self.range_weights.min()),
            azimuth_negative_weight_count=int(np.count_nonzero(self.az_weights < 0)),
            range_negative_weight_count=int(np.count_nonzero(self.range_weights < 0)),
            azimuth_most_negative_column_mass=float(
                np.minimum(self.az_weights, 0).sum(axis=0).min()),
            range_negative_weight_mass=float(np.minimum(self.range_weights, 0).sum()),
            azimuth_max_constant_preservation_residual=float(
                np.max(np.abs(self.az_weights.sum(axis=0) - 1))),
            range_constant_preservation_residual=float(abs(self.range_weights.sum() - 1)),
            weight_audit_scope="convolution vectors before boundary-index aggregation; "
                               "individual tiny negative weights can be roundoff",
        )


def toolkit_options(az_res_deg: float) -> dict[str, Any]:
    return dict(
        coordinates="polar", kernel_type="gaussian", az_res_deg=az_res_deg,
        az_boundary="periodic", range_boundary="reflective", restore_input_nan=False,
    )


def direct_fields(
    denominator: np.ndarray, numerator: np.ndarray, second_numerator: np.ndarray,
    valid: np.ndarray, width: float, az_res_deg: float, floor: float,
) -> dict[str, np.ndarray]:
    accepted = np.isfinite(denominator) & (denominator > floor)
    mean = np.full(denominator.shape, np.nan)
    moment2 = np.full(denominator.shape, np.nan)
    np.divide(numerator, denominator, out=mean, where=accepted)
    np.divide(second_numerator, denominator, out=moment2, where=accepted)
    raw_var = moment2 - mean**2
    variance = np.maximum(raw_var, 0.0)
    nominal = width * azimuth_widths(valid.shape[1], width, az_res_deg)[None, :]
    return dict(
        mean=mean, second_moment=moment2, raw_variance=raw_var,
        variance=variance, std=np.sqrt(variance), support=np.clip(denominator, 0, 1) * nominal,
        raw_mask_response=denominator, direct_acceptance=accepted.astype(float),
        fallback_eligibility=((~accepted) & bool(np.any(valid))).astype(float),
    )


def toolkit_fields(
    toolkit: Any, data: np.ndarray, width: float, az_res_deg: float, floor: float,
    denominator: np.ndarray,
) -> dict[str, np.ndarray]:
    """Test released mean/variance/std/count APIs, with private direct-only controls."""
    options = toolkit_options(az_res_deg)
    direct = dict(options, _stable_fallback=False, _denominator_floor=floor)
    mean = toolkit.dct_mean(data, width, **direct)
    moment2 = toolkit.dct_mean(data**2, width, **direct)
    variance = toolkit.dct_variance(data, width, **direct)
    std = toolkit.dct_std(data, width, **direct)
    valid = np.isfinite(data)
    accepted = np.isfinite(denominator) & (denominator > floor)
    return dict(
        mean=mean, second_moment=moment2, raw_variance=moment2 - mean**2,
        variance=variance, std=std, support=toolkit.dct_count(valid, width, **options),
        raw_mask_response=denominator, direct_acceptance=accepted.astype(float),
        fallback_eligibility=((~accepted) & bool(np.any(valid))).astype(float),
    )


def comparison_row(
    run_id: str, case_id: str, statistic: str, region: str, selection: np.ndarray,
    result: np.ndarray, reference: np.ndarray, method: str = SPECTRAL,
    reference_method: str = SPATIAL, notes: str = "",
) -> dict[str, Any]:
    a, b = result[selection], reference[selection]
    af, bf = np.isfinite(a), np.isfinite(b)
    joint = af & bf
    atol, rtol = TOLERANCES[statistic]
    diff = a[joint] - b[joint]
    # Undefined-mask mismatch must fail even if all jointly defined cells agree.
    same_defined = np.array_equal(af, bf)
    no_infinities = not (np.isinf(a).any() or np.isinf(b).any())
    good = same_defined and no_infinities and np.all(np.abs(diff) <= atol + rtol * np.abs(b[joint]))
    if not good:
        status = "fail"
    elif not np.any(joint):
        status = "no_comparable"
    else:
        status = "pass"
    return dict(
        run_id=run_id, case_id=case_id, method=method, reference_method=reference_method,
        component="DBZHC" if not case_id.startswith("selftest") else "explicit_artificial_test",
        statistic=statistic, region=region, estimate_type="direct_only_no_completion",
        kernel_match="full_finite_analytic_response_matched_no_truncation",
        n_considered=a.size, n_compared=int(joint.sum()),
        n_method_undefined=int((~af).sum()), n_reference_undefined=int((~bf).sum()),
        n_method_gap_filled=0, n_reference_gap_filled=0,
        max_abs_difference=float(np.max(np.abs(diff))) if diff.size else "",
        rmse=float(np.sqrt(np.mean(diff**2))) if diff.size else "",
        atol=atol, rtol=rtol, status=status, units=UNITS[statistic],
        notes=notes + " Undefined-mask equality is required; gap_filled counts completion, "
                      "not direct ratios evaluated at originally missing cells.",
    )


def dense_small_operator(values: np.ndarray, width: float, az_res_deg: float) -> np.ndarray:
    """Third, small-array oracle: explicit Fourier/cosine matrices, no transforms."""
    n_az, n_range = values.shape
    delta = np.arange(n_az)[:, None] - np.arange(n_az)[None, :]
    modes = np.arange(1, (n_az - 1) // 2 + 1)
    stage1 = np.empty_like(values)
    for gate, effective_width in enumerate(azimuth_widths(n_range, width, az_res_deg)):
        response = np.exp(-0.5 * (2 * np.pi * modes / n_az * effective_width / np.sqrt(12))**2)
        matrix = (1 + 2 * np.sum(
            np.cos(2 * np.pi * delta[:, :, None] * modes / n_az) * response, axis=2,
        )) / n_az
        if n_az % 2 == 0:
            matrix += (np.exp(-0.5 * (np.pi * effective_width / np.sqrt(12))**2)
                       * np.cos(np.pi * delta) / n_az)
        stage1[:, gate] = matrix @ values[:, gate]
    k = np.arange(n_range)
    c = np.sqrt(2 / n_range) * np.cos(np.pi * k[:, None] * (k[None, :] + 0.5) / n_range)
    c[0] /= np.sqrt(2)
    response = np.exp(-0.5 * (np.pi * k / n_range * width / np.sqrt(12))**2)
    range_matrix = c.T @ (response[:, None] * c)
    return stage1 @ range_matrix.T


def support_comparison_contract_test(run_id: str) -> dict[str, tuple[Any, str]]:
    """Support-only tests: matching joint values cannot hide undefined outputs."""
    selection = np.ones((1, 2), dtype=bool)
    controls = (
        ([[1.0, np.nan]], [[1.0, 2.0]], "fail"),
        ([[1.0, 2.0]], [[1.0, np.nan]], "fail"),
        ([[1.0, np.nan]], [[np.nan, 2.0]], "fail"),
        ([[np.nan, np.nan]], [[np.nan, np.nan]], "no_comparable"),
        ([[1.0, 2.0]], [[1.0, 2.0]], "pass"),
    )
    for result, reference, expected in controls:
        row = comparison_row(
            run_id, "selftest_support_comparison_contract", "support", "all", selection,
            np.asarray(result), np.asarray(reference),
        )
        if row["status"] != expected:
            raise RuntimeError(f"Support comparison contract: expected {expected}, got {row['status']}")
    return settings_values(
        support_comparison_contract_status="pass",
        support_comparison_contract_controls=len(controls),
        support_comparison_contract_scope="support-only arrays; undefined masks must match in both "
                                          "directions, including disjoint finite masks; matching "
                                          "all-undefined arrays are no_comparable",
    )


def self_tests(toolkit: Any, run_id: str) -> tuple[list[dict], dict[str, dict]]:
    rows, settings = [], {}
    contract = support_comparison_contract_test(run_id)
    rng = np.random.default_rng(20261027)
    for index, (shape, width) in enumerate((((7, 6), 0.5), ((8, 9), 5.0), ((12, 5), 15.0))):
        case = f"selftest_{index}_odd_even_width{width:g}"
        az_res = 360.0 / shape[0]
        data = rng.normal(10, 2, size=shape)
        data[0, 0] = 0.0  # Finite zero must be included, not treated as missing.
        data[1:3, 2:4] = np.nan
        valid = np.isfinite(data)
        zeroed = np.where(valid, data, 0.0)
        spatial = SpatialPolarGaussian(shape, width, az_res)
        denominator = toolkit.smooth_polar(
            valid.astype(float), width, az_res_deg=az_res, az_boundary="periodic",
        )
        reference = direct_fields(spatial(valid.astype(float)), spatial(zeroed),
                                  spatial(zeroed**2), valid, width, az_res, DEFAULT_FLOOR)
        actual = toolkit_fields(toolkit, data, width, az_res, DEFAULT_FLOOR, denominator)
        selection = np.ones(shape, dtype=bool)
        for statistic in reference:
            rows.append(comparison_row(run_id, case, statistic, "all", selection,
                                       actual[statistic], reference[statistic]))
        dense = dense_small_operator(zeroed, width, az_res)
        rows.append(comparison_row(
            run_id, case, "operator", "all", selection, spatial(zeroed), dense,
            method=SPATIAL, reference_method="explicit_dense_fourier_dct_cosine_matrices",
        ))
        settings[case] = settings_values(
            purpose="explicit artificial implementation self-test; not radar evidence",
            shape=list(shape), width=width, az_res_deg=az_res, rng_seed=20261027,
            valid_zero_count=int(np.count_nonzero(valid & (data == 0))),
            natural_missingness="not applicable; explicit 2x2 test gap",
        ) | spatial.audit() | contract
    return rows, settings


def read_radar(path: Path, quick: bool) -> tuple[np.ndarray, np.ndarray, float, dict]:
    import xarray as xr

    if not path.is_file():
        raise FileNotFoundError(
            f"Real radar input not found: {path}. Supply --radar-path or REAL_RADAR_PATH. "
            "No sample/synthetic substitution is permitted."
        )
    with xr.open_dataset(path, mask_and_scale=True, decode_times=False, decode_timedelta=False) as ds:
        if ds["DBZHC"].dims != ("time", "range"):
            raise ValueError(f"Expected DBZHC(time,range), got {ds['DBZHC'].dims}")
        start = int(ds["sweep_start_ray_index"].values[12])
        end = int(ds["sweep_end_ray_index"].values[12]) + 1
        # Crucially, no DBZHC multi-sweep .values and no filtering before isel.
        field = ds["DBZHC"].isel(time=slice(start, end))
        full = np.asarray(field.values, dtype=np.float64)
        az = np.asarray(ds["azimuth"].isel(time=slice(start, end)).values, dtype=float)
        ranges = np.asarray(ds["range"].values, dtype=float)
        fixed_angle = float(ds["fixed_angle"].values[12])
        fill_value = field.encoding.get("_FillValue")
        if fill_value is None:
            raise ValueError("DBZHC has no recorded _FillValue; decoding cannot be reconfirmed")
        az_steps = np.mod(np.diff(np.r_[az, az[0]]), 360.0)
        az_res = float(np.median(az_steps))
        range_steps = np.diff(ranges)
        nominal_step = float(ds["range"].attrs.get("meters_between_gates", np.median(range_steps)))
        if full.shape != (720, 1180):
            raise ValueError(f"Sweep 12 must reconfirm 720x1180, not {full.shape}")
        if not np.isclose(fixed_angle, 1.9927734, atol=5e-5, rtol=0):
            raise ValueError(f"Unexpected sweep-12 fixed angle {fixed_angle}")
        if not (np.all(np.isfinite(az)) and np.allclose(az_steps, 0.5, atol=5e-4, rtol=0)
                and np.isclose(az_steps.sum(), 360.0, atol=1e-6, rtol=0)):
            raise ValueError("Selected rays do not form the documented ordered complete circle")
        if not (np.all(np.isfinite(ranges)) and np.isclose(ranges[0], 95.0, atol=0.01, rtol=0)
                and np.allclose(range_steps, 74.9414, atol=0.01, rtol=0)):
            raise ValueError("Selected range coordinates do not match documented first gate/spacing")
        if np.any(full == fill_value):
            raise ValueError("Stored _FillValue survived decoding")
        metadata = settings_values(
            radar_path=str(path), radar_input_selection="CLI > REAL_RADAR_PATH > documented file",
            input_bytes=path.stat().st_size, field="DBZHC", field_units=str(field.attrs.get("units", "")),
            field_stored_dtype=str(field.encoding.get("dtype", "")), computation_dtype="float64",
            fill_value=str(fill_value), fill_value_decoding="xarray mask_and_scale=True to NaN",
            mask_rule="np.isfinite(decoded DBZHC); finite zeros valid; no extra QC thresholds",
            upstream_qc="unverified; no QC or cause of missing sectors inferred",
            public_access_redistribution="unverified; exact file remains external",
            instrument_name=str(ds.attrs.get("instrument_name", "")),
            instrument_author_confirmation="COW, C-band; not DOW6",
            sweep_zero_based=12, ray_start_inclusive=start, ray_stop_exclusive=end,
            sweep_selected_before_filtering=True, selected_full_shape=list(full.shape),
            fixed_angle_deg=fixed_angle, first_gate_m=float(ranges[0]),
            nominal_gate_spacing_m=nominal_step, median_gate_spacing_m=float(np.median(range_steps)),
            min_gate_spacing_m=float(range_steps.min()), max_gate_spacing_m=float(range_steps.max()),
            range_grid_policy="retain encoded float32 coordinate quantization; legacy filter uses indices",
            azimuth_spacing_deg=az_res, min_azimuth_step_deg=float(az_steps.min()),
            max_azimuth_step_deg=float(az_steps.max()), azimuth_circle_coverage_deg=float(az_steps.sum()),
            azimuth_first_deg=float(az[0]), azimuth_last_deg=float(az[-1]),
            azimuth_order_policy="preserve acquisition order; no sorting or resampling; nominal median step",
            periodic_seam="array endpoints in acquisition order, not necessarily geographic north",
            full_sweep_finite_count=int(np.isfinite(full).sum()),
            full_sweep_missing_fraction=float(np.mean(~np.isfinite(full))),
            full_sweep_valid_zero_count=int(np.count_nonzero(np.isfinite(full) & (full == 0))),
            full_sweep_decoded_float64_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
            quick_mode=quick, quick_range_gate_count=64 if quick else "not cropped",
            quick_scope="first 64 real gates, all 720 rays; changed reflective outer face; "
                        "NOT full-sweep evidence" if quick else "full selected sweep",
        )
    file_digest = sha256_file(path)
    metadata.update(settings_values(
        input_sha256=file_digest, documented_input_sha256=DOCUMENTED_SHA256,
        matches_documented_file_sha256=file_digest == DOCUMENTED_SHA256,
    ))
    if quick:
        return full[:, :64].copy(), ranges[:64].copy(), az_res, metadata
    return full, ranges, az_res, metadata


def regions(shape: tuple[int, int], ranges: np.ndarray) -> dict[str, np.ndarray]:
    n_az, n_range = shape
    result = {"all": np.ones(shape, dtype=bool)}
    result["centre"] = np.broadcast_to(ranges[None, :] <= 10000.0, shape)
    seam = np.zeros(shape, dtype=bool)
    seam[:min(15, n_az), :] = True
    seam[max(0, n_az - 15):, :] = True
    result["seam"] = seam
    outer = np.zeros(shape, dtype=bool)
    outer[:, max(0, n_range - 30):] = True
    result["outer"] = outer
    return result


def diagnostic_rows(
    run_id: str, case: str, method: str, fields: dict[str, np.ndarray], valid: np.ndarray,
    region_masks: dict[str, np.ndarray], ranges: np.ndarray, width: float, az_res: float,
) -> tuple[list[dict], list[dict]]:
    diagnostics, radial = [], []
    d = fields["raw_mask_response"]
    accepted = fields["direct_acceptance"].astype(bool)
    eligible = fields["fallback_eligibility"].astype(bool)
    for region, select in region_masks.items():
        raw = d[select]
        n = int(select.sum())
        finite = raw[np.isfinite(raw)]
        quantiles = np.quantile(finite, [0, .25, .5, .75, 1]) if finite.size else [""] * 5
        diagnostics.append(dict(
            run_id=run_id, case_id=case, method=method, region=region, n_considered=n,
            n_input_valid=int(np.count_nonzero(valid & select)),
            n_direct_accepted=int(np.count_nonzero(accepted & select)),
            direct_acceptance_fraction=float(np.count_nonzero(accepted & select) / n) if n else "",
            n_denominator_rejected=int(np.count_nonzero(~accepted & select)),
            n_fallback_eligible=int(np.count_nonzero(eligible & select)),
            fallback_eligible_fraction=float(np.count_nonzero(eligible & select) / n) if n else "",
            n_gap_filled=0, n_direct_at_original_missing=int(np.count_nonzero(accepted & ~valid & select)),
            raw_support_min=quantiles[0], raw_support_q25=quantiles[1], raw_support_median=quantiles[2],
            raw_support_q75=quantiles[3], raw_support_max=quantiles[4],
            n_raw_support_negative=int(np.count_nonzero(raw < 0)),
            n_raw_support_above_one=int(np.count_nonzero(raw > 1)),
            n_negative_raw_variance=int(np.count_nonzero((fields["raw_variance"] < 0) & select)),
        ))
    for gate, az_width in enumerate(azimuth_widths(valid.shape[1], width, az_res)):
        raw = d[:, gate]
        radial.append(dict(
            run_id=run_id, case_id=case, method=method, gate_index_zero_based=gate,
            range_m=float(ranges[gate]), azimuth_width_bins=float(az_width),
            nominal_support_factor=float(width * az_width), n_input_valid=int(valid[:, gate].sum()),
            raw_support_min=float(raw.min()), raw_support_median=float(np.median(raw)),
            raw_support_max=float(raw.max()), n_direct_accepted=int(accepted[:, gate].sum()),
            n_denominator_rejected=int((~accepted[:, gate]).sum()),
            n_fallback_eligible=int(eligible[:, gate].sum()), n_gap_filled=0,
            n_direct_at_original_missing=int(np.count_nonzero(accepted[:, gate] & ~valid[:, gate])),
        ))
    return diagnostics, radial


def validate_case(
    toolkit: Any, run_id: str, name: str, data: np.ndarray, ranges: np.ndarray,
    az_res: float, width: float, shared_settings: dict, save_support_arrays: bool = False,
) -> tuple[list[dict], list[dict], list[dict], dict, dict]:
    base = f"{name}_W{width:g}"
    valid = np.isfinite(data)
    zeroed = np.where(valid, data, 0.0)
    operator = SpatialPolarGaussian(data.shape, width, az_res)
    spatial_d = operator(valid.astype(float))
    spatial_num = operator(zeroed)
    spatial_num2 = operator(zeroed**2)
    spectral_d = toolkit.smooth_polar(
        valid.astype(float), width, az_res_deg=az_res, az_boundary="periodic",
        range_boundary="reflective", kernel_type="gaussian",
    )
    region_masks = regions(data.shape, ranges)
    rows, diagnostics, radial, case_settings = [], [], [], {}
    for floor in FLOORS:
        case = f"{base}_floor{floor:.0e}"
        reference = direct_fields(spatial_d, spatial_num, spatial_num2, valid, width, az_res, floor)
        actual = toolkit_fields(toolkit, data, width, az_res, floor, spectral_d)
        for statistic in reference:
            for region, selection in region_masks.items():
                rows.append(comparison_row(
                    run_id, case, statistic, region, selection, actual[statistic], reference[statistic],
                    notes=f"Strict raw d>{floor:g}; no fallback or mask restoration. "
                          "Finite analytic Gaussian weights can be signed. "
                          "All-missing moments legitimately have no comparable values.",
                ))
        for method, fields in ((SPECTRAL, actual), (SPATIAL, reference)):
            diag, by_gate = diagnostic_rows(run_id, case, method, fields, valid, region_masks,
                                            ranges, width, az_res)
            diagnostics.extend(diag)
            radial.extend(by_gate)
        widths = azimuth_widths(data.shape[1], width, az_res)
        case_settings[case] = dict(shared_settings) | settings_values(
            input_pattern=name, shape=list(data.shape), n_input_valid=int(valid.sum()),
            missing_fraction=float(np.mean(~valid)), finite_zero_count=int(np.count_nonzero(valid & (data == 0))),
            width_azimuth_legacy_index_unit=width, width_range_gates=width,
            azimuth_width_formula="W_az/((j+1)*deg2rad(az_res_deg)); j zero-based",
            azimuth_width_first_gate_bins=float(widths[0]), azimuth_width_last_gate_bins=float(widths[-1]),
            n_gates_azimuth_width_above_full_sweep=int(np.count_nonzero(widths > data.shape[0])),
            n_gates_azimuth_width_fractional=int(np.count_nonzero(widths != np.rint(widths))),
            denominator_floor=floor, denominator_comparison="finite(d) and d > floor, STRICT",
            raw_support_archive="raw_support.npz" if save_support_arrays else "not saved; CSV diagnostics only",
            raw_support_spectral_key=f"{base}_toolkit" if save_support_arrays else "",
            raw_support_spatial_key=f"{base}_spatial" if save_support_arrays else "",
            valid_mask_archive_key=f"{base}_valid" if save_support_arrays else "",
            fallback_eligibility_definition="denominator rejected AND at least one finite input; "
                                            "all-missing input is ineligible",
            fallback_eligibility_does_not_mean_fallback_executed=True,
            actual_gap_filled_count=0, default_completion_tested=False,
            benchmark_case_eligible=name == "natural" and floor == DEFAULT_FLOOR,
            threshold_sensitivity_scope="direct acceptance coverage and both moments vs spatial reference; "
                                        "NOT completed/default fallback outputs",
        ) | operator.audit()
    archive = ({f"{base}_toolkit": spectral_d, f"{base}_spatial": spatial_d,
                f"{base}_valid": valid} if save_support_arrays else {})
    return rows, diagnostics, radial, case_settings, archive


def benchmark(toolkit: Any, run_id: str, data: np.ndarray, az_res: float,
              repeats: int, warmups: int) -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    for width in WIDTHS:
        case = f"natural_W{width:g}_floor{DEFAULT_FLOOR:.0e}"
        methods = {
            SPECTRAL: (OPERATION, lambda: toolkit.dct_mean(
                data, width, **toolkit_options(az_res), _stable_fallback=False,
                _denominator_floor=DEFAULT_FLOOR,
            )),
        }
        for method, (operation, calculation) in methods.items():
            try:
                for _ in range(warmups):
                    calculation()
            except Exception as exc:
                rows.append(dict(
                    run_id=run_id, case_id=case, method=method, operation=operation,
                    repetition=0, status="failed", elapsed_s="", ram_before_mib="", peak_ram_mib="",
                    reason=f"warmup failure: {type(exc).__name__}: {exc}",
                ))
        for repetition in range(1, repeats + 1):
            order = list(methods) if repetition % 2 else list(reversed(methods))
            for method in order:
                row = dict(run_id=run_id, case_id=case, method=method, operation=methods[method][0],
                           repetition=repetition, ram_before_mib="", peak_ram_mib="", reason="")
                try:
                    start = time.perf_counter()
                    output = methods[method][1]()
                    elapsed = time.perf_counter() - start
                    if output.shape != data.shape:
                        raise RuntimeError("Mean-only timing returned wrong output shape")
                    row.update(status="success", elapsed_s=elapsed)
                except Exception as exc:
                    row.update(status="failed", elapsed_s="", reason=f"{type(exc).__name__}: {exc}")
                rows.append(row)
        for method in methods:
            measured = [row for row in rows if row["case_id"] == case and row["method"] == method]
            times = [row["elapsed_s"] for row in measured if row["status"] == "success"]
            summaries.append(dict(
                run_id=run_id, case_id=case, method=method, operation=methods[method][0],
                n_success=len(times), n_failure=sum(row["status"] == "failed" for row in measured),
                median_time_s=float(np.median(times)) if times else "",
                q25_time_s=float(np.quantile(times, .25)) if times else "",
                q75_time_s=float(np.quantile(times, .75)) if times else "", max_peak_ram_mib="",
            ))
    return rows, summaries


FIGURE_RUN_ID = "lead-review-radar-exact-only-20261027"
FIGURE_RUN_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "results" / "round1" / "2d_radar_gaussian" / FIGURE_RUN_ID
)
FIGURE_CASE_ID = "natural_W15_floor1e-03"
FIGURE_ORIGINAL_SCRIPT_SHA256 = "d97dec212b22ad83a36542821f08d34b3c450ad0807e986705d1f1b8a43e1332"
FIGURE_SAVED_CSV_SHA256 = {
    "settings.csv": "2a8db158e18ef02549afd158d0418bbf3ec4d1a3bfffe35db97aaacc25664c46",
    "comparison.csv": "0e6891daa304592125cf67659421bb7a0b5e2243cc8350814b9024969da1d919",
    "support_diagnostics.csv": "b519a1eeae6890fe270831ce65135a1433258121382d2adac240580f252f0fbf",
    "support_by_range.csv": "c948fd29b2ec3560bc3de05917ec33301905d6c46b4a8ea0099787a1f3e424a5",
}
FIGURE_STEM = "radar_gaussian_statistics"
FIGURE_SIZE = (6.5, 7.4)
FIGURE_LAYOUT_HEIGHT_IN = 7.8
FIGURE_TOP_TRIM_IN = 0.4
FIGURE_REFLECTIVITY_LIMITS = (0.0, 40.0)
FIGURE_ZOOM_HALF_WIDTH_KM = 10.0
FIGURE_STD_LIMITS = (0.0, 10.0)
FIGURE_COVERAGE_LIMITS = (0.0, 1.0)
FIGURE_MISSING_COLOR = "#D0D0D0"
FIGURE_NEGATIVE_COLOR = "#D32232"
FIGURE_CLAMPED_LABEL = "Clamped"


def _figure_saved_evidence(saved_run: Path = FIGURE_RUN_DIRECTORY):
    """Read only the four immutable files for the one approved case."""
    import csv

    from _review_common import SETTINGS_FIELDS

    selected = {}
    schemas = {"settings.csv": SETTINGS_FIELDS, "comparison.csv": COMPARISON_FIELDS,
               "support_diagnostics.csv": DIAGNOSTIC_FIELDS, "support_by_range.csv": RADIAL_FIELDS}
    for filename, fields in schemas.items():
        path = saved_run / filename
        if sha256_file(path) != FIGURE_SAVED_CSV_SHA256[filename]:
            raise ValueError(f"Saved figure evidence SHA-256 mismatch: {filename}")
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(fields):
                raise ValueError(f"Unexpected figure evidence columns: {filename}")
            selected[filename] = [row for row in reader if row["case_id"] == FIGURE_CASE_ID]
        if not selected[filename] or any(row["run_id"] != FIGURE_RUN_ID for row in selected[filename]):
            raise ValueError(f"Missing or mismatched approved figure case: {filename}")
    frozen = {}
    for row in selected["settings.csv"]:
        if row["setting"] in frozen:
            raise ValueError("Duplicate saved figure setting")
        frozen[row["setting"]] = row["value"]
    return frozen, selected


def _figure_reflectivity_colormap():
    """Require the original cmweather palette, never a silent substitute."""
    try:
        from cmweather.cm_colorblind import ChaseSpectral
    except ImportError as exc:
        raise RuntimeError("Radar figure requires cmweather with ChaseSpectral; no substitute colormap is used.") from exc
    return ChaseSpectral


def _figure_check_settings(frozen):
    """Reject changed scientific settings before reading or filtering radar."""
    from _review_common import TOOLKIT_COMMIT

    expected = {
        "dct_toolkit_commit": TOOLKIT_COMMIT, "dct_toolkit_version": "1.0.0",
        "script_sha256": FIGURE_ORIGINAL_SCRIPT_SHA256,
        "common_helper_sha256": sha256_file(Path(__file__).with_name("_review_common.py")),
        "input_sha256": DOCUMENTED_SHA256, "documented_input_sha256": DOCUMENTED_SHA256,
        "matches_documented_file_sha256": "True", "input_pattern": "natural",
        "field": "DBZHC", "field_units": "dBZ", "shape": "[720, 1180]",
        "sweep_zero_based": "12", "sweep_selected_before_filtering": "True", "quick_mode": "False",
        "width_azimuth_legacy_index_unit": "15.0", "width_range_gates": "15.0",
        "azimuth_width_formula": "W_az/((j+1)*deg2rad(az_res_deg)); j zero-based",
        "kernel": "finite analytic Gaussian H=exp(-0.5*(omega*W/sqrt(12))^2)",
        "boundary_azimuth": "periodic circular complete sweep", "boundary_range": "half-sample reflective",
        "operator_order": "azimuth first (range-dependent), range second; normalize only AFTER both",
        "denominator_floor": "0.001", "denominator_comparison": "finite(d) and d > floor, STRICT",
        "input_mask_restoration": "False", "completion_executed": "False", "actual_gap_filled_count": "0",
        "toolkit_private_controls": "_stable_fallback=False; explicit _denominator_floor for every call",
        "centre_definition": "range <=10000m, all azimuths", "n_input_valid": "508956",
        "mask_rule": "np.isfinite(decoded DBZHC); finite zeros valid; no extra QC thresholds",
        "full_sweep_decoded_float64_sha256": "e47d1b8e07791951a602344edcdcf60542a517f385e54c452fd81117f676d9a0",
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"Saved figure setting mismatch: {key}")
    if json.loads(frozen["numerical_tolerances"]) != {key: list(value) for key, value in TOLERANCES.items()}:
        raise ValueError("Saved figure numerical tolerances changed")


def _figure_selected_azimuth(path, metadata):
    """Read only the selected rays' coordinate, not the multi-sweep field."""
    import xarray as xr

    start, stop = (metadata[key][0] for key in ("ray_start_inclusive", "ray_stop_exclusive"))
    with xr.open_dataset(path, mask_and_scale=True, decode_times=False, decode_timedelta=False) as ds:
        azimuth = np.asarray(ds["azimuth"].isel(time=slice(start, stop)).values, dtype=np.float64)
    if (azimuth.shape != (720,) or azimuth[0] != metadata["azimuth_first_deg"][0]
            or azimuth[-1] != metadata["azimuth_last_deg"][0]):
        raise ValueError("Figure acquisition-order azimuth coordinate changed")
    return azimuth


def _figure_polar_mesh(azimuth, ranges):
    """Native corners from actual centers; x=R sin(az), y=R cos(az), in km.

    The first/last angular edges share the midpoint across the circular seam.
    Range edges use decoded coordinate midpoints, retaining their quantization.
    Slant range is drawn on a plane, not converted to a surface/geodesic range.
    """
    azimuth, ranges = np.asarray(azimuth, dtype=float), np.asarray(ranges, dtype=float)
    if (azimuth.ndim != 1 or ranges.ndim != 1 or azimuth.size < 3 or ranges.size < 2
            or not np.isfinite(azimuth).all() or not np.isfinite(ranges).all()
            or ranges[0] < 0 or np.any(np.diff(ranges) <= 0)):
        raise ValueError("Figure requires finite ordered native polar centers")
    steps = np.mod(np.diff(np.r_[azimuth, azimuth[0]]), 360.0)
    if np.any(steps <= 0) or not np.isclose(steps.sum(), 360.0, rtol=0, atol=1e-6):
        raise ValueError("Figure azimuth must be one acquisition-order complete circle")
    centers = np.deg2rad(azimuth[0] + np.r_[0.0, np.cumsum(steps[:-1])])
    first = (centers[-1] - 2 * np.pi + centers[0]) / 2
    angular_edges = np.r_[first, (centers[:-1] + centers[1:]) / 2, first + 2 * np.pi]
    radial_edges = np.r_[max(0.0, ranges[0] - (ranges[1] - ranges[0]) / 2),
                         (ranges[:-1] + ranges[1:]) / 2,
                         ranges[-1] + (ranges[-1] - ranges[-2]) / 2] / 1000
    x = np.sin(angular_edges[:, None]) * radial_edges[None, :]
    y = np.cos(angular_edges[:, None]) * radial_edges[None, :]
    x[-1], y[-1] = x[0], y[0]  # Exact shared corners, not a repeated data ray.
    return x, y, angular_edges, radial_edges


def _figure_check_diagnostic_rows(actual, saved, fields, identity):
    """Check every regional/radial quantity, including signed raw-d counts."""
    if len(actual) != len(saved):
        raise ValueError(f"Figure {identity} diagnostic row count changed")
    indexed = {row[identity]: row for row in saved}
    if len(indexed) != len(saved):
        raise ValueError(f"Duplicate saved figure {identity} diagnostic")
    atol, rtol = TOLERANCES["raw_mask_response"]
    for row in actual:
        previous = indexed.get(str(row[identity]))
        for key in fields:
            value = row[key]
            if previous is None or (key.startswith("raw_support_") and not np.isclose(
                    float(previous[key]), value, atol=atol, rtol=rtol)):
                raise ValueError(f"Figure raw diagnostic mismatch: {identity}={row[identity]}, {key}")
            if not key.startswith("raw_support_") and previous[key] != str(value):
                raise ValueError(f"Figure diagnostic mismatch: {identity}={row[identity]}, {key}")


def _figure_check_comparison_metadata(rows, fields, region_masks):
    """Check frozen agreement metadata; do NOT repeat the spatial reference."""
    if len(rows) != len(fields) * len(region_masks):
        raise ValueError("Saved figure comparison row count changed")
    seen = set()
    for row in rows:
        statistic, region = row["statistic"], row["region"]
        if statistic not in fields or region not in region_masks or (statistic, region) in seen:
            raise ValueError("Unexpected or duplicate saved figure comparison")
        seen.add((statistic, region))
        values = fields[statistic][region_masks[region]]
        defined = int(np.isfinite(values).sum())
        expected = dict(
            method=SPECTRAL, reference_method=SPATIAL, component="DBZHC",
            estimate_type="direct_only_no_completion",
            kernel_match="full_finite_analytic_response_matched_no_truncation",
            n_considered=str(values.size), n_compared=str(defined),
            n_method_undefined=str(values.size - defined), n_reference_undefined=str(values.size - defined),
            n_method_gap_filled="0", n_reference_gap_filled="0", status="pass", units=UNITS[statistic],
        )
        if (any(row[key] != value for key, value in expected.items())
                or (float(row["atol"]), float(row["rtol"])) != TOLERANCES[statistic]
                or not 0 <= float(row["rmse"]) <= float(row["max_abs_difference"])):
            raise ValueError(f"Saved figure comparison metadata mismatch: {region}, {statistic}")


def prepare_radar_figure(saved_run: Path = FIGURE_RUN_DIRECTORY):
    """Reproduce one full-sweep spectral case; frozen diagnostics are the gate."""
    import importlib.metadata

    from _review_common import TOOLKIT_COMMIT, _RELEASE_HASHES

    frozen, evidence = _figure_saved_evidence(saved_run)
    _figure_check_settings(frozen)
    reflectivity_colormap = _figure_reflectivity_colormap()
    path = Path(os.environ.get("REAL_RADAR_PATH", str(DEFAULT_RADAR))).expanduser().resolve()
    data, ranges, az_res, metadata = read_radar(path, quick=False)
    if metadata["input_sha256"][0] != DOCUMENTED_SHA256:
        raise ValueError("Figure input does not match the documented file SHA-256")
    for key, (value, _) in metadata.items():
        if key not in ("radar_path", "radar_input_selection") and frozen.get(key) != str(value):
            raise ValueError(f"Figure decoded input/geometry mismatch: {key}")
    azimuth = _figure_selected_azimuth(path, metadata)
    x, y, angular_edges, radial_edges = _figure_polar_mesh(azimuth, ranges)
    toolkit = load_toolkit()  # Verifies all nine v1.0.0 Python source hashes.
    valid = np.isfinite(data)
    denominator = toolkit.smooth_polar(
        valid.astype(float), 15.0, az_res_deg=az_res, az_boundary="periodic",
        range_boundary="reflective", kernel_type="gaussian",
    )
    fields = toolkit_fields(toolkit, data, 15.0, az_res, DEFAULT_FLOOR, denominator)
    accepted = np.isfinite(denominator) & (denominator > DEFAULT_FLOOR)
    for statistic in ("mean", "second_moment", "raw_variance", "variance", "std"):
        if not np.array_equal(np.isfinite(fields[statistic]), accepted):
            raise ValueError(f"Figure undefined moment mask changed: {statistic}")
    if (not np.isfinite(denominator).all()
            or not np.array_equal(fields["direct_acceptance"].astype(bool), accepted)
            or not np.array_equal(fields["variance"], np.maximum(fields["raw_variance"], 0), equal_nan=True)
            or not np.array_equal(fields["std"], np.sqrt(fields["variance"]), equal_nan=True)
            or not np.allclose(fields["support"], np.clip(denominator, 0, 1)
                               * 15.0 * azimuth_widths(data.shape[1], 15.0, az_res)[None, :],
                               atol=TOLERANCES["support"][0], rtol=TOLERANCES["support"][1])):
        raise ValueError("Figure raw-denominator/clamped-moment/coverage mapping changed")
    region_masks = regions(data.shape, ranges)
    diagnostics, radial = diagnostic_rows(
        FIGURE_RUN_ID, FIGURE_CASE_ID, SPECTRAL, fields, valid, region_masks, ranges, 15.0, az_res,
    )
    saved_diag = [row for row in evidence["support_diagnostics.csv"] if row["method"] == SPECTRAL]
    saved_radial = [row for row in evidence["support_by_range.csv"] if row["method"] == SPECTRAL]
    _figure_check_diagnostic_rows(diagnostics, saved_diag, DIAGNOSTIC_FIELDS, "region")
    _figure_check_diagnostic_rows(radial, saved_radial, RADIAL_FIELDS, "gate_index_zero_based")
    _figure_check_comparison_metadata(evidence["comparison.csv"], fields, region_masks)
    negative = accepted & (fields["raw_variance"] < 0)
    central = ranges <= 10000.0
    if (int(accepted.sum()), int(negative.sum()), int(valid.sum()),
            int(accepted[:, central].sum()), int(negative[:, central].sum()), int(valid[:, central].sum())) != (
                655428, 4593, 508956, 90152, 0, 56584):
        raise ValueError("Approved figure whole/central counts changed")
    # Display the square only after the complete sweep has been filtered. Keep
    # the frozen radial diagnostic region above distinct from this viewport.
    center_x = np.sin(np.deg2rad(azimuth))[:, None] * ranges[None, :] / 1000
    center_y = np.cos(np.deg2rad(azimuth))[:, None] * ranges[None, :] / 1000
    zoom = (np.abs(center_x) <= FIGURE_ZOOM_HALF_WIDTH_KM) & (np.abs(center_y) <= FIGURE_ZOOM_HALF_WIDTH_KM)
    # Enclose all four square corners, including the straight outer mesh edges.
    enclosing_radius = np.sqrt(2) * FIGURE_ZOOM_HALF_WIDTH_KM / np.cos(np.diff(angular_edges).max() / 2)
    zoom_gate_count = int(np.searchsorted(radial_edges, enclosing_radius))
    if zoom_gate_count >= radial_edges.size:
        raise ValueError("Native mesh does not cover the requested square zoom")
    settings = {
        "purpose": "COW DBZHC Gaussian mean, reported standard deviation and local data coverage",
        "figure_kind": "reproduction of one approved full-sweep case; no new experiment/reference/timing",
        "selected_run": FIGURE_RUN_ID, "selected_case": FIGURE_CASE_ID,
        "selected_run_directory": str(saved_run.resolve()), "input_filename": path.name,
        "input_sha256": metadata["input_sha256"][0],
        "decoded_input_sha256": metadata["full_sweep_decoded_float64_sha256"][0],
        "script_sha256": sha256_file(Path(__file__)), "saved_original_script_sha256": frozen["script_sha256"],
        "common_helper_sha256": frozen["common_helper_sha256"],
        "dct_toolkit_commit": TOOLKIT_COMMIT, "dct_toolkit_version": toolkit.__version__,
        "toolkit_source_verification": "PASS: all nine released Python source files checked by load_toolkit",
        "kernel": frozen["kernel"], "width_convention": frozen["width_convention"],
        "width_azimuth_legacy_index_unit": 15.0, "width_range_gates": 15.0,
        "azimuth_width_formula": frozen["azimuth_width_formula"], "width_cap": frozen["width_cap"],
        "origin_policy": frozen["origin_policy"], "boundary_azimuth": frozen["boundary_azimuth"],
        "boundary_range": frozen["boundary_range"], "operator_order": frozen["operator_order"],
        "denominator_floor": DEFAULT_FLOOR, "denominator_rule": frozen["denominator_comparison"],
        "completion": False, "restoration": False, "new_timing_performed": False,
        "new_spatial_reference_performed": False,
        "saved_comparison_role": "previous exact spatial agreement; metadata/counts/tolerances checked, reference NOT rerun",
        "mask_rule": frozen["mask_rule"], "coverage_definition": "rho=clip(raw d,0,1), over every cell; NOT ratio-gated",
        "coverage_interpretation": "bounded availability diagnostic, not measurement quality or statistical reliability",
        "negative_overlay_rule": "accepted & (M2-mu^2 < 0), no extra cutoff; reported variance/std are zero beneath overlay",
        "clamped_overlay_label": FIGURE_CLAMPED_LABEL,
        "clamped_overlay_key_location": "panel (c) title row only; no figure-wide legend",
        "clamped_zero_interpretation": "numerical clamp, not evidence of absent local variability",
        "missing_category_explanation": "caption only: missing observations in (a); undefined estimates in (b,c,e)",
        "negative_overlay_color": FIGURE_NEGATIVE_COLOR, "missing_undefined_color": FIGURE_MISSING_COLOR,
        "negative_overlay_whole_count": int(negative.sum()), "negative_overlay_central_count": int(negative[:, central].sum()),
        "central_definition": frozen["centre_definition"], "central_gate_count": int(central.sum()),
        "central_last_gate_center_m": float(ranges[central][-1]),
        "central_outer_cell_corner_m": float(radial_edges[int(central.sum())] * 1000),
        "central_processing": "frozen radial diagnostic region of FULL 720x1180 results; NOT the square display domain",
        "zoom_definition": "-10<=x<=10 km and -10<=y<=10 km; square viewport of full-sweep results",
        "zoom_count_definition": "native cell centers inside square; boundary polygons are clipped by the viewport",
        "zoom_processing": "post-filter viewport of FULL 720x1180 results; no cropped filtering or resampling",
        "zoom_gate_count": zoom_gate_count,
        "zoom_outer_cell_corner_m": float(radial_edges[zoom_gate_count] * 1000),
        "zoom_n_cells": int(zoom.sum()), "zoom_n_observed": int(valid[zoom].sum()),
        "zoom_n_accepted": int(accepted[zoom].sum()), "zoom_n_undefined": int((~accepted)[zoom].sum()),
        "negative_overlay_zoom_count": int(negative[zoom].sum()),
        "plot_coordinates": "x=slant_range*sin(actual_azimuth), y=slant_range*cos(actual_azimuth), km; planar, not surface/geodesic",
        "native_mesh": "midpoint corners from acquisition-order azimuth and decoded range centers; circular seam exactly closed; no resampling/transposition",
        "azimuth_coordinate_sha256": hashlib.sha256(azimuth.tobytes()).hexdigest(),
        "range_coordinate_sha256": hashlib.sha256(ranges.tobytes()).hexdigest(),
        "reflectivity_colormap": reflectivity_colormap.name,
        "reflectivity_colormap_source": "cmweather.cm_colorblind.ChaseSpectral; required, no fallback",
        "std_colormap": "cividis", "coverage_colormap": "YlGnBu",
        "reflectivity_color_limits_dbz": FIGURE_REFLECTIVITY_LIMITS, "reflectivity_colorbar_extend": "both",
        "reflectivity_colorbar_ticks_dbz": [0, 10, 20, 30, 40],
        "std_color_limits_dbz": FIGURE_STD_LIMITS, "std_colorbar_extend": "max",
        "coverage_color_limits": FIGURE_COVERAGE_LIMITS, "coverage_colorbar_extend": "neither",
        "color_scale_sharing": "observed/mean identical; whole/zoom std identical; whole/zoom coverage identical",
        "panel_order": "(a) observed, (b) normalized mean, (c) full std, (d) full rho, (e) square-zoom std, (f) square-zoom rho",
        "full_axis_limits_km": [-90, 90],
        "zoom_axis_limits_km": [-FIGURE_ZOOM_HALF_WIDTH_KM, FIGURE_ZOOM_HALF_WIDTH_KM],
        "zoom_indicator": "dashed 20x20 km square in full std/coverage panels; not the radial diagnostic region",
        "dense_mesh_rendering": "rasterized native cells in PDF; labels/axes/squares/legend remain vector",
        "top_margin_trimmed_in": FIGURE_TOP_TRIM_IN,
        "layout_reference_height_in": FIGURE_LAYOUT_HEIGHT_IN,
        "data_panel_size_in": [1.95, 1.95],
        "layout_policy": "remove top-legend band only; retain physical panel/colorbar sizes, spacing and text sizes",
        "n_regional_diagnostic_rows_verified": len(diagnostics), "n_radial_diagnostic_rows_verified": len(radial),
        "n_frozen_comparison_metadata_rows_verified": len(evidence["comparison.csv"]),
    }
    for key, (value, _) in metadata.items():
        if key not in ("radar_path", "radar_input_selection"):
            settings[key] = value
    for name, digest in _RELEASE_HASHES.items():
        settings[f"toolkit_source_sha256.{name}"] = digest
    for name in ("numpy", "scipy", "astropy", "xarray", "netCDF4", "cmweather"):
        settings[f"{name}_version"] = importlib.metadata.version(name)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        settings[name] = os.environ.get(name, "not set")
    for filename, digest in FIGURE_SAVED_CSV_SHA256.items():
        settings[f"saved_{filename}_sha256"] = digest
    for row in diagnostics:
        for key in DIAGNOSTIC_FIELDS[4:]:
            settings[f"diagnostic.{row['region']}.{key}"] = row[key]
    for row in evidence["comparison.csv"]:
        for key in ("max_abs_difference", "rmse"):
            settings[f"saved_comparison.{row['region']}.{row['statistic']}.{key}"] = row[key]
    for statistic, values, limits in (("observed", data, FIGURE_REFLECTIVITY_LIMITS),
                                     ("mean", fields["mean"], FIGURE_REFLECTIVITY_LIMITS),
                                     ("std", fields["std"], FIGURE_STD_LIMITS),
                                     ("raw_variance", fields["raw_variance"], None)):
        finite = values[np.isfinite(values)]
        settings[f"{statistic}_finite_min"] = float(finite.min())
        settings[f"{statistic}_finite_max"] = float(finite.max())
        if limits is not None:
            settings[f"{statistic}_below_color_limit_count"] = int(np.count_nonzero(finite < limits[0]))
            settings[f"{statistic}_above_color_limit_count"] = int(np.count_nonzero(finite > limits[1]))
    settings["checks"] = (
        "PASS: immutable four-CSV hashes; approved case/settings and original script SHA; documented file SHA; "
        "decoded input SHA and all read_radar metadata; nine released toolkit source hashes; "
        "every quantity in four regional and 1180 radial spectral diagnostic rows; "
        "36 frozen comparison metadata/count/tolerance rows (no spatial rerun); strict raw-d acceptance; "
        "undefined moment masks; unchanged numeric clamped moments; exact negative-overlay mask; "
        "whole/radial-central counts; native coordinate mesh; separate square post-filter viewport/counts; "
        "required cmweather ChaseSpectral; no completion or new timing"
    )
    case = dict(data=data, ranges=ranges, azimuth=azimuth, x=x, y=y, angular_edges=angular_edges,
                radial_edges=radial_edges, fields=fields, accepted=accepted, negative=negative, central=central,
                zoom=zoom, zoom_gate_count=zoom_gate_count)
    return case, settings


def create_radar_figure(case):
    """Draw six native-cell panels; never alter or recalculate the statistics."""
    from _review_figures import configure_plotting

    configure_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle, Patch
    import matplotlib.patheffects as pe
    from matplotlib.text import Text

    fig = plt.figure(figsize=FIGURE_SIZE)
    # Trim only the former top legend band, keeping original physical geometry.
    vertical_scale = FIGURE_LAYOUT_HEIGHT_IN / FIGURE_SIZE[1]
    axes = [fig.add_axes((left, bottom * vertical_scale, 0.30, 0.25 * vertical_scale))
            for bottom in (0.665, 0.375, 0.085) for left in (0.105, 0.565)]
    titles = ("(a) Observed DBZHC", "(b) Normalized mean", "(c) Std",
              "(d) Local data coverage", "(e) Std (zoom)", "(f) Coverage (zoom)")
    fields = case["fields"]
    coverage = np.clip(fields["raw_mask_response"], 0, 1)
    count = case["zoom_gate_count"]
    reflectivity_colormap = _figure_reflectivity_colormap()
    meshes = []
    for index, (ax, title, values, colormap, limits) in enumerate(zip(
            axes, titles, (case["data"], fields["mean"], fields["std"], coverage,
                           fields["std"][:, :count], coverage[:, :count]),
            (reflectivity_colormap, reflectivity_colormap, "cividis", "YlGnBu", "cividis", "YlGnBu"),
            (FIGURE_REFLECTIVITY_LIMITS, FIGURE_REFLECTIVITY_LIMITS, FIGURE_STD_LIMITS,
             FIGURE_COVERAGE_LIMITS, FIGURE_STD_LIMITS, FIGURE_COVERAGE_LIMITS))):
        zoomed = index >= 4
        x, y = (case[key][:, :count + 1] if zoomed else case[key] for key in ("x", "y"))
        cmap = plt.get_cmap(colormap).copy()
        cmap.set_bad(FIGURE_MISSING_COLOR)
        mesh = ax.pcolormesh(x, y, np.ma.masked_invalid(values), shading="flat", cmap=cmap,
                             vmin=limits[0], vmax=limits[1], rasterized=True, linewidth=0,
                             antialiased=False)
        mesh.set_gid(("observed", "mean", "std_full", "rho_full", "std_zoom", "rho_zoom")[index])
        meshes.append(mesh)
        if index in (2, 4):
            negative = case["negative"][:, :count] if zoomed else case["negative"]
            overlay = ax.pcolormesh(x, y, np.ma.masked_where(~negative, np.ones(negative.shape)),
                                    shading="flat", cmap=ListedColormap([FIGURE_NEGATIVE_COLOR]),
                                    vmin=0, vmax=1, rasterized=True, linewidth=0, antialiased=False)
            overlay.set_gid("negative_raw_variance")
        if index in (2, 3):
            square = Rectangle((-FIGURE_ZOOM_HALF_WIDTH_KM, -FIGURE_ZOOM_HALF_WIDTH_KM),
                               2 * FIGURE_ZOOM_HALF_WIDTH_KM, 2 * FIGURE_ZOOM_HALF_WIDTH_KM,
                               fill=False, edgecolor="#333333", linewidth=0.9, linestyle=(0, (3, 2)),
                               path_effects=[pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
            square.set_gid("square_20km_zoom")
            ax.add_patch(square)
        extent, ticks = (FIGURE_ZOOM_HALF_WIDTH_KM, [-10, 0, 10]) if zoomed else (90, [-50, 0, 50])
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xticks=ticks, yticks=ticks, aspect="equal")
        ax.set_title(title, loc="left", fontweight="bold", pad=5)
        ax.tick_params(length=3, pad=3, labelbottom=zoomed, labelleft=index % 2 == 0)
        ax.spines[["top", "right"]].set_visible(False)
    for mesh, box, label, ticks, extend in (
            (meshes[1], (0.885, 0.685, 0.015, 0.21), "Reflectivity (dBZ)", [0, 10, 20, 30, 40], "both"),
            (meshes[2], (0.435, 0.115, 0.015, 0.48), "Reported std (dBZ)", [0, 5, 10], "max"),
            (meshes[3], (0.885, 0.115, 0.015, 0.48), r"Coverage, $\rho$", [0, 0.5, 1], "neither")):
        left, bottom, width, height = box
        cax = fig.add_axes((left, bottom * vertical_scale, width, height * vertical_scale))
        colorbar = fig.colorbar(mesh, cax=cax, ticks=ticks, extend=extend)
        colorbar.set_label(label, labelpad=5)
        colorbar.ax.tick_params(length=3, pad=3)
    fig.supxlabel("Eastward range (km)", x=0.49, y=0.012 * vertical_scale)
    fig.supylabel("Northward range (km)", x=0.014, y=0.50 * vertical_scale)
    key = axes[2].legend([Patch(facecolor=FIGURE_NEGATIVE_COLOR)], [FIGURE_CLAMPED_LABEL],
                         loc="lower right", bbox_to_anchor=(1, 1.02), frameon=False,
                         borderaxespad=0, borderpad=0, handlelength=0.8, handletextpad=0.4, fontsize=10.1)
    key.set_gid("clamped_panel_key")
    for text in fig.findobj(Text):
        if text.get_fontsize() < 10.1:
            text.set_fontsize(10.1)
    return fig


def export_radar_figure(output: Path | None = None) -> dict[str, Path]:
    """Lazy opt-in export, rejected inside numerical results before computation."""
    from _review_figures import DEFAULT_FIGURE_ROOT, export_figure

    output = (output if output is not None else DEFAULT_FIGURE_ROOT / "2d_radar_gaussian").expanduser().resolve()
    if output.is_relative_to((Path(__file__).resolve().parents[1] / "results").resolve()):
        raise ValueError("Figure output must not be inside saved experiment results.")
    case, settings = prepare_radar_figure()
    fig = create_radar_figure(case)
    import matplotlib.pyplot as plt

    try:
        return export_figure(fig, output, FIGURE_STEM, settings)
    finally:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radar-path", type=Path, help="Exact real COW NetCDF; no synthetic/sample fallback")
    parser.add_argument("--run-id", help="Unique run directory name; existing runs are never overwritten")
    parser.add_argument("--output-root", type=Path, help="Output base passed to the shared run-directory helper")
    parser.add_argument("--repeats", type=int, default=3, help="Measured mean-only repetitions per method/width (default 3)")
    parser.add_argument("--warmups", type=int, default=1, help="Unmeasured calls before each timing (default 1)")
    parser.add_argument("--skip-timing", "--correctness-only", action="store_true", help="Numerical checks only, no benchmark files")
    parser.add_argument("--quick", action="store_true", help="Real sweep first 64 gates, all rays; implies --skip-timing")
    parser.add_argument("--self-test-only", action="store_true", help="Explicit small artificial-array implementation tests ONLY")
    parser.add_argument("--save-support-arrays", action="store_true", help="Opt in to raw_support.npz; default outputs are CSV only")
    parser.add_argument("--strict-sensitivity", action="store_true", help="Also return nonzero if nondefault denominator-floor stress tests exceed the fixed tolerances")
    parser.add_argument("--controlled-gap", action="store_true", help="Additionally remove observed rays in the middle quarter-sweep; no truth claim")
    parser.add_argument("--figure-only", action="store_true", help="Reproduce/export only the approved natural-mask W15 full sweep; no new validation or timing")
    parser.add_argument("--figure-output", type=Path, help="Figure case directory (default: manuscript/figures/round1/2d_radar_gaussian)")
    args = parser.parse_args()
    if args.figure_output is not None and not args.figure_only:
        parser.error("--figure-output requires --figure-only")
    if args.figure_only:
        if (args.radar_path is not None or args.run_id is not None or args.output_root is not None
                or args.quick or args.self_test_only or args.save_support_arrays or args.strict_sensitivity
                or args.controlled_gap or args.skip_timing or args.repeats != 3 or args.warmups != 1):
            parser.error("--figure-only uses one fixed case; do not mix validation options (use REAL_RADAR_PATH to locate the documented input)")
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be >=1 and --warmups >=0")
    if args.quick or args.self_test_only:
        args.skip_timing = True
    return args


def main() -> int:
    args = parse_args()
    if args.figure_only:
        for kind, path in export_radar_figure(args.figure_output).items():
            print(f"{kind}: {path}")
        return 0
    toolkit = load_toolkit()
    # Resolve and validate input BEFORE making a run directory; missing input
    # must not leave fake radar comparison/benchmark output behind.
    radar = None
    if not args.self_test_only:
        path = args.radar_path or Path(os.environ.get("REAL_RADAR_PATH", str(DEFAULT_RADAR)))
        radar = read_radar(path.expanduser().resolve(), args.quick)
    run_id, directory = create_run_directory("2d_radar_gaussian", args.run_id, args.output_root)
    rows, cases = self_tests(toolkit, run_id)
    protocol = settings_values(
        command=json.dumps(sys.argv), script_path=str(Path(__file__).resolve()),
        script_sha256=sha256_file(Path(__file__)), numerical_tolerances=TOLERANCES,
        kernel="finite analytic Gaussian H=exp(-0.5*(omega*W/sqrt(12))^2)",
        width_convention="explicit legacy index-unit azimuth W, range W in gates; NOT physical distances",
        gaussian_fractional_width_policy="continuous spectral response; no rounding or minimum-one width",
        width_cap="none; W15 first azimuth width exceeds 720 rays; not 720 distinct observations",
        origin_policy="legacy radius j+1; actual first gate 95 m NOT used in angular width conversion",
        boundary_azimuth="periodic circular complete sweep", boundary_range="half-sample reflective",
        operator_order="azimuth first (range-dependent), range second; normalize only AFTER both",
        reference_method=SPATIAL,
        reference_construction="independent direct analytic cosine sums; no toolkit functions, FFT/DCT or filtered impulses",
        reference_periodic_weights="all N circular classes; even antipode split at centered endpoints",
        reference_range_weights="period 2R even extension; unrepresented H[R]=0; even antipode split",
        reference_range_nyquist="half-sample even input has zero Nyquist component; H[R] choice does not change operator",
        reference_truncation="NONE: every finite-period offset included, including small/signed tails",
        reference_zero_input_shortcut="exact all-zero input returns zero by linearity; only avoids useless control convolutions",
        reference_boundary_implementation="scipy.ndimage.convolve1d(mode=wrap azimuth, mode=reflect range, origin=0)",
        weight_interpretation="signed induced weights possible; no positive-weight/population/unbiasedness guarantee",
        weight_correction="none: no positivity enforcement, clipping or renormalization of weights",
        invalid_value_replacement="np.where(isfinite(decoded input), input, 0.0); second numerator squares zero-filled input",
        variance_definition="ddof=0 moment difference; max(raw_second-mean^2,0); raw difference also saved in comparisons",
        comparison_status_contract="pass / fail / no_comparable; undefined-mask mismatch always fails",
        support_statistic_name="support; nominal weighted support, not statistical ESS or a literal count",
        nominal_support_definition="clip(raw mask response,0,1)*W_range*W_az_grid(j); not statistical ESS/count of distinct observations",
        save_support_arrays=args.save_support_arrays,
        support_archive=("raw_support.npz contains both full raw denominators and input mask for every input/width"
                         if args.save_support_arrays and radar is not None else "not saved; CSV diagnostics only"),
        nominal_support_clipping="only support diagnostic; ratios always use UNCLIPPED raw denominator",
        input_mask_restoration=False, all_domain_direct_ratios="including originally missing cells when accepted",
        toolkit_api="released dct_mean, dct_variance, dct_std, dct_count; raw mask uses smooth_polar",
        toolkit_private_controls="_stable_fallback=False; explicit _denominator_floor for every call",
        completion_executed=False, fallback_gap_filled_count=0,
        default_completion_tested=False, completion_parameters="not applicable: completion disabled, no prefill/nearest call",
        denominator_floors=list(FLOORS), default_denominator_floor=DEFAULT_FLOOR,
        primary_validation_floor=DEFAULT_FLOOR,
        threshold_sensitivity_output="threshold_sensitivity.csv; nondefault floors retain exact-test pass/fail and measured errors",
        threshold_sensitivity_interpretation="small positive denominators amplify floating-point differences; failed strict stress tolerances are retained, not relaxed",
        strict_sensitivity_exit_status=args.strict_sensitivity,
        centre_definition="range <=10000m, all azimuths", seam_definition="first/last 15 acquisition-order rays, all gates",
        outer_definition="last 30 gates, all azimuths", regions_overlap=True,
        all_missing_control="same processed radar shape, no observations; moments undefined, support zero; no fallback eligible",
        controlled_gap_enabled=args.controlled_gap,
        controlled_gap_definition="middle quarter acquisition-order rays, all gates; remove ONLY originally finite bins; retain natural mask",
        ground_truth="unknown; radar discrepancies are spatial-reference agreement, NOT recovery error",
        comparison_protocol="exact_spatial_validation_only_no_sampled_gaussian_baseline",
        conventional_sampled_gaussian_comparison="removed at author request; different discrete weights are not an equivalent-method baseline",
        exact_reference_role="numerical validation only; full analytic weights applied with convolve1d; not timed as a spatial performance baseline",
        prior_method_comparison="not run; predecessor/prefill coverage remains open for author review",
        timing_status="skipped" if args.skip_timing else "requested; only after numerical checks pass",
        timing_skip_reason="--quick/--self-test-only/--skip-timing or --correctness-only" if args.skip_timing else "",
        timing_operation=OPERATION, timing_mean_only=True, timing_includes_completion=False,
        timing_operations={SPECTRAL: OPERATION},
        timing_methods=[SPECTRAL],
        timing_repeats=args.repeats, timing_unmeasured_warmup_calls=args.warmups,
        timing_method_order="spectral method only; widths 5 then 15",
        timing_boundaries="arrays resident; includes mask, zero replacement, ALL kernels/responses, filters, normalization/output allocation; excludes I/O, CSV, self-tests",
        timing_kernel_reuse="no inter-call reuse; released mean builds responses in each smooth_polar call, included in timing",
        timing_speed_ratio="not calculated; no spatial performance comparison is made in 2D",
        radar_memory_measurement="not measured; memory columns blank",
        full_sweep_timing_performed=False,
    )
    for case in cases:
        cases[case] = dict(protocol) | cases[case]
    diagnostics, radial, archive = [], [], {}
    if radar is not None:
        data, ranges, az_res, metadata = radar
        shared = dict(protocol) | metadata
        patterns = {"natural": data, "all_missing_control": np.full_like(data, np.nan)}
        if args.controlled_gap:
            gapped = data.copy()
            middle = slice(3 * data.shape[0] // 8, 5 * data.shape[0] // 8)
            removed = np.zeros(data.shape, dtype=bool)
            removed[middle, :] = np.isfinite(data[middle, :])
            gapped[removed] = np.nan
            patterns["controlled_extra_gap"] = gapped
            shared.update(settings_values(controlled_gap_removed_observed_count=int(removed.sum())))
        for name, values in patterns.items():
            for width in WIDTHS:
                result = validate_case(toolkit, run_id, name, values, ranges, az_res, width, shared,
                                       save_support_arrays=args.save_support_arrays)
                new_rows, new_diag, new_radial, new_cases, new_archive = result
                rows.extend(new_rows)
                diagnostics.extend(new_diag)
                radial.extend(new_radial)
                cases.update(new_cases)
                archive.update(new_archive)
    # The primary check is the released default. Keep all nondefault-floor
    # stress results, including strict-tolerance failures, in a separate CSV.
    # This separation changes neither the numerical calculations nor tolerances.
    default_suffix = f"_floor{DEFAULT_FLOOR:.0e}"
    sensitivity = [row for row in rows if "_floor" in row["case_id"]
                   and not row["case_id"].endswith(default_suffix)]
    primary = [row for row in rows if "_floor" not in row["case_id"]
               or row["case_id"].endswith(default_suffix)]
    write_csv(directory / "comparison.csv", COMPARISON_FIELDS, primary)
    if sensitivity:
        write_csv(directory / "threshold_sensitivity.csv", COMPARISON_FIELDS, sensitivity)
    if diagnostics:
        write_csv(directory / "support_diagnostics.csv", DIAGNOSTIC_FIELDS, diagnostics)
        write_csv(directory / "support_by_range.csv", RADIAL_FIELDS, radial)
        if args.save_support_arrays:
            np.savez_compressed(directory / "raw_support.npz", **archive)
    sensitivity_failures = sum(row["status"] == "fail" for row in sensitivity)
    # A stress-test change in mask response/acceptance is still a correctness
    # failure. Only ratio/moment precision under smaller denominators is kept
    # descriptive unless --strict-sensitivity requests a failing exit status.
    mask_failures = any(row["status"] == "fail" and row["statistic"] in
                        ("raw_mask_response", "direct_acceptance", "fallback_eligibility")
                        for row in sensitivity)
    failed = (any(row["status"] == "fail" for row in primary) or mask_failures
              or (args.strict_sensitivity and sensitivity_failures > 0))
    for case in cases:
        is_stress = "_floor" in case and not case.endswith(default_suffix)
        cases[case].update(settings_values(
            comparison_output="threshold_sensitivity.csv" if is_stress else "comparison.csv",
            strict_sensitivity_failures_this_case=sum(
                row["status"] == "fail" for row in sensitivity if row["case_id"] == case),
        ))
    if not args.skip_timing and not failed:
        assert radar is not None
        runs, summary = benchmark(toolkit, run_id, radar[0], radar[2], args.repeats, args.warmups)
        write_csv(directory / "benchmark_runs.csv", BENCHMARK_FIELDS, runs)
        write_csv(directory / "benchmark_summary.csv", SUMMARY_FIELDS, summary)
        failed |= any(row["status"] == "failed" for row in runs)
        for case in cases:
            eligible = cases[case].get("benchmark_case_eligible", (False, ""))[0]
            cases[case].update(settings_values(
                full_sweep_timing_performed=eligible,
                timing_status=("measured; see benchmark_runs.csv" if eligible
                               else "not timed: control, self-test, or nondefault floor"),
            ))
    elif failed:
        for case in cases:
            cases[case].update(settings_values(timing_status="not run: numerical verification failed",
                                              full_sweep_timing_performed=False))
    write_settings(directory / "settings.csv", run_id, cases)
    print(f"{directory}: {'fail' if failed else 'pass'}; "
          f"{len(primary)} primary comparisons; {len(sensitivity)} sensitivity comparisons, "
          f"{sensitivity_failures} strict stress-tolerance failures retained; "
          f"{'no timings' if args.skip_timing or failed else 'mean-only timings saved'}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
