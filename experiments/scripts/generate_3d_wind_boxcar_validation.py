#!/usr/bin/env python3
"""Componentwise 3-D boxcar checks and matched seven-filter benchmarks.

Examples (use the pinned runtime and thread settings described in the plan)::

    python generate_3d_wind_boxcar_validation.py --mode numerical --cases smoke
    python generate_3d_wind_boxcar_validation.py --mode all --cases small20 --repeats 3

The opt-in ``--figure-only`` reproduces just the approved baseline20 case and
exports observed/mean 3D views and a mean/std section, without timing or numerical
CSV writes. Its sidecar also records separate component TP/FP/FN diagnostics for
the Discussion; no iterative rejection or threshold calibration is performed.
Otherwise no plots are generated. ``--methods`` limits the timed/comparison methods;
NumPy, shared-moment DCT, and the released statistics API are always checked
numerically. The API is a validation target, not a timed method. Small plane,
block, all-missing, and single-observation controls are always checked before
timing. Each timed repetition gets a new interpreter, with no filter warm-up.
Every method prepares its filter once inside the timed call and applies it
seven times: one common mask, then x and x**2 for each component. All methods
use exactly the same normalization and output construction. DCT transfer
functions come from the verified release; their reuse is implemented here,
not an optimization claimed for the released statistics API. All workers
preload the same libraries; RAM is total process RSS, not filter allocations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import psutil

from _review_common import (
    BENCHMARK_FIELDS,
    COMPARISON_FIELDS,
    SUMMARY_FIELDS,
    create_run_directory,
    load_toolkit,
    write_csv,
    write_settings,
)


COMPONENTS = ("u", "v", "w")
STATISTICS = ("mean", "variance", "std", "support")
SPECTRAL = "dct_shared_moments"
API_VALIDATION = "dct_toolkit_api_validation_only"
METHODS = (SPECTRAL, "numpy_windows", "scipy_uniform", "astropy_direct", "astropy_fft")
OPERATION = "componentwise_moments_shared_mask_seven_filters_setup_included"
FULL_FILTER_CALLS = 7
DENOMINATOR_FLOOR = 1e-3
ATOL = {"mean": 2e-11, "variance": 2e-11, "std": 2e-10, "support": 2e-10}
RTOL = 2e-10
NOISE_SEED, OUTLIER_SEED, MISSING_SEED = 4101, 4102, 4103
NOISE_STD, OUTLIER_FRACTION = 0.14, 0.003
ANOMALY_K, ANOMALY_DENSITY_MIN, ANOMALY_STD_MIN = 3.0, 0.35, 1e-8
RSS_SAMPLE_INTERVAL_S = 0.001
MIB = 1024**2
APPLICATION_FIELDS = [
    "run_id", "case_id", "method", "component", "n_cells", "n_observed",
    "n_missing", "n_injected", "n_observed_injected", "n_eligible",
    "n_eligible_injected", "n_flagged", "k", "density_min", "std_min", "notes",
]


@dataclass(frozen=True)
class Case:
    shape: tuple[int, int, int]
    width: tuple[int, int, int] = (3, 7, 7)
    pattern: str = "random"
    missing_fraction: float = 0.2
    benchmark: bool = True


CASES = {
    "smoke": Case((6, 16, 16)),
    "small20": Case((18, 48, 48)),
    "baseline_complete": Case((18, 96, 96), missing_fraction=0.0),
    "baseline20": Case((18, 96, 96)),
    "baseline50": Case((18, 96, 96), missing_fraction=0.5),
    "larger20": Case((24, 128, 128)),
    "narrow20": Case((18, 96, 96), width=(3, 3, 3)),
    "broad20": Case((18, 96, 96), width=(5, 15, 15)),
    "plane_control": Case((18, 48, 48), pattern="plane", missing_fraction=0.0),
    "block_control": Case((10, 24, 24), pattern="block", missing_fraction=0.0, benchmark=False),
    "all_missing": Case((5, 12, 12), pattern="all_missing", missing_fraction=1.0, benchmark=False),
    "single_observation": Case((5, 12, 12), pattern="single_observation", benchmark=False),
}
DEFAULT_CASES = (
    "small20", "baseline_complete", "baseline20", "baseline50", "larger20",
    "narrow20", "broad20", "plane_control",
)
CONTROLS = ("plane_control", "block_control", "all_missing", "single_observation")


def synthetic_cyclone_wind_3d(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Field copied, not imported, from generate_synthetic_validation.py."""
    nz, ny, nx = shape
    z = np.linspace(-1.0, 1.0, nz)
    y = np.linspace(-1.0, 1.0, ny)
    x = np.linspace(-1.0, 1.0, nx)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    r = np.sqrt(xx**2 + yy**2) + 1e-6
    r0 = 0.45
    vtan = 1.15 * (r / r0) * np.exp(1.0 - (r / r0))
    shear = 1.0 + 0.30 * np.sin(np.pi * zz)
    u_tan = -vtan * (yy / r) * shear
    v_tan = vtan * (xx / r) * shear
    w_core = 0.24 * np.exp(-((r / 0.35) ** 2)) * np.cos(0.5 * np.pi * zz)
    u_bg = 0.05 * np.sin(2.0 * np.pi * zz) + 0.03 * xx
    v_bg = -0.04 * np.cos(1.5 * np.pi * zz) - 0.02 * yy
    w_bg = 0.04 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    return u_tan + u_bg, v_tan + v_bg, w_core + w_bg


def generate_input(case: Case):
    """Independent noise, vector-anomaly, and missingness streams."""
    truth = synthetic_cyclone_wind_3d(case.shape)
    noise_rng = np.random.default_rng(NOISE_SEED)
    outlier_rng = np.random.default_rng(OUTLIER_SEED)
    missing_rng = np.random.default_rng(MISSING_SEED)
    injected = outlier_rng.random(case.shape) < OUTLIER_FRACTION
    phi = outlier_rng.uniform(0.0, 2.0 * np.pi, case.shape)
    cos_theta = outlier_rng.uniform(-1.0, 1.0, case.shape)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    amplitude = outlier_rng.uniform(1.25, 2.25, case.shape)
    directions = (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
    valid = np.ones(case.shape, dtype=bool)
    mask_description = "independent uniform variates >= requested missing fraction"
    if case.pattern == "random":
        valid = missing_rng.random(case.shape) >= case.missing_fraction
    elif case.pattern == "plane":
        valid[case.shape[0] // 2, :, :] = False
        mask_description = f"one complete z plane at index {case.shape[0] // 2}"
    elif case.pattern == "block":
        block = tuple(slice(n // 4, 3 * n // 4) for n in case.shape)
        valid[block] = False
        mask_description = "central block: slice(n//4, 3*n//4) on every spatial axis"
    elif case.pattern == "all_missing":
        valid[:] = False
        mask_description = "all observations missing"
    elif case.pattern == "single_observation":
        valid[:] = False
        valid[tuple(n // 2 for n in case.shape)] = True
        mask_description = "only the central cell observed; zero-variance/sqrt control"
    else:
        raise ValueError(f"Unknown missingness pattern: {case.pattern}")
    observed = tuple(
        np.where(
            valid,
            component + noise_rng.normal(0.0, NOISE_STD, case.shape)
            + injected * amplitude * direction,
            np.nan,
        )
        for component, direction in zip(truth, directions)
    )
    input_hash = hashlib.sha256()
    for component in observed:
        input_hash.update(memoryview(component))
    settings = {
        "actual_missing_fraction": float(np.mean(~valid)),
        "n_observed_per_component": int(np.count_nonzero(valid)),
        "n_missing_per_component": int(np.count_nonzero(~valid)),
        "n_injected_before_missingness": int(np.count_nonzero(injected)),
        "n_injected_observed": int(np.count_nonzero(injected & valid)),
        "n_injected_missing": int(np.count_nonzero(injected & ~valid)),
        "missing_mask_definition": mask_description,
        "input_sha256": input_hash.hexdigest(),
    }
    return observed, injected, settings


def prepare_backend(method: str):
    """Dependency import/release verification, excluded from calculation timing."""
    if method in (SPECTRAL, API_VALIDATION):
        return load_toolkit()
    if method == "scipy_uniform":
        from scipy.ndimage import uniform_filter
        return uniform_filter
    if method.startswith("astropy_"):
        from astropy.convolution import convolve, convolve_fft
        return convolve if method == "astropy_direct" else convolve_fft
    if method == "numpy_windows":
        return None
    raise ValueError(f"Unknown method: {method}")


def prepare_filter(method: str, shape, width: tuple[int, int, int], backend, chunk_planes: int):
    """Prepare once per calculation, inside timing; reuse for all seven fields.

    DCT follows released cartesian.py's dctn/multiply/idctn sequence, but moves
    its one-dimensional transfer construction outside the seven applications.
    Spatial references never use toolkit functions or spectral weights.
    """
    if len(width) != len(shape) or any(w < 1 or w % 2 != 1 for w in width):
        raise ValueError("Matched filters require one positive odd width per spatial axis")
    if method == SPECTRAL:
        from scipy.fft import dctn, idctn

        transfers = []
        for axis, (length, axis_width) in enumerate(zip(shape, width)):
            transfer_shape = [1] * len(shape)
            transfer_shape[axis] = length
            transfers.append(backend.get_dct_transfer_function(
                length, "boxcar_discrete", float(axis_width),
            ).reshape(transfer_shape))

        def dct_filter(a):
            spectrum = dctn(a, type=2, norm="ortho", workers=1)
            for transfer in transfers:
                spectrum *= transfer
            return idctn(spectrum, type=2, norm="ortho", workers=1)

        return dct_filter

    padding = tuple((n // 2, n // 2) for n in width)
    if method == "scipy_uniform":
        return lambda a: backend(a, size=width, mode="reflect", origin=0)
    if method == "numpy_windows":
        def window_mean(a):
            padded = np.pad(a, padding, mode="symmetric")
            windows = np.lib.stride_tricks.sliding_window_view(padded, width)
            result = np.empty(a.shape, dtype=np.float64)
            for start in range(0, a.shape[0], chunk_planes):
                end = min(start + chunk_planes, a.shape[0])
                # The windows are a strided view, not a materialized 6-D array.
                result[start:end] = np.mean(windows[start:end], axis=(-3, -2, -1))
            return result
        return window_mean
    kernel = np.full(width, 1.0 / np.prod(width), dtype=np.float64)

    def astropy_filter(a):
        # NumPy symmetric is half-sample reflection, matching ndimage.reflect.
        padded = np.pad(a, padding, mode="symmetric")
        options = dict(
            boundary="fill", fill_value=0.0, nan_treatment="fill",
            normalize_kernel=False, preserve_nan=False,
        )
        if method == "astropy_fft":
            options.update(fft_pad=True, psf_pad=True, crop=True, allow_huge=False)
        filtered = backend(padded, kernel, **options)
        crop = tuple(slice(n // 2, n // 2 + size) for n, size in zip(width, a.shape))
        return filtered[crop].copy()

    return astropy_filter


def common_valid_mask(observed):
    valid = np.ones(observed[0].shape, dtype=bool)
    for component in observed:
        np.logical_and(valid, np.isfinite(component), out=valid)
    return valid


def calculate_toolkit_api(observed, width, toolkit):
    """Released 19-filter statistics workflow: correctness only, never timed."""
    valid = common_valid_mask(observed)
    common = dict(width=width, coordinates="cartesian", kernel_type="boxcar_discrete",
                  restore_input_nan=False)
    support = toolkit.dct_count(valid, **common)
    moments = dict(common, mask=valid, _stable_fallback=False,
                   _denominator_floor=DENOMINATOR_FLOOR)
    result = {}
    for name, component in zip(COMPONENTS, observed):
        mean = toolkit.dct_mean(component, **moments)
        variance = toolkit.dct_variance(component, **moments)
        result[name] = dict(mean=mean, variance=variance, std=np.sqrt(variance), support=support)
    return result


def calculate(method: str, observed, width, backend, chunk_planes: int = 1, *, audit=None):
    """Same preparation, seven filter calls, and 12 outputs for every method."""
    valid = common_valid_mask(observed)
    result = {}
    prepared = prepare_filter(method, valid.shape, width, backend, chunk_planes)
    calls = 0

    def linear_filter(values):
        nonlocal calls
        calls += 1
        return prepared(values)

    density = linear_filter(valid.astype(np.float64))
    supported = np.isfinite(density) & (density > DENOMINATOR_FLOOR)
    support = np.clip(density, 0.0, 1.0) * np.prod(width)
    for name, component in zip(COMPONENTS, observed):
        filled = np.where(valid, component, 0.0)
        numerator = linear_filter(filled)
        second_numerator = linear_filter(filled * filled)
        mean = np.full(component.shape, np.nan)
        second = np.full(component.shape, np.nan)
        np.divide(numerator, density, out=mean, where=supported)
        np.divide(second_numerator, density, out=second, where=supported)
        variance = np.maximum(second - mean * mean, 0.0)
        result[name] = dict(mean=mean, variance=variance, std=np.sqrt(variance),
                             support=support)
    if audit is not None:
        audit.update(full_nd_filter_calls=calls, filter_preparations=1)
    if calls != FULL_FILTER_CALLS:
        raise RuntimeError(f"Matched workload requires seven filters, received {calls}")
    return result


def compare_outputs(run_id, case_id, method, output, reference, observed):
    rows = []
    valid = np.logical_and.reduce([np.isfinite(a) for a in observed])
    regions = {"whole_domain": np.ones(valid.shape, bool), "observed": valid, "missing": ~valid}
    for component in COMPONENTS:
        for statistic in STATISTICS:
            a, b = output[component][statistic], reference[component][statistic]
            comparison_regions = dict(regions)
            if statistic == "std":
                near_zero = reference[component]["variance"] <= ATOL["variance"]
                comparison_regions["near_zero_variance"] = near_zero
                comparison_regions["nonzero_variance"] = ~near_zero & np.isfinite(b)
            for region, selection in comparison_regions.items():
                aa, bb = a[selection], b[selection]
                finite_a, finite_b = np.isfinite(aa), np.isfinite(bb)
                jointly_finite = finite_a & finite_b
                delta = aa[jointly_finite] - bb[jointly_finite]
                tolerance = ATOL[statistic] + RTOL * np.abs(bb[jointly_finite])
                notes = "No fallback; undefined masks must agree, not only overlapping values."
                reported_atol = ATOL[statistic]
                if statistic == "std":
                    # sqrt is ill-conditioned at zero. The separately checked
                    # variance tolerance is NOT relaxed. Use its implied sqrt
                    # bound only when the reference variance is near zero.
                    ref_var = reference[component]["variance"][selection][jointly_finite]
                    near_zero = ref_var <= ATOL["variance"]
                    tolerance[near_zero] = np.maximum(
                        tolerance[near_zero],
                        np.sqrt(ATOL["variance"] + RTOL * np.abs(ref_var[near_zero])),
                    )
                    if np.any(near_zero):
                        reported_atol = float(np.sqrt(ATOL["variance"]))
                    notes += " Near-zero std uses sqrt(variance tolerance); variance is checked independently."
                masks_match = np.array_equal(finite_a, finite_b)
                no_infinities = not (np.any(np.isinf(aa)) or np.any(np.isinf(bb)))
                matches = masks_match and no_infinities and np.all(np.abs(delta) <= tolerance)
                status = "pass" if matches else "fail"
                if matches and not np.any(jointly_finite):
                    status = "no_comparable"
                rows.append(dict(
                    run_id=run_id, case_id=case_id, method=method, reference_method="numpy_windows",
                    component=component, statistic=statistic, region=region,
                    estimate_type="mask_response" if statistic == "support" else "direct_ratio",
                    kernel_match="exact_odd_discrete_boxcar", n_considered=int(selection.sum()),
                    n_compared=int(jointly_finite.sum()), n_method_undefined=int((~finite_a).sum()),
                    n_reference_undefined=int((~finite_b).sum()), n_method_gap_filled=0,
                    n_reference_gap_filled=0,
                    max_abs_difference=float(np.max(np.abs(delta))) if delta.size else None,
                    rmse=float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
                    atol=reported_atol, rtol=RTOL, status=status,
                    units={"mean": "velocity_unit", "std": "velocity_unit",
                           "variance": "velocity_unit^2", "support": "nominal_grid_cells"}[statistic],
                    notes=notes,
                ))
    return rows


def error_comparisons(run_id, case_id, method, reason):
    return [dict(
        run_id=run_id, case_id=case_id, method=method, reference_method="numpy_windows",
        component=component, statistic=statistic, region="whole_domain", status="method_error",
        estimate_type="direct_ratio" if statistic != "support" else "mask_response",
        kernel_match="exact_odd_discrete_boxcar", notes=reason,
    ) for component in COMPONENTS for statistic in STATISTICS]


def application_counts(run_id, case_id, observed, injected, output, width):
    """Descriptive counts, not a tuned detector or a detection leaderboard."""
    rows = []
    for name, data in zip(COMPONENTS, observed):
        stats = output[name]
        finite = np.isfinite(data)
        density = stats["support"] / np.prod(width)
        eligible = (finite & (density >= ANOMALY_DENSITY_MIN)
                    & np.isfinite(stats["mean"]) & np.isfinite(stats["std"])
                    & (stats["std"] > ANOMALY_STD_MIN))
        flagged = eligible & (np.abs(data - stats["mean"]) > ANOMALY_K * stats["std"])
        rows.append(dict(
            run_id=run_id, case_id=case_id, method=SPECTRAL, component=name,
            n_cells=data.size, n_observed=int(finite.sum()), n_missing=int((~finite).sum()),
            n_injected=int(injected.sum()), n_observed_injected=int((injected & finite).sum()),
            n_eligible=int(eligible.sum()), n_eligible_injected=int((injected & eligible).sum()),
            n_flagged=int(flagged.sum()), k=ANOMALY_K, density_min=ANOMALY_DENSITY_MIN,
            std_min=ANOMALY_STD_MIN,
            notes="Separate component flags; candidate included in own moments; fixed illustrative threshold, not optimized.",
        ))
    return rows


def lifetime_peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux (and the usual other Unix runtimes) report KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def worker(args):
    """One fresh process, one measured calculation, no previous filter call."""
    payload: dict[str, Any] = dict(status="error", elapsed_s=None, ram_before_mib=None,
                                   peak_ram_mib=None, reason="worker did not finish")
    stop = threading.Event()
    sampler = None
    try:
        case_id = split_names(args.cases, CASES)[0]
        method = split_names(args.methods, METHODS)[0]
        case = CASES[case_id]
        # Match the imported library footprint without any filter warm-up.
        # Imports/release verification are outside the timed calculation.
        backends = {name: prepare_backend(name) for name in METHODS}
        backend = backends[method]
        observed, injected, generation = generate_input(case)
        payload.update(worker_pid=psutil.Process().pid, case_id=case_id, method=method,
                       input_sha256=generation["input_sha256"])
        del injected, generation
        gc.collect()
        process = psutil.Process()
        before = process.memory_info().rss
        payload["ram_before_mib"] = before / MIB
        peak = [before]
        ready = threading.Event()

        def sample():
            peak[0] = max(peak[0], process.memory_info().rss)
            ready.set()
            while not stop.wait(RSS_SAMPLE_INTERVAL_S):
                peak[0] = max(peak[0], process.memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        ready.wait()
        precall_threads = process.num_threads()
        before = process.memory_info().rss
        payload["ram_before_mib"] = before / MIB
        peak[0] = max(peak[0], before)
        hwm_before = lifetime_peak_bytes()
        audit = {}
        start = time.perf_counter()
        output = calculate(method, observed, case.width, backend, args.numpy_chunk_planes, audit=audit)
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        hwm_after = lifetime_peak_bytes()
        stop.set()
        sampler.join()
        # A lifetime high-water mark is usable only if THIS call exceeded it.
        # Otherwise use in-call samples, never a generation/import memory peak.
        new_high_water = hwm_after > hwm_before
        if new_high_water:
            peak[0] = max(peak[0], hwm_after)
        payload.update(
            status="success", elapsed_s=elapsed, ram_before_mib=before / MIB,
            peak_ram_mib=peak[0] / MIB, reason="",
            precall_lifetime_peak_mib=hwm_before / MIB,
            final_lifetime_peak_mib=hwm_after / MIB,
            peak_source="in-call RSS samples + newly exceeded ru_maxrss" if new_high_water else "in-call RSS samples",
            rss_sample_interval_s=RSS_SAMPLE_INTERVAL_S,
            precall_process_threads_including_rss_sampler=precall_threads,
            n_outputs=sum(len(stats) for stats in output.values()),
            calculation_protocol="shared_mask_seven_filters",
            preloaded_methods=list(backends),
            full_nd_filter_calls=audit["full_nd_filter_calls"],
            filter_preparations=audit["filter_preparations"],
        )
        if args.max_ram_mib > 0 and peak[0] / MIB > args.max_ram_mib:
            payload.update(status="resource_limit", reason="measured calculation peak exceeded max_ram_mib")
    except Exception as error:
        payload.update(reason=f"{type(error).__name__}: {error}")
    finally:
        stop.set()
        if sampler is not None:
            sampler.join()
    print(json.dumps(payload), flush=True)
    return 0 if payload["status"] == "success" else 1


def measure_worker(args, case_id, method):
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--cases", case_id,
        "--methods", method, "--numpy-chunk-planes", str(args.numpy_chunk_planes),
        "--max-ram-mib", str(args.max_ram_mib),
    ]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process = psutil.Process(child.pid)
    start = time.monotonic()
    forced_status, reason = None, ""
    observed_peak = 0
    while child.poll() is None:
        try:
            observed_peak = max(observed_peak, process.memory_info().rss)
        except psutil.NoSuchProcess:
            pass
        if args.timeout_s > 0 and time.monotonic() - start > args.timeout_s:
            forced_status, reason = "timeout", f"worker lifetime exceeded {args.timeout_s:g} s (includes setup)"
        elif args.max_ram_mib > 0 and observed_peak / MIB > args.max_ram_mib:
            forced_status, reason = "resource_limit", f"worker RSS exceeded {args.max_ram_mib:g} MiB (includes setup)"
        if forced_status:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
            break
        time.sleep(0.01)
    stdout, stderr = child.communicate()
    if forced_status:
        return dict(status=forced_status, elapsed_s=None, ram_before_mib=None,
                    peak_ram_mib=None, reason=reason, worker_lifetime_observed_peak_mib=observed_peak / MIB)
    try:
        payload = json.loads(stdout)
    except (ValueError, TypeError):
        return dict(status="error", elapsed_s=None, ram_before_mib=None, peak_ram_mib=None,
                    reason=f"worker exit {child.returncode}; {stderr.strip() or stdout.strip() or 'no result'}")
    if child.returncode and payload.get("status") == "success":
        payload.update(status="error", reason=f"worker exited {child.returncode} despite success payload")
    if stderr.strip():
        payload["worker_stderr"] = stderr.strip()
    return payload


def summarize_benchmarks(rows):
    summaries = []
    for case_id, method in dict.fromkeys((row["case_id"], row["method"]) for row in rows):
        group = [row for row in rows if row["case_id"] == case_id and row["method"] == method]
        success = [row for row in group if row["status"] == "success"]
        times = [row["elapsed_s"] for row in success]
        summaries.append(dict(
            run_id=group[0]["run_id"], case_id=case_id, method=method, operation=OPERATION,
            n_success=len(success), n_failure=len(group) - len(success),
            median_time_s=float(np.median(times)) if times else None,
            q25_time_s=float(np.quantile(times, 0.25)) if times else None,
            q75_time_s=float(np.quantile(times, 0.75)) if times else None,
            max_peak_ram_mib=max(row["peak_ram_mib"] for row in success) if success else None,
        ))
    return summaries


def case_settings(case, args):
    return {
        "shape_zyx": list(case.shape), "component_axis_filtered": False,
        "components": list(COMPONENTS), "velocity_units": "arbitrary normalized velocity units",
        "field_source": "generate_synthetic_validation.py::synthetic_cyclone_wind_3d (copied without import)",
        "field_definition_sha256": hashlib.sha256(inspect.getsource(synthetic_cyclone_wind_3d).encode()).hexdigest(),
        "field_randomness": "deterministic analytic cyclone; no random field seed",
        "noise_seed": NOISE_SEED, "outlier_seed": OUTLIER_SEED, "missing_seed": MISSING_SEED,
        "noise_std": (NOISE_STD, "velocity_unit"), "outlier_fraction_requested": OUTLIER_FRACTION,
        "outlier_vector_amplitude_interval": [1.25, 2.25],
        "outlier_direction": "isotropic unit direction: uniform phi and cos(theta)",
        "seed_reuse_across_cases": "same seeds; common realizations for equal shapes; nested random missing masks",
        "missing_pattern": case.pattern, "requested_random_missing_fraction": case.missing_fraction,
        "validity_rule": "logical AND of isfinite(u), isfinite(v), isfinite(w); common genuine NaN mask",
        "missingness_independent_of_anomalies": True, "dtype": "float64",
        "kernel": "boxcar_discrete", "width_zyx": list(case.width),
        "window_volume": int(np.prod(case.width)), "population_variance_ddof": 0,
        "boundary": "half-sample symmetric; scipy.ndimage reflect on z,y,x",
        "astropy_boundary": "explicit numpy symmetric half-window padding, fill convolution, crop to original domain",
        "astropy_nan_treatment": "fill (inputs explicitly zero-replaced); preserve_nan=False; normalize_kernel=False",
        "astropy_fft_settings": "fft_pad=True; psf_pad=True; crop=True; allow_huge=False",
        "denominator_floor": DENOMINATOR_FLOOR, "denominator_rule": "finite d strictly > floor after full N-D operator",
        "stable_fallback": False, "restore_input_nan": False, "gap_filling_benchmarked": False,
        "unsupported_output": "mean/variance/std NaN; nominal support remains available (zero at no support)",
        "nominal_support_definition": "clip(filtered common finite mask,0,1)*product(requested odd widths); not ESS or distinct count",
        "variance_negative_handling": "maximum(second moment - mean squared, 0), same as released API",
        "absolute_tolerances": ATOL, "relative_tolerance": RTOL,
        "std_near_zero_rule": "when reference variance <= variance atol, use max(std tolerance,sqrt(variance tolerance)); independent variance test unchanged",
        "comparison_reference": "NumPy padded sliding-window reductions, no toolkit filter calls",
        "numpy_chunk_planes": args.numpy_chunk_planes,
        "numpy_window_storage": "sliding_window_view; reduce at most chunk_planes output z planes; no window materialization",
        "calculation_protocol": "shared_mask_seven_filters",
        "shared_reuse_all_methods": "one filtered common mask and acceptance mask; each component's mean reused for variance; one support array shared across components",
        "filter_setup_reuse": "once per timed calculation: DCT one transfer per axis, Astropy one normalized dense boxcar, NumPy/SciPy implicit boxcar; reused for all seven fields",
        "astropy_fft_internal_preparation": "dense spatial kernel reused; convolve_fft retains its own per-call kernel FFT and padding work, included in measured time; no cached-kernel FFT claim",
        "reference_reuse": "identical shared-moment calculation for all methods; only the linear filtering backend differs",
        "reference_padding_reuse": "each filtered field is padded separately where needed; padding and crop copies are timed",
        "spectral_method": SPECTRAL,
        "spectral_implementation": "experiment-local shared moments and prepared N-D DCT filter; NOT timing the released statistics API",
        "spectral_source": "released get_dct_transfer_function(boxcar_discrete); cartesian.py dctn/type2/ortho, broadcast transfers, idctn/type2/ortho; transfers moved before seven applications",
        "spectral_fft_workers": 1,
        "number_full_nd_filter_calls_all_benchmarked_methods": FULL_FILTER_CALLS,
        "spectral_number_full_nd_filter_calls": FULL_FILTER_CALLS,
        "reference_number_full_nd_filter_calls": FULL_FILTER_CALLS,
        "released_api_validation_method": API_VALIDATION,
        "released_api_validation_calls": "1 dct_count; 3 dct_mean; 3 dct_variance; std=sqrt(returned variance); compared numerically only, excluded from timing/RAM",
        "released_api_validation_full_nd_filter_calls": 19,
        "anomaly_k_fixed_before_run": ANOMALY_K, "anomaly_density_min": ANOMALY_DENSITY_MIN,
        "anomaly_std_min": (ANOMALY_STD_MIN, "velocity_unit"),
        "anomaly_criterion": "observed & density>=0.35 & finite mean/std & std>1e-8; flag abs(x-mean)>3*std",
        "anomaly_scope": "component flags separate; inclusive moments retain candidate self-influence; descriptive illustration, no threshold tuning, F1 or superiority claim",
        "mode": args.mode, "requested_methods": split_names(args.methods, METHODS),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "machine_total_ram_mib": (psutil.virtual_memory().total / MIB, "MiB"),
        "benchmark_operation": OPERATION, "benchmark_repeats": args.repeats,
        "benchmark_warmups": 0, "benchmark_concurrency": 1,
        "benchmark_method_order": "rotate methods by case index and repetition; fresh worker for every call",
        "benchmark_timer": "time.perf_counter, wall seconds",
        "benchmark_includes": "common mask, zero replacement, padding, kernels/transfers, full filters, ratios, variance, sqrt, output allocations",
        "benchmark_excludes": "interpreter startup/import/release check, generation/loading, GC before call, CSV, plots; illustrative anomaly counts",
        "benchmark_input_resident_before_timer": True,
        "benchmark_library_import_policy": "preload the same verified toolkit, SciPy and Astropy backends in every worker; imports excluded; no filter warm-up",
        "memory_units": "MiB = 1048576 bytes; total RSS, not Python-only allocations or peak-minus-baseline",
        "memory_measurement": "fresh interpreter per repetition; psutil in-call RSS sampling; resource ru_maxrss only if call exceeds precall lifetime peak; macOS bytes/Linux KiB converted",
        "memory_sampling_interval_s": (RSS_SAMPLE_INTERVAL_S, "s"),
        "memory_limitation": "if lifetime high-water mark is not exceeded, sub-sampling-interval transient peaks can be missed; no earlier-case/warm-up peaks included",
        "worker_timeout_s": (args.timeout_s, "s"), "worker_max_ram_mib": (args.max_ram_mib, "MiB"),
        "case_max_voxels": args.max_voxels, "case_max_window_volume": args.max_window_volume,
        "resource_limit_policy": "parent checks worker total RSS every 10ms including setup; worker also checks in-call peak; failures retained with blank unavailable measurements",
    }


def split_names(value, choices):
    names = list(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    unknown = set(names) - set(choices)
    if not names or unknown:
        raise ValueError(f"Invalid selection {value!r}; choices: {', '.join(choices)}")
    return names


FIGURE_CASE_ID = "baseline20"
FIGURE_RUN_ID = "lead-review-3d-matched-20261027"
FIGURE_RUN_DIRECTORY = Path(__file__).resolve().parents[1] / "results" / "round1" / "3d_wind_boxcar" / FIGURE_RUN_ID
FIGURE_SOURCE_COMMIT = "c66106488a4e17bfb45260f4764106c5682254de"
FIGURE_ORIGINAL_SCRIPT_SHA256 = "4182f4b86d52ed0e5f9d244e19310e2b345da4557d884235dbc6acf52ab3e4b9"
FIGURE_SAVED_CSV_SHA256 = {
    "settings.csv": "88bbd9faf46b9f4e4a8a671eda5511ac35b6dab072b90728e335bd8c62bd02bd",
    "comparison.csv": "1d80023522a6aa37e0ee3a59b9fd152d6095338b6cf6381211c699871bce991e",
    "application_counts.csv": "a51c24dded32411a15c5eb19d48188e16c5bc7a7270a39c5bc031c51013c4dc1",
    "benchmark_runs.csv": "e02f7e626d1e570b3968f39a5e3eddbc3ea23061440c40ddfc5a4a30fc805294",
    "benchmark_summary.csv": "f040790b5831ee1296eb8770ca9ca50b9f777c154daf69e89c8c781e35a5b045",
}
FIGURE_STEM = "wind_component_statistics"
FIGURE_SIZE = (6.5, 7.0)
FIGURE_SLICE_INDEX = 9
FIGURE_ARROW_STRIDE = 6
FIGURE_ARROW_OFFSET = 3
FIGURE_ARROW_SCALE = 10.0
FIGURE_ARROW_KEY = 1.0
FIGURE_VOLUME_STRIDE = (3, 8, 8)
FIGURE_VOLUME_OFFSET = (1, 4, 4)
FIGURE_VOLUME_ARROW_LENGTH = 0.11
FIGURE_INJECTION_SAMPLE_MAX = 100
FIGURE_VIEW = (22, 35)
FIGURE_SPEED_LIMITS = (0.0, 1.6)
FIGURE_W_LIMITS = (-0.3, 0.3)
FIGURE_STD_LIMITS = (0.0, 0.4)
FIGURE_MISSING_COLOR = "#bdbdbd"
FIGURE_INJECTION_COLOR = "#d62728"


def _figure_saved_evidence(saved_run: Path = FIGURE_RUN_DIRECTORY):
    """Fail closed on changed frozen CSVs or an unverified historical source."""
    import csv

    saved_run = Path(saved_run).resolve()
    tables = {}
    for filename, expected in FIGURE_SAVED_CSV_SHA256.items():
        path = saved_run / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Saved figure evidence SHA-256 mismatch: {filename}")
        with path.open(newline="", encoding="utf-8") as stream:
            tables[filename] = [row for row in csv.DictReader(stream)
                                if row["case_id"] == FIGURE_CASE_ID]
        if not tables[filename] or any(row["run_id"] != FIGURE_RUN_ID for row in tables[filename]):
            raise ValueError(f"Saved figure case/run mismatch: {filename}")
    settings = {row["setting"]: row["value"] for row in tables["settings.csv"]}
    if len(settings) != len(tables["settings.csv"]):
        raise ValueError("Duplicate saved figure settings")
    _figure_check_settings(settings)
    original = subprocess.run(
        ["git", "show", f"{FIGURE_SOURCE_COMMIT}:experiments/scripts/{Path(__file__).name}"],
        cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True,
    ).stdout
    if hashlib.sha256(original).hexdigest() != settings["script_sha256"]:
        raise ValueError("Immutable pre-figure source does not match recorded script_sha256")
    return settings, tables


def _figure_check_settings(settings):
    """Compare the single selected case to its recorded scientific settings."""
    case = CASES[FIGURE_CASE_ID]
    expected = {
        "script_sha256": FIGURE_ORIGINAL_SCRIPT_SHA256,
        "common_helper_sha256": hashlib.sha256(Path(__file__).with_name("_review_common.py").read_bytes()).hexdigest(),
        "field_definition_sha256": hashlib.sha256(inspect.getsource(synthetic_cyclone_wind_3d).encode()).hexdigest(),
        "shape_zyx": json.dumps(list(case.shape)), "width_zyx": json.dumps(list(case.width)),
        "components": json.dumps(list(COMPONENTS)), "component_axis_filtered": "False",
        "noise_seed": str(NOISE_SEED), "outlier_seed": str(OUTLIER_SEED), "missing_seed": str(MISSING_SEED),
        "noise_std": str(NOISE_STD), "outlier_fraction_requested": str(OUTLIER_FRACTION),
        "missing_pattern": case.pattern, "requested_random_missing_fraction": str(case.missing_fraction),
        "window_volume": str(int(np.prod(case.width))), "kernel": "boxcar_discrete",
        "boundary": "half-sample symmetric; scipy.ndimage reflect on z,y,x",
        "dtype": "float64", "population_variance_ddof": "0",
        "denominator_floor": str(DENOMINATOR_FLOOR),
        "denominator_rule": "finite d strictly > floor after full N-D operator",
        "stable_fallback": "False", "restore_input_nan": "False", "numerical_status": "pass",
        "calculation_protocol": "shared_mask_seven_filters", "numpy_chunk_planes": "1",
        "spectral_number_full_nd_filter_calls": str(FULL_FILTER_CALLS),
        "reference_number_full_nd_filter_calls": str(FULL_FILTER_CALLS),
        "absolute_tolerances": json.dumps(ATOL, sort_keys=True), "relative_tolerance": str(RTOL),
        "anomaly_k_fixed_before_run": str(ANOMALY_K), "anomaly_density_min": str(ANOMALY_DENSITY_MIN),
        "anomaly_std_min": str(ANOMALY_STD_MIN),
        "dct_toolkit_commit": "9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c", "dct_toolkit_version": "1.0.0",
    }
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Saved figure setting mismatch: {key}")


def _figure_check_comparisons(actual, saved):
    """Verify all component/region masks, counts and fixed-tolerance agreements."""
    identity = ("component", "statistic", "region")
    indexed = {tuple(row[key] for key in identity): row for row in saved}
    if len(indexed) != len(saved) or len(actual) != len(saved):
        raise ValueError("Saved component comparison row count changed")
    for row in actual:
        old = indexed.get(tuple(row[key] for key in identity), {})
        if row["status"] not in ("pass", "no_comparable"):
            raise ValueError(f"Full-3D figure comparison failed: {row}")
        for key in COMPARISON_FIELDS:
            if key in ("max_abs_difference", "rmse") and row[key] is not None:
                if not old.get(key) or not np.isclose(
                        float(old[key]), row[key], atol=ATOL[row["statistic"]], rtol=RTOL):
                    raise ValueError(f"Saved component discrepancy changed: {identity}, {key}")
            elif old.get(key) != ("" if row[key] is None else str(row[key])):
                raise ValueError(f"Saved component comparison mismatch: {key}")


def _figure_slice_fields(case):
    """Extract the plotted plane only AFTER the full-volume moments are checked."""
    observed, output = case["observed"], case["dct"]
    z = FIGURE_SLICE_INDEX
    ny, nx = observed[0].shape[1:]
    x, y = np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny)
    x_edges = np.r_[x[0] - (x[1] - x[0]) / 2, (x[:-1] + x[1:]) / 2, x[-1] + (x[-1] - x[-2]) / 2]
    y_edges = np.r_[y[0] - (y[1] - y[0]) / 2, (y[:-1] + y[1:]) / 2, y[-1] + (y[-1] - y[-2]) / 2]
    return dict(
        x=x, y=y, x_edges=x_edges, y_edges=y_edges,
        scalars=(output["w"]["mean"][z], output["w"]["std"][z]),
        vectors=(output["u"]["mean"][z], output["v"]["mean"][z]),
        vector_mask=case["accepted"][z],
    )


def _figure_volume_fields(case):
    """Sample one common spatial lattice; keep missing inputs out of the arrows."""
    shape = case["observed"][0].shape
    lattice = np.meshgrid(*(np.arange(offset, n, stride) for n, stride, offset
                            in zip(shape, FIGURE_VOLUME_STRIDE, FIGURE_VOLUME_OFFSET)), indexing="ij")
    indices = np.column_stack([a.ravel() for a in lattice])
    grid = tuple(np.linspace(-1, 1, n) for n in shape)

    def positions(zyx):
        return np.column_stack([grid[axis][zyx[:, axis]] for axis in (2, 1, 0)])

    def vectors(fields, zyx):
        return np.column_stack([a[tuple(zyx.T)] for a in fields])

    injection_indices = np.argwhere(case["injected"] & case["valid"])
    if len(injection_indices) > FIGURE_INJECTION_SAMPLE_MAX:
        selected = np.linspace(0, len(injection_indices) - 1, FIGURE_INJECTION_SAMPLE_MAX, dtype=int)
        injection_indices = injection_indices[selected]
    return dict(
        indices=indices, positions=positions(indices),
        observed=vectors(case["observed"], indices),
        mean=vectors(tuple(case["dct"][c]["mean"] for c in COMPONENTS), indices),
        observed_mask=case["valid"][tuple(indices.T)],
        mean_mask=case["accepted"][tuple(indices.T)],
        injection_indices=injection_indices, injection_positions=positions(injection_indices),
        injection_vectors=vectors(case["observed"], injection_indices),
    )


def _figure_screening_diagnostics(case):
    """Partition the unchanged component flags against eligible vector injections.

    The known injection mask is used only for these labels and the display, never
    to exclude observations from the moments or to choose a threshold. A vector
    injection need not cause a large departure in each signed component.
    """
    diagnostics = {}
    volume = np.prod(CASES[FIGURE_CASE_ID].width)
    for name, data in zip(COMPONENTS, case["observed"]):
        stats = case["dct"][name]
        eligible = (np.isfinite(data) & (stats["support"] / volume >= ANOMALY_DENSITY_MIN)
                    & np.isfinite(stats["mean"]) & np.isfinite(stats["std"])
                    & (stats["std"] > ANOMALY_STD_MIN))
        flagged = eligible & (np.abs(data - stats["mean"]) > ANOMALY_K * stats["std"])
        injected = case["injected"]
        masks = dict(eligible=eligible, flagged=flagged, tp=flagged & injected,
                     fp=flagged & ~injected, fn=eligible & injected & ~flagged)
        diagnostics[name] = dict(masks=masks, counts={f"n_{key}": int(mask.sum())
                                                   for key, mask in masks.items()})
    return diagnostics


def prepare_wind_figure(saved_run: Path = FIGURE_RUN_DIRECTORY):
    """Reproduce only baseline20, untimed, using unchanged full-3D calculations."""
    import importlib.metadata
    import os
    import platform

    from _review_common import TOOLKIT_COMMIT, _RELEASE_HASHES

    saved, tables = _figure_saved_evidence(saved_run)
    case = CASES[FIGURE_CASE_ID]
    toolkit = load_toolkit()
    observed, injected, generation = generate_input(case)
    for key, value in generation.items():
        if saved.get(key) != str(value):
            raise ValueError(f"Full approved input identity mismatch: {key}")
    valid = common_valid_mask(observed)
    if any(a.shape != case.shape or a.dtype != np.float64
           or not np.array_equal(np.isfinite(a), valid) for a in observed):
        raise ValueError("Full-3D input shape/dtype/common component mask changed")
    audits = {method: {} for method in (SPECTRAL, "numpy_windows")}
    reference = calculate("numpy_windows", observed, case.width, None, audit=audits["numpy_windows"])
    output = calculate(SPECTRAL, observed, case.width, toolkit, audit=audits[SPECTRAL])
    # Two separate, untimed mask-only checks expose raw d without altering
    # calculate() or deriving acceptance from its clipped CSV 'support' output.
    raw_density = prepare_filter(SPECTRAL, case.shape, case.width, toolkit, 1)(valid.astype(np.float64))
    reference_density = prepare_filter("numpy_windows", case.shape, case.width, None, 1)(valid.astype(np.float64))
    accepted = np.isfinite(raw_density) & (raw_density > DENOMINATOR_FLOOR)
    reference_accepted = np.isfinite(reference_density) & (reference_density > DENOMINATOR_FLOOR)
    if not np.array_equal(accepted, reference_accepted) or not np.allclose(
            raw_density, reference_density, atol=ATOL["support"] / np.prod(case.width), rtol=RTOL):
        raise ValueError("Full-3D raw mask response/strict accepted mask mismatch")
    for method, result, density, mask in (
            (SPECTRAL, output, raw_density, accepted),
            ("numpy_windows", reference, reference_density, reference_accepted)):
        if audits[method] != dict(full_nd_filter_calls=7, filter_preparations=1):
            raise ValueError("Figure calculation changed the matched seven-filter algorithm")
        for name in COMPONENTS:
            for statistic in ("mean", "variance", "std"):
                a = result[name][statistic]
                if a.shape != case.shape or not np.array_equal(np.isfinite(a), mask) or np.isinf(a).any():
                    raise ValueError(f"Undefined full-3D component moment mask changed: {method}/{name}/{statistic}")
            if (result[name]["support"] is not result["u"]["support"] or not np.array_equal(
                    result[name]["support"], np.clip(density, 0, 1) * np.prod(case.width))):
                raise ValueError("Shared coverage / N_eq = 147 rho mapping changed")
            if not np.array_equal(result[name]["std"], np.sqrt(result[name]["variance"]), equal_nan=True):
                raise ValueError("Component standard-deviation mapping changed")
    comparisons = compare_outputs(FIGURE_RUN_ID, FIGURE_CASE_ID, SPECTRAL, output, reference, observed)
    frozen_comparisons = [r for r in tables["comparison.csv"] if r["method"] == SPECTRAL]
    _figure_check_comparisons(comparisons, frozen_comparisons)
    applications = application_counts(FIGURE_RUN_ID, FIGURE_CASE_ID, observed, injected, output, case.width)
    frozen_applications = {r["component"]: r for r in tables["application_counts.csv"]}
    if len(frozen_applications) != 3 or any(
            frozen_applications.get(row["component"]) != {key: str(value) for key, value in row.items()}
            for row in applications):
        raise ValueError("Saved component application counts mismatch")
    prepared = dict(observed=observed, injected=injected, dct=output, numpy=reference, valid=valid,
                    raw_density=raw_density, numpy_raw_density=reference_density, accepted=accepted)
    sliced = _figure_slice_fields(prepared)
    sampled = _figure_volume_fields(prepared)
    diagnostics = _figure_screening_diagnostics(prepared)
    for row in applications:
        counts = diagnostics[row["component"]]["counts"]
        if (counts["n_flagged"] != row["n_flagged"] or counts["n_eligible"] != row["n_eligible"]
                or counts["n_tp"] + counts["n_fn"] != row["n_eligible_injected"]
                or counts["n_tp"] + counts["n_fp"] != row["n_flagged"]):
            raise ValueError("Discussion diagnostic does not partition the frozen component flags")
    settings = {
        "purpose": "Componentwise 3D wind boxcar moments on the approved incomplete baseline20 input",
        "selected_run": FIGURE_RUN_ID, "selected_case": FIGURE_CASE_ID,
        "saved_run_directory": str(Path(saved_run).resolve()),
        "source_script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "saved_original_script_sha256": saved["script_sha256"],
        "original_script_git_commit": FIGURE_SOURCE_COMMIT,
        "original_script_commit_hash_verified": True,
        "input_sha256": generation["input_sha256"],
        "input_hash_scope": "all three float64 full-volume observed components, in u/v/w order, including NaNs",
        "common_mask_sha256": hashlib.sha256(valid.tobytes()).hexdigest(),
        "injected_mask_sha256": hashlib.sha256(injected.tobytes()).hexdigest(),
        "accepted_mask_sha256": hashlib.sha256(accepted.tobytes()).hexdigest(),
        "accepted_mask_rule": "isfinite(raw d) AND raw d > 1e-3, before clipping",
        "calculation": "unchanged calculate(): seven full-3D filters, one preparation, common mask, separate component moments",
        "additional_untimed_checks": "one raw-mask-only filter per backend; original application counts; 42 frozen regional comparisons",
        "new_timing_performed": False, "new_api_validation_performed": False,
        "completion": False, "stable_fallback": False, "restoration": False,
        "denominator_floor": DENOMINATOR_FLOOR, "denominator_additive_term": False,
        "population_variance_ddof": 0, "component_axis_filtered": False,
        "density_definition": "rho = clip(full-3D filtered common finite mask, 0, 1); not confidence or ESS",
        "csv_support_mapping": "support is N_eq = product(width_zyx) * rho = 147 * rho, not rho",
        "slice_index_zero_based": FIGURE_SLICE_INDEX,
        "slice_normalized_z": float(np.linspace(-1, 1, case.shape[0])[FIGURE_SLICE_INDEX]),
        "slice_stage": "extract after full-3D DCT and independent NumPy moments and all full-domain checks",
        "axes": "normalized z/y/x centers linspace(-1,1,n); not metres or km",
        "scalar_cell_faces_x": [float(sliced["x_edges"][0]), float(sliced["x_edges"][-1])],
        "scalar_cell_faces_y": [float(sliced["y_edges"][0]), float(sliced["y_edges"][-1])],
        "orientation": "rows=y increasing upward; columns=x increasing rightward; no flip; quiver x=u, y=v",
        "panels": ["3D observed (u,v,w), sampled observed injections and missing positions",
                   "3D componentwise mean (mu_u,mu_v,mu_w)",
                   "horizontal (mu_u,mu_v) arrows and mu_w", "horizontal std_w"],
        "omitted_panels": "coverage and std_u/std_v omitted from display only; all component moments still verified",
        "velocity_units": saved["velocity_units"],
        "arrow_stride_cells": FIGURE_ARROW_STRIDE, "arrow_offset_cells": FIGURE_ARROW_OFFSET,
        "arrow_sampling": "slice [3::6,3::6] for drawing only; no additional filtering or field averaging",
        "arrow_scale": FIGURE_ARROW_SCALE, "arrow_scale_units": "inches", "arrow_angles": "xy",
        "arrow_key_velocity_units": FIGURE_ARROW_KEY,
        "arrow_length_in_per_velocity_unit": 1.0 / FIGURE_ARROW_SCALE,
        "arrow_masks": "section mean: strict accepted moment mask; no raw section arrows",
        "volume_stride_zyx": list(FIGURE_VOLUME_STRIDE), "volume_offset_zyx": list(FIGURE_VOLUME_OFFSET),
        "volume_arrow_length_normalized": FIGURE_VOLUME_ARROW_LENGTH,
        "volume_arrow_length_encodes_magnitude": False,
        "volume_view_elevation_azimuth_deg": list(FIGURE_VIEW),
        "volume_color": "viridis; norm of observed vector or norm of componentwise mean vector, NOT mean wind speed",
        "volume_magnitude_color_limits": list(FIGURE_SPEED_LIMITS), "volume_colorbar_extend": "max",
        "volume_sampling": "one regular z/y/x lattice for both panels; missing inputs omitted, accepted missing-input means retained",
        "volume_n_lattice_positions": len(sampled["indices"]),
        "volume_n_observed_arrows": int(sampled["observed_mask"].sum()),
        "volume_n_mean_arrows": int(sampled["mean_mask"].sum()),
        "volume_missing_sample_marker": "gray point; not a zero vector or a detected anomaly",
        "injection_display": "red observed vectors at up to 100 evenly spaced indices in lexicographic z/y/x injection order; display only",
        "injection_display_count": len(sampled["injection_indices"]),
        "injection_display_indices_sha256": hashlib.sha256(sampled["injection_indices"].tobytes()).hexdigest(),
        "injection_mask_used_for_moments_or_flags": False,
        "mean_w_color_limits": list(FIGURE_W_LIMITS), "w_colorbar_extend": "neither",
        "std_w_color_limits": list(FIGURE_STD_LIMITS), "std_colorbar_extend": "neither",
        "missing_undefined_color": FIGURE_MISSING_COLOR,
        "dense_scalar_fields": "rasterized native 96x96 cell faces; no interpolation; vector text/axes/2D and 3D quiver",
        "diagnostic_scope": "Discussion only; separate component flags vs known vector-injection labels on eligible observed cells; no vector OR rule",
        "diagnostic_k": ANOMALY_K, "diagnostic_density_min": ANOMALY_DENSITY_MIN,
        "diagnostic_std_min": ANOMALY_STD_MIN,
        "diagnostic_exclusions": "missing or ineligible injections are not false negatives",
        "diagnostic_threshold_calibration_performed": False,
        "diagnostic_self_influence_mitigation_performed": False,
        "iterative_rejection_or_reweighting_performed": False,
        "n_regional_comparisons_verified": len(comparisons),
        "n_application_rows_verified": len(applications),
        "raw_density_min": float(raw_density.min()), "raw_density_max": float(raw_density.max()),
        "n_accepted_full_volume_per_component": int(accepted.sum()),
        "n_accepted_at_missing_full_volume_per_component": int((accepted & ~valid).sum()),
        "n_observed_slice": int(valid[FIGURE_SLICE_INDEX].sum()),
        "n_accepted_slice": int(accepted[FIGURE_SLICE_INDEX].sum()),
        "dct_toolkit_commit": TOOLKIT_COMMIT, "dct_toolkit_version": toolkit.__version__,
        "dct_toolkit_source": str(Path(toolkit.__file__).resolve().parent),
        "python_version": platform.python_version(), "python_executable": sys.executable,
    }
    for key in ("shape_zyx", "width_zyx", "window_volume", "kernel", "boundary", "field_source",
                "field_definition_sha256", "noise_seed", "outlier_seed", "missing_seed", "noise_std",
                "outlier_fraction_requested", "requested_random_missing_fraction", "actual_missing_fraction",
                "n_observed_per_component", "n_missing_per_component", "n_injected_before_missingness",
                "n_injected_observed", "n_injected_missing", "numpy_chunk_planes"):
        settings[key] = saved[key]
    for key, value in saved.items():
        if key.endswith("_version"):
            settings[f"saved_{key}"] = value
    for package in ("numpy", "scipy", "astropy", "psutil"):
        settings[f"{package}_version"] = importlib.metadata.version(package)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        settings[name] = os.environ.get(name, "not set")
    for filename, expected in _RELEASE_HASHES.items():
        actual = hashlib.sha256(Path(toolkit.__file__).with_name(filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Toolkit source SHA-256 mismatch after reproduction: {filename}")
        settings[f"toolkit_source_sha256.{filename}"] = actual
    for filename, expected in FIGURE_SAVED_CSV_SHA256.items():
        settings[f"saved_{filename}_sha256"] = expected
    for row in applications:
        for key in ("n_cells", "n_observed", "n_missing", "n_eligible", "n_flagged"):
            settings[f"application.{row['component']}.{key}"] = row[key]
    for name, diagnostic in diagnostics.items():
        for key, value in diagnostic["counts"].items():
            settings[f"diagnostic.{name}.{key}"] = value
        for key, mask in diagnostic["masks"].items():
            settings[f"diagnostic.{name}.{key}_sha256"] = hashlib.sha256(mask.tobytes()).hexdigest()
    for name in ("observed", "mean"):
        values = np.linalg.norm(sampled[name][sampled[f"{name}_mask"]], axis=1)
        settings[f"volume_{name}_magnitude_min"] = float(values.min())
        settings[f"volume_{name}_magnitude_max"] = float(values.max())
        settings[f"volume_{name}_above_color_limit_count"] = int((values > FIGURE_SPEED_LIMITS[1]).sum())
    for name, data in zip(COMPONENTS, observed):
        settings[f"observed_{name}_sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
        for statistic in STATISTICS:
            settings[f"full3d_{name}_{statistic}_sha256"] = hashlib.sha256(output[name][statistic].tobytes()).hexdigest()
    for name, values, limits in zip(
            ("mean_w", "std_w"), sliced["scalars"], (FIGURE_W_LIMITS, FIGURE_STD_LIMITS)):
        finite = values[np.isfinite(values)]
        settings[f"slice_{name}_min"] = float(finite.min())
        settings[f"slice_{name}_max"] = float(finite.max())
        settings[f"slice_{name}_below_color_limit_count"] = int((finite < limits[0]).sum())
        settings[f"slice_{name}_above_color_limit_count"] = int((finite > limits[1]).sum())
        settings[f"slice_{name}_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
        if (finite < limits[0]).any() or (finite > limits[1]).any():
            raise ValueError("Section colour limits would hide values without colorbar extensions")
    settings["checks"] = (
        "PASS: five immutable saved CSV hashes; historical committed script SHA; selected case/settings; "
        "full approved three-component input SHA and all generation counts; nine released toolkit source hashes; "
        "full-3D DCT/independent NumPy seven-filter audits; raw-d acceptance; undefined masks; shared coverage; "
        "42 frozen component/regional comparison masks/counts/tolerances/discrepancies; all three application rows; "
        "N_eq = 147 rho; post-filter section/volume sampling; component TP/FP/FN partition of frozen flags; "
        "explicit arrow and colour scales; no timing, completion or iterative rejection"
    )
    return prepared, settings


def _figure_draw_volume_arrows(ax, positions, vectors, norm, cmap, gid):
    """Colour each shaft and both heads by that vector's magnitude."""
    rgba = cmap(norm(np.linalg.norm(vectors, axis=1)))
    # Matplotlib orders shafts, then all first heads, then all second heads.
    colors = np.tile(rgba, (3, 1))
    arrow = ax.quiver(*positions.T, *vectors.T, colors=colors,
                      length=FIGURE_VOLUME_ARROW_LENGTH, normalize=True,
                      linewidths=0.55, arrow_length_ratio=0.28, alpha=0.8)
    arrow.set_gid(gid)
    return arrow


def create_wind_figure(case):
    """Restore the 3D visual context and show one matched mean/std section."""
    from _review_figures import configure_plotting

    configure_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.text import Text

    sliced = _figure_slice_fields(case)
    sampled = _figure_volume_fields(case)
    fig = plt.figure(figsize=FIGURE_SIZE)
    volume_axes = [fig.add_axes((left, 0.51, 0.39, 0.42), projection="3d") for left in (0.055, 0.50)]
    axes = [fig.add_axes((left, 0.10, 0.32, 0.30)) for left in (0.08, 0.56)]
    norm = Normalize(*FIGURE_SPEED_LIMITS)
    volume_cmap = plt.get_cmap("viridis")
    z = np.linspace(-1, 1, case["observed"][0].shape[0])[FIGURE_SLICE_INDEX]
    for ax, field, title in zip(volume_axes, ("observed", "mean"),
                               ("(a) Observed wind", "(b) Local-mean wind")):
        keep = sampled[f"{field}_mask"]
        _figure_draw_volume_arrows(ax, sampled["positions"][keep], sampled[field][keep],
                                  norm, volume_cmap, f"{field}_uvw")
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [z] * 5,
                color="#555555", linestyle="--", linewidth=0.7, alpha=0.7, gid="section_plane")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
               xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[-1, 0, 1])
        ax.set_xlabel("x", labelpad=-4)
        ax.set_ylabel("y", labelpad=-4)
        ax.set_zlabel("z", labelpad=-4)
        ax.set_title(title, loc="left", fontweight="bold", pad=5)
        ax.tick_params(pad=-1)
        ax.view_init(elev=FIGURE_VIEW[0], azim=FIGURE_VIEW[1])
        ax.set_box_aspect((1, 1, 0.75))
    missing = sampled["positions"][~sampled["observed_mask"]]
    volume_axes[0].scatter(*missing.T, color=FIGURE_MISSING_COLOR, s=4,
                           marker=".", depthshade=False, gid="missing_sample_positions")
    injection = volume_axes[0].quiver(
        *sampled["injection_positions"].T, *sampled["injection_vectors"].T,
        color=FIGURE_INJECTION_COLOR, length=FIGURE_VOLUME_ARROW_LENGTH,
        normalize=True, linewidths=1.0, arrow_length_ratio=0.28, alpha=0.95)
    injection.set_gid("observed_injection_sample")
    titles = (r"(c) Mean section, $\mu_w$", r"(d) Std section, $\sigma_w$")
    meshes, arrows = [], []
    sample = slice(FIGURE_ARROW_OFFSET, None, FIGURE_ARROW_STRIDE)
    xx, yy = np.meshgrid(sliced["x"][sample], sliced["y"][sample])
    for index, (ax, title, values, cmap_name, limits) in enumerate(zip(
            axes, titles, sliced["scalars"], ("RdBu_r", "cividis"),
            (FIGURE_W_LIMITS, FIGURE_STD_LIMITS))):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(FIGURE_MISSING_COLOR)
        mesh = ax.pcolormesh(sliced["x_edges"], sliced["y_edges"], np.ma.masked_invalid(values),
                             shading="flat", cmap=cmap, vmin=limits[0], vmax=limits[1],
                             rasterized=True, linewidth=0, antialiased=False)
        mesh.set_gid(("mean_w", "std_w")[index])
        meshes.append(mesh)
        if index == 0:
            u, v = sliced["vectors"]
            mask = ~sliced["vector_mask"][sample, sample]
            arrow = ax.quiver(xx, yy, np.ma.array(u[sample, sample], mask=mask),
                              np.ma.array(v[sample, sample], mask=mask), angles="xy",
                              scale_units="inches", scale=FIGURE_ARROW_SCALE,
                              units="inches", width=0.006, color="#202020", pivot="middle")
            arrow.set_gid("mean_uv")
            arrows.append(arrow)
        ax.set(xlim=(sliced["x_edges"][0], sliced["x_edges"][-1]),
               ylim=(sliced["y_edges"][0], sliced["y_edges"][-1]),
               xticks=[-1, 0, 1], yticks=[-1, 0, 1], aspect="equal")
        ax.set_title(title, loc="left", fontweight="bold", pad=5)
        ax.tick_params(length=3, pad=3)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].quiverkey(arrows[0], 0.08, 0.455, FIGURE_ARROW_KEY, r"1 unit: $(\mu_u,\mu_v)$",
                      coordinates="figure", labelpos="E", labelsep=0.08,
                      fontproperties={"family": "DejaVu Sans", "size": 10.4})
    for mesh, box, label, ticks, extend in (
            (ScalarMappable(norm=norm, cmap=volume_cmap), (0.90, 0.60, 0.012, 0.25),
             "Vector magnitude", [0, 0.8, 1.6], "max"),
            (meshes[0], (0.42, 0.10, 0.014, 0.30), r"Mean, $\mu_w$", [-0.3, 0, 0.3], "neither"),
            (meshes[1], (0.90, 0.10, 0.014, 0.30), r"Std, $\sigma_w$", [0, 0.2, 0.4], "neither")):
        colorbar = fig.colorbar(mesh, cax=fig.add_axes(box), label=label, ticks=ticks, extend=extend)
        colorbar.ax.tick_params(length=3, pad=3)
    fig.legend([Line2D([], [], color=FIGURE_INJECTION_COLOR, linewidth=1.4),
                Line2D([], [], color=FIGURE_MISSING_COLOR, marker=".", linestyle="none"),
                Line2D([], [], color="#555555", linestyle="--", linewidth=0.7)],
               ["Injected (sample)", "Missing (sample)", f"Section z = {z:.4f}"],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               columnspacing=1.1, handlelength=1.3, handletextpad=0.5)
    for text in fig.findobj(Text):
        if text.get_fontsize() < 10.4:
            text.set_fontsize(10.4)
    return fig


def _figure_check_plot(fig, case):
    """Check actual 3D shafts, masks, colours, plane heights and section arrays."""
    sliced = _figure_slice_fields(case)
    sampled = _figure_volume_fields(case)
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt

    norm = Normalize(*FIGURE_SPEED_LIMITS)
    for ax, field in zip(fig.axes[:2], ("observed", "mean")):
        keep = sampled[f"{field}_mask"]
        positions, vectors = sampled["positions"][keep], sampled[field][keep]
        collection = next(a for a in ax.collections if a.get_gid() == f"{field}_uvw")
        _figure_check_volume_arrows(collection, positions, vectors)
        colors = plt.get_cmap("viridis")(norm(np.linalg.norm(vectors, axis=1)))
        colors[:, 3] = 0.8
        expected_colors = np.tile(colors, (3, 1))
        if not np.array_equal(collection.get_edgecolors(), expected_colors):
            raise ValueError("3D vector-magnitude colours changed")
        line, = [a for a in ax.lines if a.get_gid() == "section_plane"]
        if not np.array_equal(line.get_data_3d()[2], np.repeat(
                np.linspace(-1, 1, case["observed"][0].shape[0])[FIGURE_SLICE_INDEX], 5)):
            raise ValueError("3D section plane differs from the extracted slice height")
    injection = next(a for a in fig.axes[0].collections if a.get_gid() == "observed_injection_sample")
    _figure_check_volume_arrows(injection, sampled["injection_positions"], sampled["injection_vectors"])
    missing = next(a for a in fig.axes[0].collections if a.get_gid() == "missing_sample_positions")
    if not np.array_equal(np.column_stack(missing._offsets3d), sampled["positions"][~sampled["observed_mask"]]):
        raise ValueError("Missing-input positions differ from the common 3D sample mask")
    sample = slice(FIGURE_ARROW_OFFSET, None, FIGURE_ARROW_STRIDE)
    for index, (ax, expected) in enumerate(zip(fig.axes[2:4], sliced["scalars"])):
        mesh = ax.collections[0]
        if not np.array_equal(mesh.get_array().filled(np.nan), expected, equal_nan=True):
            raise ValueError("Plotted scalar array differs from checked full-3D slice")
        coords = mesh.get_coordinates()
        if not (np.array_equal(coords[0, :, 0], sliced["x_edges"])
                and np.array_equal(coords[:, 0, 1], sliced["y_edges"])):
            raise ValueError("Scalar cell-face coordinates changed")
        if index == 0:
            arrow = ax.collections[1]
            mask = ~sliced["vector_mask"][sample, sample].ravel()
            u, v = sliced["vectors"]
            if (not np.array_equal(np.ma.getmaskarray(np.ma.array(arrow.U, mask=arrow.Umask)), mask)
                    or not np.array_equal(arrow.U[~mask], u[sample, sample].ravel()[~mask])
                    or not np.array_equal(arrow.V[~mask], v[sample, sample].ravel()[~mask])
                    or arrow.scale != FIGURE_ARROW_SCALE or arrow.scale_units != "inches"
                    or arrow.angles != "xy" or arrow.get_rasterized()):
                raise ValueError("Component arrow mapping/masks/shared scale changed")


def _figure_check_volume_arrows(collection, positions, vectors):
    """Verify actual plotted shafts, rather than auxiliary artist attributes."""
    lengths = np.linalg.norm(vectors, axis=1)
    direction = np.divide(vectors, lengths[:, None], out=np.zeros_like(vectors), where=lengths[:, None] > 0)
    segments = np.asarray(collection._segments3d)
    expected = np.stack([positions + FIGURE_VOLUME_ARROW_LENGTH * direction, positions], axis=1)
    if (segments.shape != (3 * len(positions), 2, 3)
            or not np.allclose(segments[:len(positions)], expected, rtol=0, atol=2e-15)
            or not np.allclose(segments[len(positions):2 * len(positions), 0], expected[:, 0], rtol=0, atol=2e-15)
            or not np.allclose(segments[2 * len(positions):, 0], expected[:, 0], rtol=0, atol=2e-15)
            or collection.get_rasterized()):
        raise ValueError("3D arrow coordinates/directions/fixed lengths changed")


def export_wind_figure(output: Path | None = None) -> dict[str, Path]:
    """Export just the new assets/sidecar, never a numerical or benchmark CSV."""
    from _review_figures import DEFAULT_FIGURE_ROOT, export_figure

    output = (output if output is not None else DEFAULT_FIGURE_ROOT / "3d_wind_boxcar").expanduser().resolve()
    if output.is_relative_to((Path(__file__).resolve().parents[1] / "results").resolve()):
        raise ValueError("Figure output must not be inside saved experiment results.")
    case, settings = prepare_wind_figure()
    fig = create_wind_figure(case)
    import matplotlib.pyplot as plt

    try:
        _figure_check_plot(fig, case)
        settings["plot_checks"] = "PASS: exact 3D shafts/directions/fixed lengths/magnitude colours; common sample masks; observed-only injection sample; missing positions; matching section height; two exact section arrays/native cell faces; x=mu_u/y=mu_v arrows; vector arrows"
        return export_figure(fig, output, FIGURE_STEM, settings)
    finally:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, help="Parent of 3d_wind_boxcar/run_id (default: results/round1)")
    parser.add_argument("--mode", choices=("numerical", "benchmark", "all"), default="numerical",
                        help="benchmark/all also verify numerical agreement before each case is eligible for timing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated case IDs; small controls are always verified")
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma-separated methods; numerical NumPy/toolkit checks are mandatory")
    parser.add_argument("--max-cases", type=int, default=0, help="Trim selected cases to first N; 0 disables trimming")
    parser.add_argument("--max-voxels", type=int, default=400000, help="Record larger selected cases as resource_limit; 0 disables cap")
    parser.add_argument("--max-window-volume", type=int, default=1200, help="0 disables cap")
    parser.add_argument("--numpy-chunk-planes", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=180.0, help="Worker lifetime including setup; 0 disables limit")
    parser.add_argument("--max-ram-mib", type=float, default=2048.0, help="Total worker RSS limit; 0 disables limit")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--figure-only", action="store_true", help="Reproduce/export only the approved baseline20 full-3D case; no timing or numerical CSV writes")
    parser.add_argument("--figure-output", type=Path, help="Figure case directory (default: manuscript/figures/round1/3d_wind_boxcar)")
    args = parser.parse_args()
    if args.figure_output is not None and not args.figure_only:
        parser.error("--figure-output requires --figure-only")
    if args.figure_only:
        if (args.run_id is not None or args.output_root is not None or args.mode != "numerical"
                or args.cases != "small20,baseline_complete,baseline20,baseline50,larger20,narrow20,broad20,plane_control"
                or args.methods != "dct_shared_moments,numpy_windows,scipy_uniform,astropy_direct,astropy_fft"
                or args.repeats != 3 or args.max_cases != 0 or args.max_voxels != 400000
                or args.max_window_volume != 1200 or args.numpy_chunk_planes != 1
                or args.timeout_s != 180.0 or args.max_ram_mib != 2048.0 or args.list_cases or args.worker):
            parser.error("--figure-only uses the fixed approved baseline20 case; do not mix numerical, case, worker or benchmark options")
    if args.repeats < 1 or args.numpy_chunk_planes < 1:
        parser.error("--repeats and --numpy-chunk-planes must be >= 1")
    for option in ("max_cases", "max_voxels", "max_window_volume", "timeout_s", "max_ram_mib"):
        if getattr(args, option) < 0:
            parser.error(f"--{option.replace('_', '-')} must be >= 0")
    try:
        split_names(args.cases, CASES)
        split_names(args.methods, METHODS)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    if args.figure_only:
        for kind, path in export_wind_figure(args.figure_output).items():
            print(f"{kind}: {path}")
        return 0
    if args.list_cases:
        for name, case in CASES.items():
            print(f"{name}: shape={case.shape} width={case.width} mask={case.pattern} missing={case.missing_fraction:g} benchmark={case.benchmark}")
        return 0
    if args.worker:
        return worker(args)
    # Fail closed on an unapproved editable checkout before writing results.
    load_toolkit()
    run_id, directory = create_run_directory("3d_wind_boxcar", args.run_id, args.output_root)
    requested = split_names(args.cases, CASES)
    if args.max_cases:
        requested = requested[:args.max_cases]
    case_ids = list(dict.fromkeys(requested + list(CONTROLS)))
    methods = split_names(args.methods, METHODS)
    checked_methods = list(dict.fromkeys(["numpy_windows", SPECTRAL, API_VALIDATION] + methods))
    settings = {name: case_settings(CASES[name], args) for name in case_ids}
    comparisons, applications, benchmark_rows = [], [], []
    verified = {}
    failed = False
    for case_id in case_ids:
        case = CASES[case_id]
        reason = ""
        if args.max_voxels and np.prod(case.shape) > args.max_voxels:
            reason = f"case exceeds max_voxels={args.max_voxels}"
        if args.max_window_volume and np.prod(case.width) > args.max_window_volume:
            reason = f"case exceeds max_window_volume={args.max_window_volume}"
        settings[case_id]["requested_for_benchmark"] = case_id in requested and case.benchmark
        if reason:
            settings[case_id]["numerical_status"] = "resource_limit"
            settings[case_id]["numerical_reason"] = reason
            for method in checked_methods:
                comparisons.extend(error_comparisons(run_id, case_id, method, reason))
            verified[case_id] = False
            failed = True
            continue
        try:
            observed, injected, generated_settings = generate_input(case)
            settings[case_id].update(generated_settings)
            reference = calculate("numpy_windows", observed, case.width, None, args.numpy_chunk_planes)
        except Exception as error:
            reason = f"Input generation or NumPy reference failed: {type(error).__name__}: {error}"
            comparisons.extend(error_comparisons(run_id, case_id, "numpy_windows", reason))
            settings[case_id].update(numerical_status="error", numerical_reason=reason)
            verified[case_id] = False
            failed = True
            continue
        case_ok = True
        for method in checked_methods:
            if method == "numpy_windows":
                continue
            try:
                if method == API_VALIDATION:
                    output = calculate_toolkit_api(observed, case.width, prepare_backend(method))
                else:
                    output = calculate(method, observed, case.width, prepare_backend(method), args.numpy_chunk_planes)
                rows = compare_outputs(run_id, case_id, method, output, reference, observed)
                comparisons.extend(rows)
                case_ok = case_ok and all(row["status"] in ("pass", "no_comparable") for row in rows)
                if method == SPECTRAL:
                    applications.extend(application_counts(run_id, case_id, observed, injected, output, case.width))
                del output
            except Exception as error:
                comparisons.extend(error_comparisons(run_id, case_id, method, f"{type(error).__name__}: {error}"))
                case_ok = False
        verified[case_id] = case_ok
        settings[case_id]["numerical_status"] = "pass" if case_ok else "fail"
        failed = failed or not case_ok
        del observed, injected, reference
        gc.collect()
    write_csv(directory / "comparison.csv", COMPARISON_FIELDS, comparisons)
    if applications:
        write_csv(directory / "application_counts.csv", APPLICATION_FIELDS, applications)
    write_settings(directory / "settings.csv", run_id, settings)
    if args.mode in ("benchmark", "all"):
        # ALL selected cases and mandatory controls must pass before ANY timing.
        controls_ok = all(verified.get(name, False) for name in CONTROLS)
        for case_index, case_id in enumerate(requested):
            if not CASES[case_id].benchmark:
                continue
            for repetition in range(1, args.repeats + 1):
                offset = (case_index + repetition - 1) % len(methods)
                for method in methods[offset:] + methods[:offset]:
                    if not verified.get(case_id, False) or not controls_ok:
                        status = settings[case_id]["numerical_status"]
                        payload = dict(status="resource_limit" if status == "resource_limit" else "blocked_numerical",
                                       elapsed_s=None, ram_before_mib=None, peak_ram_mib=None,
                                       reason=settings[case_id].get("numerical_reason", "case or mandatory control failed numerical verification"))
                    else:
                        try:
                            payload = measure_worker(args, case_id, method)
                            if payload.get("status") == "success" and (
                                payload.get("input_sha256") != settings[case_id]["input_sha256"]
                                or payload.get("n_outputs") != 12
                                or payload.get("full_nd_filter_calls") != FULL_FILTER_CALLS
                                or payload.get("filter_preparations") != 1
                            ):
                                payload.update(status="error", elapsed_s=None,
                                               reason="worker input hash, outputs, or matched seven-filter preparation differs from verified case")
                        except Exception as error:
                            payload = dict(status="error", elapsed_s=None, ram_before_mib=None,
                                           peak_ram_mib=None, reason=f"worker launch/monitor failed: {type(error).__name__}: {error}")
                    benchmark_rows.append(dict(
                        run_id=run_id, case_id=case_id, method=method, operation=OPERATION,
                        repetition=repetition, **{name: payload.get(name) for name in (
                            "status", "elapsed_s", "ram_before_mib", "peak_ram_mib", "reason")},
                    ))
                    settings[case_id][f"worker_{method}_{repetition}"] = payload
                    failed = failed or payload["status"] != "success"
                    # Preserve completed rows even if a later worker is interrupted.
                    write_csv(directory / "benchmark_runs.csv", BENCHMARK_FIELDS, benchmark_rows)
                    write_csv(directory / "benchmark_summary.csv", SUMMARY_FIELDS, summarize_benchmarks(benchmark_rows))
                    write_settings(directory / "settings.csv", run_id, settings)
    statuses = {status: sum(row["status"] == status for row in comparisons)
                for status in sorted({row["status"] for row in comparisons})}
    print(json.dumps(dict(run_id=run_id, directory=str(directory), comparisons=statuses,
                          benchmark_rows=len(benchmark_rows), success=not failed), sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
