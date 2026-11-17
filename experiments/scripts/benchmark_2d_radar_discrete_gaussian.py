#!/usr/bin/env python3
"""Sampled Gaussian equivalence and a matched three-filter radar benchmark.

This separate experiment preserves all earlier calculations. Nonnegative sampled
Gaussian kernels use sigma=W_grid/sqrt(12) and radius=ceil(4*sigma). Their exact
discrete responses are compared with direct spatial convolution and an untimed,
independently constructed centered-residual reference. The released analytical
Gaussian is a different-operator comparison, not the sampled-kernel oracle.

Only the original radar sweep and W=5,15 are used. Five fresh sequential workers
per method/width include preparation and all three complete filters. Existing
toolkit release checks, strict d>1e-3, output policies and tolerances are retained.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import psutil
from scipy.fft import dct, idct, rfft, irfft
from scipy.ndimage import convolve1d

import benchmark_2d_radar_statistics as base
from _review_common import (
    BENCHMARK_FIELDS, COMPARISON_FIELDS, SUMMARY_FIELDS,
    create_run_directory, write_csv, write_settings,
)


METHODS = ("analytical_spectral", "discrete_spectral", "discrete_spatial")
REFERENCE = "independent_centered_spatial_gaussian"
EXPERIMENT = "2d_radar_discrete_gaussian"
OPERATION = "radar_gaussian_statistics_three_filters_setup_included"
SHAPE = base.SHAPE
WIDTHS = base.WIDTHS
FIELDS = base.FIELDS
FLOOR = base.FLOOR
TRUNCATE = 4.0
LIBRARIES = ["numpy", "scipy.fft", "scipy.ndimage", "astropy.convolution", "psutil", "dct_toolkit"]
DIAGNOSTIC_FIELDS = [
    "run_id", "case_id", "method", "region", "n_considered", "n_defined",
    "n_negative_raw_variance", "min_raw_variance", "most_negative_raw_variance",
    "median_negative_raw_variance", "n_negative_beyond_reference_tolerance",
    "max_abs_centered_variance_difference", "raw_mask_min", "raw_mask_max",
]
KERNEL_FIELDS = [
    "run_id", "case_id", "axis", "gate", "sigma", "radius", "n_weights",
    "min_weight", "weight_sum", "max_abs_astropy_weight_difference",
    "max_abs_response_cosine_difference", "weights_sha256", "status",
]
AGREEMENT_FIELDS = [
    "run_id", "case_id", "width", "region", "n_considered", "n_defined",
    "max_abs_mean_difference", "max_abs_variance_difference", "min_raw_variance",
    "n_negative_raw_variance", "n_negative_beyond_reference_tolerance",
]
TIMING_FIELDS = ["run_id", "width", *METHODS]
PHASE_FIELDS = ["run_id", "case_id", "method", "repetition", "status", "preparation_s", "total_s"]


def source_hashes():
    return {name: base.radar.sha256_file(path) for name, path in (
        ("experiment", Path(__file__)), ("benchmark_helper", Path(base.__file__)),
        ("radar_input", Path(base.radar.__file__)),
        ("common", Path(__file__).with_name("_review_common.py")),
    )}


def gaussian_kernel(sigma):
    """Centered, normalized sampled Gaussian; no minimum-one width replacement."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    radius = int(np.ceil(TRUNCATE * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def astropy_kernel(sigma):
    """Independent model discretization, only used outside benchmark timing."""
    from astropy.convolution import Gaussian1DKernel

    radius = int(np.ceil(TRUNCATE * sigma))
    return np.asarray(Gaussian1DKernel(sigma, x_size=2 * radius + 1, mode="center").array).copy()


def widths_and_kernels(shape, width, az_res, constructor=gaussian_kernel):
    base.check_geometry(shape, width, az_res)
    sigma_az = base.radar.azimuth_widths(shape[1], width, az_res) / np.sqrt(12)
    sigma_range = float(width) / np.sqrt(12)
    return sigma_az, [constructor(s) for s in sigma_az], constructor(sigma_range)


def discrete_response(weights, period):
    """Exact circular response, folding all repeated donors when support > period.

    A reflective axis of length N uses period 2*N; its DCT-II modes are the
    first N modes of that even extension. This is not a DCT of kernel samples.
    """
    weights = np.asarray(weights, dtype=float)
    if period < 1 or weights.ndim != 1 or weights.size % 2 != 1:
        raise ValueError("Expected a positive period and an odd centered kernel")
    radius = weights.size // 2
    offsets = np.arange(-radius, radius + 1)
    folded = np.bincount(offsets % period, weights=weights, minlength=period)
    response = rfft(folded, workers=1)
    if np.max(np.abs(response.imag)) > 5e-14:
        raise ValueError("The centered Gaussian response should be real")
    return response.real


def expected_audit(method, shape):
    audit = dict(filter_preparations=1, full_filter_calls=3)
    if method != METHODS[0]:
        audit.update(azimuth_kernels=shape[1], range_kernels=1)
    if method == "discrete_spectral":
        audit.update(azimuth_kernel_ffts=shape[1], range_kernel_ffts=1)
    elif method == "discrete_spatial":
        audit.update(azimuth_convolutions=3 * shape[1], range_convolutions=3)
    elif method != METHODS[0]:
        raise ValueError("Unknown method")
    return audit


def prepare_filter(method, shape, width, az_res, toolkit, audit):
    if method == METHODS[0]:
        return base.prepare_filter("prepared_dct_rfft", shape, width, az_res, toolkit, audit)
    if method not in METHODS:
        raise ValueError("Unknown method")
    _, angular, radial = widths_and_kernels(shape, width, az_res)
    audit.update(filter_preparations=1, azimuth_kernels=shape[1], range_kernels=1)
    if method == "discrete_spectral":
        h_az = np.stack([discrete_response(g, shape[0]) for g in angular], axis=1)
        h_range = discrete_response(radial, 2 * shape[1])[:shape[1]]
        audit.update(azimuth_kernel_ffts=shape[1], range_kernel_ffts=1)

    def linear(values):
        if values.shape != tuple(shape):
            raise ValueError("Input shape differs from prepared geometry")
        audit["full_filter_calls"] = audit.get("full_filter_calls", 0) + 1
        if method == "discrete_spatial":
            stage = np.empty(shape, dtype=np.float64)
            for gate, kernel in enumerate(angular):
                stage[:, gate] = convolve1d(values[:, gate], kernel, mode="wrap")
            audit["azimuth_convolutions"] = audit.get("azimuth_convolutions", 0) + shape[1]
            audit["range_convolutions"] = audit.get("range_convolutions", 0) + 1
            return convolve1d(stage, radial, axis=1, mode="reflect")
        spectrum = rfft(values, axis=0, norm="ortho", workers=1) * h_az
        stage = irfft(spectrum, n=shape[0], axis=0, norm="ortho", workers=1)
        spectrum = dct(stage, axis=1, type=2, norm="ortho", workers=1) * h_range
        return idct(spectrum, axis=1, type=2, norm="ortho", workers=1)

    return linear


def calculate(method, data, width, az_res, toolkit, audit=None, phases=None):
    audit = audit if audit is not None else {}
    valid = np.isfinite(data)
    zeroed = np.where(valid, data, 0.0)
    start = time.perf_counter()
    linear = prepare_filter(method, data.shape, width, az_res, toolkit, audit)
    if phases is not None:
        phases["preparation_s"] = time.perf_counter() - start
    density = linear(valid.astype(np.float64))
    numerator = linear(zeroed)
    second = linear(zeroed**2)
    return base.moment_fields(density, numerator, second, width, az_res)


def reflect_indices(indices, length):
    """Explicit half-sample reflection, independent of convolution/FFT routines."""
    folded = np.asarray(indices) % (2 * length)
    return np.where(folded < length, folded, 2 * length - 1 - folded)


def centered_reference(data, width, az_res):
    """Full-array population variance from nonnegative squared residuals.

    Calculate angular centered moments, then combine their within/between
    variances with the original angular mask responses as range weights. No
    raw-moment subtraction, toolkit filter, FFT, DCT or convolve call is used.
    """
    _, angular, radial = widths_and_kernels(data.shape, width, az_res, astropy_kernel)
    n_az, n_range = data.shape
    density = np.zeros(data.shape)
    mean = np.zeros(data.shape)
    variance = np.zeros(data.shape)
    rows = np.arange(n_az)[:, None]
    for gate, kernel in enumerate(angular):
        offsets = np.arange(-(kernel.size // 2), kernel.size // 2 + 1)
        # Sum repeated circular donors, retaining nonnegative weights.
        folded = np.zeros(n_az)
        np.add.at(folded, offsets % n_az, kernel)
        offsets = np.flatnonzero(folded)
        weights = folded[offsets][None, :]
        donors = data[(rows - offsets[None, :]) % n_az, gate]
        valid = np.isfinite(donors)
        safe = np.where(valid, donors, 0.0)
        mass = (weights * valid).sum(axis=1)
        mu = np.divide((weights * safe).sum(axis=1), mass,
                       out=np.zeros(n_az), where=mass > 0)
        residual = np.where(valid, donors - mu[:, None], 0.0)
        var = np.divide((weights * residual**2).sum(axis=1), mass,
                        out=np.zeros(n_az), where=mass > 0)
        density[:, gate], mean[:, gate], variance[:, gate] = mass, mu, var
    denominator = np.zeros(data.shape)
    numerator = np.zeros(data.shape)
    radius = radial.size // 2
    for offset, weight in zip(range(-radius, radius + 1), radial):
        indices = reflect_indices(np.arange(n_range) - offset, n_range)
        mass = weight * density[:, indices]
        denominator += mass
        numerator += mass * mean[:, indices]
    mu = np.divide(numerator, denominator, out=np.zeros(data.shape), where=denominator > 0)
    centered = np.zeros(data.shape)
    for offset, weight in zip(range(-radius, radius + 1), radial):
        indices = reflect_indices(np.arange(n_range) - offset, n_range)
        mass = weight * density[:, indices]
        centered += mass * (variance[:, indices] + (mean[:, indices] - mu)**2)
    accepted = np.isfinite(denominator) & (denominator > FLOOR)
    var = np.divide(centered, denominator, out=np.full(data.shape, np.nan), where=accepted)
    mu = np.where(accepted, mu, np.nan)
    coverage = np.clip(denominator, 0, 1)
    nominal = width * base.radar.azimuth_widths(n_range, width, az_res)[None, :]
    return dict(mean=mu, second_moment=var + mu**2, raw_variance=var,
                variance=var, std=np.sqrt(var), raw_mask_response=denominator,
                direct_acceptance=accepted, coverage=coverage, support=coverage * nominal)


def compare(run_id, case, method, key, region, actual, reference, reference_name=REFERENCE):
    row = base.compare_arrays(run_id, case, method, key, region, actual, reference)
    row.update(reference_method=reference_name, kernel_match="normalized_sampled_gaussian_4sigma",
               notes="Full-array check against centered residuals; identical defined-output masks required. "
                     "Raw variance is checked before clamping; tolerances are the existing radar tolerances.")
    if key == "std" and reference_name == REFERENCE:
        # As in the existing 3D centered/near-zero checks, do not confuse the
        # ill-conditioned square root at zero with a failure of the variance.
        a, b = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
        af, bf = np.isfinite(a), np.isfinite(b)
        joint = af & bf
        variance_atol, variance_rtol = base.TOLERANCES["variance"]
        atol, rtol = base.TOLERANCES["std"]
        allowed = atol + rtol * np.abs(b[joint])
        near_zero = b[joint]**2 <= variance_atol
        allowed[near_zero] = np.maximum(allowed[near_zero], np.sqrt(
            variance_atol + variance_rtol * b[joint][near_zero]**2))
        passed = (np.array_equal(af, bf) and not np.isinf(a).any() and not np.isinf(b).any()
                  and np.all(np.abs(a[joint] - b[joint]) <= allowed))
        row["status"] = ("pass" if joint.any() else "no_comparable") if passed else "fail"
        row["notes"] += (" For reference variance <= variance_atol, std uses the square root "
                         "of the fixed variance tolerance; nonzero-variance std and variance tests are unchanged.")
    return row


def variance_diagnostics(run_id, case, method, region, fields, reference, selection):
    raw = fields["raw_variance"][selection]
    truth = reference["variance"][selection]
    finite = np.isfinite(raw)
    negative = raw[finite & (raw < 0)]
    joint = finite & np.isfinite(truth)
    atol, rtol = base.TOLERANCES["raw_variance"]
    exceeded = joint & (raw < -(atol + rtol * np.abs(truth)))
    density = fields["raw_mask_response"][selection]
    return dict(run_id=run_id, case_id=case, method=method, region=region,
                n_considered=raw.size, n_defined=int(finite.sum()),
                n_negative_raw_variance=int(negative.size),
                min_raw_variance=float(raw[finite].min()) if finite.any() else None,
                most_negative_raw_variance=float(negative.min()) if negative.size else None,
                median_negative_raw_variance=float(np.median(negative)) if negative.size else None,
                n_negative_beyond_reference_tolerance=int(exceeded.sum()),
                max_abs_centered_variance_difference=float(np.max(np.abs(raw[joint] - truth[joint]))) if joint.any() else None,
                raw_mask_min=float(density.min()) if density.size else None,
                raw_mask_max=float(density.max()) if density.size else None)


def kernel_checks(run_id, case, shape, width, az_res):
    sigmas, angular, radial = widths_and_kernels(shape, width, az_res)
    rows = []
    kernels = [("azimuth", j, sigma, kernel, shape[0]) for j, (sigma, kernel) in enumerate(zip(sigmas, angular))]
    kernels.append(("range", -1, width / np.sqrt(12), radial, 2 * shape[1]))
    for axis, gate, sigma, kernel, period in kernels:
        radius = kernel.size // 2
        offsets = np.arange(-radius, radius + 1)
        omega = 2 * np.pi * np.arange(period // 2 + 1) / period
        cosine = np.cos(omega[:, None] * offsets[None, :]) @ kernel
        difference = float(np.max(np.abs(discrete_response(kernel, period) - cosine)))
        weight_difference = float(np.max(np.abs(kernel - astropy_kernel(sigma))))
        passed = (np.all(kernel >= 0) and abs(kernel.sum() - 1) < 5e-14
                  and difference < 5e-13 and weight_difference < 5e-14)
        rows.append(dict(run_id=run_id, case_id=case, axis=axis, gate=gate, sigma=float(sigma),
                         radius=radius, n_weights=kernel.size, min_weight=float(kernel.min()),
                         weight_sum=float(kernel.sum()), max_abs_astropy_weight_difference=weight_difference,
                         max_abs_response_cosine_difference=difference,
                         weights_sha256=base.array_sha256(kernel), status="pass" if passed else "fail"))
    return rows


def validate_case(run_id, case, data, ranges, width, az_res, toolkit):
    reference = centered_reference(data, width, az_res)
    regions = dict(all=np.ones(data.shape, dtype=bool), centre=np.broadcast_to(ranges <= 10000, data.shape))
    comparisons, diagnostics, operator_differences, hashes = [], [], [], {}
    outputs = {}
    for method in METHODS:
        audit = {}
        fields = calculate(method, data, width, az_res, toolkit, audit)
        if audit != expected_audit(method, data.shape):
            raise ValueError("Preparation/filter counts differ from the matched workload")
        hashes[method] = base.fingerprints(fields)
        outputs[method] = fields
        for region, selection in regions.items():
            diagnostics.append(variance_diagnostics(run_id, case, method, region, fields, reference, selection))
            for key in FIELDS:
                row = compare(run_id, case, method, key, region, fields[key][selection], reference[key][selection])
                if method == METHODS[0]:
                    row.update(status="different_operator", kernel_match="analytical_vs_sampled_gaussian",
                               notes="Descriptive difference between operators; not a numerical correctness gate.")
                    operator_differences.append(row)
                else:
                    comparisons.append(row)
    # Preserve numerical verification against the released analytical API.
    valid = np.isfinite(data)
    density = toolkit.smooth_polar(valid.astype(float), width, az_res_deg=az_res,
                                   az_boundary="periodic", range_boundary="reflective", kernel_type="gaussian")
    released = base.radar.toolkit_fields(toolkit, data, width, az_res, FLOOR, density)
    released["coverage"] = np.clip(density, 0, 1)
    for key in FIELDS:
        row = compare(run_id, case, METHODS[0], key, "all", outputs[METHODS[0]][key], released[key], "released_analytical_api")
        row.update(kernel_match="released_analytical_gaussian", notes="Separate untimed release equivalence check.")
        comparisons.append(row)
    checked = all(r["status"] in ("pass", "no_comparable") for r in comparisons)
    checked = checked and all(r["n_negative_beyond_reference_tolerance"] == 0
                              for r in diagnostics if r["method"] != METHODS[0])
    descriptor = dict(verified=checked, source_sha256=source_hashes(), input_sha256=base.array_sha256(data),
                      shape=list(data.shape), width=width, az_res_deg=az_res, output_sha256=hashes)
    summary = []
    def maximum_difference(selected, statistic):
        values = [r["max_abs_difference"] for r in selected
                  if r["statistic"] == statistic and r["max_abs_difference"] is not None]
        return max(values) if values else None

    for region, selection in regions.items():
        rows = [r for r in comparisons if r["region"] == region and r["method"] != METHODS[0]]
        diag = next(r for r in diagnostics if r["region"] == region and r["method"] == "discrete_spectral")
        summary.append(dict(run_id=run_id, case_id=case, width=width, region=region,
                            n_considered=int(selection.sum()), n_defined=diag["n_defined"],
                            max_abs_mean_difference=maximum_difference(rows, "mean"),
                            max_abs_variance_difference=maximum_difference(rows, "raw_variance"),
                            min_raw_variance=diag["min_raw_variance"],
                            n_negative_raw_variance=diag["n_negative_raw_variance"],
                            n_negative_beyond_reference_tolerance=diag["n_negative_beyond_reference_tolerance"]))
    return comparisons, diagnostics, operator_differences, descriptor, summary


def preload():
    import astropy.convolution  # noqa: F401 -- identical imports in all workers

    return base.load_toolkit()


def worker(args):
    payload = dict(status="error", reason="worker did not finish")
    stop, sampler = threading.Event(), None
    try:
        if any(os.environ.get(k) != "1" for k in base.THREAD_VARIABLES):
            raise ValueError("All numerical-library thread limits must be set to 1")
        toolkit = preload()
        descriptor = json.loads(args.manifest.read_text())[args.case]
        data = np.load(args.input, allow_pickle=False)
        if (not descriptor["verified"] or descriptor["source_sha256"] != source_hashes()
                or data.dtype != np.float64 or data.shape != SHAPE or list(data.shape) != descriptor["shape"]
                or base.array_sha256(data) != descriptor["input_sha256"]):
            raise ValueError("Worker source/input is not the verified radar case")
        gc.collect()
        process = psutil.Process()
        peak, ready = [process.memory_info().rss], threading.Event()

        def sample():
            peak[0] = max(peak[0], process.memory_info().rss)
            ready.set()
            while not stop.wait(base.RSS_INTERVAL):
                peak[0] = max(peak[0], process.memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        ready.wait()
        before = process.memory_info().rss
        peak[0] = max(peak[0], before)
        hwm_before = base.lifetime_peak_bytes()
        audit, phases = {}, {}
        start = time.perf_counter()
        output = calculate(args.method, data, descriptor["width"], descriptor["az_res_deg"], toolkit, audit, phases)
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        hwm_after = base.lifetime_peak_bytes()
        stop.set()
        sampler.join()
        if hwm_after > hwm_before:
            peak[0] = max(peak[0], hwm_after)
        hashes = base.fingerprints(output)
        if audit != expected_audit(args.method, data.shape) or hashes != descriptor["output_sha256"][args.method]:
            raise ValueError("Worker filter counts or output hashes differ from the verified calculation")
        payload.update(status="success", elapsed_s=elapsed, preparation_s=phases["preparation_s"],
                       ram_before_mib=before / base.MIB, peak_ram_mib=peak[0] / base.MIB, reason="",
                       case_id=args.case, method=args.method, worker_pid=os.getpid(),
                       source_sha256=descriptor["source_sha256"], input_sha256=descriptor["input_sha256"],
                       output_sha256=hashes, output_fingerprints_verified=True, audit=audit,
                       n_outputs=len(output), preloaded_libraries=LIBRARIES,
                       rss_sample_interval_s=base.RSS_INTERVAL,
                       precall_lifetime_peak_mib=hwm_before / base.MIB, final_lifetime_peak_mib=hwm_after / base.MIB)
        if peak[0] / base.MIB > args.max_ram_mib:
            payload.update(status="resource_limit", reason="Calculation exceeded RSS budget")
    except Exception as error:
        payload.update(status="error", reason=f"{type(error).__name__}: {error}")
    finally:
        stop.set()
        if sampler is not None:
            sampler.join()
    return payload


def measure_worker(args, case, method, data_path, manifest):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--case", case,
               "--method", method, "--input", str(data_path), "--manifest", str(manifest),
               "--max-ram-mib", str(args.max_ram_mib)]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process, start, reason = psutil.Process(child.pid), time.monotonic(), ""
    while child.poll() is None:
        try:
            if process.memory_info().rss / base.MIB > args.max_ram_mib:
                reason = "Worker exceeded RSS budget, including untimed setup"
        except psutil.NoSuchProcess:
            pass
        if time.monotonic() - start > args.timeout_s:
            reason = "Worker exceeded lifetime limit, including untimed setup"
        if reason:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
            break
        time.sleep(.01)
    stdout, stderr = child.communicate()
    if reason:
        return dict(status="resource_limit", reason=reason)
    try:
        payload = json.loads(stdout)
    except (ValueError, TypeError):
        return dict(status="error", reason=f"Worker exit {child.returncode}: {stderr.strip() or stdout.strip()}")
    if child.returncode and payload.get("status") == "success":
        payload.update(status="error", reason=f"Worker exited {child.returncode}")
    if stderr.strip():
        payload["worker_stderr"] = stderr.strip()
    return payload


def check_worker(payload, descriptor, case, method):
    if payload.get("status") != "success":
        return payload
    valid = (payload.get("case_id") == case and payload.get("method") == method
             and payload.get("source_sha256") == descriptor["source_sha256"]
             and payload.get("input_sha256") == descriptor["input_sha256"]
             and payload.get("output_sha256") == descriptor["output_sha256"][method]
             and payload.get("output_fingerprints_verified") is True
             and payload.get("audit") == expected_audit(method, descriptor["shape"])
             and payload.get("n_outputs") == len(FIELDS) and payload.get("preloaded_libraries") == LIBRARIES)
    for key in ("elapsed_s", "ram_before_mib", "peak_ram_mib", "preparation_s"):
        value = payload.get(key)
        valid = valid and isinstance(value, (int, float)) and np.isfinite(value) and value > 0
    valid = valid and payload.get("peak_ram_mib", 0) >= payload.get("ram_before_mib", 0)
    valid = valid and payload.get("elapsed_s", 0) >= payload.get("preparation_s", 0)
    if not valid:
        payload.update(status="invalid_worker", reason="Worker differs from verified source/input/output/workload")
    return payload


def summarize(rows):
    summaries = base.summarize(rows)
    for row in summaries:
        row["operation"] = OPERATION
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--radar-path", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--max-ram-mib", type=float, default=1024)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats < 1 or not np.isfinite(args.timeout_s) or args.timeout_s <= 0 or not np.isfinite(args.max_ram_mib) or args.max_ram_mib <= 0:
        parser.error("Repetitions, timeout and RSS limit must be finite and positive")
    if args.worker:
        if (not all((args.case, args.method, args.input, args.manifest)) or args.validate_only
                or args.run_id or args.output_root or args.radar_path or args.scratch_root or args.repeats != 5):
            parser.error("--worker requires only case/method/input/manifest and resource limits")
    elif any((args.case, args.method, args.input, args.manifest)):
        parser.error("Worker arguments require --worker")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        payload = worker(args)
        print(json.dumps(payload), flush=True)
        return 0 if payload["status"] == "success" else 1
    if not args.validate_only and any(os.environ.get(k) != "1" for k in base.THREAD_VARIABLES):
        raise ValueError("Set all documented numerical-library thread limits to 1 before timing")
    toolkit = preload()
    path = (args.radar_path or Path(os.environ.get("REAL_RADAR_PATH", str(base.radar.DEFAULT_RADAR)))).expanduser().resolve()
    patterns, ranges, az_res, metadata = base.load_verified_inputs(path)
    data = patterns["natural"]  # No new controlled-gap or threshold-sensitivity cases.
    del patterns
    run_id, directory = create_run_directory(EXPERIMENT, args.run_id, args.output_root)
    scratch_root = args.scratch_root or Path(tempfile.gettempdir()) / "opencode"
    rows, diagnostics, differences, kernels, agreement = [], [], [], [], []
    cases, descriptors, measurements, phase_rows = {}, {}, [], []
    with tempfile.TemporaryDirectory(prefix="radar-discrete-gaussian-", dir=scratch_root) as scratch:
        scratch = Path(scratch)
        input_path = scratch / "natural.npy"
        np.save(input_path, data, allow_pickle=False)
        for width in WIDTHS:
            case = f"natural_W{width:g}"
            checks = kernel_checks(run_id, case, SHAPE, width, az_res)
            kernels.extend(checks)
            checked, diag, diff, descriptor, compact = validate_case(run_id, case, data, ranges, width, az_res, toolkit)
            descriptor["verified"] = descriptor["verified"] and all(r["status"] == "pass" for r in checks)
            rows.extend(checked)
            diagnostics.extend(diag)
            differences.extend(diff)
            agreement.extend(compact)
            descriptors[case] = descriptor
            cases[case] = {k: value[0] for k, value in metadata.items()} | descriptor | dict(
                methods=list(METHODS), numerical_reference=REFERENCE, operation=OPERATION,
                gaussian_sigma="converted_grid_width/sqrt(12)", kernel_radius="ceil(4*sigma)",
                kernel_discretization="normalized exp(-0.5*(integer_offset/sigma)^2), center sampled",
                kernel_truncate=TRUNCATE, independent_kernel_check="Astropy Gaussian1DKernel, matching explicit size and center sampling",
                discrete_response="RFFT of circularly accumulated kernel offsets; reflective period 2*N, first N modes",
                centered_variance="full-array angular squared residuals plus range within/between variance; no moment subtraction",
                analytical_role="released analytical response, different operator from sampled Gaussian; verified separately against released API",
                negative_variance_check="raw values before clamp; min, negative count, and exceedances of unchanged radar variance tolerance",
                near_zero_std_rule="when centered reference variance <= variance_atol, max(std tolerance,sqrt(variance tolerance)); variance tests unchanged",
                regions="all cells; range <= 10000 m", numerical_tolerances=base.TOLERANCES,
                boundaries="periodic azimuth; half-sample reflective range", operator_order="azimuth then range, then one normalization",
                width_convention="legacy index widths, not actual physical distance", denominator_floor=FLOOR,
                completion=False, restoration=False, full_filter_calls=3, filter_preparations=1,
                audits={method: expected_audit(method, SHAPE) for method in METHODS},
                repetitions=args.repeats, warmups=0, requested_library_threads=1, fft_workers=1,
                worker_mode="fresh sequential interpreters, rotated method order, identical imports",
                preloaded_libraries=LIBRARIES, timing_includes="masking, zero replacement, squares, kernel/response preparation, three filters, normalized moments, coverage/count, allocation",
                timing_excludes="I/O, imports, release verification, references, hashes, diagnostics, CSV, plotting",
                preparation_time="subset of total time: weights and responses constructed once; no cache across calls",
                memory="total process RSS, 1-ms in-call samples plus newly exceeded lifetime peak, not allocation-only RAM",
                worker_timeout_s=args.timeout_s, worker_rss_limit_mib=args.max_ram_mib,
                timing_status="not run: validation pending")
            print(f"Validated {case}: {'pass' if descriptor['verified'] else 'FAIL'}", flush=True)
        for name, fields, values in (
            ("comparison.csv", COMPARISON_FIELDS, rows), ("kernel_comparison.csv", COMPARISON_FIELDS, differences),
            ("kernel_checks.csv", KERNEL_FIELDS, kernels), ("variance_diagnostics.csv", DIAGNOSTIC_FIELDS, diagnostics),
            ("agreement_summary.csv", AGREEMENT_FIELDS, agreement),
        ):
            write_csv(directory / name, fields, values)
        failed = not all(d["verified"] for d in descriptors.values())
        manifest = scratch / "verified.json"
        manifest.write_text(json.dumps(descriptors, sort_keys=True))
        if not failed and not args.validate_only:
            for case_index, (case, descriptor) in enumerate(descriptors.items()):
                for repetition in range(1, args.repeats + 1):
                    start = (case_index + repetition - 1) % len(METHODS)
                    for method in METHODS[start:] + METHODS[:start]:
                        payload = check_worker(measure_worker(args, case, method, input_path, manifest), descriptor, case, method)
                        cases[case][f"worker_{method}_{repetition}"] = payload
                        measurements.append(dict(run_id=run_id, case_id=case, method=method, operation=OPERATION,
                                                 repetition=repetition, **{k: payload.get(k) for k in (
                                                     "status", "elapsed_s", "ram_before_mib", "peak_ram_mib", "reason")}))
                        phase_rows.append(dict(run_id=run_id, case_id=case, method=method, repetition=repetition,
                                               status=payload["status"], preparation_s=payload.get("preparation_s"), total_s=payload.get("elapsed_s")))
                        write_csv(directory / "benchmark_runs.csv", BENCHMARK_FIELDS, measurements)
                        write_csv(directory / "benchmark_summary.csv", SUMMARY_FIELDS, summarize(measurements))
                        write_csv(directory / "benchmark_phases.csv", PHASE_FIELDS, phase_rows)
                        print(f"{case} {method} repetition {repetition}: {payload['status']} ({payload.get('elapsed_s')} s)", flush=True)
                        if payload["status"] != "success":
                            failed = True
                            break
                    if failed:
                        break
                if failed:
                    break
        timing = []
        summaries = summarize(measurements) if measurements else []
        for case, values in cases.items():
            complete = all(any(r["case_id"] == case and r["method"] == m and r["n_success"] == args.repeats and not r["n_failure"] for r in summaries) for m in METHODS)
            values["timing_status"] = ("measured and verified" if complete else "not run: --validate-only" if args.validate_only and not failed else "incomplete or not run: verification/resource failure")
            timing.append(dict(run_id=run_id, width=values["width"], **{
                m: next((r["median_time_s"] for r in summaries if r["case_id"] == case and r["method"] == m), None) for m in METHODS}))
        write_csv(directory / "timing_summary.csv", TIMING_FIELDS, timing)
        write_settings(directory / "settings.csv", run_id, cases)
    print(f"{directory}: {'FAIL' if failed else 'pass'}; {len(rows)} correctness comparisons; {len(measurements)} measurements", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
