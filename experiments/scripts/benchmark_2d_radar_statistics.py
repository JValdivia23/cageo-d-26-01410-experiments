#!/usr/bin/env python3
"""Matched Gaussian radar statistics: three filters, identical missing-data rules.

Use the exact v1.0.0 runtime documented in experiments/README.md. This separate
script preserves the earlier radar validation and mean-only timing results.
Only the existing 720x1180 sweep, W=5/15, and natural/controlled-gap masks are
measured. Five repetitions use fresh, sequential workers, without filter warm-up.

The prepared DCT/RFFT operator follows released polar.py with vectorized azimuth
response construction and one preparation shared by mask/data/squared data.
The independent FFT uses circular azimuth and a half-sample even range extension.
Full-support direct convolution applies the SAME finite-grid weights, constructed
efficiently by inverse FFT; it is NOT ordinary truncated spatial Gaussian
smoothing. The original direct-cosine-sum reference remains an untimed oracle.

Before timing, full-array outputs, raw denominators, accepted masks, weights and
released API outputs must pass the unchanged default-floor tolerances. Workers
check input identity and the exact output fingerprints of the verified method.
Setup, masking, three filters, normalization and output allocation are timed;
input reading, imports, verification, plotting and completion are not. No figure
or old result is written. Run --validate-only for correctness checks alone.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import psutil
from scipy.fft import dct, idct, rfft, irfft
from scipy.ndimage import convolve1d

import generate_2d_radar_gaussian_validation as radar
from _review_common import (
    BENCHMARK_FIELDS, COMPARISON_FIELDS, SUMMARY_FIELDS, TOOLKIT_COMMIT,
    create_run_directory, load_toolkit, write_csv, write_settings,
)


METHODS = ("prepared_dct_rfft", "symmetric_fft", "full_support_direct")
API = "released_api_validation_only"
REFERENCE = "independent_cosine_sum_spatial_oracle"
WIDTHS = (5.0, 15.0)
PATTERNS = ("natural", "controlled_extra_gap")
SHAPE = (720, 1180)
DECODED_SHA256 = "e47d1b8e07791951a602344edcdcf60542a517f385e54c452fd81117f676d9a0"
FLOOR = 1e-3
OPERATION = "scalar_polar_gaussian_statistics_three_filters_setup_included"
FIELDS = ("mean", "second_moment", "raw_variance", "variance", "std",
          "raw_mask_response", "direct_acceptance", "coverage", "support")
TOLERANCES = dict(radar.TOLERANCES, coverage=radar.TOLERANCES["raw_mask_response"])
MIB = 1024**2
RSS_INTERVAL = 0.001
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def array_sha256(values):
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def input_patterns(data):
    """Exactly the existing middle-quarter removal, retaining natural missingness."""
    gapped = data.copy()
    middle = slice(3 * data.shape[0] // 8, 5 * data.shape[0] // 8)
    gapped[middle, :] = np.nan
    return dict(natural=data, controlled_extra_gap=gapped)


def check_geometry(shape, width, az_res):
    if (len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape)
            or not np.isfinite(width) or width <= 0 or not np.isfinite(az_res) or az_res <= 0):
        raise ValueError("Expected a positive 2D shape, finite positive width and azimuth spacing")


def independent_responses(shape, width, az_res):
    """Analytic periodic responses, without toolkit response/filter calls."""
    check_geometry(shape, width, az_res)
    n_az, n_range = shape
    sigma_az = width / ((np.arange(n_range) + 1) * np.deg2rad(az_res) * np.sqrt(12))
    az_frequency = 2 * np.pi * np.arange(n_az // 2 + 1) / n_az
    h_az = np.exp(-0.5 * (az_frequency[:, None] * sigma_az[None, :]) ** 2)
    range_frequency = np.pi * np.arange(n_range + 1) / n_range
    h_range = np.exp(-0.5 * (range_frequency * width / np.sqrt(12)) ** 2)
    # A half-sample even extension has zero Nyquist coefficient. Match the
    # original cosine-sum oracle's convention, including in direct weights.
    h_range[-1] = 0.0
    return h_az, h_range


def independent_weights(shape, width, az_res):
    """All centered offsets, including signed tails and split even antipodes."""
    h_az, h_range = independent_responses(shape, width, az_res)
    n_az, n_range = shape
    circular = irfft(h_az, n=n_az, axis=0, workers=1)
    offsets = np.arange(-(n_az // 2), n_az // 2 + 1)
    az_weights = circular[offsets % n_az].copy()
    if n_az % 2 == 0:
        az_weights[[0, -1]] *= 0.5
    circular_range = irfft(h_range, n=2 * n_range, workers=1)
    offsets = np.arange(-n_range, n_range + 1)
    range_weights = circular_range[offsets % (2 * n_range)].copy()
    range_weights[[0, -1]] *= 0.5
    return az_weights, range_weights


def prepare_filter(method, shape, width, az_res, toolkit, audit):
    """Prepare once per measured call; every invocation is one complete filter."""
    check_geometry(shape, width, az_res)
    if method not in METHODS:
        raise ValueError(f"Unknown benchmark method: {method}")
    audit["filter_preparations"] = audit.get("filter_preparations", 0) + 1
    n_az, n_range = shape
    if method == METHODS[0]:
        # Vectorized form of released polar.py's periodic Gaussian formula.
        widths = width / ((np.arange(n_range) + 1) * np.deg2rad(az_res))
        omega = (2 * np.pi * np.arange(n_az // 2 + 1)) / n_az
        h_az = np.exp(-0.5 * (omega[:, None] * (widths / np.sqrt(12))[None, :]) ** 2)
        h_range = toolkit.get_dct_transfer_function(n_range, "gaussian", float(width))
    elif method == METHODS[1]:
        h_az, h_range = independent_responses(shape, width, az_res)
    else:
        az_weights, range_weights = independent_weights(shape, width, az_res)

    def linear_filter(values):
        if values.shape != shape:
            raise ValueError("Input shape differs from the prepared filter geometry")
        audit["full_filter_calls"] = audit.get("full_filter_calls", 0) + 1
        if method == METHODS[2]:
            stage = np.empty(shape, dtype=np.float64)
            for gate in range(n_range):
                stage[:, gate] = convolve1d(values[:, gate], az_weights[:, gate], mode="wrap", origin=0)
            return convolve1d(stage, range_weights, axis=1, mode="reflect", origin=0)
        spectrum = rfft(values, axis=0, norm="ortho", workers=1)
        spectrum *= h_az
        stage = irfft(spectrum, n=n_az, axis=0, norm="ortho", workers=1)
        if method == METHODS[0]:
            spectrum = dct(stage, axis=1, type=2, norm="ortho", workers=1)
            spectrum *= h_range[None, :]
            return idct(spectrum, axis=1, type=2, norm="ortho", workers=1)
        extended = np.concatenate([stage, stage[:, ::-1]], axis=1)
        spectrum = rfft(extended, axis=1, norm="ortho", workers=1)
        spectrum *= h_range[None, :]
        filtered = irfft(spectrum, n=2 * n_range, axis=1, norm="ortho", workers=1)
        return filtered[:, :n_range].copy()

    return linear_filter


def moment_fields(denominator, numerator, squared_numerator, width, az_res):
    """Common output construction for every backend; ratios use unclipped d."""
    accepted = np.isfinite(denominator) & (denominator > FLOOR)
    mean = np.full(denominator.shape, np.nan)
    second = np.full(denominator.shape, np.nan)
    np.divide(numerator, denominator, out=mean, where=accepted)
    np.divide(squared_numerator, denominator, out=second, where=accepted)
    raw_variance = second - mean**2
    variance = np.maximum(raw_variance, 0.0)
    coverage = np.clip(denominator, 0, 1)
    nominal = width * radar.azimuth_widths(denominator.shape[1], width, az_res)[None, :]
    return dict(mean=mean, second_moment=second, raw_variance=raw_variance,
                variance=variance, std=np.sqrt(variance), raw_mask_response=denominator,
                direct_acceptance=accepted, coverage=coverage, support=coverage * nominal)


def calculate(method, data, width, az_res, toolkit, audit=None):
    audit = audit if audit is not None else {}
    valid = np.isfinite(data)
    zeroed = np.where(valid, data, 0.0)
    linear = prepare_filter(method, data.shape, width, az_res, toolkit, audit)
    denominator = linear(valid.astype(np.float64))
    numerator = linear(zeroed)
    squared_numerator = linear(zeroed**2)
    return moment_fields(denominator, numerator, squared_numerator, width, az_res)


def compare_arrays(run_id, case_id, method, statistic, region, actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Comparison shapes differ")
    af, bf = np.isfinite(a), np.isfinite(b)
    joint = af & bf
    delta = a[joint] - b[joint]
    atol, rtol = TOLERANCES[statistic]
    passed = (np.array_equal(af, bf) and not np.isinf(a).any() and not np.isinf(b).any()
              and np.all(np.abs(delta) <= atol + rtol * np.abs(b[joint])))
    return dict(run_id=run_id, case_id=case_id, method=method, reference_method=REFERENCE,
                component="DBZHC", statistic=statistic, region=region,
                estimate_type="direct_ratio_no_completion", kernel_match="full_finite_analytic_gaussian",
                n_considered=a.size, n_compared=int(joint.sum()),
                n_method_undefined=int((~af).sum()), n_reference_undefined=int((~bf).sum()),
                n_method_gap_filled=0, n_reference_gap_filled=0,
                max_abs_difference=float(np.max(np.abs(delta))) if delta.size else None,
                rmse=float(np.sqrt(np.mean(delta**2))) if delta.size else None,
                atol=atol, rtol=rtol, status=("pass" if joint.any() else "no_comparable") if passed else "fail",
                units=radar.UNITS.get(statistic, "1"),
                notes="All selected cells; exact defined-mask equality; original default-floor tolerances. "
                      "Coverage is availability, not confidence; signed weights and raw variance retained.")


def fingerprints(fields):
    if set(fields) != set(FIELDS):
        raise ValueError("Required statistics outputs changed")
    return {key: array_sha256(fields[key]) for key in FIELDS}


def validate_case(run_id, case_id, data, ranges, width, az_res, toolkit, oracle):
    valid = np.isfinite(data)
    zeroed = np.where(valid, data, 0.0)
    expected = radar.direct_fields(oracle(valid.astype(float)), oracle(zeroed), oracle(zeroed**2),
                                   valid, width, az_res, FLOOR)
    expected["coverage"] = np.clip(expected["raw_mask_response"], 0, 1)
    regions = radar.regions(data.shape, ranges)
    rows, verified = [], {}
    for method in (*METHODS, API):
        if method == API:
            options = dict(az_res_deg=az_res, az_boundary="periodic", range_boundary="reflective", kernel_type="gaussian")
            density = toolkit.smooth_polar(valid.astype(float), width, **options)
            fields = radar.toolkit_fields(toolkit, data, width, az_res, FLOOR, density)
            fields["coverage"] = np.clip(density, 0, 1)
            fields = {key: fields[key] for key in FIELDS}
        else:
            audit = {}
            fields = calculate(method, data, width, az_res, toolkit, audit)
            if audit != dict(filter_preparations=1, full_filter_calls=3):
                raise ValueError("The benchmark must prepare once and apply three complete filters")
        for key in FIELDS:
            for region, selection in regions.items():
                rows.append(compare_arrays(run_id, case_id, method, key, region,
                                           fields[key][selection], expected[key][selection]))
        if method in METHODS:
            verified[method] = fingerprints(fields)
        del fields
    descriptor = dict(verified=all(r["status"] in ("pass", "no_comparable") for r in rows),
                      benchmark_script_sha256=radar.sha256_file(Path(__file__)),
                      input_sha256=array_sha256(data), shape=list(data.shape), width=width,
                      az_res_deg=az_res, output_sha256=verified,
                      n_input_valid=int(valid.sum()), missing_fraction=float((~valid).mean()),
                      n_accepted=int(expected["direct_acceptance"].sum()),
                      n_negative_raw_variance=int(np.count_nonzero(expected["raw_variance"] < 0)))
    return rows, descriptor


def load_verified_inputs(path):
    data, ranges, az_res, metadata = radar.read_radar(path, quick=False)
    if (metadata["input_sha256"][0] != radar.DOCUMENTED_SHA256
            or array_sha256(data) != DECODED_SHA256 or data.shape != SHAPE):
        raise ValueError("Radar file or full-sweep decoded input identity changed")
    # Check all old evidence bytes, without changing or rerunning that experiment.
    radar._figure_saved_evidence()
    with (radar.FIGURE_RUN_DIRECTORY / "settings.csv").open(newline="") as stream:
        frozen = {}
        for row in csv.DictReader(stream):
            frozen.setdefault(row["case_id"], {})[row["setting"]] = row["value"]
    patterns = input_patterns(data)
    for pattern, values in patterns.items():
        for width in WIDTHS:
            old = frozen[f"{pattern}_W{width:g}_floor{FLOOR:.0e}"]
            if (int(np.isfinite(values).sum()) != int(old["n_input_valid"])
                    or float((~np.isfinite(values)).mean()) != float(old["missing_fraction"])):
                raise ValueError("Previously approved radar input mask changed")
    return patterns, ranges, az_res, metadata


def lifetime_peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def worker(args):
    """Only the checked calculation is timed; checksums follow timer/sampler stop."""
    payload = dict(status="error", elapsed_s=None, ram_before_mib=None, peak_ram_mib=None, reason="worker did not finish")
    stop, sampler = threading.Event(), None
    try:
        if any(os.environ.get(key) != "1" for key in THREAD_VARIABLES):
            raise ValueError("Set all documented numerical-library thread limits to 1")
        toolkit = load_toolkit()  # Same imports/source checks for every method.
        descriptor = json.loads(args.manifest.read_text())[args.case]
        data = np.load(args.input, allow_pickle=False)
        if (not descriptor["verified"] or data.dtype != np.float64
                or list(data.shape) != descriptor["shape"] or data.shape != SHAPE
                or descriptor["benchmark_script_sha256"] != radar.sha256_file(Path(__file__))
                or array_sha256(data) != descriptor["input_sha256"]):
            raise ValueError("Worker input is not the numerically verified radar case")
        payload.update(input_sha256=descriptor["input_sha256"], worker_pid=os.getpid(),
                       method=args.method, case_id=args.case,
                       benchmark_script_sha256=descriptor["benchmark_script_sha256"])
        gc.collect()
        process = psutil.Process()
        before = process.memory_info().rss
        peak, ready = [before], threading.Event()

        def sample():
            peak[0] = max(peak[0], process.memory_info().rss)
            ready.set()
            while not stop.wait(RSS_INTERVAL):
                peak[0] = max(peak[0], process.memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        ready.wait()
        before = process.memory_info().rss
        peak[0] = max(peak[0], before)
        hwm_before = lifetime_peak_bytes()
        audit = {}
        start = time.perf_counter()
        fields = calculate(args.method, data, descriptor["width"], descriptor["az_res_deg"], toolkit, audit)
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        hwm_after = lifetime_peak_bytes()
        stop.set()
        sampler.join()
        if hwm_after > hwm_before:
            peak[0] = max(peak[0], hwm_after)
        if audit != dict(filter_preparations=1, full_filter_calls=3):
            raise ValueError("Worker did not perform one preparation and three filters")
        actual = fingerprints(fields)
        if actual != descriptor["output_sha256"][args.method]:
            raise ValueError("Worker full-array output fingerprints differ from verified method")
        payload.update(status="success", elapsed_s=elapsed, ram_before_mib=before / MIB,
                       peak_ram_mib=peak[0] / MIB, reason="", audit=audit, n_outputs=len(fields),
                       output_sha256=actual, output_fingerprints_verified=True,
                       precall_lifetime_peak_mib=hwm_before / MIB, final_lifetime_peak_mib=hwm_after / MIB,
                       peak_source="in-call RSS + newly exceeded lifetime peak" if hwm_after > hwm_before else "in-call RSS",
                       preloaded_libraries=["numpy", "scipy.fft", "scipy.ndimage", "psutil", "dct_toolkit"],
                       rss_sample_interval_s=RSS_INTERVAL)
        if peak[0] / MIB > args.max_ram_mib:
            payload.update(status="resource_limit", reason="Calculation exceeded worker RSS budget")
    except Exception as error:
        payload.update(reason=f"{type(error).__name__}: {error}")
    finally:
        stop.set()
        if sampler is not None:
            sampler.join()
    return payload


def measure_worker(args, case_id, method, input_path, manifest):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--case", case_id,
               "--method", method, "--input", str(input_path), "--manifest", str(manifest),
               "--max-ram-mib", str(args.max_ram_mib)]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process, start = psutil.Process(child.pid), time.monotonic()
    reason = ""
    while child.poll() is None:
        try:
            if process.memory_info().rss / MIB > args.max_ram_mib:
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
        time.sleep(0.01)
    stdout, stderr = child.communicate()
    if reason:
        return dict(status="resource_limit", elapsed_s=None, ram_before_mib=None, peak_ram_mib=None, reason=reason)
    try:
        payload = json.loads(stdout)
    except (ValueError, TypeError):
        return dict(status="error", elapsed_s=None, ram_before_mib=None, peak_ram_mib=None,
                    reason=f"Worker exit {child.returncode}: {stderr.strip() or stdout.strip()}")
    if child.returncode and payload.get("status") == "success":
        payload.update(status="error", reason=f"Worker returned success but exited {child.returncode}")
    if stderr.strip():
        payload["worker_stderr"] = stderr.strip()
    return payload


def check_worker(payload, descriptor, case_id, method):
    """A timing cannot enter successful summaries without the verified workload."""
    if payload.get("status") != "success":
        return payload
    required = (payload.get("input_sha256") == descriptor["input_sha256"]
                and payload.get("benchmark_script_sha256") == descriptor["benchmark_script_sha256"]
                and payload.get("case_id") == case_id and payload.get("method") == method
                and payload.get("audit") == dict(filter_preparations=1, full_filter_calls=3)
                and payload.get("n_outputs") == len(FIELDS)
                and payload.get("output_fingerprints_verified") is True
                and payload.get("output_sha256") == descriptor["output_sha256"][method]
                and isinstance(payload.get("elapsed_s"), (int, float))
                and np.isfinite(payload["elapsed_s"]) and payload["elapsed_s"] > 0
                and isinstance(payload.get("ram_before_mib"), (int, float))
                and isinstance(payload.get("peak_ram_mib"), (int, float))
                and np.isfinite(payload["ram_before_mib"]) and np.isfinite(payload["peak_ram_mib"])
                and payload["peak_ram_mib"] >= payload["ram_before_mib"] > 0
                and payload.get("preloaded_libraries") == ["numpy", "scipy.fft", "scipy.ndimage", "psutil", "dct_toolkit"])
    if not required:
        payload.update(status="invalid_worker", reason="Worker input/outputs/imports/filter audit differ from verified workload")
    return payload


def summarize(rows):
    summaries = []
    for case_id, method in dict.fromkeys((row["case_id"], row["method"]) for row in rows):
        group = [r for r in rows if r["case_id"] == case_id and r["method"] == method]
        success = [r for r in group if r["status"] == "success"]
        times = [r["elapsed_s"] for r in success]
        summaries.append(dict(run_id=group[0]["run_id"], case_id=case_id, method=method, operation=OPERATION,
                              n_success=len(success), n_failure=len(group) - len(success),
                              median_time_s=float(np.median(times)) if times else None,
                              q25_time_s=float(np.quantile(times, .25)) if times else None,
                              q75_time_s=float(np.quantile(times, .75)) if times else None,
                              max_peak_ram_mib=max(r["peak_ram_mib"] for r in success) if success else None))
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--radar-path", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
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
        parser.error("case/method/input/manifest require --worker")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        payload = worker(args)
        print(json.dumps(payload), flush=True)
        return 0 if payload["status"] == "success" else 1
    if not args.validate_only and any(os.environ.get(key) != "1" for key in THREAD_VARIABLES):
        raise ValueError("Set all documented numerical-library thread limits to 1 before timing")
    toolkit = load_toolkit()
    path = (args.radar_path or Path(os.environ.get("REAL_RADAR_PATH", str(radar.DEFAULT_RADAR)))).expanduser().resolve()
    patterns, ranges, az_res, metadata = load_verified_inputs(path)
    run_id, directory = create_run_directory("2d_radar_statistics_benchmark", args.run_id, args.output_root)
    rows, descriptors, cases, input_paths = [], {}, {}, {}
    scratch_root = args.scratch_root
    if scratch_root is None:
        preferred = Path(tempfile.gettempdir()) / "opencode"
        scratch_root = preferred if preferred.is_dir() else Path(tempfile.gettempdir())
    # Only this process's own small temporary input/manifest directory is removed.
    with tempfile.TemporaryDirectory(prefix="radar-statistics-", dir=scratch_root) as scratch:
        scratch = Path(scratch)
        for pattern, values in patterns.items():
            input_paths[pattern] = scratch / f"{pattern}.npy"
            np.save(input_paths[pattern], values, allow_pickle=False)
        for width in WIDTHS:
            oracle = radar.SpatialPolarGaussian(SHAPE, width, az_res)
            fast_weights = independent_weights(SHAPE, width, az_res)
            for axis, actual, expected in zip(("azimuth_weights", "range_weights"), fast_weights,
                                              (oracle.az_weights, oracle.range_weights)):
                rows.append(compare_arrays(run_id, f"natural_W{width:g}", "inverse_fft_weight_construction",
                                           "operator", axis, actual, expected))
            for pattern, values in patterns.items():
                case_id = f"{pattern}_W{width:g}"
                checked, descriptor = validate_case(run_id, case_id, values, ranges, width, az_res, toolkit, oracle)
                rows.extend(checked)
                descriptors[case_id] = descriptor
                cases[case_id] = {key: value[0] for key, value in metadata.items()} | dict(
                    case_pattern=pattern, width=width, shape=list(SHAPE), **{key: value for key, value in descriptor.items()
                                                                         if key not in ("shape", "width")},
                    radar_file_sha256=metadata["input_sha256"][0],
                    methods=list(METHODS), numerical_reference=REFERENCE, numerical_tolerances=TOLERANCES,
                    oracle_role="untimed original cosine-sum weights and full-support spatial application",
                    radar_reference_script_sha256=radar.sha256_file(Path(radar.__file__)),
                    frozen_validation_run=radar.FIGURE_RUN_ID,
                    full_filter_calls=3, filter_preparations=1, required_outputs=list(FIELDS),
                    operation=OPERATION, denominator_floor=FLOOR, denominator_rule="finite raw d strictly > floor",
                    completion=False, restoration=False, population_variance_ddof=0,
                    coverage="clip(raw d,0,1); availability, not confidence or ESS",
                    support="coverage * W * W/((j+1)*deg2rad(az_res)); equivalent count, not ESS",
                    boundaries="periodic azimuth; half-sample reflective range",
                    operator_order="azimuth first with range-dependent widths; range second; then normalization",
                    width_convention="unchanged legacy index units, NOT actual physical distances",
                    dct_implementation="experiment-local shared moments; vectorized released periodic Gaussian formula and released get_dct_transfer_function; NOT released statistics API timing",
                    dct_source="v1.0.0 polar.py periodic RFFT/IRFFT axis 0 then DCT-II/IDCT-II axis 1; same analytic Gaussian responses",
                    fft_implementation="independent analytic RFFT azimuth; concatenate(stage,stage reversed in range); 2R RFFT/IRFFT; crop",
                    direct_implementation="all finite-grid weights from inverse FFT, then convolve1d wrap/reflect; no tail truncation; NOT sampled Gaussian filtering",
                    direct_weight_lengths=[SHAPE[0] + 1, 2 * SHAPE[1] + 1],
                    filter_setup="once per calculation INSIDE timer; same reuse across three fields; no inter-call cache",
                    timing_includes="finite mask, zero replacement, squared input, response/weight construction, three complete filters, normalization, variance, std, coverage/equivalent count, output allocation",
                    timing_excludes="input file read/load/hash, imports/release check, independent verification, output fingerprints, CSV, plots",
                    repetitions=args.repeats, warmups=0, worker_mode="fresh interpreter for each repetition, one worker at a time",
                    method_order="rotate by case index and repetition; no simultaneous measurements",
                    fft_workers=1, requested_library_threads=1,
                    physical_cpu_cores=psutil.cpu_count(logical=False), machine_total_ram_mib=psutil.virtual_memory().total / MIB,
                    memory="total process RSS; 1-ms in-call samples plus newly exceeded lifetime high-water mark; not allocation-only RAM",
                    rss_sample_interval_s=RSS_INTERVAL, worker_timeout_s=args.timeout_s, worker_rss_limit_mib=args.max_ram_mib,
                    numerical_scope="all cells and centre/seam/outer regions; unchanged default-floor tolerances; no new floor sensitivity or recovery experiment",
                    timing_status="not run: validation pending",
                )
                print(f"Validated {case_id}: {'pass' if descriptor['verified'] else 'FAIL'}", flush=True)
            del oracle
        write_csv(directory / "comparison.csv", COMPARISON_FIELDS, rows)
        failed = any(row["status"] == "fail" for row in rows)
        manifest = scratch / "verified.json"
        manifest.write_text(json.dumps(descriptors, sort_keys=True))
        measurements = []
        if not failed and not args.validate_only:
            for case_index, (case_id, descriptor) in enumerate(descriptors.items()):
                pattern = cases[case_id]["case_pattern"]
                for repetition in range(1, args.repeats + 1):
                    start = (case_index + repetition - 1) % len(METHODS)
                    order = METHODS[start:] + METHODS[:start]
                    for method in order:
                        payload = check_worker(measure_worker(args, case_id, method, input_paths[pattern], manifest),
                                               descriptor, case_id, method)
                        cases[case_id][f"worker_{method}_{repetition}"] = payload
                        measurements.append(dict(run_id=run_id, case_id=case_id, method=method,
                                                 operation=OPERATION, repetition=repetition,
                                                 **{key: payload.get(key) for key in ("status", "elapsed_s", "ram_before_mib", "peak_ram_mib", "reason")}))
                        print(f"{case_id} {method} repetition {repetition}: {payload['status']}"
                              f" ({payload.get('elapsed_s')} s)", flush=True)
                        write_csv(directory / "benchmark_runs.csv", BENCHMARK_FIELDS, measurements)
                        write_csv(directory / "benchmark_summary.csv", SUMMARY_FIELDS, summarize(measurements))
                        if payload["status"] != "success":
                            failed = True
                            break
                    if failed:
                        break
                if failed:
                    break
        for case_id, values in cases.items():
            count = sum(r["case_id"] == case_id for r in measurements)
            values["timing_status"] = ("measured and verified" if count == args.repeats * len(METHODS) and not failed
                                       else "not run: --validate-only" if args.validate_only and not failed
                                       else "incomplete or not run: verification/resource failure")
        write_settings(directory / "settings.csv", run_id, cases)
    print(f"{directory}: {'FAIL' if failed else 'pass'}; {len(rows)} full-array comparisons; "
          f"{len(measurements)} worker measurements", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
