#!/usr/bin/env python3
"""Prepared spectral radar statistics versus NumPy/SciPy range-adaptive loops.

Extend the matched radar experiment without modifying its earlier source/results.
Every method prepares once and filters mask, zero-filled data, and squared data.
The direct methods use the same full finite-grid weights, not sampled/truncated
Gaussians. Both loop over range gates with a different azimuth kernel per gate.
NumPy uses convolve with wrap padding, then a row loop with symmetric padding;
SciPy uses convolve1d wrap per gate, then its compiled reflective range pass.
Shared inverse-FFT weight construction is included for both direct methods.

All numerical comparisons must pass before five fresh sequential workers per
method/case. Preparation, padding, filtering, moments and allocation are timed;
input I/O, imports, reference calculations and fingerprints are not. The pinned
released statistics API is checked, not timed. No figure or earlier CSV is written.
"""

from __future__ import annotations

import argparse
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
from scipy.ndimage import convolve1d

import benchmark_2d_radar_statistics as base
from _review_common import BENCHMARK_FIELDS, COMPARISON_FIELDS, SUMMARY_FIELDS, create_run_directory, write_csv, write_settings


METHODS = ("prepared_dct_rfft", "numpy_range_loop", "scipy_range_loop")
SHAPE = base.SHAPE
FIELDS = base.FIELDS
OPERATION = base.OPERATION
LIBRARIES = ["numpy", "scipy.fft", "scipy.ndimage", "psutil", "dct_toolkit"]


def source_hashes():
    return {name: base.radar.sha256_file(path) for name, path in (
        ("benchmark", Path(__file__)), ("spectral_helper", Path(base.__file__)),
        ("radar_reference", Path(base.radar.__file__)),
        ("common_helper", Path(__file__).with_name("_review_common.py")),
    )}


def expected_audit(method, shape):
    audit = dict(filter_preparations=1, full_filter_calls=3)
    if method in METHODS[1:]:
        audit["azimuth_gate_convolutions"] = 3 * shape[1]
        audit["range_convolutions"] = 3 * shape[0] if method == METHODS[1] else 3
    return audit


def prepare_filter(method, shape, width, az_res, toolkit, audit):
    if method == METHODS[0]:
        return base.prepare_filter(method, shape, width, az_res, toolkit, audit)
    if method not in METHODS[1:]:
        raise ValueError(f"Unknown range-loop method: {method}")
    base.check_geometry(shape, width, az_res)
    audit["filter_preparations"] = audit.get("filter_preparations", 0) + 1
    # Identical analytic weights/setup for both direct methods. No truncation,
    # positive-weight substitution, or toolkit filter is used by either loop.
    az_weights, range_weights = base.independent_weights(shape, width, az_res)
    half_az, half_range = az_weights.shape[0] // 2, range_weights.size // 2

    def linear(values):
        if values.shape != shape:
            raise ValueError("Input shape differs from prepared loop geometry")
        audit["full_filter_calls"] = audit.get("full_filter_calls", 0) + 1
        stage = np.empty(shape, dtype=np.float64)
        for gate in range(shape[1]):
            if method == METHODS[1]:
                padded = np.pad(values[:, gate], (half_az, half_az), mode="wrap")
                stage[:, gate] = np.convolve(padded, az_weights[:, gate], mode="valid")
            else:
                stage[:, gate] = convolve1d(values[:, gate], az_weights[:, gate], mode="wrap", origin=0)
            audit["azimuth_gate_convolutions"] = audit.get("azimuth_gate_convolutions", 0) + 1
        if method == METHODS[2]:
            audit["range_convolutions"] = audit.get("range_convolutions", 0) + 1
            return convolve1d(stage, range_weights, axis=1, mode="reflect", origin=0)
        result = np.empty(shape, dtype=np.float64)
        for ray in range(shape[0]):
            # NumPy symmetric padding repeats the endpoint, matching SciPy's
            # half-sample `reflect`, NOT NumPy's whole-sample `reflect` mode.
            padded = np.pad(stage[ray], (half_range, half_range), mode="symmetric")
            result[ray] = np.convolve(padded, range_weights, mode="valid")
            audit["range_convolutions"] = audit.get("range_convolutions", 0) + 1
        return result

    return linear


def calculate(method, data, width, az_res, toolkit, audit=None):
    audit = audit if audit is not None else {}
    valid = np.isfinite(data)
    zeroed = np.where(valid, data, 0.0)
    linear = prepare_filter(method, data.shape, width, az_res, toolkit, audit)
    denominator = linear(valid.astype(np.float64))
    numerator = linear(zeroed)
    squared_numerator = linear(zeroed**2)
    return base.moment_fields(denominator, numerator, squared_numerator, width, az_res)


def validate_case(run_id, case_id, data, ranges, width, az_res, toolkit, oracle):
    valid = np.isfinite(data)
    zeroed = np.where(valid, data, 0.0)
    reference = base.radar.direct_fields(oracle(valid.astype(float)), oracle(zeroed), oracle(zeroed**2),
                                         valid, width, az_res, base.FLOOR)
    reference["coverage"] = np.clip(reference["raw_mask_response"], 0, 1)
    regions, rows, hashes = base.radar.regions(data.shape, ranges), [], {}
    for method in (*METHODS, base.API):
        if method == base.API:
            density = toolkit.smooth_polar(valid.astype(float), width, az_res_deg=az_res,
                                           az_boundary="periodic", range_boundary="reflective", kernel_type="gaussian")
            output = base.radar.toolkit_fields(toolkit, data, width, az_res, base.FLOOR, density)
            output["coverage"] = np.clip(density, 0, 1)
            output = {key: output[key] for key in FIELDS}
        else:
            audit = {}
            output = calculate(method, data, width, az_res, toolkit, audit)
            if audit != expected_audit(method, data.shape):
                raise ValueError("Preparation, full filters or range-loop calls differ from the required workload")
            hashes[method] = base.fingerprints(output)
        for key in FIELDS:
            for region, selection in regions.items():
                rows.append(base.compare_arrays(run_id, case_id, method, key, region,
                                                 output[key][selection], reference[key][selection]))
    descriptor = dict(verified=all(row["status"] in ("pass", "no_comparable") for row in rows),
                      source_sha256=source_hashes(), input_sha256=base.array_sha256(data),
                      shape=list(data.shape), width=width, az_res_deg=az_res, output_sha256=hashes,
                      n_input_valid=int(valid.sum()), missing_fraction=float((~valid).mean()),
                      n_accepted=int(reference["direct_acceptance"].sum()))
    return rows, descriptor


def worker(args):
    payload = dict(status="error", elapsed_s=None, ram_before_mib=None, peak_ram_mib=None, reason="worker did not finish")
    stop, sampler = threading.Event(), None
    try:
        if any(os.environ.get(key) != "1" for key in base.THREAD_VARIABLES):
            raise ValueError("Set all documented numerical-library thread limits to 1")
        toolkit = base.load_toolkit()
        descriptor = json.loads(args.manifest.read_text())[args.case]
        data = np.load(args.input, allow_pickle=False)
        if (not descriptor["verified"] or descriptor["source_sha256"] != source_hashes()
                or data.dtype != np.float64 or data.shape != SHAPE or list(data.shape) != descriptor["shape"]
                or base.array_sha256(data) != descriptor["input_sha256"]):
            raise ValueError("Worker source/input is not the numerically verified case")
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
        audit = {}
        start = time.perf_counter()
        output = calculate(args.method, data, descriptor["width"], descriptor["az_res_deg"], toolkit, audit)
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        hwm_after = base.lifetime_peak_bytes()
        stop.set()
        sampler.join()
        if hwm_after > hwm_before:
            peak[0] = max(peak[0], hwm_after)
        hashes = base.fingerprints(output)
        if audit != expected_audit(args.method, data.shape) or hashes != descriptor["output_sha256"][args.method]:
            raise ValueError("Worker loop/filter audit or full-array fingerprints differ from verified calculation")
        payload.update(status="success", elapsed_s=elapsed, ram_before_mib=before / base.MIB,
                       peak_ram_mib=peak[0] / base.MIB, reason="", case_id=args.case, method=args.method,
                       worker_pid=os.getpid(), source_sha256=descriptor["source_sha256"],
                       input_sha256=descriptor["input_sha256"], output_sha256=hashes,
                       output_fingerprints_verified=True, audit=audit, n_outputs=len(output),
                       preloaded_libraries=LIBRARIES, rss_sample_interval_s=base.RSS_INTERVAL,
                       precall_lifetime_peak_mib=hwm_before / base.MIB, final_lifetime_peak_mib=hwm_after / base.MIB)
        if peak[0] / base.MIB > args.max_ram_mib:
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


def check_worker(payload, descriptor, case_id, method):
    if payload.get("status") != "success":
        return payload
    correct = (payload.get("case_id") == case_id and payload.get("method") == method
               and payload.get("source_sha256") == descriptor["source_sha256"]
               and payload.get("input_sha256") == descriptor["input_sha256"]
               and payload.get("output_sha256") == descriptor["output_sha256"][method]
               and payload.get("output_fingerprints_verified") is True
               and payload.get("audit") == expected_audit(method, descriptor["shape"])
               and payload.get("n_outputs") == len(FIELDS) and payload.get("preloaded_libraries") == LIBRARIES)
    for key in ("elapsed_s", "ram_before_mib", "peak_ram_mib"):
        value = payload.get(key)
        correct = correct and isinstance(value, (int, float)) and np.isfinite(value) and value > 0
    if not correct or payload.get("peak_ram_mib", 0) < payload.get("ram_before_mib", 0):
        payload.update(status="invalid_worker", reason="Worker differs from verified source/input/output/loop workload")
    return payload


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
    if not args.validate_only and any(os.environ.get(key) != "1" for key in base.THREAD_VARIABLES):
        raise ValueError("Set all documented numerical-library thread limits to 1 before timing")
    toolkit = base.load_toolkit()
    path = (args.radar_path or Path(os.environ.get("REAL_RADAR_PATH", str(base.radar.DEFAULT_RADAR)))).expanduser().resolve()
    patterns, ranges, az_res, metadata = base.load_verified_inputs(path)
    run_id, directory = create_run_directory("2d_radar_range_loops", args.run_id, args.output_root)
    scratch_root = args.scratch_root
    if scratch_root is None:
        preferred = Path(tempfile.gettempdir()) / "opencode"
        scratch_root = preferred if preferred.is_dir() else Path(tempfile.gettempdir())
    rows, cases, descriptors, measurements = [], {}, {}, []
    with tempfile.TemporaryDirectory(prefix="radar-range-loops-", dir=scratch_root) as scratch:
        scratch = Path(scratch)
        paths = {}
        for pattern, data in patterns.items():
            paths[pattern] = scratch / f"{pattern}.npy"
            np.save(paths[pattern], data, allow_pickle=False)
        for width in base.WIDTHS:
            oracle = base.radar.SpatialPolarGaussian(SHAPE, width, az_res)
            weights = base.independent_weights(SHAPE, width, az_res)
            for axis, actual, reference in zip(("azimuth_weights", "range_weights"), weights,
                                                (oracle.az_weights, oracle.range_weights)):
                rows.append(base.compare_arrays(run_id, f"natural_W{width:g}", "inverse_fft_weight_construction",
                                                 "operator", axis, actual, reference))
            widths = base.radar.azimuth_widths(SHAPE[1], width, az_res)
            for pattern, data in patterns.items():
                case_id = f"{pattern}_W{width:g}"
                checked, descriptor = validate_case(run_id, case_id, data, ranges, width, az_res, toolkit, oracle)
                rows.extend(checked)
                descriptors[case_id] = descriptor
                cases[case_id] = {key: value[0] for key, value in metadata.items()} | descriptor | dict(
                    case_pattern=pattern, methods=list(METHODS), operation=OPERATION,
                    radar_file_sha256=metadata["input_sha256"][0], reference_method=base.REFERENCE,
                    numerical_tolerances=base.TOLERANCES, denominator_floor=base.FLOOR,
                    denominator_rule="finite raw d strictly > 1e-3; normalization AFTER both axes",
                    full_filter_calls=3, filter_preparations=1, required_outputs=list(FIELDS),
                    completion=False, restoration=False, population_variance_ddof=0,
                    boundaries="periodic azimuth; half-sample reflective range",
                    operator_order="azimuth first with range-dependent kernels, range second",
                    width_convention="unchanged legacy W/((j+1)*az_res_radians), NOT physical gate ranges",
                    range_dependent_azimuth_kernel=True, azimuth_width_first=float(widths[0]),
                    azimuth_width_last=float(widths[-1]), azimuth_widths_sha256=base.array_sha256(widths),
                    azimuth_weights_sha256=base.array_sha256(weights[0]), range_weights_sha256=base.array_sha256(weights[1]),
                    direct_weight_lengths=[weights[0].shape[0], weights[1].size],
                    dct_implementation="unchanged prepared DCT/RFFT from benchmark_2d_radar_statistics.py; NOT released statistics API timing",
                    numpy_implementation="range-gate loop: np.pad wrap + np.convolve valid with gate-specific weights; row loop: np.pad symmetric + np.convolve valid",
                    scipy_implementation="range-gate loop: scipy.ndimage.convolve1d wrap with gate-specific weights; compiled axis-1 convolve1d reflect",
                    direct_setup="identical analytic inverse-FFT weight construction for NumPy/SciPy once INSIDE timer; no truncation; cosine-sum oracle untimed",
                    audits={method: expected_audit(method, SHAPE) for method in METHODS},
                    comparison_scope="prepared spectral calculation versus these range-adaptive loop implementations; not all NumPy/SciPy algorithms",
                    repetitions=args.repeats, warmups=0, worker_mode="fresh sequential interpreters, rotated method order, identical imports",
                    timing_includes="masking, zero replacement, squared input, preparation, padding, three complete filters, normalization, moments, coverage/count, allocation",
                    timing_excludes="input I/O/hash, imports/release checks, independent verification, output fingerprints, plotting, CSV",
                    requested_library_threads=1, fft_workers=1, rss_sample_interval_s=base.RSS_INTERVAL,
                    memory="total process RSS: 1-ms in-call samples plus newly exceeded lifetime high-water mark, not allocation-only RAM",
                    physical_cpu_cores=psutil.cpu_count(logical=False), machine_total_ram_mib=psutil.virtual_memory().total / base.MIB,
                    worker_timeout_s=args.timeout_s, worker_rss_limit_mib=args.max_ram_mib,
                    timing_status="not run: validation pending")
                print(f"Validated {case_id}: {'pass' if descriptor['verified'] else 'FAIL'}", flush=True)
            del oracle
        write_csv(directory / "comparison.csv", COMPARISON_FIELDS, rows)
        failed = any(row["status"] == "fail" for row in rows)
        manifest = scratch / "verified.json"
        manifest.write_text(json.dumps(descriptors, sort_keys=True))
        if not failed and not args.validate_only:
            for index, (case_id, descriptor) in enumerate(descriptors.items()):
                for repetition in range(1, args.repeats + 1):
                    offset = (index + repetition - 1) % len(METHODS)
                    for method in METHODS[offset:] + METHODS[:offset]:
                        payload = check_worker(measure_worker(args, case_id, method, paths[cases[case_id]["case_pattern"]], manifest),
                                               descriptor, case_id, method)
                        cases[case_id][f"worker_{method}_{repetition}"] = payload
                        measurements.append(dict(run_id=run_id, case_id=case_id, method=method, operation=OPERATION,
                                                 repetition=repetition, **{key: payload.get(key) for key in
                                                                           ("status", "elapsed_s", "ram_before_mib", "peak_ram_mib", "reason")}))
                        write_csv(directory / "benchmark_runs.csv", BENCHMARK_FIELDS, measurements)
                        write_csv(directory / "benchmark_summary.csv", SUMMARY_FIELDS, base.summarize(measurements))
                        print(f"{case_id} {method} repetition {repetition}: {payload['status']} ({payload.get('elapsed_s')} s)", flush=True)
                        if payload["status"] != "success":
                            failed = True
                            break
                    if failed:
                        break
                if failed:
                    break
        for case_id, case in cases.items():
            count = sum(row["case_id"] == case_id and row["status"] == "success" for row in measurements)
            case["timing_status"] = ("measured and verified" if count == args.repeats * len(METHODS)
                                     else "not run: --validate-only" if args.validate_only and not failed
                                     else "incomplete or not run: verification/resource failure")
        write_settings(directory / "settings.csv", run_id, cases)
    print(f"{directory}: {'FAIL' if failed else 'pass'}; {len(rows)} comparisons; {len(measurements)} worker measurements", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
