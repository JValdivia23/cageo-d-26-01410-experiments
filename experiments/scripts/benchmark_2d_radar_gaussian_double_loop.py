#!/usr/bin/env python3
"""Match polar Gaussian spectral filtering to explicit spatial double loops.

Both axis stages visit every (azimuth, range) cell in nested Python loops and
evaluate a 1D Gaussian neighbourhood with numpy.dot. No convolution routine or
transform evaluates this spatial filter. The azimuth-first/range-second operator,
sampled kernels and missing-data rules are identical to the frozen Gaussian
experiment. SciPy's composed 1D convolutions remain an additional comparison.

Only the original 720x1180 sweep at W=5,15 is used. Independent centered-residual
checks precede five fresh sequential one-thread workers per method/width.
Preparation and three complete filters are included. Earlier evidence is never
overwritten; Figure 3's spectral arrays must match their previous saved hashes.
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

import benchmark_2d_radar_discrete_gaussian as gaussian
from _review_common import (
    BENCHMARK_FIELDS, COMPARISON_FIELDS, SUMMARY_FIELDS,
    create_run_directory, write_csv, write_settings,
)


SPECTRAL = "discrete_spectral"
LOOP = "explicit_spatial_double_loop"
SCIPY = "discrete_spatial"
METHODS = (SPECTRAL, LOOP, SCIPY)
EXPERIMENT = "2d_radar_gaussian_double_loop"
OPERATION = "polar_gaussian_statistics_three_filters_explicit_cell_loops"
SHAPE, WIDTHS, FIELDS, FLOOR = gaussian.SHAPE, gaussian.WIDTHS, gaussian.FIELDS, gaussian.FLOOR
LIBRARIES = gaussian.LIBRARIES
PREVIOUS_RUN = Path(__file__).resolve().parents[1] / "results/round1/2d_radar_discrete_gaussian/radar-discrete-gaussian-20261110"
PREVIOUS_SETTINGS_SHA256 = "d6bde42ce270e1bda6c381fcb0c07579950334f81b2ddd4bd3679375d45b0afc"
TIMING_FIELDS = ["run_id", "width", *METHODS, "double_loop_over_spectral"]


def source_hashes():
    return dict(double_loop_driver=gaussian.base.radar.sha256_file(Path(__file__)), **gaussian.source_hashes())


def expected_audit(method, shape):
    if method != LOOP:
        return gaussian.expected_audit(method, shape)
    return dict(filter_preparations=1, full_filter_calls=3,
                azimuth_kernels=shape[1], range_kernels=1,
                azimuth_padding_calls=3 * shape[1], range_padding_calls=3,
                azimuth_cell_updates=3 * shape[0] * shape[1],
                range_cell_updates=3 * shape[0] * shape[1])


def prepare_filter(method, shape, width, az_res, toolkit, audit):
    if method in (SPECTRAL, SCIPY):
        return gaussian.prepare_filter(method, shape, width, az_res, toolkit, audit)
    if method != LOOP:
        raise ValueError("Unknown double-loop comparison method")
    _, angular, radial = gaussian.widths_and_kernels(shape, width, az_res)
    lengths = [kernel.size for kernel in angular]
    radii = [length // 2 for length in lengths]
    range_radius = radial.size // 2
    audit.update(filter_preparations=1, azimuth_kernels=shape[1], range_kernels=1)
    n_az, n_range = shape

    def linear(values):
        if values.shape != tuple(shape):
            raise ValueError("Input shape differs from prepared double-loop geometry")
        audit["full_filter_calls"] = audit.get("full_filter_calls", 0) + 1
        # Keep each azimuth neighbourhood contiguous without constructing a
        # fixed 2D kernel or a dense matrix of all boundary-donor indices.
        padded_columns = [np.pad(values[:, gate], (radius, radius), mode="wrap")
                          for gate, radius in enumerate(radii)]
        audit["azimuth_padding_calls"] = audit.get("azimuth_padding_calls", 0) + n_range
        angular_result = np.empty(shape, dtype=np.float64)
        angular_updates = 0
        for azimuth in range(n_az):
            for gate in range(n_range):
                # Gaussian weights are symmetric; the ascending centred
                # window is equivalent to summing x[azimuth-offset, gate].
                angular_result[azimuth, gate] = np.dot(
                    angular[gate], padded_columns[gate][azimuth:azimuth + lengths[gate]])
                angular_updates += 1
        audit["azimuth_cell_updates"] = audit.get("azimuth_cell_updates", 0) + angular_updates
        del padded_columns
        # NumPy symmetric padding repeats the endpoint, matching half-sample
        # reflection (SciPy mode='reflect'), including support wider than N.
        padded_range = np.pad(angular_result, ((0, 0), (range_radius, range_radius)), mode="symmetric")
        audit["range_padding_calls"] = audit.get("range_padding_calls", 0) + 1
        result = np.empty(shape, dtype=np.float64)
        range_updates = 0
        for azimuth in range(n_az):
            for gate in range(n_range):
                result[azimuth, gate] = np.dot(radial, padded_range[azimuth, gate:gate + radial.size])
                range_updates += 1
        audit["range_cell_updates"] = audit.get("range_cell_updates", 0) + range_updates
        return result

    return linear


def calculate(method, data, width, az_res, toolkit, audit=None, phases=None):
    audit = audit if audit is not None else {}
    valid = np.isfinite(data)
    zeroed = np.where(valid, data, 0.)
    start = time.perf_counter()
    linear = prepare_filter(method, data.shape, width, az_res, toolkit, audit)
    if phases is not None:
        phases["preparation_s"] = time.perf_counter() - start
    denominator = linear(valid.astype(np.float64))
    numerator = linear(zeroed)
    second = linear(zeroed**2)
    return gaussian.base.moment_fields(denominator, numerator, second, width, az_res)


def compare(run_id, case, method, key, region, actual, reference, reference_name):
    # Retain the existing centered-reference near-zero square-root rule. Raw
    # variance always keeps its original tolerance; above-threshold std also
    # retains the original std tolerance. Label the actual reference below.
    row = gaussian.compare(run_id, case, method, key, region, actual, reference)
    row["reference_method"] = reference_name
    row["notes"] = row["notes"].replace("against centered residuals", "against the stated reference")
    return row


def validate_case(run_id, case, data, ranges, width, az_res, toolkit):
    reference = gaussian.centered_reference(data, width, az_res)
    regions = dict(all=np.ones(data.shape, bool), centre=np.broadcast_to(ranges <= 10000, data.shape))
    outputs, hashes, rows, diagnostics = {}, {}, [], []
    for method in METHODS:
        audit = {}
        fields = calculate(method, data, width, az_res, toolkit, audit)
        if audit != expected_audit(method, data.shape):
            raise ValueError("Preparation/filter/cell-loop audit differs from the matched workload")
        outputs[method], hashes[method] = fields, gaussian.base.fingerprints(fields)
        for region, selection in regions.items():
            diagnostics.append(gaussian.variance_diagnostics(
                run_id, case, method, region, fields, reference, selection))
            for key in FIELDS:
                rows.append(compare(run_id, case, method, key, region,
                                    fields[key][selection], reference[key][selection], gaussian.REFERENCE))
    matched, compact = [], []
    for region, selection in regions.items():
        selected = []
        for key in FIELDS:
            row = compare(run_id, case, SPECTRAL, key, region, outputs[SPECTRAL][key][selection],
                          outputs[LOOP][key][selection], LOOP)
            matched.append(row)
            selected.append(row)
        diag = next(r for r in diagnostics if r["method"] == SPECTRAL and r["region"] == region)
        compact.append(dict(run_id=run_id, case_id=case, width=width, region=region,
                            n_considered=int(selection.sum()), n_defined=diag["n_defined"],
                            max_abs_mean_difference=next(r["max_abs_difference"] for r in selected if r["statistic"] == "mean"),
                            max_abs_variance_difference=next(r["max_abs_difference"] for r in selected if r["statistic"] == "raw_variance"),
                            min_raw_variance=diag["min_raw_variance"],
                            n_negative_raw_variance=diag["n_negative_raw_variance"],
                            n_negative_beyond_reference_tolerance=diag["n_negative_beyond_reference_tolerance"]))
    verified = (all(r["status"] in ("pass", "no_comparable") for r in (*rows, *matched))
                and all(r["n_negative_beyond_reference_tolerance"] == 0 for r in diagnostics))
    descriptor = dict(verified=verified, shape=list(data.shape), width=width, az_res_deg=az_res,
                      source_sha256=source_hashes(), input_sha256=gaussian.base.array_sha256(data),
                      output_sha256=hashes)
    return rows, matched, diagnostics, descriptor, compact


def previous_spectral_hashes():
    path = PREVIOUS_RUN / "settings.csv"
    if gaussian.base.radar.sha256_file(path) != PREVIOUS_SETTINGS_SHA256:
        raise ValueError("Previous Gaussian evidence identity changed")
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["case_id"]: json.loads(row["value"])[SPECTRAL]
                for row in csv.DictReader(stream) if row["setting"] == "output_sha256"}


def worker(args):
    payload = dict(status="error", reason="worker did not finish")
    stop, sampler = threading.Event(), None
    try:
        if any(os.environ.get(key) != "1" for key in gaussian.base.THREAD_VARIABLES):
            raise ValueError("Set all numerical-library thread limits to 1")
        toolkit = gaussian.preload()
        descriptor = json.loads(args.manifest.read_text())[args.case]
        data = np.load(args.input, allow_pickle=False)
        if (not descriptor["verified"] or descriptor["source_sha256"] != source_hashes()
                or data.dtype != np.float64 or data.shape != SHAPE or list(data.shape) != descriptor["shape"]
                or gaussian.base.array_sha256(data) != descriptor["input_sha256"]):
            raise ValueError("Worker source/input is not the verified radar case")
        gc.collect()
        process = psutil.Process()
        peak, ready = [process.memory_info().rss], threading.Event()

        def sample():
            peak[0] = max(peak[0], process.memory_info().rss)
            ready.set()
            while not stop.wait(gaussian.base.RSS_INTERVAL):
                peak[0] = max(peak[0], process.memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        ready.wait()
        before = process.memory_info().rss
        peak[0] = max(peak[0], before)
        high_before = gaussian.base.lifetime_peak_bytes()
        audit, phases = {}, {}
        start = time.perf_counter()
        output = calculate(args.method, data, descriptor["width"], descriptor["az_res_deg"], toolkit, audit, phases)
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        high_after = gaussian.base.lifetime_peak_bytes()
        stop.set()
        sampler.join()
        if high_after > high_before:
            peak[0] = max(peak[0], high_after)
        hashes = gaussian.base.fingerprints(output)
        if audit != expected_audit(args.method, data.shape) or hashes != descriptor["output_sha256"][args.method]:
            raise ValueError("Worker output hashes or cell/filter counts differ from verified calculation")
        payload.update(status="success", elapsed_s=elapsed, preparation_s=phases["preparation_s"], reason="",
                       ram_before_mib=before / gaussian.base.MIB, peak_ram_mib=peak[0] / gaussian.base.MIB,
                       case_id=args.case, method=args.method, worker_pid=os.getpid(),
                       source_sha256=descriptor["source_sha256"], input_sha256=descriptor["input_sha256"],
                       output_sha256=hashes, output_fingerprints_verified=True, audit=audit,
                       n_outputs=len(output), preloaded_libraries=LIBRARIES,
                       rss_sample_interval_s=gaussian.base.RSS_INTERVAL,
                       precall_lifetime_peak_mib=high_before / gaussian.base.MIB,
                       final_lifetime_peak_mib=high_after / gaussian.base.MIB)
        if peak[0] / gaussian.base.MIB > args.max_ram_mib:
            payload.update(status="resource_limit", reason="Calculation exceeded worker RSS budget")
    except Exception as error:
        payload.update(status="error", reason=f"{type(error).__name__}: {error}")
    finally:
        stop.set()
        if sampler is not None:
            sampler.join()
    return payload


def measure_worker(args, case, method, input_path, manifest):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--case", case,
               "--method", method, "--input", str(input_path), "--manifest", str(manifest),
               "--max-ram-mib", str(args.max_ram_mib)]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process, start, reason = psutil.Process(child.pid), time.monotonic(), ""
    while child.poll() is None:
        try:
            if process.memory_info().rss / gaussian.base.MIB > args.max_ram_mib:
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
    for key in ("elapsed_s", "preparation_s", "ram_before_mib", "peak_ram_mib"):
        value = payload.get(key)
        valid = valid and isinstance(value, (int, float)) and np.isfinite(value) and value > 0
    valid = (valid and payload.get("elapsed_s", 0) >= payload.get("preparation_s", 0)
             and payload.get("peak_ram_mib", 0) >= payload.get("ram_before_mib", 0))
    if not valid:
        payload.update(status="invalid_worker", reason="Worker differs from verified source/input/output/cell-loop workload")
    return payload


def summarize(rows):
    result = gaussian.base.summarize(rows)
    for row in result:
        row["operation"] = OPERATION
    return result


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
    if not args.validate_only and any(os.environ.get(key) != "1" for key in gaussian.base.THREAD_VARIABLES):
        raise ValueError("Set all documented numerical-library thread limits to 1 before timing")
    toolkit = gaussian.preload()
    previous = previous_spectral_hashes()
    path = Path(args.radar_path or os.environ.get("REAL_RADAR_PATH", str(gaussian.base.radar.DEFAULT_RADAR))).expanduser().resolve()
    patterns, ranges, az_res, metadata = gaussian.base.load_verified_inputs(path)
    data = patterns["natural"]
    del patterns
    run_id, directory = create_run_directory(EXPERIMENT, args.run_id, args.output_root)
    scratch_root = args.scratch_root or Path(tempfile.gettempdir()) / "opencode"
    rows, matched, diagnostics, kernels, compact, measured, phases = [], [], [], [], [], [], []
    cases, descriptors = {}, {}
    with tempfile.TemporaryDirectory(prefix="radar-gaussian-double-loop-", dir=scratch_root) as scratch:
        scratch = Path(scratch)
        input_path = scratch / "natural.npy"
        np.save(input_path, data, allow_pickle=False)
        for width in WIDTHS:
            case = f"natural_W{width:g}"
            checks = gaussian.kernel_checks(run_id, case, SHAPE, width, az_res)
            checked, comparisons, diag, descriptor, summary = validate_case(run_id, case, data, ranges, width, az_res, toolkit)
            if descriptor["output_sha256"][SPECTRAL] != previous[case]:
                raise ValueError("Spectral outputs differ from the previous Gaussian calculation and Figure 3")
            descriptor["verified"] = descriptor["verified"] and all(row["status"] == "pass" for row in checks)
            rows.extend(checked)
            matched.extend(comparisons)
            diagnostics.extend(diag)
            kernels.extend(checks)
            compact.extend(summary)
            descriptors[case] = descriptor
            cases[case] = {key: value[0] for key, value in metadata.items()} | descriptor | dict(
                methods=list(METHODS), operation=OPERATION,
                radar_file_sha256=gaussian.base.radar.DOCUMENTED_SHA256,
                reference_method=gaussian.REFERENCE, primary_pair=[SPECTRAL, LOOP],
                numerical_tolerances=gaussian.base.TOLERANCES,
                near_zero_std_rule="reference variance <= variance_atol: sqrt(variance-error bound); raw variance tolerance unchanged",
                gaussian_sigma="converted_grid_width/sqrt(12)", kernel_radius="ceil(4*sigma)",
                kernel_discretization="normalized integer Gaussian samples, radius ceil(4*sigma)",
                boundaries="periodic azimuth; half-sample reflective range",
                operator_order="azimuth first (donor-gate-dependent widths), range second; normalize only after both",
                width_convention="legacy W/((j+1)*azimuth_spacing_radians), not fixed physical distance",
                denominator_floor=FLOOR, denominator_rule="finite raw d > 0.001, strict",
                completion=False, restoration=False, filter_preparations=1, full_filter_calls=3,
                explicit_loop_implementation="nested Python azimuth/range cell loops for each axis stage; numpy.dot per 1D neighbourhood; np.pad wrap then symmetric; no convolve/FFT/DCT",
                scipy_implementation="gate-dependent convolve1d wrap followed by compiled range-axis convolve1d reflect; composed implementation, not a library polar API",
                loop_audit="actual cell-update increments inside both nested loops; 3*N_az*N_range per stage",
                audits={method: expected_audit(method, SHAPE) for method in METHODS},
                comparison_scope="spectral versus this explicit Python spatial implementation; not a universal spatial-convolution ranking",
                previous_gaussian_run=PREVIOUS_RUN.name, previous_settings_sha256=PREVIOUS_SETTINGS_SHA256,
                spectral_matches_previous_output_hashes=True,
                repetitions=args.repeats, warmups=0, requested_library_threads=1, fft_workers=1,
                worker_mode="fresh sequential interpreters, rotated method order, identical preloaded libraries",
                preloaded_libraries=LIBRARIES, rss_sample_interval_s=gaussian.base.RSS_INTERVAL,
                timing_includes="masking, zero replacement, squares, kernel/response preparation, boundary padding, three complete filters, normalized moments, coverage/count, allocation",
                timing_excludes="I/O, imports, source/input checks, references, diagnostics, output hashes, CSV, plotting",
                preparation_time="subset of total: construct kernels/responses once; per-filter padding remains in total",
                memory="total process RSS: 1-ms in-call samples plus newly exceeded lifetime peak; not allocation-only RAM",
                worker_timeout_s=args.timeout_s, worker_rss_limit_mib=args.max_ram_mib,
                timing_status="not run: validation pending")
            print(f"Validated {case}: {'pass' if descriptor['verified'] else 'FAIL'}", flush=True)
        for name, fields, values in (
            ("comparison.csv", COMPARISON_FIELDS, rows), ("spectral_loop_comparison.csv", COMPARISON_FIELDS, matched),
            ("variance_diagnostics.csv", gaussian.DIAGNOSTIC_FIELDS, diagnostics),
            ("kernel_checks.csv", gaussian.KERNEL_FIELDS, kernels),
            ("agreement_summary.csv", gaussian.AGREEMENT_FIELDS, compact),
        ):
            write_csv(directory / name, fields, values)
        failed = not all(case["verified"] for case in descriptors.values())
        manifest = scratch / "verified.json"
        manifest.write_text(json.dumps(descriptors, sort_keys=True))
        if not failed and not args.validate_only:
            for index, (case, descriptor) in enumerate(descriptors.items()):
                for repetition in range(1, args.repeats + 1):
                    offset = (index + repetition - 1) % len(METHODS)
                    for method in METHODS[offset:] + METHODS[:offset]:
                        payload = check_worker(measure_worker(args, case, method, input_path, manifest), descriptor, case, method)
                        cases[case][f"worker_{method}_{repetition}"] = payload
                        measured.append(dict(run_id=run_id, case_id=case, method=method, operation=OPERATION,
                                             repetition=repetition, **{key: payload.get(key) for key in (
                                                 "status", "elapsed_s", "ram_before_mib", "peak_ram_mib", "reason")}))
                        phases.append(dict(run_id=run_id, case_id=case, method=method, repetition=repetition,
                                           status=payload["status"], preparation_s=payload.get("preparation_s"), total_s=payload.get("elapsed_s")))
                        write_csv(directory / "benchmark_runs.csv", BENCHMARK_FIELDS, measured)
                        write_csv(directory / "benchmark_summary.csv", SUMMARY_FIELDS, summarize(measured))
                        write_csv(directory / "benchmark_phases.csv", gaussian.PHASE_FIELDS, phases)
                        print(f"{case} {method} repetition {repetition}: {payload['status']} ({payload.get('elapsed_s')} s)", flush=True)
                        if payload["status"] != "success":
                            failed = True
                            break
                    if failed:
                        break
                if failed:
                    break
        summaries, timing = summarize(measured) if measured else [], []
        for case, values in cases.items():
            complete = all(any(row["case_id"] == case and row["method"] == method
                               and row["n_success"] == args.repeats and not row["n_failure"]
                               for row in summaries) for method in METHODS)
            values["timing_status"] = ("measured and verified" if complete else "not run: --validate-only" if args.validate_only and not failed else "incomplete or not run: verification/resource failure")
            medians = {method: next((row["median_time_s"] for row in summaries
                                    if row["case_id"] == case and row["method"] == method), None) for method in METHODS}
            ratio = medians[LOOP] / medians[SPECTRAL] if complete else None
            timing.append(dict(run_id=run_id, width=values["width"], **medians, double_loop_over_spectral=ratio))
        write_csv(directory / "timing_summary.csv", TIMING_FIELDS, timing)
        write_settings(directory / "settings.csv", run_id, cases)
    print(f"{directory}: {'FAIL' if failed else 'pass'}; {len(rows)} centered comparisons; {len(matched)} spectral/loop comparisons; {len(measured)} measurements", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
