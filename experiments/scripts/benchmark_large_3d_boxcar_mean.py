#!/usr/bin/env python3
"""Quick large-array, in-place normalized-mean comparison; not full statistics.

Uses one float64 scalar field with 20% independently seeded NaNs, reflective
3-D boxcars of side 3/31, and two measured repetitions in fresh processes.
Both methods overwrite the input and the float mask with their filtered values
and divide in place. This keeps two main float arrays rather than the full
three-component statistics workload. Preparation and both filters are timed;
generation, input hashing, and verification are not. No plots or large arrays
are saved. Large outputs are checked at deterministic samples, not every cell.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
from scipy.fft import dctn, idctn
from scipy.ndimage import uniform_filter

from _review_common import (
    BENCHMARK_FIELDS, COMPARISON_FIELDS, SUMMARY_FIELDS,
    create_run_directory, load_toolkit, write_csv, write_settings,
)


GIB = 1024**3
METHODS = ("scipy_uniform_inplace", "dct_prepared_inplace")
OPERATION = "scalar_normalized_mean_two_filters_inplace_setup_included"
VALUE_SEED, MASK_SEED, SAMPLE_SEED = 9071, 9072, 9073
FLOOR, ATOL, RTOL = 1e-3, 2e-11, 2e-10


def memory_state():
    """Native macOS page counts: Rosetta psutil VM page sizes can be wrong."""
    if sys.platform != "darwin":
        swap = psutil.swap_memory()
        return dict(available_bytes=psutil.virtual_memory().available,
                    swapins=swap.sin, swapouts=swap.sout, pressure_level=0)
    output = subprocess.check_output(["vm_stat"], text=True)
    page_size = int(re.search(r"page size of (\d+) bytes", output).group(1))
    pages = {name.strip(): int(count) for name, count in
             re.findall(r"^([^:\n]+):\s+(\d+)\.?$", output, re.MULTILINE)}
    # Conservative cache estimate, not an assertion that every inactive page
    # is available without compression. Keep an additional 1-GiB reserve.
    available = pages["Pages free"] + pages.get("Pages speculative", 0)
    available += min(pages["Pages inactive"], pages["File-backed pages"])
    pressure = int(subprocess.check_output(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True))
    return dict(available_bytes=available * page_size, native_page_size=page_size,
                swapins=pages["Swapins"], swapouts=pages["Swapouts"],
                compressor_pages=pages["Pages occupied by compressor"], pressure_level=pressure)


def prepare_filter(method, shape, width, toolkit):
    if method == METHODS[0]:
        return lambda a: uniform_filter(a, size=width, output=a, mode="reflect")
    transfers = []
    for axis, length in enumerate(shape):
        broadcast = [1] * len(shape)
        broadcast[axis] = length
        transfers.append(toolkit.get_dct_transfer_function(
            length, "boxcar_discrete", float(width),
        ).reshape(broadcast))

    def filter_inplace(a):
        spectrum = dctn(a, type=2, norm="ortho", workers=1, overwrite_x=True)
        if not np.shares_memory(spectrum, a):
            raise RuntimeError("DCT unexpectedly copied the full input")
        for transfer in transfers:
            spectrum *= transfer
        result = idctn(spectrum, type=2, norm="ortho", workers=1, overwrite_x=True)
        if not np.shares_memory(result, a):
            raise RuntimeError("Inverse DCT unexpectedly copied the full input")
        return result

    return filter_inplace


def normalized_mean(data, width, method, toolkit):
    valid = np.isfinite(data)
    density = valid.astype(np.float64)
    data[~valid] = 0.0
    del valid
    linear_filter = prepare_filter(method, data.shape, width, toolkit)
    density = linear_filter(density)
    mean = linear_filter(data)
    accepted = np.isfinite(density) & (density > FLOOR)
    np.divide(mean, density, out=mean, where=accepted)
    mean[~accepted] = np.nan
    return mean, density


def generate_input(shape):
    data = np.empty(shape, dtype=np.float64)
    values, masks = np.random.default_rng(VALUE_SEED), np.random.default_rng(MASK_SEED)
    digest, missing = hashlib.sha256(), 0
    for plane in data:
        values.standard_normal(out=plane)
        invalid = masks.random(plane.shape) < 0.20
        plane[invalid] = np.nan
        missing += int(invalid.sum())
        digest.update(memoryview(plane))
    return data, digest.hexdigest(), missing


def sample_indices(shape):
    edges = np.array(np.meshgrid(*[(0, 1, n // 2, n - 2, n - 1) for n in shape],
                                indexing="ij")).reshape(3, -1)
    random = np.vstack([np.random.default_rng(SAMPLE_SEED + axis).integers(0, n, 2048)
                        for axis, n in enumerate(shape)])
    return tuple(np.concatenate([edges, random], axis=1))


def self_tests(toolkit):
    for width in (3, 31):
        data, _, _ = generate_input((12, 13, 14))
        data[0, 0, 0] = 0.0
        outputs = [normalized_mean(data.copy(), width, method, toolkit) for method in METHODS]
        for actual, reference in zip(outputs[1], outputs[0]):
            np.testing.assert_allclose(actual, reference, atol=ATOL, rtol=RTOL, equal_nan=True)
        expected = toolkit.dct_mean(data, width, kernel_type="boxcar_discrete",
                                    restore_input_nan=False, _stable_fallback=False,
                                    _denominator_floor=FLOOR)
        np.testing.assert_allclose(outputs[1][0], expected, atol=ATOL, rtol=RTOL, equal_nan=True)
        for method in METHODS:
            empty = np.full((5, 6, 7), np.nan)
            mean, density = normalized_mean(empty, width, method, toolkit)
            assert np.isnan(mean).all() and np.count_nonzero(density) == 0


def worker(args):
    toolkit = load_toolkit()  # Both backends are imported in every fresh worker.
    data, digest, missing = generate_input(tuple(args.shape))
    gc.collect()
    before_vm = memory_state()
    before_rss = psutil.Process().memory_info().rss
    before_hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    mean, density = normalized_mean(data, args.width, args.method, toolkit)
    elapsed = time.perf_counter() - start
    after_hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    after_rss = psutil.Process().memory_info().rss
    after_vm = memory_state()
    hwm_bytes = after_hwm if sys.platform == "darwin" else after_hwm * 1024
    peak = max(before_rss, after_rss, hwm_bytes if after_hwm > before_hwm else 0)
    indices = sample_indices(data.shape)
    swapping = any(after_vm[key] > before_vm[key] for key in ("swapins", "swapouts"))
    return dict(status="paging_detected" if swapping else "success", elapsed_s=elapsed,
                ram_before_mib=before_rss / 1024**2, peak_ram_mib=peak / 1024**2,
                reason="system swap activity during timing; excluded from successful summaries" if swapping else "",
                peak_source="new lifetime high-water mark" if after_hwm > before_hwm else "before/after RSS only",
                input_sha256=digest, actual_missing_fraction=missing / data.size,
                worker_pid=psutil.Process().pid, native_vm_before=before_vm, native_vm_after=after_vm,
                mean_samples=mean[indices].tolist(), density_samples=density[indices].tolist(),
                filter_calls=2, input_overwritten=True)


def measure(args, shape, method, width):
    before = memory_state()
    input_bytes = int(np.prod(shape)) * 8
    # Two float arrays, up to two bool arrays, library/scratch allowance,
    # plus 1 GiB reserved for other applications. No override is offered.
    required = 2.3 * input_bytes + 0.5 * GIB + GIB
    if before["available_bytes"] < required or before["pressure_level"] >= 4:
        return dict(status="skipped_memory", elapsed_s=None, ram_before_mib=None, peak_ram_mib=None,
                    reason=f"native conservative headroom {before['available_bytes']/GIB:.2f} GiB < required {required/GIB:.2f} GiB, or critical pressure",
                    native_vm_preflight=before)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--shape",
               *map(str, shape), "--method", method, "--width", str(width)]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Sample arrays can exceed a pipe buffer. Drain both streams while the
    # parent watches resources, rather than waiting for the child to exit.
    captured = {}

    def drain(name):
        captured[name] = getattr(child, name).read()

    readers = [threading.Thread(target=drain, args=(name,)) for name in ("stdout", "stderr")]
    for reader in readers:
        reader.start()
    process, start = psutil.Process(child.pid), time.monotonic()
    reason, next_vm_check = "", 0.0
    while child.poll() is None:
        try:
            if process.memory_info().rss > 2.7 * input_bytes + 0.5 * GIB:
                reason = "worker exceeded conservative in-place RSS budget"
        except psutil.NoSuchProcess:
            pass
        now = time.monotonic()
        if now - start > 600:
            reason = "worker exceeded 600-second lifetime limit"
        if now >= next_vm_check:
            state = memory_state()
            if state["pressure_level"] >= 4 or state["available_bytes"] < 0.5 * GIB:
                reason = "host reached critical pressure or less than 0.5 GiB conservative headroom"
            next_vm_check = now + 1
        if reason:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
            break
        time.sleep(0.05)
    child.wait()
    for reader in readers:
        reader.join()
    stdout, stderr = captured["stdout"], captured["stderr"]
    if reason or child.returncode:
        return dict(status="resource_limit" if reason else "error", elapsed_s=None,
                    ram_before_mib=None, peak_ram_mib=None, reason=reason or stderr.strip() or stdout.strip())
    payload = json.loads(stdout)
    payload["native_vm_preflight"] = before
    return payload


def comparison(run_id, case_id, repetition, statistic, actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    fa, fb = np.isfinite(a), np.isfinite(b)
    joint = fa & fb
    delta = a[joint] - b[joint]
    good = np.array_equal(fa, fb) and not np.isinf(a).any() and not np.isinf(b).any()
    good = good and np.all(np.abs(delta) <= ATOL + RTOL * np.abs(b[joint]))
    return dict(run_id=run_id, case_id=case_id, method=METHODS[1], reference_method=METHODS[0],
                component="scalar", statistic=statistic, region="sampled_edges_and_random",
                estimate_type="direct_normalized_mean" if statistic == "mean" else "raw_mask_response",
                kernel_match="exact_odd_discrete_boxcar", n_considered=a.size, n_compared=int(joint.sum()),
                n_method_undefined=int((~fa).sum()), n_reference_undefined=int((~fb).sum()),
                n_method_gap_filled=0, n_reference_gap_filled=0,
                max_abs_difference=float(np.max(np.abs(delta))) if delta.size else None,
                rmse=float(np.sqrt(np.mean(delta**2))) if delta.size else None,
                atol=ATOL, rtol=RTOL, status=("pass" if joint.any() else "no_comparable") if good else "fail",
                units="arbitrary_signal_unit" if statistic == "mean" else "1",
                notes=f"Repetition {repetition}; deterministic sampled agreement only, not all-cell validation; no fallback")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--input-gib", type=int, choices=(1, 2), nargs="+", default=[1])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shape", type=int, nargs=3, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, choices=(3, 31), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)), flush=True)
        return 0
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    toolkit = load_toolkit()
    self_tests(toolkit)
    run_id, directory = create_run_directory("3d_large_boxcar_mean", args.run_id, args.output_root)
    rows, comparisons, settings, hashes = [], [], {}, {}
    stop = False
    for size in args.input_gib:
        shape = (512, 512, 512 * size)
        for width in (3, 31):
            case = f"gib{size}_m{width}"
            settings[case] = dict(shape_zyx=list(shape), input_bytes=size * GIB, dtype="float64",
                width_zyx=[width] * 3, missing_probability=0.20, value_seed=VALUE_SEED,
                mask_seed=MASK_SEED, sample_seed=SAMPLE_SEED, kernel="boxcar_discrete",
                boundary="half-sample reflect on all axes", operation=OPERATION,
                scope="quick exploratory scalar mean only; NOT three-component mean/variance/std/support",
                denominator_floor=FLOOR, fallback=False, restore_input_nan=False,
                input_policy="overwritten in place by zero replacement, filter, normalization; same for both methods",
                filters_per_call=2, filter_setup="once inside timer; reused for mask and data",
                libraries="same verified toolkit, SciPy FFT, and SciPy ndimage imported in every worker",
                workers=1, warmups=0, repeats=args.repeats,
                timing_includes="finite mask, float mask, invalid replacement, transfer preparation, two filters, divide/output",
                timing_excludes="generation, input hash, imports, sampled verification, native memory snapshots, CSV",
                numerical_validation="full small-array check including W31 and all-missing control before workers; only fixed edge/random samples on large arrays",
                memory_preflight="native vm_stat page size; free+speculative+min(inactive,file-backed); additional 1 GiB reserve",
                memory_results="total RSS; newly exceeded ru_maxrss; not filter-only allocations",
                swap_policy="any system swapin/swapout increment during timing excludes the run from success summaries; stop campaign on resource/paging failure",
                interpretation="kernel-width scaling at fixed input size; no universal speed or large-data crossover claim")
            paired = {}
            for repetition in range(1, args.repeats + 1):
                order = METHODS if repetition % 2 else tuple(reversed(METHODS))
                for method in order:
                    payload = measure(args, shape, method, width)
                    if payload["status"] == "success":
                        previous = hashes.setdefault(size, payload["input_sha256"])
                        if previous != payload["input_sha256"]:
                            payload.update(status="error", reason="worker input hash differs across methods/windows")
                    rows.append(dict(run_id=run_id, case_id=case, method=method, operation=OPERATION,
                                     repetition=repetition, **{key: payload.get(key) for key in BENCHMARK_FIELDS[5:]}))
                    samples = {name: payload.pop(name) for name in ("mean_samples", "density_samples") if name in payload}
                    settings[case][f"worker_{method}_{repetition}"] = payload
                    if payload["status"] == "success":
                        paired[method] = samples
                    print(json.dumps(dict(case=case, method=method, repetition=repetition,
                                          status=payload["status"], elapsed_s=payload.get("elapsed_s"),
                                          peak_ram_mib=payload.get("peak_ram_mib"), reason=payload.get("reason"))), flush=True)
                    if payload["status"] != "success":
                        stop = True
                        break
                if not stop:
                    for name, statistic in (("mean_samples", "mean"), ("density_samples", "raw_mask_response")):
                        comparisons.append(comparison(run_id, case, repetition, statistic,
                                                      paired[METHODS[1]][name], paired[METHODS[0]][name]))
                    stop = any(row["status"] == "fail" for row in comparisons)
                write_csv(directory / "benchmark_runs.csv", BENCHMARK_FIELDS, rows)
                write_csv(directory / "comparison.csv", COMPARISON_FIELDS, comparisons)
                write_settings(directory / "settings.csv", run_id, settings)
                if stop:
                    break
            if stop:
                break
        if stop:
            break
    summaries = []
    for case, method in dict.fromkeys((r["case_id"], r["method"]) for r in rows):
        group = [r for r in rows if r["case_id"] == case and r["method"] == method]
        good = [r for r in group if r["status"] == "success"]
        times = [r["elapsed_s"] for r in good]
        summaries.append(dict(run_id=run_id, case_id=case, method=method, operation=OPERATION,
            n_success=len(good), n_failure=len(group)-len(good),
            median_time_s=float(np.median(times)) if times else None,
            q25_time_s=float(np.quantile(times, .25)) if times else None,
            q75_time_s=float(np.quantile(times, .75)) if times else None,
            max_peak_ram_mib=max(r["peak_ram_mib"] for r in good) if good else None))
    write_csv(directory / "benchmark_summary.csv", SUMMARY_FIELDS, summaries)
    print(f"Results: {directory}", flush=True)
    return int(stop)


if __name__ == "__main__":
    raise SystemExit(main())
