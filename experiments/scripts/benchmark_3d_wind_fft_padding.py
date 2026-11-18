#!/usr/bin/env python3
"""Compare native Astropy FFT convolution with and without the outer zero pad.

Reuse the frozen v2.0.0 wind benchmark. The five reported cases are timed with
the four unchanged non-FFT methods and both Astropy FFT settings. The symmetric
halo, crop, and shared seven-filter calculation are unchanged.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

import run_review_v2 as review


ENTRYPOINT = Path(__file__).resolve()
PADDED_FFT = "astropy_fft"
HALO_FFT = "astropy_fft_halo"
REPORTED_CASES = ("small20", "baseline20", "larger20", "narrow20", "broad20")
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def prepare_halo_fft(shape, width, convolve_fft):
    """Use Astropy's native FFT on the symmetric halo and retain the interior."""
    if len(width) != len(shape) or any(w < 1 or w % 2 != 1 for w in width):
        raise ValueError("Matched filters require one positive odd width per spatial axis")
    padding = tuple((n // 2, n // 2) for n in width)
    kernel = np.full(width, 1.0 / np.prod(width), dtype=np.float64)

    def astropy_filter(a):
        padded = np.pad(a, padding, mode="symmetric")
        # Interior centers are at least one kernel radius from the outer edge.
        # Circular and linear convolution therefore agree in the retained crop.
        options = dict(
            boundary="wrap", fill_value=0.0, nan_treatment="fill",
            normalize_kernel=False, preserve_nan=False,
        )
        options.update(fft_pad=False, psf_pad=False, crop=True, allow_huge=False)
        filtered = convolve_fft(padded, kernel, **options)
        crop = tuple(slice(n // 2, n // 2 + size) for n, size in zip(width, a.shape))
        return filtered[crop].copy()

    return astropy_filter


def source_hashes():
    return {**review.source_hashes("wind"), ENTRYPOINT.name: review.release.sha256(ENTRYPOINT)}


def write_settings(path, run_id, cases):
    hashes = source_hashes()
    common = review.release.environment_settings()
    common.update(
        script=(str(ENTRYPOINT), ""),
        script_sha256=(hashes[ENTRYPOINT.name], ""),
        version_bridge=("frozen v2.0.0 wind calculation plus a separate native Astropy FFT padding variant", ""),
        review_sources_sha256=(hashes, ""),
    )
    rows = []
    for case, values in cases.items():
        for name, value in sorted({**values, **common}.items()):
            value, unit = value if isinstance(value, tuple) and len(value) == 2 else (value, "")
            rows.append(dict(run_id=run_id, case_id=case, setting=name, value=value, unit=unit))
    review.release.write_csv(path, review.release.SETTINGS_FIELDS, rows)


def measure_worker(args, case, method):
    command = [
        sys.executable, "-B", str(ENTRYPOINT), "--worker", "--cases", case,
        "--methods", method, "--numpy-chunk-planes", str(args.numpy_chunk_planes),
        "--max-ram-mib", str(args.max_ram_mib),
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
        process, start, reason = psutil.Process(child.pid), time.monotonic(), ""
        while child.poll() is None:
            try:
                if args.max_ram_mib and process.memory_info().rss / 1024**2 > args.max_ram_mib:
                    reason = "Worker exceeded total RSS budget, including setup"
            except psutil.NoSuchProcess:
                pass
            if args.timeout_s and time.monotonic() - start > args.timeout_s:
                reason = "Worker exceeded lifetime limit, including setup"
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
    except ValueError:
        return dict(status="error", reason=f"Worker exit {child.returncode}: {stderr or stdout}")
    if payload.get("status") == "success" and (
        child.returncode or payload.get("v2_review_sources_sha256") != source_hashes()
        or payload.get("toolkit_source_sha256") != review.release.RELEASE_HASHES
        or payload.get("dct_toolkit_commit") != review.release.TOOLKIT_COMMIT
        or payload.get("dct_toolkit_version") != review.release.TOOLKIT_VERSION
    ):
        payload.update(status="error", reason="Worker exit or v2.0.0 source identity mismatch")
    if stderr.strip():
        payload["worker_stderr"] = stderr.strip()
    return payload


def configure():
    wind = review.configure("wind")
    frozen_backend, frozen_filter, frozen_settings = (
        wind.prepare_backend, wind.prepare_filter, wind.case_settings,
    )
    wind.DEFAULT_CASES = REPORTED_CASES
    wind.METHODS = (*wind.METHODS, HALO_FFT)

    def prepare_backend(method):
        return frozen_backend(PADDED_FFT if method == HALO_FFT else method)

    def prepare_filter(method, shape, width, backend, chunk_planes):
        if method == HALO_FFT:
            return prepare_halo_fft(shape, width, backend)
        return frozen_filter(method, shape, width, backend, chunk_planes)

    def case_settings(case, args):
        values = frozen_settings(case, args)
        values.update(
            astropy_fft_halo_settings="native convolve_fft; boundary=wrap; fft_pad=False; psf_pad=False; crop=True; allow_huge=False",
            astropy_fft_halo_shape_zyx=[n + 2 * (w // 2) for n, w in zip(case.shape, case.width)],
            astropy_fft_halo_boundary="same NumPy symmetric halo and original-grid crop; wrap is outside the retained windows",
            astropy_fft_halo_reuse="same dense spatial kernel reused; Astropy still transforms the kernel on each call",
        )
        return values

    wind.prepare_backend = prepare_backend
    wind.prepare_filter = prepare_filter
    wind.case_settings = case_settings
    wind.measure_worker = measure_worker
    wind.write_settings = write_settings
    return wind


def main():
    if "--figure-only" in sys.argv:
        raise ValueError("This timing comparison does not generate figures")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise ValueError("Set the five documented numerical-library thread limits to 1")
    wind = configure()
    if "--worker" not in sys.argv:
        return wind.main()
    output = io.StringIO()
    with redirect_stdout(output):
        status = wind.worker(wind.parse_args())
    payload = json.loads(output.getvalue())
    payload.update(
        v2_review_sources_sha256=source_hashes(),
        toolkit_source_sha256=review.release.RELEASE_HASHES,
        dct_toolkit_commit=review.release.TOOLKIT_COMMIT,
        dct_toolkit_version=review.release.TOOLKIT_VERSION,
    )
    print(json.dumps(payload), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
