#!/usr/bin/env python3
"""Revalidate the selected review experiments with the exact v2.0.0 release.

Usage: python run_review_v2.py {boxcar1d,boundary,radar,wind,visual} [driver options]

The v1.0.0 sources remain immutable. Only release loading, provenance, output
directories, and untimed fresh-worker launching are rebound here. Numerical
functions, cases, tolerances, preparation, filter counts, and timers are reused
unchanged. The radar path deliberately selects only the four-sigma methods;
it never calls an inherited analytical-Gaussian branch with the new default.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import _review_v2_common as release


SCRIPTS = Path(__file__).resolve().parent
FROZEN_HASHES = {
    "_review_common.py": "a8520fd9ca1306ff289dca46a278f3eb047608d0fa86f2231461da400a48b17f",
    "_review_figures.py": "b93d58ee7d36c707f434c529053f254b22117122da8cb1157ca1e6d14a2167a6",
    "generate_1d_boxcar_validation.py": "bf445a42560ae32e4db2dcf108c2f661a1f5e41f3fc142af585d8f4b347621bd",
    "generate_1d_boundary_comparison.py": "1eb251b879172449b4428bb8595382bd75fb39328988149acbbb5d4dd517ede2",
    "generate_3d_wind_boxcar_validation.py": "401558f018c44a862dea7cb6a8485322b3e0b4fa951cface02efdb866c31cee1",
    "benchmark_2d_radar_gaussian_double_loop.py": "2ed6354c6e21fd91ca119d7323e808322361335b4835f2e53e32cb196b55559f",
    "benchmark_2d_radar_discrete_gaussian.py": "301fd08241b095773b30d055d007a2697636ddd4c364cb716ee8b316f2cfec21",
    "benchmark_2d_radar_statistics.py": "17c3232b4a24fa635d56f8958bdc3fa3eee82ced2bc1e06a062c60e253297f16",
    "generate_2d_radar_gaussian_validation.py": "98cc65594de08a3f79a478a459f45fb7232e7cb1bbdd75677de2364f1fe52c0d",
    "generate_visual_pipeline_figure.py": "508f6933fc9728c773c40ab93862944f502c009406faefc7d1ee664c372b6023",
}
MODULES = {
    "boxcar1d": ["generate_1d_boxcar_validation"],
    "boundary": ["generate_1d_boxcar_validation", "generate_1d_boundary_comparison"],
    "radar": ["generate_2d_radar_gaussian_validation", "benchmark_2d_radar_statistics",
              "benchmark_2d_radar_discrete_gaussian", "benchmark_2d_radar_gaussian_double_loop"],
    "wind": ["generate_3d_wind_boxcar_validation"],
    "visual": ["generate_visual_pipeline_figure"],
}


def source_hashes(kind):
    names = ["_review_common.py", "_review_figures.py", *[f"{n}.py" for n in MODULES[kind]],
             "_review_v2_common.py", "run_review_v2.py"]
    return {name: release.sha256(SCRIPTS / name) for name in names}


def configure(kind):
    """Bind the new release explicitly, without replacing the legacy module."""
    for name, digest in FROZEN_HASHES.items():
        if release.sha256(SCRIPTS / name) != digest:
            raise RuntimeError(f"Frozen review source changed: {name}")
    release.load_toolkit()
    modules = [importlib.import_module(name) for name in MODULES[kind]]
    for module in modules:
        for name in ("load_toolkit", "environment_settings", "create_run_directory"):
            if hasattr(module, name):
                setattr(module, name, getattr(release, name))

        def write_settings(path, run_id, cases, kind=kind):
            cases = {case: {**values, "review_sources_sha256": source_hashes(kind)}
                     for case, values in cases.items()}
            release.write_settings(path, run_id, cases)

        if hasattr(module, "write_settings"):
            module.write_settings = write_settings
    driver = modules[-1]
    if kind == "radar":
        driver.source_hashes = lambda: source_hashes(kind)
        driver.measure_worker = lambda args, case, method, input_path, manifest: launch_worker(
            kind, ["--worker", "--case", case, "--method", method, "--input", str(input_path),
                   "--manifest", str(manifest), "--max-ram-mib", str(args.max_ram_mib)], args)
    elif kind == "wind":
        driver.measure_worker = lambda args, case, method: launch_worker(
            kind, ["--worker", "--cases", case, "--methods", method,
                   "--numpy-chunk-planes", str(args.numpy_chunk_planes),
                   "--max-ram-mib", str(args.max_ram_mib)], args)
    return driver


def launch_worker(kind, options, args):
    """Use the same lifetime/RSS limits outside the unchanged timed calculation."""
    import psutil

    command = [sys.executable, "-B", str(Path(__file__).resolve()), kind, *options]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process, start, reason = psutil.Process(child.pid), time.monotonic(), ""
    while child.poll() is None:
        try:
            if args.max_ram_mib > 0 and process.memory_info().rss / 1024**2 > args.max_ram_mib:
                reason = "Worker exceeded total RSS budget, including setup"
        except psutil.NoSuchProcess:
            pass
        if args.timeout_s > 0 and time.monotonic() - start > args.timeout_s:
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
    except (ValueError, TypeError):
        return dict(status="error", reason=f"Worker exit {child.returncode}: {stderr or stdout}")
    if (child.returncode or payload.get("v2_review_sources_sha256") != source_hashes(kind)
            or payload.get("toolkit_source_sha256") != release.RELEASE_HASHES
            or payload.get("dct_toolkit_commit") != release.TOOLKIT_COMMIT
            or payload.get("dct_toolkit_version") != release.TOOLKIT_VERSION):
        payload.update(status="error", reason="Worker exit or v2.0.0 source identity mismatch")
    if stderr.strip():
        payload["worker_stderr"] = stderr.strip()
    return payload


def run_worker(kind, driver, options):
    if kind == "radar":
        payload = driver.worker(driver.parse_args(options))
    else:
        sys.argv = [str(Path(__file__).resolve()), *options]
        output = io.StringIO()
        with redirect_stdout(output):
            driver.worker(driver.parse_args())
        payload = json.loads(output.getvalue())
    payload.update(v2_review_sources_sha256=source_hashes(kind),
                   toolkit_source_sha256=release.RELEASE_HASHES,
                   dct_toolkit_commit=release.TOOLKIT_COMMIT,
                   dct_toolkit_version=release.TOOLKIT_VERSION)
    print(json.dumps(payload), flush=True)
    return 0 if payload["status"] == "success" else 1


def run_boundary(driver, options):
    # Reuse historical-input checks and all numerical functions, not its v1-only
    # figure/export guards. The old TOOLKIT_COMMIT remains the historical guard.
    parser = argparse.ArgumentParser(description="Refresh the approved boundary pair; no figure overwrite")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(options)
    cases = driver.prepare_comparison()
    run_id, directory = release.create_run_directory("1d_boundary_comparison", args.run_id, args.output_root)
    rows = driver.evidence_rows(cases, run_id)
    for name, fields, values in zip(
        ("comparison.csv", "truth_errors.csv", "profiles.csv", "interior_agreement.csv"),
        (driver.COMPARISON_FIELDS, driver.TRUTH_FIELDS, driver.PROFILE_FIELDS, driver.COMPARISON_FIELDS), rows,
    ):
        release.write_csv(directory / name, fields, values)
    settings = {}
    for case in cases:
        values = driver.calculation_settings(case)
        values = {k: v for k, v in values.items() if not k.startswith("toolkit_source_sha256.")}
        values.update(review_sources_sha256=source_hashes("boundary"),
                      checks="Frozen v1 input/coverage/comparisons verified; ten v2 sources checked; unchanged numerical tolerances",
                      retained_figure="Figure 2 retained; profiles revalidated without exporting")
        settings[case["case_id"]] = values
    release.write_settings(directory / "settings.csv", run_id, settings)
    driver.verify_saved(cases, driver.SELECTED_RUN)
    print(f"{directory}: pass; retained Figure 2 profiles verified")
    return 0


def run_visual(driver, options):
    import numpy as np
    from scipy.ndimage import convolve1d
    from _review_figures import DEFAULT_FIGURE_ROOT, export_figure

    parser = argparse.ArgumentParser(description="Figure 1 with the v2.0.0 full Gaussian; preserve old assets")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FIGURE_ROOT / "v2.0.0")
    args = parser.parse_args(options)
    data = driver.calculate()  # New full sampled default, not gaussian_spectral.
    sigma = driver.WIDTH / np.sqrt(12)
    offsets = np.arange(-int(np.ceil(12 * sigma)), int(np.ceil(12 * sigma)) + 1)
    weights = np.exp(-.5 * (offsets / sigma)**2)
    weights /= weights.sum()
    for key, values in (("a1", data["values"]), ("a2", data["values"]**2), ("d", data["mask"])):
        expected = convolve1d(convolve1d(values, weights, axis=0, mode="reflect"),
                              weights, axis=1, mode="reflect")
        np.testing.assert_allclose(data[key], expected, atol=2e-15, rtol=2e-14)
    toolkit = release.load_toolkit()
    observed = np.where(data["mask"], data["field"], np.nan)
    options = dict(kernel_type="gaussian", restore_input_nan=False, _stable_fallback=False,
                   _denominator_floor=driver.FLOOR)
    mean = toolkit.dct_mean(observed, driver.WIDTH, **options)
    variance = toolkit.dct_variance(observed, driver.WIDTH, **options)
    np.testing.assert_allclose(data["mean"], mean, atol=3e-14, rtol=3e-13, equal_nan=True)
    np.testing.assert_allclose(data["std"]**2, variance, atol=3e-14, rtol=3e-13, equal_nan=True)
    metadata = {k: v[0] for k, v in release.environment_settings().items()}
    metadata.update(purpose="Illustrative direct normalized statistics; no benchmark",
                    operator="v2.0.0 full normalized sampled Gaussian; width 7; reflective both axes",
                    input_rule="seed 42; unchanged 80x80 field and original validity mask; explicit zero replacement",
                    denominator="finite raw d > 0.001; no completion or restoration",
                    reference="independent 12-sigma spatial evaluation below float64 tail precision; released statistics API",
                    review_sources_sha256=source_hashes("visual"),
                    array_sha256={key: __import__("hashlib").sha256(data[key].tobytes()).hexdigest()
                                  for key in ("field", "mask", "values", "a1", "a2", "d", "mean", "std", "coverage", "transfer")},
                    checks="PASS: spatial raw filters and released direct mean/variance; unchanged layout/palette/crop")
    directory = args.output_root / "visual_pipeline"
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite figure directory: {directory}")
    fig = driver.create_figure(data)
    paths = export_figure(fig, directory, driver.STEM, metadata)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(json.dumps({key: str(path) for key, path in paths.items()}))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in MODULES:
        raise SystemExit(__doc__)
    kind, options = argv[0], argv[1:]
    driver = configure(kind)
    if "--figure-only" in options:
        raise ValueError("Legacy figure-only guards select v1.0.0; use the explicit v2 visual path instead.")
    if "--worker" in options:
        if kind not in ("radar", "wind"):
            raise ValueError("Only radar/wind have timed workers")
        return run_worker(kind, driver, options)
    if kind == "boundary":
        return run_boundary(driver, options)
    if kind == "visual":
        return run_visual(driver, options)
    if kind == "wind":
        sys.argv = [str(Path(__file__).resolve()), *options]
        return driver.main()
    return driver.main(options)


if __name__ == "__main__":
    raise SystemExit(main())
