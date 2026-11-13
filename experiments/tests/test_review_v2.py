"""The new release is tested in child interpreters, not mixed with v1 tests."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "experiments/scripts"
RESULTS = ROOT / "experiments/results/round1/v2.0.0"
SOURCE = os.environ.get("DCT_TOOLKIT_V2_SOURCE")


def run_code(code):
    if not SOURCE:
        pytest.skip("set DCT_TOOLKIT_V2_SOURCE to the exact release")
    prefix = f"import sys; sys.path.insert(0, {str(SCRIPTS)!r});\n"
    result = subprocess.run([sys.executable, "-B", "-c", prefix + code],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_frozen_source_checks_and_version_separation():
    run_code("""
import run_review_v2 as runner
import _review_common as old
import _review_v2_common as new
for name, digest in runner.FROZEN_HASHES.items():
    assert new.sha256(runner.SCRIPTS / name) == digest
assert old.TOOLKIT_COMMIT == '9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c'
assert new.load_toolkit().__version__ == '2.0.0'
assert len(new.RELEASE_HASHES) == 10 and len(old._RELEASE_HASHES) == 9
assert new.TOOLKIT_COMMIT == '7ff29e998dd1102d3fef2a84050402e86dc7ef67'
""")


@pytest.mark.parametrize("kind", ["boxcar1d", "boundary", "radar", "wind", "visual"])
def test_explicit_runtime_bindings(kind):
    run_code(f"""
import importlib
import run_review_v2 as runner
import _review_v2_common as release
driver = runner.configure({kind!r})
for name in runner.MODULES[{kind!r}]:
    module = importlib.import_module(name)
    if hasattr(module, 'load_toolkit'):
        assert module.load_toolkit is release.load_toolkit
assert runner.source_hashes({kind!r})['run_review_v2.py'] == release.sha256(runner.__file__)
""")


def test_counterfeit_source_is_rejected(tmp_path):
    (tmp_path / "dct_toolkit").mkdir()
    (tmp_path / "dct_toolkit/__init__.py").write_text('__version__="2.0.0"\n')
    run_code(f"""
import os
import _review_v2_common as release
os.environ['DCT_TOOLKIT_V2_SOURCE'] = {str(tmp_path)!r}
try:
    release.load_toolkit()
except RuntimeError as error:
    assert 'source files' in str(error)
else:
    raise AssertionError('Counterfeit release accepted')
""")


def test_full_default_and_explicit_analytical_compatibility():
    run_code("""
import numpy as np
import _review_v2_common as release
toolkit = release.load_toolkit()
from dct_toolkit._gaussian import _sampled_gaussian_transfer
for n in (1, 2, 7, 25):
    for width in (.3, 1.9, 5., 80.):
        sigma = width / np.sqrt(12)
        offsets = np.arange(-int(np.ceil(12*sigma)), int(np.ceil(12*sigma))+1)
        weights = np.exp(-.5*(offsets/sigma)**2)
        weights /= weights.sum()
        for periodic in (False, True):
            omega = ((2*np.pi*np.arange(n//2+1)/n) if periodic else np.pi*np.arange(n)/n)
            expected = np.cos(omega[:, None]*offsets) @ weights
            np.testing.assert_allclose(_sampled_gaussian_transfer(n, width, periodic), expected,
                                       atol=3e-14, rtol=3e-13)
        old = np.exp(-.5*(np.pi*np.arange(n)/n*sigma)**2)
        np.testing.assert_allclose(toolkit.get_dct_transfer_function(n, 'gaussian_spectral', width),
                                   old, atol=2e-15, rtol=2e-14)
""")


def test_unchanged_four_sigma_loop_and_spectral_filters():
    run_code("""
import numpy as np
import run_review_v2 as runner
driver = runner.configure('radar')
assert driver.gaussian.TRUNCATE == 4.0
assert driver.METHODS == ('discrete_spectral', 'explicit_spatial_double_loop', 'discrete_spatial')
for shape in ((1, 7), (7, 1), (9, 11)):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    data.flat[::4] = np.nan
    reference = driver.gaussian.centered_reference(data, 1.5, 10.)
    for method in driver.METHODS:
        audit = {}
        result = driver.calculate(method, data, 1.5, 10., None, audit)
        assert audit == driver.expected_audit(method, shape)
        for key in driver.FIELDS:
            row = driver.compare('test', 'small', method, key, 'all',
                                 result[key], reference[key], driver.gaussian.REFERENCE)
            assert row['status'] in ('pass', 'no_comparable'), row
""")


def test_wind_shared_and_released_api_are_still_matched():
    run_code("""
import run_review_v2 as runner
driver = runner.configure('wind')
case = driver.CASES['plane_control']
observed, _, _ = driver.generate_input(case)
reference = driver.calculate('numpy_windows', observed, case.width, None)
for method in (driver.SPECTRAL, driver.API_VALIDATION):
    backend = driver.prepare_backend(method)
    result = (driver.calculate_toolkit_api(observed, case.width, backend)
              if method == driver.API_VALIDATION else driver.calculate(method, observed, case.width, backend))
    rows = driver.compare_outputs('test', 'plane_control', method, result, reference, observed)
    assert all(row['status'] in ('pass', 'no_comparable') for row in rows)
""")


def test_v2_results_have_separate_release_and_worker_provenance():
    directories = list(RESULTS.glob("*/release-v2-20261111"))
    if len(directories) < 4:
        pytest.skip("complete the four selected release campaigns first")
    for directory in directories:
        with (directory / "settings.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for case in {row["case_id"] for row in rows}:
            settings = {row["setting"]: row["value"] for row in rows if row["case_id"] == case}
            assert settings["dct_toolkit_version"] == "2.0.0"
            assert settings["dct_toolkit_commit"] == "7ff29e998dd1102d3fef2a84050402e86dc7ef67"
            assert len(json.loads(settings["toolkit_source_sha256"])) == 10
            assert "run_review_v2.py" in json.loads(settings["review_sources_sha256"])
            for name, digest in json.loads(settings["review_sources_sha256"]).items():
                assert hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() == digest
            for name, value in settings.items():
                if name.startswith("worker_") and name.rsplit("_", 1)[-1].isdigit():
                    worker = json.loads(value)
                    assert worker["status"] == "success"
                    assert worker["dct_toolkit_version"] == "2.0.0"
                    assert worker["v2_review_sources_sha256"] == json.loads(settings["review_sources_sha256"])
        comparison = directory / "comparison.csv"
        with comparison.open(newline="") as stream:
            assert all(row["status"] in ("pass", "no_comparable") for row in csv.DictReader(stream))


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("experiment,old_run,filenames", [
    ("1d_boxcar", "lead-review-1d-20261027", ["comparison.csv", "coverage.csv"]),
    ("1d_boundary_comparison", "boundary-truncated-20261030",
     ["comparison.csv", "truth_errors.csv", "profiles.csv", "interior_agreement.csv"]),
    ("3d_wind_boxcar", "lead-review-3d-matched-20261027", ["comparison.csv", "application_counts.csv"]),
    ("2d_radar_gaussian_double_loop", "radar-gaussian-double-loop-20261111",
     ["comparison.csv", "spectral_loop_comparison.csv", "agreement_summary.csv",
      "variance_diagnostics.csv", "kernel_checks.csv"]),
])
def test_selected_numerical_evidence_is_unchanged(experiment, old_run, filenames):
    for filename in filenames:
        old = read_rows(RESULTS.parent / experiment / old_run / filename)
        new = read_rows(RESULTS / experiment / "release-v2-20261111" / filename)
        clean = lambda rows: [{k: v for k, v in row.items() if k != "run_id"} for row in rows]
        assert clean(old) == clean(new)


@pytest.mark.parametrize("experiment,repeats,count", [
    ("3d_wind_boxcar", 3, 120), ("2d_radar_gaussian_double_loop", 5, 30),
])
def test_all_new_benchmark_aggregations(experiment, repeats, count):
    directory = RESULTS / experiment / "release-v2-20261111"
    runs = read_rows(directory / "benchmark_runs.csv")
    assert len(runs) == count and all(r["status"] == "success" for r in runs)
    for summary in read_rows(directory / "benchmark_summary.csv"):
        selected = [r for r in runs if (r["case_id"], r["method"]) ==
                    (summary["case_id"], summary["method"])]
        assert len(selected) == repeats and int(summary["n_failure"]) == 0
        values = [float(r["elapsed_s"]) for r in selected]
        for key, quantile in (("median_time_s", .5), ("q25_time_s", .25), ("q75_time_s", .75)):
            assert float(summary[key]) == np.quantile(values, quantile)
        assert float(summary["max_peak_ram_mib"]) == max(float(r["peak_ram_mib"]) for r in selected)


def test_retained_wind_figure_arrays_match_v2_full_volume():
    run_code("""
import csv, hashlib
import run_review_v2 as runner
driver = runner.configure('wind')
path = runner.release.ROOT / 'manuscript/figures/round1/3d_wind_boxcar/figure_settings.csv'
with path.open(newline='') as stream:
    saved = {r['setting']: r['value'] for r in csv.DictReader(stream)}
case = driver.CASES['baseline20']
observed, injected, generation = driver.generate_input(case)
output = driver.calculate(driver.SPECTRAL, observed, case.width, driver.load_toolkit())
digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
assert generation['input_sha256'] == saved['input_sha256']
for component, values in zip(driver.COMPONENTS, observed):
    assert digest(values) == saved[f'observed_{component}_sha256']
    for statistic in driver.STATISTICS:
        assert digest(output[component][statistic]) == saved[f'full3d_{component}_{statistic}_sha256']
assert digest(output['w']['mean'][driver.FIGURE_SLICE_INDEX]) == saved['slice_mean_w_sha256']
assert digest(output['w']['std'][driver.FIGURE_SLICE_INDEX]) == saved['slice_std_w_sha256']
""")


def test_current_release_and_default_are_explicit_in_manuscript():
    text = (ROOT / "manuscript/main.tex").read_text()
    assert "v2.0.0" in text and r"\texttt{7ff29e9}" in text
    assert r"\texttt{gaussian\_spectral}" in text
    assert "Gaussian samples at all integer offsets, normalized by their infinite sum" in text
    assert "same four-standard-deviation cutoff in both implementations" in text
    assert "its end-to-end runtime was not measured here" in text
    assert "exploratory check retained from v1.0.0" in text
    assert "figures/round1/v2.0.0/visual_pipeline/pdf/visual_statistics_pipeline.pdf" in text


def test_current_3d_tables_match_new_raw_aggregations():
    directory = RESULTS / "3d_wind_boxcar/release-v2-20261111"
    raw = read_rows(directory / "benchmark_runs.csv")
    settings = read_rows(directory / "settings.csv")
    text = (ROOT / "manuscript/main.tex").read_text()
    for case in dict.fromkeys(row["case_id"] for row in raw):
        values = {r["setting"]: r["value"] for r in settings if r["case_id"] == case}
        grid = r"$" + r"\times".join(str(n) for n in json.loads(values["shape_zyx"])) + "$"
        window = "$(" + ",".join(str(n) for n in json.loads(values["width_zyx"])) + ")$"
        missing = f"{100*float(values['actual_missing_fraction']):.2f}"
        if case == "plane_control":
            missing = "$" + missing + r"^{\mathrm{p}}$"
        prefix = f"{grid} & {window} & {missing} & "
        for label, key, formatting in (("performance_time", "elapsed_s", ".4f"),
                                       ("performance_ram", "peak_ram_mib", ".1f")):
            numbers = []
            for method in ("dct_shared_moments", "numpy_windows", "astropy_direct", "astropy_fft"):
                selected = [float(r[key]) for r in raw if r["case_id"] == case and r["method"] == method]
                actual = np.median(selected) if key == "elapsed_s" else max(selected)
                numbers.append(format(actual, formatting))
            table = text.split(f"\\label{{tab:{label}}}")[1].split("\\end{table}")[0]
            assert prefix + " & ".join(numbers) + r" \\" in table


def test_new_figure1_assets_and_source_hashes():
    from PIL import Image
    directory = ROOT / "manuscript/figures/round1/v2.0.0/visual_pipeline"
    settings = {r["setting"]: r["value"] for r in read_rows(directory / "figure_settings.csv")}
    assert settings["dct_toolkit_version"] == "2.0.0"
    assert "full normalized sampled Gaussian" in settings["operator"]
    for kind in ("pdf", "png"):
        path = directory / kind / f"visual_statistics_pipeline.{kind}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == settings[f"{kind}_sha256"]
    with Image.open(directory / "png/visual_statistics_pipeline.png") as image:
        assert image.size == (1950, 1125)
        assert all(abs(dpi - 300) < .01 for dpi in image.info["dpi"])
