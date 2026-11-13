"""Range-adaptive NumPy/SciPy loops: tiny controls and selected saved evidence."""

import csv
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import benchmark_2d_radar_range_loops as loops


def forbidden(*args, **kwargs):
    pytest.fail("No reference, SciPy filtering, or real-radar campaign is allowed in this control")


@pytest.fixture(scope="module")
def toolkit():
    return loops.base.load_toolkit()


@pytest.mark.parametrize("shape,width,az_res", [((7, 8), 3.7, 360/7), ((8, 9), 5., 45.),
                                               ((5, 4), 15., 72.), ((1, 6), 1.5, 10.), ((6, 1), 5., .5)])
def test_loop_impulses_signed_arrays_and_constants_match_dense_operator(toolkit, shape, width, az_res):
    values = [np.ones(shape), np.random.default_rng(6201).normal(size=shape)]
    for position in ((0, 0), (shape[0]-1, shape[1]-1)):
        impulse = np.zeros(shape)
        impulse[position] = 1
        values.append(impulse)
    for data in values:
        expected = loops.base.radar.dense_small_operator(data, width, az_res)
        for method in loops.METHODS:
            audit = {}
            linear = loops.prepare_filter(method, shape, width, az_res, toolkit, audit)
            actual = linear(data)
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
            assert audit["filter_preparations"] == audit["full_filter_calls"] == 1
            if method in loops.METHODS[1:]:
                assert audit["azimuth_gate_convolutions"] == shape[1]
                assert audit["range_convolutions"] == (shape[0] if method == loops.METHODS[1] else 1)


@pytest.mark.parametrize("pattern", ("incomplete", "all_missing", "finite_zero", "single_observation"))
def test_three_filters_identical_missingness_moments_and_audits(toolkit, pattern):
    data = np.random.default_rng(6202).normal(size=(8, 9))
    data[1:4, 2:6] = np.nan
    if pattern == "all_missing":
        data[:] = np.nan
    elif pattern == "finite_zero":
        data[:] = 0
    elif pattern == "single_observation":
        data[:] = np.nan
        data[3, 4] = 2
    before = data.tobytes()
    expected = loops.base.calculate(loops.METHODS[0], data, 5., 45., toolkit)
    for method in loops.METHODS:
        audit = {}
        output = loops.calculate(method, data, 5., 45., toolkit, audit)
        assert audit == loops.expected_audit(method, data.shape)
        for key in loops.FIELDS:
            checked = loops.base.compare_arrays("unit", pattern, method, key, "all", output[key], expected[key])
            assert checked["status"] in ("pass", "no_comparable"), checked
        assert data.tobytes() == before


def test_numpy_loop_uses_gate_specific_weights_numpy_convolve_and_correct_padding(monkeypatch):
    data = np.arange(72, dtype=float).reshape(8, 9)
    weights = loops.base.independent_weights(data.shape, 5., 45.)
    calls, padding = [], []
    convolve, pad = np.convolve, np.pad

    def convolution(values, kernel, mode):
        calls.append((values.size, kernel.copy(), mode))
        return convolve(values, kernel, mode)

    def padded(values, widths, mode):
        padding.append(mode)
        return pad(values, widths, mode)

    monkeypatch.setattr(loops, "convolve1d", forbidden)
    monkeypatch.setattr(loops.base, "prepare_filter", forbidden)
    monkeypatch.setattr(np, "convolve", convolution)
    monkeypatch.setattr(np, "pad", padded)
    audit = {}
    output = loops.prepare_filter("numpy_range_loop", data.shape, 5., 45., object(), audit)(data)
    assert output.shape == data.shape
    assert len(calls) == sum(data.shape)
    assert padding == ["wrap"] * data.shape[1] + ["symmetric"] * data.shape[0]
    for gate in range(data.shape[1]):
        np.testing.assert_array_equal(calls[gate][1], weights[0][:, gate])
    assert not np.allclose(calls[0][1], calls[data.shape[1]-1][1])
    for item in calls[data.shape[1]:]:
        np.testing.assert_array_equal(item[1], weights[1])
    assert all(item[2] == "valid" for item in calls)


@pytest.mark.parametrize("method", loops.METHODS)
def test_fresh_worker_verifies_sources_outputs_and_actual_loop_counts(tmp_path, monkeypatch, toolkit, method):
    data = np.random.default_rng(6203).normal(size=(8, 9))
    data[2:4, 3:6] = np.nan
    output = loops.calculate(method, data, 5., 45., toolkit)
    descriptor = dict(verified=True, source_sha256=loops.source_hashes(), input_sha256=loops.base.array_sha256(data),
                      shape=list(data.shape), width=5., az_res_deg=45., output_sha256={method: loops.base.fingerprints(output)})
    path, manifest = tmp_path/"input.npy", tmp_path/"manifest.json"
    np.save(path, data)
    manifest.write_text(json.dumps(dict(unit=descriptor)))
    monkeypatch.setattr(loops, "SHAPE", data.shape)
    for key in loops.base.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    args = SimpleNamespace(case="unit", method=method, input=path, manifest=manifest, max_ram_mib=4096)
    payload = loops.worker(args)
    assert loops.check_worker(payload, descriptor, "unit", method)["status"] == "success", payload
    assert payload["audit"] == loops.expected_audit(method, data.shape)
    corrupted = dict(payload, audit=dict(filter_preparations=1, full_filter_calls=3))
    if method in loops.METHODS[1:]:
        assert loops.check_worker(corrupted, descriptor, "unit", method)["status"] == "invalid_worker"
    monkeypatch.setattr(loops, "calculate", forbidden)
    descriptor["source_sha256"]["benchmark"] = "wrong"
    manifest.write_text(json.dumps(dict(unit=descriptor)))
    assert loops.worker(args)["status"] == "error"


def test_failed_validation_blocks_every_timed_worker(tmp_path, monkeypatch, toolkit):
    data = np.ones((8, 9))
    monkeypatch.setattr(loops, "SHAPE", data.shape)
    monkeypatch.setattr(loops.base, "load_verified_inputs", lambda path: (
        loops.base.input_patterns(data), np.arange(9), 45., {"input_sha256": ("file-sha", "")}))
    monkeypatch.setattr(loops.base, "load_toolkit", lambda: toolkit)
    monkeypatch.setattr(loops, "validate_case", lambda *args: ([dict(status="fail")], dict(verified=False)))
    monkeypatch.setattr(loops, "measure_worker", forbidden)
    captured = []
    monkeypatch.setattr(loops, "write_settings", lambda *args: captured.append(args))
    for key in loops.base.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    assert loops.main(["--run-id", "failed", "--output-root", str(tmp_path), "--scratch-root", str(tmp_path)]) == 1
    directory = tmp_path/"2d_radar_range_loops"/"failed"
    assert (directory/"comparison.csv").exists() and not (directory/"benchmark_runs.csv").exists()
    assert all("failure" in case["timing_status"] for case in captured[0][2].values())


@pytest.mark.parametrize("argv", (["--worker"], ["--repeats", "0"], ["--timeout-s", "nan"],
                                   ["--max-ram-mib", "0"], ["--method", "numpy_range_loop"]))
def test_cli_rejects_incomplete_or_unsafe_worker_options(argv):
    with pytest.raises(SystemExit):
        loops.parse_args(argv)


RESULT = os.environ.get("RADAR_LOOP_RESULT_DIR")


def test_abstract_focuses_on_integrated_statistics_and_implementation():
    text = (SCRIPTS.parents[1]/"manuscript/main.tex").read_text()
    abstract = text.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    opening = abstract.strip().split(".", 1)[0]
    assert "integrated spectral method" in opening
    assert "local mean, variance, and standard deviation" in opening
    assert "Cartesian and polar grids with missing data" in opening
    assert "selected independently for each axis" in abstract
    assert "range-dependent angular filter widths" in abstract
    assert "prepared spectral computation of all three statistics" in abstract
    assert "For three-dimensional Cartesian grids, prepared spectral computation of all three statistics" in abstract
    assert "largest gains for wider windows" in abstract
    assert "running-sum" not in abstract
    assert "Python library" in abstract and r"\texttt{dct\_toolkit}" in abstract
    assert all(detail not in abstract for detail in ("data coverage", "denominator", "clamp", "v1.0.0"))


def saved_rows(name):
    with (Path(RESULT)/name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(None not in row and None not in row.values() for row in rows)
    return rows


@pytest.mark.skipif(not RESULT, reason="set RADAR_LOOP_RESULT_DIR to selected range-loop comparison")
def test_saved_full_array_checks_and_unchanged_tolerances():
    rows = saved_rows("comparison.csv")
    assert len(rows) == 580 and all(row["status"] == "pass" for row in rows)
    assert {row["method"] for row in rows} == {*loops.METHODS, loops.base.API, "inverse_fft_weight_construction"}
    for row in rows:
        assert (float(row["atol"]), float(row["rtol"])) == loops.base.TOLERANCES[row["statistic"]]
        assert row["n_method_undefined"] == row["n_reference_undefined"]


@pytest.mark.skipif(not RESULT, reason="set RADAR_LOOP_RESULT_DIR to selected range-loop comparison")
def test_saved_source_scaling_inputs_outputs_and_loop_audits():
    cases = {}
    for row in saved_rows("settings.csv"):
        cases.setdefault(row["case_id"], {})[row["setting"]] = row["value"]
    assert len(cases) == 4
    pids, inputs = [], set()
    for case_id, case in cases.items():
        assert json.loads(case["source_sha256"]) == loops.source_hashes()
        assert case["script_sha256"] == loops.source_hashes()["benchmark"]
        assert case["radar_file_sha256"] == loops.base.radar.DOCUMENTED_SHA256
        assert case["dct_toolkit_version"] == "1.0.0" and case["dct_toolkit_commit"] == loops.base.TOOLKIT_COMMIT
        assert case["range_dependent_azimuth_kernel"] == "True"
        assert float(case["azimuth_width_first"])/float(case["azimuth_width_last"]) == pytest.approx(1180)
        assert json.loads(case["direct_weight_lengths"]) == [721, 2361]
        assert case["completion"] == case["restoration"] == "False"
        assert case["timing_status"] == "measured and verified" and case["verified"] == "True"
        assert case["warmups"] == "0" and case["repetitions"] == "5"
        for key in loops.base.THREAD_VARIABLES:
            assert case[key] == "1"
        hashes = json.loads(case["output_sha256"])
        workers = [json.loads(value) for key, value in case.items() if key.startswith("worker_") and key.rsplit("_", 1)[-1].isdigit()]
        assert len(workers) == 15
        for item in workers:
            assert item["audit"] == loops.expected_audit(item["method"], loops.SHAPE)
            assert item["source_sha256"] == loops.source_hashes()
            assert item["input_sha256"] == case["input_sha256"]
            assert item["output_sha256"] == hashes[item["method"]]
            assert item["status"] == "success" and item["output_fingerprints_verified"] is True
            assert item["preloaded_libraries"] == loops.LIBRARIES and item["n_outputs"] == 9
            pids.append(item["worker_pid"])
            inputs.add(item["input_sha256"])
    assert len(pids) == len(set(pids)) == 60 and len(inputs) == 2


@pytest.mark.skipif(not RESULT, reason="set RADAR_LOOP_RESULT_DIR to selected range-loop comparison")
def test_saved_time_and_rss_summaries_from_every_repetition():
    runs, summaries = saved_rows("benchmark_runs.csv"), saved_rows("benchmark_summary.csv")
    assert len(runs) == 60 and len(summaries) == 12
    assert all(row["status"] == "success" and row["operation"] == loops.OPERATION for row in runs)
    for row in summaries:
        group = [r for r in runs if (r["case_id"], r["method"]) == (row["case_id"], row["method"])]
        assert {r["repetition"] for r in group} == {"1", "2", "3", "4", "5"}
        times = [float(r["elapsed_s"]) for r in group]
        for key, q in (("median_time_s", .5), ("q25_time_s", .25), ("q75_time_s", .75)):
            assert float(row[key]) == pytest.approx(np.quantile(times, q), rel=1e-12)
        assert float(row["max_peak_ram_mib"]) == max(float(r["peak_ram_mib"]) for r in group)
        assert row["n_success"] == "5" and row["n_failure"] == "0"


@pytest.mark.skipif(not RESULT, reason="set RADAR_LOOP_RESULT_DIR to selected range-loop comparison")
def test_archived_analytical_manuscript_table_matches_times_speedups_and_rss():
    # The current manuscript selects Gaussian convolution instead. Retain this
    # check against the last manuscript that selected these analytical weights.
    text = subprocess.check_output(["git", "show", "c4924a4:manuscript/main.tex"], cwd=SCRIPTS.parents[1]).decode()
    table = text.split("\\label{tab:radar_performance}")[1].split("\\end{table}")[0]
    indexed = {(r["case_id"], r["method"]): r for r in saved_rows("benchmark_summary.csv")}
    for pattern, label in (("natural", "Natural"), ("controlled_extra_gap", "Added gap")):
        for width in loops.base.WIDTHS:
            case = f"{pattern}_W{width:g}"
            own = float(indexed[case, loops.METHODS[0]]["median_time_s"])
            times = [f"{float(indexed[case, method]['median_time_s']):.4f}" for method in loops.METHODS]
            ratios = [f"{float(indexed[case, method]['median_time_s'])/own:.2f}" for method in loops.METHODS[1:]]
            rss = [f"{float(indexed[case, method]['max_peak_ram_mib']):.1f}" for method in loops.METHODS]
            assert f"{label} & {width:g} & " + " & ".join(times+ratios+rss) + r" \\" in table


@pytest.mark.skipif(not RESULT, reason="set RADAR_LOOP_RESULT_DIR to selected range-loop comparison")
def test_archived_analytical_manuscript_speedups_and_range_adaptive_loop_framing():
    text = subprocess.check_output(["git", "show", "c4924a4:manuscript/main.tex"], cwd=SCRIPTS.parents[1]).decode()
    abstract = text.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    section = text.split("\\label{sec:radar_performance}", 1)[1].split("\\subsection{", 1)[0]
    indexed = {(r["case_id"], r["method"]): float(r["median_time_s"]) for r in saved_rows("benchmark_summary.csv")}
    abstract_speedups = []
    for method in loops.METHODS[1:]:
        values = {f"{p}_W{w:g}": indexed[f"{p}_W{w:g}", method]/indexed[f"{p}_W{w:g}", loops.METHODS[0]]
                  for p in loops.base.PATTERNS for w in loops.base.WIDTHS}
        natural = [value for key, value in values.items() if key.startswith("natural_")]
        # The concise abstract rounds both natural-mask widths to the same
        # integer; the section/table still retain the full case-specific ratios.
        rounded_natural = {f"{value:.0f}" for value in natural}
        assert len(rounded_natural) == 1
        abstract_speedups.append(rounded_natural.pop())
        assert f"{min(values.values()):.1f}--{max(values.values()):.1f}" in section
    assert f"approximately {abstract_speedups[0]} and {abstract_speedups[1]} times faster" in abstract
    assert "range-adaptive" in abstract and "loop over range gates" in section
    assert "NumPy" in abstract and "SciPy" in abstract
    assert "not time separate statistics-API calls" in section
    assert "full" in section and "not ordinary" in section
