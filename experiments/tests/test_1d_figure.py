"""Focused export checks for two frozen 1D cases; no validation campaign/timing."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import generate_1d_boxcar_validation as one


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def prepared():
    return one.prepare_missingness_figure()


@pytest.fixture
def drawing(prepared):
    fig = one.create_missingness_figure(prepared[0])
    yield fig
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_import_and_default_self_test_branch_do_not_load_plotting():
    code = """
import sys
import generate_1d_boxcar_validation as one
assert not any(name in sys.modules for name in ('_review_figures', 'matplotlib', 'PIL'))
one.load_toolkit = lambda: object()
one.small_tests = lambda toolkit: 309
assert one.main(['--self-test-only']) == 0
assert not any(name in sys.modules for name in ('_review_figures', 'matplotlib', 'PIL'))
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=SCRIPTS, check=True)


def test_exact_saved_input_identity_and_undefined_interiors(prepared):
    cases, metadata = prepared
    expected = (
        ("19bb742436b9636cbbbc80de9854a115f3f0ca5f213353aed578cdd680276d04",
         "9009d4d32c6ecf984688a63d16e7a7037439136c13744e66c6e26f17d595f10e", 174, 14, 87, 113),
        ("ae17bd18fcb84282a72586fe8c0ee223ba20ea7e528b27ebc6cddb9f5bceb661",
         "9e228f407525d0f693169ad9fdb0e214fb3ebed2298f8bd52bb43c0538ed42ef", 167, 7, 0, 33),
    )
    assert tuple(case["case_id"] for case in cases) == one.FIGURE_CASE_IDS
    for case, (input_hash, mask_hash, n_defined, n_at_missing, start, stop) in zip(cases, expected):
        data = case["data"]
        missing = ~np.isfinite(data)
        assert hashlib.sha256(data.tobytes()).hexdigest() == input_hash
        assert hashlib.sha256(missing.tobytes()).hexdigest() == mask_hash
        assert missing.sum() == 40 and data.size == 200
        undefined = np.zeros(200, dtype=bool)
        undefined[start:stop] = True
        for result in (case["dct"], case["numpy"]):
            accepted = np.isfinite(result["denominator"]) & (result["denominator"] > 1e-3)
            np.testing.assert_array_equal(accepted, ~undefined)
            for statistic in ("mean", "variance", "std"):
                np.testing.assert_array_equal(np.isnan(result[statistic]), undefined)
                assert np.isfinite(result[statistic]).sum() == n_defined
                assert (np.isfinite(result[statistic]) & missing).sum() == n_at_missing
            assert np.isfinite(result["support"]).all()
            np.testing.assert_allclose(result["support"], 15 * np.clip(result["denominator"], 0, 1),
                                       atol=one.TOLERANCES["support"][0], rtol=one.TOLERANCES["support"][1])
        for statistic, (atol, rtol) in one.TOLERANCES.items():
            np.testing.assert_allclose(case["dct"][statistic], case["numpy"][statistic],
                                       atol=atol, rtol=rtol, equal_nan=True)
        assert metadata[f"{case['case_id']}.input_sha256"] == input_hash
        assert metadata[f"{case['case_id']}.mask_sha256"] == mask_hash
        assert metadata[f"{case['case_id']}.n_direct_defined"] == n_defined
    # Same noise realization, not independently regenerated noisy profiles.
    shared = np.isfinite(cases[0]["data"]) & np.isfinite(cases[1]["data"])
    np.testing.assert_array_equal(cases[0]["data"][shared], cases[1]["data"][shared])


def test_all_24_frozen_regional_comparisons_and_coverage_rows(prepared):
    cases, metadata = prepared
    with (one.FIGURE_RUN_DIRECTORY / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["case_id"] in one.FIGURE_CASE_IDS
                and row["method"] == "toolkit_dct_direct"]
    assert len(rows) == metadata["n_regional_comparisons_verified"] == 24
    for case in cases:
        for region, mask in one.regions(case["data"]).items():
            for statistic in one.TOLERANCES:
                saved, = [row for row in rows if row["case_id"] == case["case_id"]
                          and row["region"] == region and row["statistic"] == statistic]
                actual = one.comparison_row(
                    one.FIGURE_RUN_ID, case["case_id"], case["signal"], "toolkit_dct_direct",
                    statistic, region, mask, case["dct"][statistic], case["numpy"][statistic],
                )
                assert actual["status"] == saved["status"] == "pass"
                for key, value in actual.items():
                    if key in ("max_abs_difference", "rmse"):
                        # Exact in the recorded environment; retain the original
                        # fixed tolerance for platform-level floating point changes.
                        atol, rtol = one.TOLERANCES[statistic]
                        np.testing.assert_allclose(float(saved[key]), value, atol=atol, rtol=rtol)
                    else:
                        assert saved[key] == str(value), (case["case_id"], region, statistic, key)
    with (one.FIGURE_RUN_DIRECTORY / "coverage.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["case_id"] in one.FIGURE_CASE_IDS
                and row["method"] in ("toolkit_dct_direct", one.REFERENCE)]
    assert len(rows) == 12
    for case in cases:
        for method, result in (("toolkit_dct_direct", case["dct"]), (one.REFERENCE, case["numpy"])):
            for region, mask in one.regions(case["data"]).items():
                saved, = [row for row in rows if row["case_id"] == case["case_id"]
                          and row["method"] == method and row["region"] == region]
                actual = one.coverage_row(one.FIGURE_RUN_ID, case["case_id"], method, region, mask, case["data"], result)
                for key, value in actual.items():
                    if isinstance(value, float):
                        np.testing.assert_allclose(float(saved[key]), value, atol=5e-11, rtol=2e-11)
                    else:
                        assert saved[key] == str(value), (case["case_id"], method, region, key)


@pytest.mark.parametrize("filename,match,field,value,message", [
    ("settings.csv", {"setting": "input_sha256"}, "value", "wrong", "input_sha256"),
    ("settings.csv", {"setting": "mask_sha256"}, "value", "wrong", "mask_sha256"),
    ("settings.csv", {"setting": "master_seed"}, "value", "20261028", "master_seed"),
    ("settings.csv", {"setting": "std_atol"}, "value", "0.1", "std_atol"),
    ("coverage.csv", {"method": "toolkit_dct_direct", "region": "all"},
     "n_direct_defined", "200", "coverage mismatch"),
    ("comparison.csv", {"method": "toolkit_dct_direct", "region": "gap", "statistic": "mean"},
     "n_compared", "40", "comparison failed"),
    ("comparison.csv", {"method": "toolkit_dct_direct", "region": "all", "statistic": "variance"},
     "atol", "0.1", "comparison failed"),
    ("comparison.csv", {"method": "toolkit_dct_direct", "region": "all", "statistic": "std"},
     "status", "fail", "comparison failed"),
    ("comparison.csv", {"method": "toolkit_dct_direct", "region": "edge", "statistic": "std"},
     "max_abs_difference", "0.1", "discrepancy changed"),
    ("comparison.csv", {"method": "toolkit_dct_direct", "region": "gap", "statistic": "support"},
     "rmse", "0.1", "discrepancy changed"),
])
def test_rejects_changed_saved_identity_counts_or_tolerances(tmp_path, filename, match,
                                                          field, value, message):
    saved = tmp_path / one.FIGURE_RUN_ID
    saved.mkdir()
    for name in ("settings.csv", "coverage.csv", "comparison.csv"):
        shutil.copyfile(one.FIGURE_RUN_DIRECTORY / name, saved / name)
    path = saved / filename
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    amended = 0
    for row in rows:
        if row["case_id"] == one.FIGURE_CASE_IDS[0] and all(row[k] == v for k, v in match.items()):
            row[field] = value
            amended += 1
    assert amended == 1
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=message):
        one.prepare_missingness_figure(saved)


def test_rejects_missing_saved_case(tmp_path):
    for filename in ("settings.csv", "coverage.csv", "comparison.csv"):
        with (one.FIGURE_RUN_DIRECTORY / filename).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [row for row in reader if row["case_id"] != one.FIGURE_CASE_IDS[1]]
        with (tmp_path / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ValueError, match="Saved figure case mismatch"):
        one.prepare_missingness_figure(tmp_path)


@pytest.mark.parametrize("statistic", ("mean", "variance", "std", "support", "denominator", "filled_mean"))
def test_export_gate_checks_every_statistic_and_forbids_filling(monkeypatch, statistic):
    original = one.toolkit_moments

    def corrupted(*args):
        result = original(*args)
        if statistic == "filled_mean":
            result["mean"] = np.nan_to_num(result["mean"])
        else:
            result[statistic] = result[statistic] + 0.1
        return result

    monkeypatch.setattr(one, "toolkit_moments", corrupted)
    with pytest.raises(ValueError, match="Undefined figure moments changed|comparison failed|coverage/count mapping changed"):
        one.prepare_missingness_figure()


def test_plot_uses_checked_arrays_no_truth_no_bridging_and_original_gap_shading(drawing, prepared):
    from matplotlib.path import Path as PlotPath
    from matplotlib.text import Text

    assert len(drawing.axes) == 6 and len(drawing.legends) == 1
    assert tuple(drawing.get_size_inches()) == one.FIGURE_SIZE
    assert min(text.get_fontsize() for text in drawing.findobj(Text)
               if text.get_visible() and text.get_text().strip()) >= 10.1
    x = np.linspace(0, 10, 200)
    edges = np.r_[0, (x[:-1] + x[1:]) / 2, 10]
    for column, case in enumerate(prepared[0]):
        missing = np.flatnonzero(~np.isfinite(case["data"]))
        for row in range(3):
            ax = drawing.axes[2 * row + column]
            assert ax.get_xlim() == (0, 10)
            assert ax.get_ylim() == one.FIGURE_Y_LIMITS[row]
            assert ax.get_title(loc="left").startswith(f"({chr(97 + 2 * row + column)})")
            assert not ax.images
            patch, = ax.patches
            assert patch.get_gid() == "original_missing_interval"
            assert patch.get_x() == edges[missing[0]]
            assert patch.get_width() == pytest.approx(edges[missing[-1] + 1] - edges[missing[0]])
            lines = {line.get_gid(): line for line in ax.lines}
            expected = ({"observed", "dct_mean", "numpy_mean"} if row == 0 else
                        {"dct_std", "numpy_std"} if row == 1 else {"dct_rho", "numpy_rho"})
            assert set(lines) == expected
            for line in lines.values():
                np.testing.assert_array_equal(line.get_xdata(), x)
            if row == 0:
                np.testing.assert_array_equal(lines["observed"].get_ydata(), case["data"])
                assert lines["observed"].get_linestyle() == "None"
                observed = case["data"][np.isfinite(case["data"])]
                assert observed.min() >= ax.get_ylim()[0] and observed.max() <= ax.get_ylim()[1]
            if row < 2:
                statistic = ("mean", "std")[row]
                for method in ("dct", "numpy"):
                    values = lines[f"{method}_{statistic}"].get_ydata()
                    np.testing.assert_array_equal(values, case[method][statistic])
                    assert values[np.isfinite(values)].min() >= ax.get_ylim()[0]
                    assert values[np.isfinite(values)].max() <= ax.get_ylim()[1]
                    # Inspect the renderer path as well as the source arrays:
                    # a middle gap has two subpaths, never a joining segment.
                    moves = sum(code == PlotPath.MOVETO for _, code in
                                lines[f"{method}_{statistic}"].get_path().iter_segments())
                    assert moves == (2 if column == 0 else 1)
                assert lines[f"numpy_{statistic}"].is_dashed()
            else:
                for method in ("dct", "numpy"):
                    values = lines[f"{method}_rho"].get_ydata()
                    np.testing.assert_array_equal(values, np.clip(case[method]["denominator"], 0, 1))
                    assert np.isfinite(values).all()
                assert lines["numpy_rho"].is_dashed()
    assert not any(artist.get_rasterized() for artist in drawing.findobj())


def test_figure_only_exports_metadata_without_validation_or_result_csv_writes(tmp_path, monkeypatch):
    from PIL import Image

    def forbidden(*args, **kwargs):
        pytest.fail("Figure-only must not enter validation, self-tests or result writers")

    for name in ("create_run_directory", "small_tests", "methods", "write_csv", "write_settings"):
        monkeypatch.setattr(one, name, forbidden)
    generated, calculated = [], []
    make_cases, toolkit_moments = one.make_cases, one.toolkit_moments

    def tracked_cases(*args):
        for case in make_cases(*args):
            generated.append(case[0])
            yield case

    def tracked_moments(data, *args):
        calculated.append(hashlib.sha256(data.tobytes()).hexdigest())
        return toolkit_moments(data, *args)

    monkeypatch.setattr(one, "make_cases", tracked_cases)
    monkeypatch.setattr(one, "toolkit_moments", tracked_moments)
    before = {path.name: digest(path) for path in one.FIGURE_RUN_DIRECTORY.iterdir()}
    assert one.main(["--figure-only", "--figure-output", str(tmp_path)]) == 0
    assert len(calculated) == 2 and len(set(calculated)) == 2
    assert len(generated) == 4 and generated[-1] == one.FIGURE_CASE_IDS[1]
    assert {path.name: digest(path) for path in one.FIGURE_RUN_DIRECTORY.iterdir()} == before
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()) == [
        "figure_settings.csv", "pdf/boxcar_missingness.pdf", "png/boxcar_missingness.png",
    ]
    with (tmp_path / "figure_settings.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["figure", "setting", "value"]
        rows = list(reader)
    assert {row["figure"] for row in rows} == {one.FIGURE_STEM}
    values = {row["setting"]: row["value"] for row in rows}
    assert len(values) == len(rows)
    assert json.loads(values["selected_cases"]) == list(one.FIGURE_CASE_IDS)
    assert values["selected_run"] == one.FIGURE_RUN_ID
    assert values["completion"] == values["restoration"] == "False"
    assert values["master_seed"] == "20261027" and values["replicate"] == "0"
    assert values["requested_width"] == values["discrete_width"] == "15"
    assert values["denominator_floor"] == "0.001"
    assert values["script_sha256"] == digest(Path(one.__file__))
    assert values["figure_helper_sha256"] == digest(SCRIPTS / "_review_figures.py")
    assert values["figure_width_in"] == "6.5" and values["figure_height_in"] == "6.2"
    assert float(values["min_text_size_pt"]) >= 10 and values["pdf_fonttype"] == "42"
    assert len([name for name in values if name.startswith("toolkit_source_sha256.")]) == 9
    assert values["checks"].startswith("PASS:")
    for statistic, (atol, rtol) in one.TOLERANCES.items():
        assert float(values[f"{statistic}_atol"]) == atol
        assert float(values[f"{statistic}_rtol"]) == rtol
    for filename in ("settings.csv", "coverage.csv", "comparison.csv"):
        assert values[f"saved_{filename}_sha256"] == before[filename]
    for kind in ("pdf", "png"):
        assert values[f"{kind}_sha256"] == digest(tmp_path / values[f"{kind}_path"])
    with Image.open(tmp_path / values["png_path"]) as image:
        assert image.size == (1950, 1860)
        assert all(abs(value - 300) < 0.01 for value in image.info["dpi"])
        assert image.getpixel((0, 0)) == (255, 255, 255, 255)
    pdf_bytes = (tmp_path / values["pdf_path"]).read_bytes()
    assert b"/FontFile2" in pdf_bytes and b"/Subtype /Type3" not in pdf_bytes


def test_figure_only_early_branch_and_default_output(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Figure-only must branch before toolkit loading/self-tests/run creation")

    for name in ("load_toolkit", "small_tests", "create_run_directory", "make_cases"):
        monkeypatch.setattr(one, name, forbidden)
    calls = []
    monkeypatch.setattr(one, "export_missingness_figure", lambda output: calls.append(output) or {})
    assert one.main(["--figure-only"]) == 0
    assert calls == [None]


@pytest.mark.parametrize("args", [
    ["--figure-output", "unused"],
    ["--figure-only", "--master-seed", "1"],
    ["--figure-only", "--mode", "smoke"],
    ["--figure-only", "--self-test-only"],
    ["--figure-only", "--run-id", "new-run"],
    ["--figure-only", "--output-root", "unused"],
])
def test_rejects_figure_validation_option_mixing(args, monkeypatch):
    monkeypatch.setattr(one, "export_missingness_figure", lambda *args: pytest.fail("No export expected"))
    with pytest.raises(SystemExit) as error:
        one.main(args)
    assert error.value.code == 2


def test_cannot_export_into_frozen_result_folders(monkeypatch):
    monkeypatch.setattr(one, "prepare_missingness_figure", lambda *args: pytest.fail("Must reject before calculating"))
    with pytest.raises(ValueError, match="must not be inside saved experiment results"):
        one.export_missingness_figure(one.FIGURE_RUN_DIRECTORY)
