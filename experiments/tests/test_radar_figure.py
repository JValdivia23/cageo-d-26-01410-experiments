"""One frozen radar figure: native geometry, signed diagnostics and lazy export.

The module fixture hashes/reads the large radar file and reproduces its full
spectral case once. No test runs the dense reference or a timing campaign.
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import generate_2d_radar_gaussian_validation as radar


def forbidden(*args, **kwargs):
    pytest.fail("Figure mode must not run a reference, self-test, benchmark or numerical result writer")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def evidence():
    return radar._figure_saved_evidence()


@pytest.fixture(scope="module")
def prepared():
    with pytest.MonkeyPatch.context() as patch:
        for name in ("SpatialPolarGaussian", "dense_small_operator", "validate_case", "benchmark",
                     "self_tests", "create_run_directory", "write_csv", "write_settings"):
            patch.setattr(radar, name, forbidden)
        return radar.prepare_radar_figure()


@pytest.fixture(scope="module")
def drawing(prepared):
    fig = radar.create_radar_figure(prepared[0])
    yield fig
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_import_and_legacy_self_test_dispatch_do_not_import_plotting():
    code = """
import sys
from pathlib import Path
import generate_2d_radar_gaussian_validation as radar
assert not any(n in sys.modules for n in ('_review_figures', 'matplotlib', 'PIL', 'xarray', 'cmweather'))
radar.load_toolkit = lambda: object()
radar.create_run_directory = lambda *args: ('test-only', Path('.'))
radar.self_tests = lambda *args: ([], {})
radar.write_csv = radar.write_settings = lambda *args: None
sys.argv = ['radar', '--self-test-only']
assert radar.main() == 0
assert not any(n in sys.modules for n in ('_review_figures', 'matplotlib', 'PIL', 'xarray', 'cmweather'))
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=SCRIPTS, check=True)


def test_frozen_file_decoded_hashes_release_and_full_central_counts(prepared):
    case, settings = prepared
    assert case["data"].shape == (720, 1180)
    assert settings["input_sha256"] == radar.DOCUMENTED_SHA256
    assert hashlib.sha256(case["data"].tobytes()).hexdigest() == (
        settings["decoded_input_sha256"]
    ) == "e47d1b8e07791951a602344edcdcf60542a517f385e54c452fd81117f676d9a0"
    assert np.isfinite(case["data"]).sum() == 508956
    assert case["accepted"].sum() == 655428
    assert case["negative"].sum() == 4593
    assert case["central"].sum() == 133
    assert case["accepted"][:, case["central"]].sum() == 90152
    assert not case["negative"][:, case["central"]].any()
    assert np.isfinite(case["data"][:, case["central"]]).sum() == 56584
    assert settings["central_last_gate_center_m"] == 9987.265625
    assert settings["central_outer_cell_corner_m"] == 10024.736328125
    assert settings["saved_original_script_sha256"] == radar.FIGURE_ORIGINAL_SCRIPT_SHA256
    assert settings["script_sha256"] == digest(Path(radar.__file__))
    assert settings["dct_toolkit_commit"] == "9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c"
    assert settings["numpy_version"] == np.__version__
    assert settings["astropy_version"] == "8.0.2"
    assert len([key for key in settings if key.startswith("toolkit_source_sha256.")]) == 9
    assert settings["n_regional_diagnostic_rows_verified"] == 4
    assert settings["n_radial_diagnostic_rows_verified"] == 1180
    assert settings["n_frozen_comparison_metadata_rows_verified"] == 36
    assert settings["new_spatial_reference_performed"] is settings["new_timing_performed"] is False
    assert settings["completion"] is settings["restoration"] is False


def test_signed_denominator_undefined_moments_and_exact_negative_overlay(prepared):
    case, settings = prepared
    fields = case["fields"]
    raw = fields["raw_mask_response"]
    accepted = np.isfinite(raw) & (raw > 1e-3)
    np.testing.assert_array_equal(case["accepted"], accepted)
    np.testing.assert_array_equal(case["negative"], accepted & (fields["raw_variance"] < 0))
    for statistic in ("mean", "second_moment", "variance", "std", "raw_variance"):
        np.testing.assert_array_equal(np.isfinite(fields[statistic]), accepted)
    np.testing.assert_array_equal(fields["variance"], np.maximum(fields["raw_variance"], 0))
    np.testing.assert_array_equal(fields["std"], np.sqrt(fields["variance"]))
    assert np.all(fields["std"][case["negative"]] == 0)
    assert (raw < 0).sum() == 97941 and (raw > 1).sum() == 93873
    assert raw.min() == pytest.approx(-0.03669788270965666, abs=2e-12)
    assert raw.max() == pytest.approx(1.026742079260644, abs=2e-12)
    assert settings["mean_finite_min"] == pytest.approx(-39.36499608640338)
    assert settings["mean_finite_max"] == pytest.approx(71.16126147424193)
    assert settings["raw_variance_finite_min"] == pytest.approx(-3645.1029666865334)
    assert settings["raw_variance_finite_max"] == pytest.approx(113.86524145193061)
    assert settings["mean_below_color_limit_count"] > 0
    assert settings["mean_above_color_limit_count"] > 0
    assert settings["std_below_color_limit_count"] == 0
    assert settings["std_above_color_limit_count"] == np.sum(fields["std"] > 10) > 0


def test_square_zoom_keeps_corners_and_separate_frozen_radial_diagnostics(prepared):
    from matplotlib.path import Path as PolygonPath

    case, settings = prepared
    radius = case["ranges"][None, :] / 1000
    theta = np.deg2rad(case["azimuth"])[:, None]
    expected = (np.abs(radius * np.sin(theta)) <= 10) & (np.abs(radius * np.cos(theta)) <= 10)
    np.testing.assert_array_equal(case["zoom"], expected)
    assert expected.sum() == settings["zoom_n_cells"] == 107268
    assert np.isfinite(case["data"])[expected].sum() == settings["zoom_n_observed"] == 64534
    assert case["accepted"][expected].sum() == settings["zoom_n_accepted"] == 99973
    assert (~case["accepted"])[expected].sum() == settings["zoom_n_undefined"] == 7295
    assert case["negative"][expected].sum() == settings["negative_overlay_zoom_count"] == 0
    assert settings["negative_overlay_central_count"] == 0  # Unchanged R<=10 km diagnostic, not a square.
    assert not (~expected[:, case["central"]]).any()
    assert (expected & ~case["central"][None, :]).sum() == 11508
    assert case["zoom_gate_count"] == settings["zoom_gate_count"] == 188 > case["central"].sum()
    assert not expected[:, case["zoom_gate_count"]:].any()
    count = case["zoom_gate_count"]
    assert not case["negative"][:, :count].any()  # No red boundary polygon can enter the square either.
    perimeter = PolygonPath(np.column_stack((case["x"][:, count], case["y"][:, count])))
    assert perimeter.contains_points([[-10, -10], [-10, 10], [10, -10], [10, 10]]).all()
    assert settings["zoom_outer_cell_corner_m"] == pytest.approx(14146.513671875)
    assert settings["zoom_axis_limits_km"] == [-10, 10]
    assert "native cell centers" in settings["zoom_count_definition"]
    assert "no cropped filtering" in settings["zoom_processing"]
    assert "NOT the square display" in settings["central_processing"]


def test_required_chasespectral_palette_and_original_limits_are_shared(drawing, prepared):
    from cmweather.cm_colorblind import ChaseSpectral
    from matplotlib.colors import to_rgba

    samples = np.linspace(0, 1, 256)
    for ax in drawing.axes[:2]:
        mesh = ax.collections[0]
        assert mesh.cmap.name == "ChaseSpectral"
        np.testing.assert_array_equal(mesh.cmap(samples), ChaseSpectral(samples))
        np.testing.assert_array_equal(mesh.cmap.get_bad(), to_rgba(radar.FIGURE_MISSING_COLOR))
        assert mesh.get_clim() == (0, 40)
    assert drawing.axes[-3]._colorbar.extend == "both"
    assert drawing.axes[2].collections[0].get_clim() == drawing.axes[4].collections[0].get_clim() == (0, 10)
    assert drawing.axes[-2]._colorbar.extend == "max"
    np.testing.assert_array_equal(drawing.axes[-2]._colorbar.get_ticks(), [0, 5, 10])
    np.testing.assert_array_equal(drawing.axes[-3]._colorbar.get_ticks(), [0, 10, 20, 30, 40])
    case, settings = prepared
    assert settings["reflectivity_colormap"] == "ChaseSpectral"
    assert "cmweather.cm_colorblind.ChaseSpectral" in settings["reflectivity_colormap_source"]
    assert settings["cmweather_version"] == "0.3.2"
    for name, values, counts in (("observed", case["data"], (158, 551)),
                                 ("mean", case["fields"]["mean"], (75, 97))):
        assert settings[f"{name}_below_color_limit_count"] == np.sum(values < 0) == counts[0]
        assert settings[f"{name}_above_color_limit_count"] == np.sum(values > 40) == counts[1]


def test_missing_cmweather_is_rejected_before_reading_without_palette_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "cmweather.cm_colorblind", None)
    monkeypatch.setattr(radar, "read_radar", forbidden)
    with pytest.raises(RuntimeError, match="requires cmweather with ChaseSpectral; no substitute"):
        radar.prepare_radar_figure()


def test_every_regional_and_radial_diagnostic_matches_saved_evidence(prepared, evidence):
    case, settings = prepared
    masks = radar.regions(case["data"].shape, case["ranges"])
    diagnostics, radial = radar.diagnostic_rows(
        radar.FIGURE_RUN_ID, radar.FIGURE_CASE_ID, radar.SPECTRAL, case["fields"],
        np.isfinite(case["data"]), masks, case["ranges"], 15, 0.5,
    )
    for actual, filename, fields, identity in (
            (diagnostics, "support_diagnostics.csv", radar.DIAGNOSTIC_FIELDS, "region"),
            (radial, "support_by_range.csv", radar.RADIAL_FIELDS, "gate_index_zero_based")):
        saved = [row for row in evidence[1][filename] if row["method"] == radar.SPECTRAL]
        radar._figure_check_diagnostic_rows(actual, saved, fields, identity)
        for row, old in zip(actual, saved):
            for key in fields:
                if key.startswith("raw_support_"):
                    np.testing.assert_allclose(float(old[key]), row[key], atol=2e-12, rtol=2e-11)
                else:
                    assert old[key] == str(row[key])
    for row in diagnostics:
        for key in radar.DIAGNOSTIC_FIELDS[4:]:
            assert settings[f"diagnostic.{row['region']}.{key}"] == row[key]
    # Rejection is tested on copies, not by altering any saved file.
    wrong = [dict(row) for row in diagnostics]
    wrong[0]["n_negative_raw_variance"] -= 1
    saved = [row for row in evidence[1]["support_diagnostics.csv"] if row["method"] == radar.SPECTRAL]
    with pytest.raises(ValueError, match="diagnostic mismatch"):
        radar._figure_check_diagnostic_rows(wrong, saved, radar.DIAGNOSTIC_FIELDS, "region")
    wrong = [dict(row) for row in radial]
    wrong[500]["raw_support_min"] += 1e-5
    saved = [row for row in evidence[1]["support_by_range.csv"] if row["method"] == radar.SPECTRAL]
    with pytest.raises(ValueError, match="raw diagnostic mismatch"):
        radar._figure_check_diagnostic_rows(wrong, saved, radar.RADIAL_FIELDS, "gate_index_zero_based")


@pytest.mark.parametrize("key,value", [
    ("input_sha256", "wrong"), ("full_sweep_decoded_float64_sha256", "wrong"),
    ("script_sha256", "wrong"), ("dct_toolkit_commit", "wrong"), ("completion_executed", "True"),
    ("input_mask_restoration", "True"), ("width_range_gates", "5.0"), ("denominator_floor", "0.01"),
    ("boundary_range", "periodic"), ("operator_order", "range first"), ("quick_mode", "True"),
])
def test_rejects_changed_scientific_settings_without_rereading_radar(evidence, key, value):
    settings = dict(evidence[0], **{key: value})
    with pytest.raises(ValueError, match=key):
        radar._figure_check_settings(settings)


@pytest.mark.parametrize("filename", radar.FIGURE_SAVED_CSV_SHA256)
def test_rejects_modified_evidence_before_any_calculation(tmp_path, monkeypatch, filename):
    monkeypatch.setattr(radar, "read_radar", forbidden)
    for name in radar.FIGURE_SAVED_CSV_SHA256:
        (tmp_path / name).write_bytes((radar.FIGURE_RUN_DIRECTORY / name).read_bytes())
    with (tmp_path / filename).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match=f"SHA-256 mismatch: {filename}"):
        radar.prepare_radar_figure(tmp_path)


def test_missing_file_and_bad_file_or_decoded_hash_have_no_fallback(evidence, prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(radar, "_figure_saved_evidence", lambda *args: evidence)
    monkeypatch.setattr(radar, "load_toolkit", forbidden)
    monkeypatch.setenv("REAL_RADAR_PATH", str(tmp_path / "missing.nc"))
    with pytest.raises(FileNotFoundError, match="No sample/synthetic substitution"):
        radar.prepare_radar_figure()
    case = prepared[0]
    metadata = {"input_sha256": ("wrong", "")}
    monkeypatch.setattr(radar, "read_radar", lambda *args, **kwargs: (case["data"], case["ranges"], 0.5, metadata))
    with pytest.raises(ValueError, match="documented file SHA-256"):
        radar.prepare_radar_figure()
    metadata["input_sha256"] = (radar.DOCUMENTED_SHA256, "")
    metadata["full_sweep_decoded_float64_sha256"] = ("wrong", "")
    with pytest.raises(ValueError, match="decoded input/geometry mismatch"):
        radar.prepare_radar_figure()


def test_native_corner_mesh_cardinal_orientation_and_closed_acquisition_seam():
    # Small coordinate-only check: no artificial radar field or filter is used.
    azimuth = np.array([0., 90., 180., 270.])
    ranges = np.array([95., 169.94140625, 244.8828125])
    x, y, angular, radial = radar._figure_polar_mesh(azimuth, ranges)
    np.testing.assert_allclose(angular, np.deg2rad([-45, 45, 135, 225, 315]))
    np.testing.assert_array_equal(radial * 1000, [57.529296875, 132.470703125, 207.412109375, 282.353515625])
    assert x.shape == y.shape == (5, 4)
    np.testing.assert_array_equal(x[-1], x[0])
    np.testing.assert_array_equal(y[-1], y[0])
    assert x[0, -1] < 0 < x[1, -1] and y[0, -1] > 0 and y[1, -1] > 0  # North ray.
    assert x[1, -1] > 0 and x[2, -1] > 0 and y[2, -1] < 0  # East ray.
    rotated = np.mod(azimuth + 103.6, 360)
    xr, yr, ar, rr = radar._figure_polar_mesh(rotated, ranges)
    assert ar[0] == pytest.approx(np.deg2rad(103.6 - 45))
    np.testing.assert_allclose(xr[1:-1], np.sin(ar[1:-1, None]) * rr)
    np.testing.assert_allclose(yr[1:-1], np.cos(ar[1:-1, None]) * rr)


def test_six_native_panels_postfilter_subsets_and_visible_exact_negative_masks(drawing, prepared):
    from matplotlib.text import Text

    case = prepared[0]
    n = case["zoom_gate_count"]
    fields = case["fields"]
    rho = np.clip(fields["raw_mask_response"], 0, 1)
    values = (case["data"], fields["mean"], fields["std"], rho, fields["std"][:, :n], rho[:, :n])
    assert len(drawing.axes) == 9  # Six data axes and three shared colorbars.
    assert tuple(drawing.get_size_inches()) == (6.5, 7.4)
    assert min(t.get_fontsize() for t in drawing.findobj(Text)
               if t.get_visible() and t.get_text().strip()) >= 10.1
    drawing.canvas.draw()
    renderer = drawing.canvas.get_renderer()
    for i, (ax, expected) in enumerate(zip(drawing.axes[:6], values)):
        np.testing.assert_allclose(ax.get_position().size * drawing.get_size_inches(), [1.95, 1.95], atol=1e-12)
        assert ax.get_title(loc="left").startswith(f"({chr(97 + i)})")
        assert ax._left_title.get_window_extent(renderer).x1 <= ax.get_window_extent(renderer).x1 + 0.5
        assert not ax.images  # No imshow interpolation/resampling.
        mesh = ax.collections[0]
        np.testing.assert_array_equal(mesh.get_array().filled(np.nan), expected)
        coordinates = mesh.get_coordinates()
        np.testing.assert_array_equal(coordinates[..., 0], case["x"][:, :n + 1] if i >= 4 else case["x"])
        np.testing.assert_array_equal(coordinates[..., 1], case["y"][:, :n + 1] if i >= 4 else case["y"])
        assert mesh.get_rasterized()
        assert ax.get_xlim() == ax.get_ylim() == ((-10, 10) if i >= 4 else (-90, 90))
        if i >= 4:
            assert "(zoom)" in ax.get_title(loc="left")
        if i in (2, 4):
            overlay = ax.collections[1]
            assert overlay.get_gid() == "negative_raw_variance" and overlay.get_rasterized()
            negative = case["negative"][:, :n] if i == 4 else case["negative"]
            np.testing.assert_array_equal(~np.ma.getmaskarray(overlay.get_array()), negative)
            assert overlay.cmap.colors == [radar.FIGURE_NEGATIVE_COLOR]
        else:
            assert len(ax.collections) == 1
        if i in (2, 3):
            square, = ax.patches
            assert square.get_gid() == "square_20km_zoom" and square.get_xy() == (-10, -10)
            assert square.get_width() == square.get_height() == 20
    for a, b in ((0, 1), (2, 4), (3, 5)):
        assert drawing.axes[a].collections[0].get_clim() == drawing.axes[b].collections[0].get_clim()
    assert drawing.axes[-3]._colorbar.extend == "both"


def test_clamped_key_is_local_to_std_title_row_without_covering_data(drawing):
    from matplotlib.colors import to_rgba

    assert not drawing.legends  # No global missing/clamped legend.
    assert drawing.axes[2].get_title(loc="left") == "(c) Std"
    for i, ax in enumerate(drawing.axes[:6]):
        assert (ax.get_legend() is not None) == (i == 2)
    key = drawing.axes[2].get_legend()
    assert key.get_gid() == "clamped_panel_key" and not key.get_frame_on()
    assert [text.get_text() for text in key.get_texts()] == ["Clamped"]
    np.testing.assert_array_equal(key.legend_handles[0].get_facecolor(), to_rgba(radar.FIGURE_NEGATIVE_COLOR))
    drawing.canvas.draw()
    renderer = drawing.canvas.get_renderer()
    box = key.get_window_extent(renderer)
    title = drawing.axes[2]._left_title.get_window_extent(renderer)
    data = drawing.axes[2].get_window_extent(renderer)
    assert box.y0 > data.y1 and box.x0 > title.x1 + 3
    assert box.x1 <= data.x1 + 0.5
    assert box.y0 < title.y1 and box.y1 > title.y0


def test_export_has_fixed_canvas_300dpi_embedded_vector_text_and_complete_sidecar(
        tmp_path, monkeypatch, drawing, prepared):
    from PIL import Image

    monkeypatch.setattr(radar, "prepare_radar_figure", lambda: prepared)
    monkeypatch.setattr(radar, "create_radar_figure", lambda case: drawing)
    for name in ("SpatialPolarGaussian", "validate_case", "benchmark", "self_tests",
                 "create_run_directory", "write_csv", "write_settings"):
        monkeypatch.setattr(radar, name, forbidden)
    before = {name: digest(radar.FIGURE_RUN_DIRECTORY / name) for name in radar.FIGURE_SAVED_CSV_SHA256}
    paths = radar.export_radar_figure(tmp_path)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()) == [
        "figure_settings.csv", "pdf/radar_gaussian_statistics.pdf", "png/radar_gaussian_statistics.png",
    ]
    with paths["settings"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["figure", "setting", "value"]
        rows = list(reader)
    assert {row["figure"] for row in rows} == {radar.FIGURE_STEM}
    settings = {row["setting"]: row["value"] for row in rows}
    assert len(settings) == len(rows) and settings["checks"].startswith("PASS:")
    assert settings["selected_case"] == radar.FIGURE_CASE_ID
    assert settings["selected_run"] == radar.FIGURE_RUN_ID
    assert settings["input_sha256"] == radar.DOCUMENTED_SHA256
    assert settings["script_sha256"] == digest(Path(radar.__file__))
    assert settings["saved_original_script_sha256"] == radar.FIGURE_ORIGINAL_SCRIPT_SHA256
    assert settings["figure_helper_sha256"] == digest(SCRIPTS / "_review_figures.py")
    assert settings["completion"] == settings["restoration"] == settings["new_timing_performed"] == "False"
    assert settings["negative_overlay_whole_count"] == "4593" and settings["negative_overlay_central_count"] == "0"
    assert len([key for key in settings if key.startswith("toolkit_source_sha256.")]) == 9
    assert settings["reflectivity_colormap"] == "ChaseSpectral"
    assert settings["cmweather_version"] == "0.3.2"
    assert json.loads(settings["reflectivity_color_limits_dbz"]) == [0, 40]
    assert json.loads(settings["reflectivity_colorbar_ticks_dbz"]) == [0, 10, 20, 30, 40]
    assert settings["reflectivity_colorbar_extend"] == "both"
    assert json.loads(settings["std_color_limits_dbz"]) == [0, 10]
    assert settings["std_colorbar_extend"] == "max"
    assert int(settings["std_above_color_limit_count"]) == prepared[1]["std_above_color_limit_count"] > 0
    assert json.loads(settings["zoom_axis_limits_km"]) == [-10, 10]
    assert settings["zoom_gate_count"] == "188"
    assert settings["zoom_n_cells"] == "107268" and settings["zoom_n_accepted"] == "99973"
    assert settings["negative_overlay_zoom_count"] == "0"
    assert settings["clamped_overlay_label"] == "Clamped"
    assert settings["clamped_overlay_key_location"] == "panel (c) title row only; no figure-wide legend"
    assert settings["missing_category_explanation"].startswith("caption only:")
    assert settings["clamped_zero_interpretation"] == "numerical clamp, not evidence of absent local variability"
    assert float(settings["min_text_size_pt"]) >= 10.1
    assert settings["figure_width_in"] == "6.5" and settings["figure_height_in"] == "7.4"
    assert settings["top_margin_trimmed_in"] == "0.4"
    assert json.loads(settings["data_panel_size_in"]) == [1.95, 1.95]
    for name, old in before.items():
        assert digest(radar.FIGURE_RUN_DIRECTORY / name) == settings[f"saved_{name}_sha256"] == old
    for kind in ("pdf", "png"):
        assert settings[f"{kind}_sha256"] == digest(paths[kind])
    with Image.open(paths["png"]) as image:
        assert image.size == (1950, 2220)
        assert all(abs(v - 300) < 0.01 for v in image.info["dpi"])
        assert image.getpixel((0, 0)) == (255, 255, 255, 255)
    pdf = paths["pdf"].read_bytes()
    assert b"/FontFile2" in pdf and b"/Subtype /Type3" not in pdf
    assert b"/Subtype /Image" in pdf  # Dense cells, not labels, are rasterized.


def test_figure_dispatch_branches_before_legacy_work(monkeypatch):
    for name in ("load_toolkit", "read_radar", "self_tests", "validate_case", "benchmark", "create_run_directory"):
        monkeypatch.setattr(radar, name, forbidden)
    calls = []
    monkeypatch.setattr(radar, "export_radar_figure", lambda output: calls.append(output) or {})
    monkeypatch.setattr(sys, "argv", ["radar", "--figure-only"])
    assert radar.main() == 0 and calls == [None]


@pytest.mark.parametrize("args", [
    ["--figure-output", "unused"], ["--figure-only", "--radar-path", "unused"],
    ["--figure-only", "--quick"], ["--figure-only", "--self-test-only"],
    ["--figure-only", "--controlled-gap"], ["--figure-only", "--save-support-arrays"],
    ["--figure-only", "--strict-sensitivity"], ["--figure-only", "--skip-timing"],
    ["--figure-only", "--run-id", "unused"], ["--figure-only", "--output-root", "unused"],
    ["--figure-only", "--repeats", "2"], ["--figure-only", "--warmups", "0"],
])
def test_rejects_mixed_validation_flags(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["radar", *args])
    monkeypatch.setattr(radar, "export_radar_figure", forbidden)
    with pytest.raises(SystemExit) as error:
        radar.main()
    assert error.value.code == 2


def test_results_output_is_rejected_before_computation(monkeypatch):
    monkeypatch.setattr(radar, "prepare_radar_figure", forbidden)
    with pytest.raises(ValueError, match="must not be inside saved experiment results"):
        radar.export_radar_figure(radar.FIGURE_RUN_DIRECTORY)
