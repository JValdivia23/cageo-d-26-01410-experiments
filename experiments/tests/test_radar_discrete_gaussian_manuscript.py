"""Selected Gaussian manuscript/figure evidence; no new reference or benchmark."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import generate_2d_radar_discrete_gaussian_figure as figure


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    pytest.fail("Figure reproduction must not run references, validation or timing campaigns")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def prepared():
    radar = figure.experiment.base.radar
    path = Path(os.environ.get("REAL_RADAR_PATH", str(radar.DEFAULT_RADAR))).expanduser()
    if not path.is_file():
        pytest.skip("Exact processed radar input is not available locally")
    with pytest.MonkeyPatch.context() as patch:
        for name in ("centered_reference", "validate_case", "kernel_checks", "measure_worker", "worker", "main"):
            patch.setattr(figure.experiment, name, forbidden)
        return figure.prepare_figure()


@pytest.fixture(scope="module")
def drawing(prepared):
    fig = figure.create_figure(prepared[0])
    yield fig
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_generator_import_has_no_plotting_or_input_side_effects():
    code = """
import sys
import generate_2d_radar_discrete_gaussian_figure
assert not any(n in sys.modules for n in ('matplotlib', 'xarray', 'cmweather', '_review_figures'))
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=SCRIPTS, check=True)


@pytest.mark.parametrize("name", figure.SAVED_SHA256)
def test_changed_evidence_is_rejected_before_radar_io(name, tmp_path, monkeypatch):
    monkeypatch.setattr(figure.experiment.base.radar, "read_radar", forbidden)
    for filename in figure.SAVED_SHA256:
        (tmp_path / filename).write_bytes((figure.RUN_DIRECTORY / filename).read_bytes())
    with (tmp_path / name).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match=f"SHA-256 mismatch: {name}"):
        figure.prepare_figure(tmp_path)


def test_changed_numerical_source_is_rejected_before_radar_io(monkeypatch):
    monkeypatch.setattr(figure.experiment.base.radar, "read_radar", forbidden)
    monkeypatch.setattr(figure.experiment, "source_hashes", lambda: {"experiment": "wrong"})
    with pytest.raises(ValueError, match="numerical sources changed"):
        figure.prepare_figure()


def test_output_cannot_overwrite_numerical_results(monkeypatch):
    monkeypatch.setattr(figure, "prepare_figure", forbidden)
    with pytest.raises(ValueError, match="not be inside saved numerical results"):
        figure.export(figure.RUN_DIRECTORY / "figures")


def test_required_palette_has_no_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "cmweather.cm_colorblind", None)
    monkeypatch.setattr(figure.experiment.base.radar, "read_radar", forbidden)
    with pytest.raises(RuntimeError, match="requires cmweather"):
        figure.prepare_figure()


def test_figure_arrays_are_the_saved_discrete_spectral_outputs_not_analytical(prepared):
    case, settings = prepared
    frozen, _ = figure.saved_evidence()
    saved = json.loads(frozen["output_sha256"])
    actual = figure.experiment.base.fingerprints(case["fields"])
    assert actual == saved["discrete_spectral"] and actual != saved["analytical_spectral"]
    assert case["data"].shape == (720, 1180)
    assert case["accepted"].sum() == settings["n_accepted"] == 640682
    assert case["accepted"][:, case["central"]].sum() == settings["central_n_accepted"] == 90140
    assert case["negative"].sum() == settings["n_negative_raw_variance"] == 66
    assert settings["raw_variance_finite_min"] == pytest.approx(-1.8115642319571634e-10, abs=1e-20)
    assert settings["negative_overlay"].startswith("none;")
    assert settings["new_timing_performed"] is settings["new_spatial_reference_performed"] is False
    assert settings["completion"] is settings["restoration"] is False
    assert settings["dct_toolkit_commit"] == figure.experiment.base.TOOLKIT_COMMIT
    assert len([key for key in settings if key.startswith("toolkit_source_sha256.")]) == 9
    raw = case["fields"]["raw_mask_response"]
    np.testing.assert_array_equal(case["accepted"], np.isfinite(raw) & (raw > 1e-3))
    np.testing.assert_array_equal(case["fields"]["variance"], np.maximum(case["fields"]["raw_variance"], 0))
    np.testing.assert_array_equal(case["fields"]["std"], np.sqrt(case["fields"]["variance"]))
    assert np.nanmin(case["fields"]["mean"]) >= np.nanmin(case["data"]) - 1e-9
    assert np.nanmax(case["fields"]["mean"]) <= np.nanmax(case["data"]) + 1e-9


def test_square_zoom_uses_full_sweep_with_all_corners(prepared):
    from matplotlib.path import Path as PolygonPath

    case, settings = prepared
    az = np.deg2rad(case["azimuth"])[:, None]
    radius = case["ranges"][None, :] / 1000
    expected = (np.abs(radius * np.sin(az)) <= 10) & (np.abs(radius * np.cos(az)) <= 10)
    np.testing.assert_array_equal(case["zoom"], expected)
    assert settings["zoom_n_cells"] == 107268
    assert settings["zoom_n_accepted"] == 99961
    assert settings["zoom_n_undefined"] == 7307
    assert settings["zoom_n_observed"] == 64534
    assert settings["zoom_n_negative_raw_variance"] == 0
    count = case["zoom_gate_count"]
    assert count == 188 and count > case["central"].sum()
    perimeter = PolygonPath(np.column_stack((case["x"][:, count], case["y"][:, count])))
    assert perimeter.contains_points([[-10, -10], [-10, 10], [10, -10], [10, 10]]).all()
    assert not expected[:, count:].any()


def test_six_maps_preserve_style_and_display_only_new_arrays_without_red_overlays(drawing, prepared):
    from cmweather.cm_colorblind import ChaseSpectral
    from matplotlib.text import Text

    case = prepared[0]
    n = case["zoom_gate_count"]
    fields = case["fields"]
    rho = np.clip(fields["raw_mask_response"], 0, 1)
    expected = (case["data"], fields["mean"], fields["std"], rho, fields["std"][:, :n], rho[:, :n])
    assert len(drawing.axes) == 9 and not drawing.legends
    assert tuple(drawing.get_size_inches()) == (6.5, 7.4)
    for index, (ax, values) in enumerate(zip(drawing.axes[:6], expected)):
        assert len(ax.collections) == 1 and ax.get_legend() is None
        assert not ax.images
        mesh = ax.collections[0]
        np.testing.assert_array_equal(mesh.get_array().filled(np.nan), values)
        np.testing.assert_array_equal(mesh.get_coordinates()[..., 0], case["x"][:, :n+1] if index >= 4 else case["x"])
        assert mesh.get_rasterized() and mesh.get_gid() != "negative_raw_variance"
        assert ax.get_xlim() == ax.get_ylim() == ((-10, 10) if index >= 4 else (-90, 90))
        np.testing.assert_allclose(ax.get_position().size * drawing.get_size_inches(), [1.95, 1.95], atol=1e-12)
    samples = np.linspace(0, 1, 256)
    for ax in drawing.axes[:2]:
        mesh = ax.collections[0]
        np.testing.assert_array_equal(mesh.cmap(samples), ChaseSpectral(samples))
        assert mesh.get_clim() == (0, 40)
    assert drawing.axes[2].collections[0].get_clim() == drawing.axes[4].collections[0].get_clim() == (0, 10)
    assert drawing.axes[3].collections[0].get_clim() == drawing.axes[5].collections[0].get_clim() == (0, 1)
    assert drawing.axes[-2].get_ylabel() == "Std (dBZ)"
    assert drawing.axes[-2]._colorbar.extend == "max"
    assert all("Clamped" not in text.get_text() for text in drawing.findobj(Text))
    drawing.canvas.draw()
    renderer = drawing.canvas.get_renderer()
    for ax in drawing.axes[:6]:
        assert ax._left_title.get_window_extent(renderer).x1 <= ax.get_window_extent(renderer).x1 + .5


def test_export_reproduces_saved_assets_byte_for_byte_with_300dpi_vector_text(tmp_path, monkeypatch, prepared):
    from PIL import Image

    monkeypatch.setattr(figure, "prepare_figure", lambda *args: prepared)
    paths = figure.export(tmp_path)
    for kind, path in paths.items():
        original = figure.FIGURE_DIRECTORY / path.relative_to(tmp_path)
        assert path.read_bytes() == original.read_bytes(), kind
    with Image.open(paths["png"]) as image:
        assert image.size == (1950, 2220)
        assert all(abs(value - 300) < .01 for value in image.info["dpi"])
    pdf = paths["pdf"].read_bytes()
    assert b"/FontFile2" in pdf and b"/Subtype /Type3" not in pdf and b"/Subtype /Image" in pdf
    settings = {row["setting"]: row["value"] for row in read_rows(paths["settings"])}
    assert settings["selected_method"] == "discrete_spectral"
    assert float(settings["min_text_size_pt"]) >= 10.1
    for kind in ("pdf", "png"):
        assert settings[f"{kind}_sha256"] == digest(paths[kind])


def scientific(value):
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def manuscript():
    return (figure.ROOT / "manuscript/main.tex").read_text()


def test_previous_gaussian_four_row_summary_matches_independent_raw_variance_evidence():
    text = manuscript()
    compact = read_rows(figure.RUN_DIRECTORY / "agreement_summary.csv")
    comparisons = read_rows(figure.RUN_DIRECTORY / "comparison.csv")
    assert len(compact) == 4
    for row in compact:
        selected = [r for r in comparisons if r["method"].startswith("discrete_")
                    and r["case_id"] == row["case_id"] and r["region"] == row["region"]]
        for statistic, key in (("mean", "max_abs_mean_difference"), ("raw_variance", "max_abs_variance_difference")):
            assert float(row[key]) == max(float(r["max_abs_difference"]) for r in selected if r["statistic"] == statistic)
        assert row["n_defined"] == next(r["n_compared"] for r in selected if r["statistic"] == "mean")
    section = text.split("\\label{sec:radar_application}")[1].split("\\subsection{")[0]
    assert "67.63\\%" in section and "75.41\\%" in section
    assert "checked before clamping" in section
    assert "controlled-removal and floor-sensitivity tests of this Gaussian kernel were not performed" in section
    assert "2d_radar_discrete_gaussian/pdf/radar_gaussian_statistics.pdf" in section
    assert "Red in" not in section and "13,322" not in section and "4,593" not in section
    assert "1.84\\times10^{-5}" in section


def test_previous_gaussian_timings_rss_and_preparation_match_all_raw_repetitions():
    summaries = read_rows(figure.RUN_DIRECTORY / "benchmark_summary.csv")
    runs = read_rows(figure.RUN_DIRECTORY / "benchmark_runs.csv")
    phases = read_rows(figure.RUN_DIRECTORY / "benchmark_phases.csv")
    methods = ("discrete_spectral", "discrete_spatial", "analytical_spectral")
    for width in (5, 15):
        values = []
        for key, formatting in (("median_time_s", ".4f"), ("max_peak_ram_mib", ".1f")):
            for method in methods:
                summary = next(r for r in summaries if r["case_id"] == f"natural_W{width}" and r["method"] == method)
                selected = [r for r in runs if r["case_id"] == summary["case_id"] and r["method"] == method]
                actual = np.median([float(r["elapsed_s"]) for r in selected]) if key == "median_time_s" else max(float(r["peak_ram_mib"]) for r in selected)
                assert float(summary[key]) == actual
                values.append(format(actual, formatting))
        assert len(values) == 6
    for method in methods:
        preparation = [np.median([float(r["preparation_s"])*1000 for r in phases
                                 if r["case_id"] == f"natural_W{width}" and r["method"] == method]) for width in (5, 15)]
        assert all(value > 0 for value in preparation)


def test_gaussian_definition_and_release_provenance_are_distinct_without_positivity_labels():
    text = manuscript()
    assert "g_a[0]+2\\sum_{\\ell=1}^{q_a}g_a[\\ell]\\cos(\\ell\\omega_{a,k_a})" in text
    assert "q_a=\\lceil4\\sigma_{g,a}\\rceil" in text
    assert "matched four-sigma radar calculation remains in the accompanying experiment sources" in text
    assert "full sampled Gaussian default" in text
    assert r"\texttt{gaussian\_spectral}" in text
    assert "its end-to-end runtime was not measured here" in text
    assert "All numerical results reported here are based on released" not in text
    assert "conventional positive-weight Gaussian" not in text
    assert "signed Gaussian weights" not in text


def test_prior_analytical_figure_and_numerical_evidence_remain_unchanged():
    for name, expected in figure.SAVED_SHA256.items():
        assert digest(figure.RUN_DIRECTORY / name) == expected
    old = figure.ROOT / "manuscript/figures/round1/2d_radar_gaussian"
    for path in old.rglob("*"):
        if path.is_file():
            relative = path.relative_to(figure.ROOT)
            original = subprocess.check_output(["git", "show", f"c4924a4:{relative}"], cwd=figure.ROOT)
            assert path.read_bytes() == original
