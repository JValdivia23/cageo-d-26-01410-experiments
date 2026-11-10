"""One approved 3D wind case, four-panel mapping and Discussion diagnostics.

No benchmark, worker, API campaign or other CASES entry is executed. Small
in-memory filter spies exercise masks/component isolation, not new field cases.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import generate_3d_wind_boxcar_validation as wind


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    pytest.fail("Figure-only must not enter a worker, campaign, API validation or numerical result writer")


@pytest.fixture(scope="module")
def evidence():
    return wind._figure_saved_evidence()


@pytest.fixture(scope="module")
def prepared():
    with pytest.MonkeyPatch.context() as patch:
        for name in ("worker", "measure_worker", "calculate_toolkit_api", "prepare_backend",
                     "create_run_directory", "write_csv", "write_settings"):
            patch.setattr(wind, name, forbidden)
        calls, generated = [], []
        calculate, generate = wind.calculate, wind.generate_input

        def checked_calculate(method, observed, width, backend, *args, **kwargs):
            calls.append((method, observed[0].shape, width))
            return calculate(method, observed, width, backend, *args, **kwargs)

        def checked_generate(case):
            generated.append(case)
            return generate(case)

        patch.setattr(wind, "calculate", checked_calculate)
        patch.setattr(wind, "generate_input", checked_generate)
        result = wind.prepare_wind_figure()
        assert generated == [wind.CASES["baseline20"]]
        assert calls == [("numpy_windows", (18, 96, 96), (3, 7, 7)),
                         (wind.SPECTRAL, (18, 96, 96), (3, 7, 7))]
        return result


@pytest.fixture(scope="module")
def drawing(prepared):
    fig = wind.create_wind_figure(prepared[0])
    yield fig
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_import_and_ordinary_cli_path_remain_plotting_free():
    code = """
import sys
import generate_3d_wind_boxcar_validation as wind
assert not any(n in sys.modules for n in ('_review_figures', 'matplotlib', 'PIL'))
original_load = wind.load_toolkit
wind.export_wind_figure = lambda *args: (_ for _ in ()).throw(AssertionError('figure export'))
wind.load_toolkit = lambda: (_ for _ in ()).throw(AssertionError('unneeded toolkit import'))
sys.argv = ['wind', '--list-cases']
assert wind.main() == 0
sys.argv = ['wind', '--worker', '--cases', 'smoke', '--methods', 'numpy_windows']
wind.worker = lambda args: 0
assert wind.main() == 0
# Enter the ordinary benchmark branch through the real pinned-source import,
# but stop before any run creation, data generation or measured calculation.
wind.load_toolkit = original_load
wind.create_run_directory = lambda *args: (_ for _ in ()).throw(RuntimeError('before run creation'))
sys.argv = ['wind', '--mode', 'benchmark', '--cases', 'baseline20']
try:
    wind.main()
except RuntimeError as error:
    assert str(error) == 'before run creation'
else:
    raise AssertionError('ordinary branch did not stop')
assert not any(n in sys.modules for n in ('_review_figures', 'matplotlib', 'PIL'))
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=SCRIPTS, check=True, capture_output=True)


def test_full_approved_input_identity_versions_and_defined_counts(prepared, evidence):
    case, settings = prepared
    saved = evidence[0]
    full_hash = hashlib.sha256()
    for data in case["observed"]:
        assert data.shape == (18, 96, 96) and data.dtype == np.float64
        full_hash.update(memoryview(data))
        np.testing.assert_array_equal(np.isfinite(data), case["valid"])
    assert full_hash.hexdigest() == saved["input_sha256"] == settings["input_sha256"] == (
        "b831f7c4bd3d8e1e563793aedd96c1dba012baa41a720d7f7405fb79fc0afbeb")
    assert case["valid"].sum() == 132407
    assert settings["actual_missing_fraction"] == saved["actual_missing_fraction"] == "0.2018289448302469"
    assert case["accepted"].sum() == settings["n_accepted_full_volume_per_component"] == 165888
    assert settings["n_accepted_at_missing_full_volume_per_component"] == 33481
    assert settings["n_observed_slice"] == 7360 and settings["n_accepted_slice"] == 9216
    for key in ("shape_zyx", "width_zyx", "window_volume", "kernel", "boundary", "noise_seed",
                "outlier_seed", "missing_seed", "field_definition_sha256", "velocity_units"):
        assert settings[key] == saved[key]
    for key in ("numpy_version", "scipy_version", "astropy_version", "python_version", "dct_toolkit_version"):
        assert settings[f"saved_{key}"] == saved[key]
    assert settings["saved_original_script_sha256"] == saved["script_sha256"] == wind.FIGURE_ORIGINAL_SCRIPT_SHA256
    assert settings["script_sha256"] == digest(Path(wind.__file__))
    assert settings["original_script_git_commit"] == "c66106488a4e17bfb45260f4764106c5682254de"
    assert settings["dct_toolkit_commit"] == "9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c"
    assert len([k for k in settings if k.startswith("toolkit_source_sha256.")]) == 9
    assert settings["completion"] is settings["stable_fallback"] is settings["restoration"] is False
    assert settings["new_timing_performed"] is settings["new_api_validation_performed"] is False
    assert settings["injection_mask_used_for_moments_or_flags"] is False
    assert settings["iterative_rejection_or_reweighting_performed"] is False
    assert settings["diagnostic_threshold_calibration_performed"] is False
    assert settings["diagnostic_self_influence_mitigation_performed"] is False


def test_shared_raw_coverage_accepted_masks_and_component_agreement(prepared, evidence):
    case, settings = prepared
    raw = case["raw_density"]
    accepted = np.isfinite(raw) & (raw > 1e-3)
    np.testing.assert_array_equal(case["accepted"], accepted)
    np.testing.assert_array_equal(accepted, np.isfinite(case["numpy_raw_density"]) & (case["numpy_raw_density"] > 1e-3))
    for result, density in ((case["dct"], raw), (case["numpy"], case["numpy_raw_density"])):
        for name in wind.COMPONENTS:
            assert result[name]["support"] is result["u"]["support"]
            np.testing.assert_array_equal(result[name]["support"], 147 * np.clip(density, 0, 1))
            for statistic in ("mean", "variance", "std"):
                np.testing.assert_array_equal(np.isfinite(result[name][statistic]), accepted)
            np.testing.assert_array_equal(result[name]["std"], np.sqrt(result[name]["variance"]))
    actual = wind.compare_outputs(wind.FIGURE_RUN_ID, wind.FIGURE_CASE_ID, wind.SPECTRAL,
                                  case["dct"], case["numpy"], case["observed"])
    frozen = [r for r in evidence[1]["comparison.csv"] if r["method"] == wind.SPECTRAL]
    assert len(actual) == len(frozen) == settings["n_regional_comparisons_verified"] == 42
    wind._figure_check_comparisons(actual, frozen)
    assert all(row["status"] in ("pass", "no_comparable") for row in actual)
    for name, flagged in zip(wind.COMPONENTS, (378, 357, 508)):
        assert settings[f"application.{name}.n_eligible"] == 132407
        assert settings[f"application.{name}.n_flagged"] == flagged
    assert not np.array_equal(case["dct"]["u"]["mean"], case["dct"]["v"]["mean"])
    assert not np.array_equal(case["dct"]["u"]["std"], case["dct"]["v"]["std"])


def test_full_3d_window_precedes_slice_not_2d_statistics(prepared):
    case = prepared[0]
    z, y, x = 9, 48, 48
    for name, observed in zip(wind.COMPONENTS, case["observed"]):
        window = observed[z - 1:z + 2, y - 3:y + 4, x - 3:x + 4]
        retained = window[np.isfinite(window)]
        expected = case["numpy"][name]
        assert expected["mean"][z, y, x] == pytest.approx(retained.mean(), abs=2e-14)
        assert expected["std"][z, y, x] == pytest.approx(retained.std(ddof=0), abs=2e-14)
        assert expected["support"][z, y, x] == pytest.approx(retained.size)
        plane = observed[z, y - 3:y + 4, x - 3:x + 4]
        # The plotted full-volume statistic cannot be a 7x7 plane statistic.
        assert abs(np.nanmean(plane) - expected["mean"][z, y, x]) > 1e-4
    sliced = wind._figure_slice_fields(case)
    np.testing.assert_array_equal(sliced["scalars"][0], case["dct"]["w"]["mean"][9])
    np.testing.assert_array_equal(sliced["scalars"][1], case["dct"]["w"]["std"][9])
    for array, name in zip(sliced["vectors"], ("u", "v")):
        np.testing.assert_array_equal(array, case["dct"][name]["mean"][9])


def test_presentation_revision_preserves_all_full_volume_arrays(prepared):
    previous = subprocess.check_output([
        "git", "show", "f27468f:manuscript/figures/round1/3d_wind_boxcar/figure_settings.csv",
    ], cwd=SCRIPTS.parents[1]).decode()
    old = {r["setting"]: r["value"] for r in csv.DictReader(io.StringIO(previous))}
    settings = prepared[1]
    keys = [k for k in old if k.startswith(("full3d_", "observed_")) and k.endswith("_sha256")]
    assert len(keys) == 15
    for key in keys + ["input_sha256", "common_mask_sha256", "accepted_mask_sha256"]:
        assert settings[key] == old[key]


def test_component_diagnostics_partition_frozen_flags_not_missing_injections(prepared, evidence):
    case, settings = prepared
    diagnostic = wind._figure_screening_diagnostics(case)
    assert case["injected"].sum() == 521
    assert (case["injected"] & case["valid"]).sum() == 406
    assert (case["injected"] & ~case["valid"]).sum() == 115
    saved = {r["component"]: r for r in evidence[1]["application_counts.csv"]}
    for name, expected in zip(wind.COMPONENTS, ((286, 92, 120), (270, 87, 136), (290, 218, 116))):
        counts, masks = diagnostic[name]["counts"], diagnostic[name]["masks"]
        assert tuple(counts[f"n_{k}"] for k in ("tp", "fp", "fn")) == expected
        assert counts["n_tp"] + counts["n_fn"] == int(saved[name]["n_eligible_injected"]) == 406
        assert counts["n_tp"] + counts["n_fp"] == int(saved[name]["n_flagged"])
        assert counts["n_eligible"] == int(saved[name]["n_eligible"]) == 132407
        for key, mask in masks.items():
            assert not (mask & ~case["valid"]).any()
            assert counts[f"n_{key}"] == settings[f"diagnostic.{name}.n_{key}"]
            assert hashlib.sha256(mask.tobytes()).hexdigest() == settings[f"diagnostic.{name}.{key}_sha256"]
        np.testing.assert_array_equal(masks["tp"] | masks["fp"], masks["flagged"])
        np.testing.assert_array_equal(masks["tp"] | masks["fn"], masks["eligible"] & case["injected"])
        assert not (masks["tp"] & masks["fp"]).any()
        assert not (masks["tp"] & masks["fn"]).any()


def test_diagnostic_ineligible_injections_are_not_false_negatives(prepared):
    case = dict(prepared[0])
    case["dct"] = {c: {s: a.copy() for s, a in fields.items()} for c, fields in case["dct"].items()}
    positions = np.argwhere(case["injected"] & case["valid"])[:4]
    for name in wind.COMPONENTS:
        stats = case["dct"][name]
        stats["support"][tuple(positions[0])] = 147 * (wind.ANOMALY_DENSITY_MIN - 0.01)
        stats["std"][tuple(positions[1])] = wind.ANOMALY_STD_MIN
        stats["std"][tuple(positions[2])] = np.nan
        stats["mean"][tuple(positions[3])] = np.nan
    for item in wind._figure_screening_diagnostics(case).values():
        for mask in item["masks"].values():
            assert not mask[tuple(positions.T)].any()
        assert item["counts"]["n_tp"] + item["counts"]["n_fn"] == 402


def test_injection_labels_do_not_change_flags_or_moment_arrays(prepared):
    case = dict(prepared[0])
    before = {f"{c}/{s}": a.tobytes() for c, fields in case["dct"].items() for s, a in fields.items()}
    original = wind._figure_screening_diagnostics(case)
    case["injected"] = np.zeros_like(case["injected"])
    changed = wind._figure_screening_diagnostics(case)
    for name in wind.COMPONENTS:
        np.testing.assert_array_equal(changed[name]["masks"]["flagged"], original[name]["masks"]["flagged"])
        assert changed[name]["counts"]["n_tp"] == changed[name]["counts"]["n_fn"] == 0
        assert changed[name]["counts"]["n_fp"] == original[name]["counts"]["n_flagged"]
        for statistic, array in case["dct"][name].items():
            assert array.tobytes() == before[f"{name}/{statistic}"]


def test_unchanged_seven_filter_code_isolates_components_and_raw_strict_floor(prepared, monkeypatch):
    observed = prepared[0]["observed"]
    density = np.ones(observed[0].shape)
    density[9, 3, 3:9] = [0, 1e-3, np.nextafter(1e-3, np.inf), np.nan, np.inf, -0.1]
    calls = []

    def prepared_filter(method, shape, width, backend, chunk_planes):
        assert shape == (18, 96, 96) and width == (3, 7, 7)

        def linear(a):
            calls.append(a.copy())
            return density.copy() if len(calls) == 1 else np.zeros(shape)

        return linear

    monkeypatch.setattr(wind, "prepare_filter", prepared_filter)
    audit = {}
    output = wind.calculate(wind.SPECTRAL, observed, (3, 7, 7), None, audit=audit)
    assert audit == dict(full_nd_filter_calls=7, filter_preparations=1)
    common = wind.common_valid_mask(observed)
    np.testing.assert_array_equal(calls[0], common.astype(float))
    for index, (name, data) in enumerate(zip(wind.COMPONENTS, observed)):
        filled = np.where(common, data, 0)
        np.testing.assert_array_equal(calls[1 + 2 * index], filled)
        np.testing.assert_array_equal(calls[2 + 2 * index], filled * filled)
        mask = np.isfinite(density) & (density > 1e-3)
        for statistic in ("mean", "variance", "std"):
            np.testing.assert_array_equal(np.isfinite(output[name][statistic]), mask)
        assert output[name]["support"] is output["u"]["support"]
    # A lone component NaN excludes that cell from the shared mask, not just u.
    different_mask = [a.copy() for a in observed]
    position = tuple(np.argwhere(common)[0])
    assert all(np.isfinite(a[position]) for a in observed)
    different_mask[0][position] = np.nan
    assert not wind.common_valid_mask(different_mask)[position]


@pytest.mark.parametrize("key,value", [
    ("script_sha256", "wrong"), ("field_definition_sha256", "wrong"), ("noise_seed", "4100"),
    ("outlier_seed", "0"), ("missing_seed", "0"), ("width_zyx", "[1, 7, 7]"),
    ("shape_zyx", "[1, 96, 96]"), ("component_axis_filtered", "True"),
    ("stable_fallback", "True"), ("restore_input_nan", "True"), ("denominator_floor", "0.01"),
    ("dct_toolkit_commit", "wrong"), ("boundary", "periodic"), ("absolute_tolerances", "{}"),
])
def test_rejects_changed_scientific_settings(evidence, key, value):
    with pytest.raises(ValueError, match=key):
        wind._figure_check_settings(dict(evidence[0], **{key: value}))


@pytest.mark.parametrize("filename", wind.FIGURE_SAVED_CSV_SHA256)
def test_modified_frozen_evidence_is_rejected_before_generation(tmp_path, monkeypatch, filename):
    for name in wind.FIGURE_SAVED_CSV_SHA256:
        (tmp_path / name).write_bytes((wind.FIGURE_RUN_DIRECTORY / name).read_bytes())
    with (tmp_path / filename).open("ab") as stream:
        stream.write(b"\n")
    monkeypatch.setattr(wind, "generate_input", forbidden)
    monkeypatch.setattr(wind, "load_toolkit", forbidden)
    with pytest.raises(ValueError, match=f"SHA-256 mismatch: {filename}"):
        wind.prepare_wind_figure(tmp_path)


def test_corrupt_full_input_hash_is_rejected_before_calculation(prepared, evidence, monkeypatch):
    case = prepared[0]
    monkeypatch.setattr(wind, "_figure_saved_evidence", lambda *args: evidence)
    monkeypatch.setattr(wind, "load_toolkit", lambda: object())
    monkeypatch.setattr(wind, "generate_input", lambda *args: (case["observed"], None, {"input_sha256": "wrong"}))
    monkeypatch.setattr(wind, "calculate", forbidden)
    with pytest.raises(ValueError, match="input_sha256"):
        wind.prepare_wind_figure()


def test_rejects_nonmatching_comparisons_and_undefined_masks(prepared, evidence):
    case = prepared[0]
    actual = wind.compare_outputs(wind.FIGURE_RUN_ID, wind.FIGURE_CASE_ID, wind.SPECTRAL,
                                  case["dct"], case["numpy"], case["observed"])
    frozen = [r for r in evidence[1]["comparison.csv"] if r["method"] == wind.SPECTRAL]
    for key, value in (("n_compared", 1), ("n_method_undefined", 1), ("status", "fail"), ("max_abs_difference", 0.1)):
        changed = [dict(row) for row in actual]
        changed[0][key] = value
        with pytest.raises(ValueError, match="comparison|discrepancy"):
            wind._figure_check_comparisons(changed, frozen)


def test_exact_four_panel_mapping_cell_faces_color_scales_and_vector_arrows(drawing, prepared):
    from matplotlib.quiver import QuiverKey
    from matplotlib.text import Text

    case = prepared[0]
    sliced = wind._figure_slice_fields(case)
    assert len(drawing.axes) == 7 and tuple(drawing.get_size_inches()) == (6.5, 7.0)
    assert [ax.name for ax in drawing.axes[:4]] == ["3d", "3d", "rectilinear", "rectilinear"]
    assert min(t.get_fontsize() for t in drawing.findobj(Text)
               if t.get_visible() and t.get_text().strip()) >= 10.4
    np.testing.assert_array_equal(sliced["x"], np.linspace(-1, 1, 96))
    np.testing.assert_array_equal(sliced["y"], np.linspace(-1, 1, 96))
    assert sliced["x_edges"][0] == pytest.approx(-1 - 1 / 95)
    assert sliced["x_edges"][-1] == pytest.approx(1 + 1 / 95)
    sample = slice(3, None, 6)
    xx, yy = np.meshgrid(sliced["x"][sample], sliced["y"][sample])
    expected_limits = [(-0.3, 0.3), (0, 0.4)]
    for index, (ax, expected, limits) in enumerate(zip(drawing.axes[2:4], sliced["scalars"], expected_limits)):
        assert ax.get_title(loc="left").startswith(f"({chr(99 + index)})")
        assert not ax.images and len(ax.collections) == (2 if index == 0 else 1)
        mesh = ax.collections[0]
        np.testing.assert_array_equal(mesh.get_array().filled(np.nan), expected)
        assert mesh.get_clim() == limits and mesh.get_rasterized()
        coords = mesh.get_coordinates()
        np.testing.assert_array_equal(coords[..., 0], np.broadcast_to(sliced["x_edges"], (97, 97)))
        np.testing.assert_array_equal(coords[..., 1], np.broadcast_to(sliced["y_edges"][:, None], (97, 97)))
        assert ax.get_xlim() == ax.get_ylim() == (sliced["x_edges"][0], sliced["x_edges"][-1])
        if index == 0:
            arrow = ax.collections[1]
            np.testing.assert_array_equal(arrow.X, xx.ravel())
            np.testing.assert_array_equal(arrow.Y, yy.ravel())
            mask = ~sliced["vector_mask"][sample, sample].ravel()
            np.testing.assert_array_equal(np.broadcast_to(arrow.Umask, mask.shape), mask)
            u, v = sliced["vectors"]
            np.testing.assert_array_equal(arrow.U[~mask], u[sample, sample].ravel()[~mask])
            np.testing.assert_array_equal(arrow.V[~mask], v[sample, sample].ravel()[~mask])
            assert arrow.scale == 10 and arrow.scale_units == "inches" and arrow.angles == "xy"
            assert not arrow.get_rasterized()
    key, = drawing.findobj(QuiverKey)
    assert key.U == 1 and key.label == r"1 unit: $(\mu_u,\mu_v)$"
    assert key.text.get_fontsize() >= 10.4
    assert drawing.axes[-3]._colorbar.extend == "max"
    assert drawing.axes[-3]._colorbar.mappable.get_cmap().name == "viridis"
    assert drawing.axes[-3]._colorbar.mappable.get_clim() == (0, 1.6)
    assert drawing.axes[-2]._colorbar.extend == drawing.axes[-1]._colorbar.extend == "neither"
    wind._figure_check_plot(drawing, case)
    assert [t.get_text() for t in drawing.legends[0].get_texts()] == [
        "Injected (sample)", "Missing (sample)", "Section z = 0.0588"]
    for name in ("mean_w", "std_w"):
        assert prepared[1][f"slice_{name}_below_color_limit_count"] == 0
        assert prepared[1][f"slice_{name}_above_color_limit_count"] == 0


def test_3d_sampling_observed_injections_common_lattice_and_fixed_arrows(drawing, prepared):
    case, settings = prepared
    sampled = wind._figure_volume_fields(case)
    expected = np.array([(z, y, x) for z in range(1, 18, 3)
                         for y in range(4, 96, 8) for x in range(4, 96, 8)])
    np.testing.assert_array_equal(sampled["indices"], expected)
    grid = [np.linspace(-1, 1, n) for n in (18, 96, 96)]
    expected_positions = np.column_stack([grid[i][expected[:, i]] for i in (2, 1, 0)])
    np.testing.assert_array_equal(sampled["positions"], expected_positions)
    for column, name in enumerate(wind.COMPONENTS):
        np.testing.assert_array_equal(sampled["observed"][:, column], case["observed"][column][tuple(expected.T)])
        np.testing.assert_array_equal(sampled["mean"][:, column], case["dct"][name]["mean"][tuple(expected.T)])
    assert len(expected) == settings["volume_n_lattice_positions"] == 864
    assert sampled["observed_mask"].sum() == settings["volume_n_observed_arrows"]
    assert sampled["mean_mask"].sum() == settings["volume_n_mean_arrows"] == 864
    candidates = np.argwhere(case["injected"] & case["valid"])
    np.testing.assert_array_equal(sampled["injection_indices"], candidates[np.linspace(0, 405, 100, dtype=int)])
    assert settings["injection_display_count"] == 100
    assert (case["injected"] & case["valid"])[tuple(sampled["injection_indices"].T)].all()
    for ax, name in zip(drawing.axes[:2], ("observed", "mean")):
        assert ax.elev == 22 and ax.azim == 35
        assert ax.get_xlim() == ax.get_ylim() == ax.get_zlim() == (-1, 1)
        assert ax.get_title(loc="left").startswith("(a)" if name == "observed" else "(b)")
        collection = next(a for a in ax.collections if a.get_gid() == f"{name}_uvw")
        keep = sampled[f"{name}_mask"]
        shafts = np.asarray(collection._segments3d)[:int(keep.sum())]
        np.testing.assert_allclose(shafts[:, 1], expected_positions[keep], rtol=0, atol=2e-15)
        np.testing.assert_allclose(np.linalg.norm(shafts[:, 0] - shafts[:, 1], axis=1), 0.11, rtol=0, atol=2e-15)
        assert not collection.get_rasterized()
        n = int(keep.sum())
        segments = np.asarray(collection._segments3d)
        colors = collection.get_edgecolors()
        for start in (n, 2 * n):
            np.testing.assert_allclose(segments[start:start + n, 0], shafts[:, 0], rtol=0, atol=2e-15)
            np.testing.assert_array_equal(colors[start:start + n], colors[:n])
        line, = ax.lines
        np.testing.assert_array_equal(line.get_data_3d()[2], np.full(5, grid[0][9]))
    wind._figure_check_plot(drawing, case)


def test_undefined_moments_remain_gray_and_have_no_mean_arrows(prepared):
    # Drawing-only mask fixture derived from the verified arrays; no filter rerun.
    case = dict(prepared[0])
    case["dct"] = {c: {s: a.copy() for s, a in fields.items()} for c, fields in case["dct"].items()}
    case["accepted"] = case["accepted"].copy()
    for position in ((9, 3, 3), (10, 4, 4)):
        case["accepted"][position] = False
        for component in wind.COMPONENTS:
            for statistic in ("mean", "variance", "std"):
                case["dct"][component][statistic][position] = np.nan
    fig = wind.create_wind_figure(case)
    try:
        for index in (2, 3):
            assert fig.axes[index].collections[0].get_array().mask[3, 3]
        assert fig.axes[2].collections[1].Umask[0]
        wind._figure_check_plot(fig, case)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_export_writes_only_assets_and_sidecar_with_hashes_300dpi_and_fonts(tmp_path, monkeypatch, drawing, prepared):
    from PIL import Image

    monkeypatch.setattr(wind, "prepare_wind_figure", lambda: (prepared[0], dict(prepared[1])))
    monkeypatch.setattr(wind, "create_wind_figure", lambda case: drawing)
    for name in ("worker", "measure_worker", "create_run_directory", "write_csv", "write_settings"):
        monkeypatch.setattr(wind, name, forbidden)
    before = {name: digest(wind.FIGURE_RUN_DIRECTORY / name) for name in wind.FIGURE_SAVED_CSV_SHA256}
    paths = wind.export_wind_figure(tmp_path)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()) == [
        "figure_settings.csv", "pdf/wind_component_statistics.pdf", "png/wind_component_statistics.png"]
    with paths["settings"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["figure", "setting", "value"]
        rows = list(reader)
    settings = {r["setting"]: r["value"] for r in rows}
    assert len(settings) == len(rows) and {r["figure"] for r in rows} == {wind.FIGURE_STEM}
    assert settings["checks"].startswith("PASS:") and settings["plot_checks"].startswith("PASS:")
    assert settings["selected_case"] == "baseline20" and settings["selected_run"] == wind.FIGURE_RUN_ID
    assert settings["script_sha256"] == digest(Path(wind.__file__))
    assert settings["figure_helper_sha256"] == digest(SCRIPTS / "_review_figures.py")
    assert settings["slice_index_zero_based"] == "9" and float(settings["slice_normalized_z"]) == pytest.approx(1 / 17)
    assert settings["completion"] == settings["restoration"] == settings["new_timing_performed"] == "False"
    assert json.loads(settings["mean_w_color_limits"]) == [-0.3, 0.3]
    assert json.loads(settings["std_w_color_limits"]) == [0, 0.4]
    assert settings["w_colorbar_extend"] == "neither"
    assert settings["volume_colorbar_extend"] == "max"
    assert settings["diagnostic.u.n_tp"] == "286"
    assert settings["diagnostic.v.n_fp"] == "87"
    assert settings["diagnostic.w.n_fn"] == "116"
    assert float(settings["min_text_size_pt"]) >= 10.4 and settings["pdf_fonttype"] == "42"
    assert settings["figure_width_in"] == "6.5" and settings["figure_height_in"] == "7.0"
    for name, old in before.items():
        assert settings[f"saved_{name}_sha256"] == digest(wind.FIGURE_RUN_DIRECTORY / name) == old
    for kind in ("pdf", "png"):
        assert settings[f"{kind}_sha256"] == digest(paths[kind])
    with Image.open(paths["png"]) as image:
        assert image.size == (1950, 2100)
        assert all(abs(v - 300) < 0.01 for v in image.info["dpi"])
        assert image.getpixel((0, 0)) == (255, 255, 255, 255)
    pdf = paths["pdf"].read_bytes()
    assert b"/FontFile2" in pdf and b"/Subtype /Type3" not in pdf and b"/Subtype /Image" in pdf


def test_early_figure_dispatch_default_and_no_ignored_case_options(monkeypatch):
    for name in ("load_toolkit", "generate_input", "calculate", "worker", "create_run_directory"):
        monkeypatch.setattr(wind, name, forbidden)
    calls = []
    monkeypatch.setattr(wind, "export_wind_figure", lambda output: calls.append(output) or {})
    monkeypatch.setattr(sys, "argv", ["wind", "--figure-only"])
    assert wind.main() == 0 and calls == [None]


@pytest.mark.parametrize("args", [
    ["--figure-output", "unused"], ["--figure-only", "--run-id", "unused"],
    ["--figure-only", "--output-root", "unused"], ["--figure-only", "--mode", "all"],
    ["--figure-only", "--cases", "baseline50"], ["--figure-only", "--cases", "baseline20"],
    ["--figure-only", "--methods", "numpy_windows"], ["--figure-only", "--repeats", "1"],
    ["--figure-only", "--max-cases", "1"], ["--figure-only", "--max-voxels", "0"],
    ["--figure-only", "--max-window-volume", "0"], ["--figure-only", "--numpy-chunk-planes", "2"],
    ["--figure-only", "--timeout-s", "0"], ["--figure-only", "--max-ram-mib", "0"],
    ["--figure-only", "--list-cases"], ["--figure-only", "--worker"],
])
def test_rejects_mixed_figure_numerical_or_benchmark_options(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["wind", *args])
    monkeypatch.setattr(wind, "export_wind_figure", forbidden)
    with pytest.raises(SystemExit) as error:
        wind.main()
    assert error.value.code == 2


def test_results_output_is_rejected_before_reproduction(monkeypatch):
    monkeypatch.setattr(wind, "prepare_wind_figure", forbidden)
    with pytest.raises(ValueError, match="must not be inside saved experiment results"):
        wind.export_wind_figure(wind.FIGURE_RUN_DIRECTORY)


def test_manuscript_diagnostics_and_robust_extensions_match_verified_scope(prepared):
    root = SCRIPTS.parents[1]
    text = (root / "manuscript/main.tex").read_text()
    discussion = text.split("\\section{Discussion}")[1].split("\\section{Conclusion}")[0]
    for name in wind.COMPONENTS:
        counts = prepared[1]
        triple = ",".join(str(counts[f"diagnostic.{name}.n_{key}"]) for key in ("tp", "fp", "fn"))
        assert f"$({triple})$ for ${name}$" in discussion
    for phrase in ("115 missing injections are excluded", "separate component diagnostics",
                   "not an optimal or independently calibrated threshold", "original availability mask",
                   "No iterative rejection, robust reweighting, or independent detector evaluation was performed",
                   "screening bands are not confidence intervals", "\\citet{Garcia2010}", "\\citet{Vitale2021}"):
        assert phrase in discussion
    caption = text.split("\\caption{Observed wind and local statistics")[1].split("\\label{fig:wind_statistics}")[0]
    assert "not detected anomalies" in caption and "not mean wind speed" in caption
    assert "fixed display length" in caption and "uncertainty in the mean" in caption
    bibliography = (root / "manuscript/references.bib").read_text()
    for key, doi in (("Garcia2010", "10.1016/j.csda.2009.09.020"), ("Vitale2021", "10.1038/s41598-021-91002-y")):
        assert bibliography.count(f"@article{{{key},") == 1
        assert doi in bibliography.split(f"@article{{{key},")[1].split("\n}")[0]
