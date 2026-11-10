"""Independent numerical, provenance, display and no-overwrite checks for Fig. 2."""

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import generate_1d_boundary_comparison as boundary
from _review_common import _RELEASE_HASHES, load_toolkit


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def file_hashes(directory):
    return {str(path.relative_to(directory)): digest(path) for path in directory.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def prepared():
    return boundary.prepare_comparison()


@pytest.fixture(scope="module")
def saved(prepared, tmp_path_factory):
    return boundary.save_evidence(prepared, "test-two-cases", tmp_path_factory.mktemp("boundary-evidence"))


@pytest.fixture
def drawing(prepared):
    fig = boundary.create_figure(prepared)
    yield fig
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_import_is_lazy_and_does_not_start_calculations_or_io():
    code = """
import sys
import generate_1d_boxcar_validation as inherited
import _review_common
def forbidden(*args, **kwargs):
    raise AssertionError('Import must not calculate or read evidence')
inherited.make_cases = forbidden
inherited.toolkit_moments = forbidden
_review_common.load_toolkit = forbidden
from pathlib import Path
Path.read_bytes = forbidden
Path.open = forbidden
Path.mkdir = forbidden
import generate_1d_boundary_comparison
assert not any(name in sys.modules for name in ('matplotlib', 'PIL', '_review_figures'))
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=SCRIPTS, check=True)


def test_exact_original_input_hashes_shared_noise_truth_and_accepted_sets(prepared):
    expected_hashes = (
        ("b9cd1f9cd5e947b143d04238a45b7fa865a650670383ac2f094129aeed00fbf7",
         "6d9c54dee5660c46886f32d80e57e9dd0ffa57ee0cd2a762b036d9c8e0c3a33a"),
        ("19bb742436b9636cbbbc80de9854a115f3f0ca5f213353aed578cdd680276d04",
         "9009d4d32c6ecf984688a63d16e7a7037439136c13744e66c6e26f17d595f10e"),
    )
    x = np.linspace(0, 10, 200)
    truth = 1.2 * x + 2.5 * np.sin(1.5 * x)
    noise_stream = np.random.SeedSequence([20261027, 0, 0]).spawn(2)[0]
    complete = truth + np.random.default_rng(noise_stream).normal(0, 1, 200)
    np.testing.assert_array_equal(prepared[0]["data"], complete)
    for i, (case, (input_hash, mask_hash)) in enumerate(zip(prepared, expected_hashes)):
        assert case["case_id"] == boundary.CASE_IDS[i]
        np.testing.assert_array_equal(case["truth"], truth)
        np.testing.assert_array_equal(case["t"], x)
        assert hashlib.sha256(case["data"].tobytes()).hexdigest() == input_hash
        missing = ~np.isfinite(case["data"])
        assert hashlib.sha256(missing.tobytes()).hexdigest() == mask_hash
        np.testing.assert_array_equal(case["data"][~missing], complete[~missing])
        expected = np.ones(200, dtype=bool)
        if i:
            expected[87:113] = False
            np.testing.assert_array_equal(np.flatnonzero(missing), np.arange(80, 120))
        for model in boundary.MODELS:
            result = case["outputs"][model]
            assert int(np.isfinite(result["mean"]).sum()) == (174 if i else 200)
            assert int((np.isfinite(result["mean"]) & missing).sum()) == (14 if i else 0)
            for statistic in ("mean", "variance", "std"):
                np.testing.assert_array_equal(np.isfinite(result[statistic]), expected)
            if model in boundary.BOUNDARIES:
                for statistic, (atol, rtol) in boundary.TOLERANCES.items():
                    np.testing.assert_allclose(result[statistic], case["references"][model][statistic],
                                               atol=atol, rtol=rtol, equal_nan=True)
                np.testing.assert_allclose(result["denominator"], case["references"][model]["denominator"],
                                           atol=5e-13 / 15, rtol=2e-13)


def test_only_two_selected_inputs_are_filtered(monkeypatch):
    produced, periodic, reflective = [], [], []
    make_cases = boundary.inherited.make_cases
    periodic_moments, reflective_moments = boundary.periodic_moments, boundary.inherited.toolkit_moments

    def tracked_cases(*args):
        assert args == ("full", 20261027)
        for case in make_cases(*args):
            produced.append(case[0])
            yield case

    def tracked_periodic(data, *args):
        periodic.append(hashlib.sha256(data.tobytes()).hexdigest())
        return periodic_moments(data, *args)

    def tracked_reflective(data, *args):
        reflective.append(hashlib.sha256(data.tobytes()).hexdigest())
        return reflective_moments(data, *args)

    monkeypatch.setattr(boundary.inherited, "make_cases", tracked_cases)
    monkeypatch.setattr(boundary, "periodic_moments", tracked_periodic)
    monkeypatch.setattr(boundary.inherited, "toolkit_moments", tracked_reflective)
    boundary.prepare_comparison()
    assert produced == [boundary.CASE_IDS[0], "trend_sine_seed0_random_nan_n200_m15", boundary.CASE_IDS[1]]
    assert periodic == reflective and len(periodic) == 2


def test_periodic_uses_released_explicit_linear_geometry_and_actual_rfft(monkeypatch):
    import dct_toolkit.polar as polar
    import dct_toolkit.stats as stats
    toolkit = load_toolkit()
    calls, transforms, api_controls = [], [], []
    smooth, rfft, api_mean = polar.smooth_polar, polar.scipy.fft.rfft, toolkit.dct_mean

    def forbidden(*args, **kwargs):
        pytest.fail("Periodic route must not use Cartesian/DCT or completion")

    def tracked_smooth(data, *args, **kwargs):
        calls.append(kwargs.copy())
        assert kwargs["axis_types"] == ["linear"]
        assert kwargs["boundary"] == ["periodic"]
        assert kwargs["width_units"] == "grid" and kwargs["filter_axes"] == [0]
        assert kwargs["kernel_type"] == "boxcar_discrete"
        return smooth(data, *args, **kwargs)

    def tracked_fft(*args, **kwargs):
        transforms.append(kwargs["axis"])
        return rfft(*args, **kwargs)

    def tracked_api(*args, **kwargs):
        api_controls.append(kwargs)
        return api_mean(*args, **kwargs)

    monkeypatch.setattr(stats, "smooth_cartesian", forbidden)
    monkeypatch.setattr(stats, "dct_prefill", forbidden)
    monkeypatch.setattr(polar.scipy.fft, "dct", forbidden)
    monkeypatch.setattr(polar.scipy.fft, "rfft", tracked_fft)
    monkeypatch.setattr(stats, "smooth_polar", tracked_smooth)
    monkeypatch.setattr(toolkit, "smooth_polar", tracked_smooth)
    monkeypatch.setattr(toolkit, "dct_mean", tracked_api)
    data = np.r_[np.arange(80.0), np.full(40, np.nan), np.arange(120.0, 200)]
    actual = boundary.periodic_moments(data, 15, toolkit)
    assert len(calls) == len(transforms) and len(calls) > 3 and set(transforms) == {0}
    assert api_controls[0]["coordinates"] == "polar"
    assert api_controls[0]["_stable_fallback"] is False
    assert api_controls[0]["restore_input_nan"] is False
    assert api_controls[0]["_denominator_floor"] == 1e-3
    assert np.isnan(actual["mean"][87:113]).all()


def manual_windows(data, width, model):
    result = {stat: np.full(data.shape, np.nan) for stat in ("mean", "variance", "std")}
    result["support"] = np.zeros(data.shape)
    result["denominator"] = np.zeros(data.shape)
    for i in range(data.size):
        sample, available = [], 0
        for delta in range(-(width // 2), width // 2 + 1):
            if model == "truncated" and not 0 <= i + delta < data.size:
                continue
            available += 1
            j = i + delta if model == "truncated" else (i + delta) % (data.size if model == "periodic" else 2 * data.size)
            if model == "reflective" and j >= data.size:
                j = 2 * data.size - 1 - j
            if np.isfinite(data[j]):
                sample.append(data[j])
        result["denominator"][i] = len(sample) / available
        result["support"][i] = width * len(sample) / available
        if len(sample) / available > 1e-3:
            mean = sum(sample) / len(sample)
            variance = sum((x - mean)**2 for x in sample) / len(sample)
            result["mean"][i], result["variance"][i], result["std"][i] = mean, variance, variance**0.5
    return result


@pytest.mark.parametrize("model", ("periodic", "reflective", "truncated"))
@pytest.mark.parametrize("width", (1, 3, 13))
@pytest.mark.parametrize("values", ([0., 1., 4., -2., 7.], [0., np.nan, np.inf, -np.inf, 2.],
                                   [0., np.nan, np.nan, np.nan, np.nan], [np.nan] * 5))
def test_manual_wrapped_reflected_multiwrap_missing_and_valid_zero_windows(model, width, values):
    data = np.array(values)
    expected = manual_windows(data, width, model)
    if model == "truncated":
        reference = actual = boundary.truncated_numpy_moments(data, width)
    else:
        reference = (boundary.wrapped_numpy_moments if model == "periodic" else boundary.inherited.numpy_moments)(data, width)
        actual = (boundary.periodic_moments if model == "periodic" else boundary.inherited.toolkit_moments)(data, width, load_toolkit())
    for result in (reference, actual):
        for statistic, (atol, rtol) in boundary.TOLERANCES.items():
            np.testing.assert_allclose(result[statistic], expected[statistic], atol=atol, rtol=rtol, equal_nan=True)
        np.testing.assert_allclose(result["denominator"], expected["denominator"], atol=5e-13 / width, rtol=2e-13)


def test_truncated_windows_use_actual_neighbors_without_padding_or_wrap(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Truncated windows must not pad the input")

    monkeypatch.setattr(np, "pad", forbidden)
    data = np.arange(200.0)
    result = boundary.truncated_numpy_moments(data, 15)
    assert result["mean"][0] == 3.5 and result["n_available"][0] == 8
    assert result["n_available"][-1] == 8 and np.all(result["n_available"][7:193] == 15)
    assert np.all(result["denominator"] == 1)  # Domain truncation is not missing observations.
    data[-1] = 1e12
    assert boundary.truncated_numpy_moments(data, 15)["mean"][0] == 3.5
    data[:7] = np.nan
    result = boundary.truncated_numpy_moments(data, 15)
    assert result["mean"][0] == 7 and result["variance"][0] == 0
    assert result["denominator"][0] == 1 / 8
    assert result["support"][0] == 15 / 8  # Nominal equivalent count, not one observation.


def test_release_strict_finite_raw_floor_no_epsilon_or_filling(monkeypatch):
    import dct_toolkit.stats as stats
    denominator = np.array([0, 1e-3, np.nextafter(1e-3, np.inf), np.nan, np.inf, -np.inf, -1, 0.5])
    returns = iter((np.full(8, 2.), denominator))
    monkeypatch.setattr(stats, "smooth_polar", lambda *args, **kwargs: next(returns))
    monkeypatch.setattr(stats, "dct_prefill", lambda *args, **kwargs: pytest.fail("No completion"))
    result = load_toolkit().dct_mean(np.ones(8), 15, coordinates="polar", **boundary.periodic_options(),
                                   _stable_fallback=False, _denominator_floor=1e-3, restore_input_nan=False)
    np.testing.assert_array_equal(np.isfinite(result), [False, False, True, False, False, False, False, True])
    assert result[2] == 2 / denominator[2] and result[-1] == 4


@pytest.mark.parametrize("field", ("mean", "variance", "std", "support", "denominator", "filled_mean"))
def test_fixed_checks_reject_changed_values_masks_or_filling(monkeypatch, field):
    original = boundary.periodic_moments

    def broken(*args):
        result = original(*args)
        if field == "filled_mean":
            result["mean"] = np.nan_to_num(result["mean"])
        else:
            result[field] = result[field] + 0.001
        return result

    monkeypatch.setattr(boundary, "periodic_moments", broken)
    with pytest.raises(ValueError, match="policy changed|agreement failed"):
        boundary.prepare_comparison()


@pytest.mark.parametrize("filename,match,key,value", [
    ("settings.csv", {"setting": "input_sha256"}, "value", "bad"),
    ("settings.csv", {"setting": "mask_sha256"}, "value", "bad"),
    ("settings.csv", {"setting": "master_seed"}, "value", "42"),
    ("settings.csv", {"setting": "std_atol"}, "value", "0.1"),
    ("coverage.csv", {"method": "toolkit_dct_direct", "region": "all"}, "n_direct_defined", "190"),
    ("comparison.csv", {"method": "toolkit_dct_direct", "region": "all", "statistic": "mean"}, "rmse", "0.1"),
])
def test_frozen_identity_settings_counts_numerics_are_checked(tmp_path, monkeypatch, filename, match, key, value):
    copied = tmp_path / boundary.FROZEN_RUN.name
    copied.mkdir()
    for name in ("settings.csv", "coverage.csv", "comparison.csv"):
        shutil.copyfile(boundary.FROZEN_RUN / name, copied / name)
    amend_csv(copied / filename, match, key, value)
    monkeypatch.setattr(boundary, "FROZEN_RUN", copied)
    with pytest.raises(ValueError, match="Frozen"):
        boundary.prepare_comparison()


def amend_csv(path, match, key, value):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, contents = reader.fieldnames, list(reader)
    row, = [row for row in contents if row["case_id"] == boundary.CASE_IDS[0]
            and all(row[k] == v for k, v in match.items())]
    row[key] = value
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(contents)


def test_new_evidence_schemas_count_policy_fixed_tolerances_and_nonoverwrite(prepared, saved):
    assert sorted(path.name for path in saved.iterdir()) == [
        "comparison.csv", "interior_agreement.csv", "profiles.csv", "settings.csv", "truth_errors.csv"]
    comparisons = rows(saved / "comparison.csv")
    assert len(comparisons) == 100
    assert {row["method"] for row in comparisons} == {boundary.METHODS[model] for model in boundary.BOUNDARIES}
    assert {row["reference_method"] for row in comparisons} == set(boundary.REFERENCES.values())
    assert all(row["status"] in ("pass", "no_comparable") for row in comparisons)
    for row in comparisons:
        atol, rtol = boundary.tolerance(row["statistic"])
        assert float(row["atol"]) == atol and float(row["rtol"]) == rtol
        assert row["n_method_gap_filled"] == row["n_reference_gap_filled"] == "0"
        if row["n_compared"] == "0":
            assert row["rmse"] == row["max_abs_difference"] == ""
    settings = rows(saved / "settings.csv")
    for case_id in boundary.CASE_IDS:
        values = {row["setting"]: row["value"] for row in settings if row["case_id"] == case_id}
        assert values["script_sha256"] == digest(boundary.__file__)
        assert json.loads(values["boundary"]) == ["periodic", "reflective"]
        assert json.loads(values["displayed_models"]) == list(boundary.MODELS)
        assert values["completion"] == values["restoration"] == "False"
        assert values["master_seed"] == "20261027" and values["replicate"] == "0"
        for name, value in _RELEASE_HASHES.items():
            assert values[f"toolkit_source_sha256.{name}"] == value
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert values[name] == "1"
    before = file_hashes(saved)
    assert boundary.verify_saved(prepared, saved) == saved
    with pytest.raises(FileExistsError):
        boundary.save_evidence(prepared, saved.name, saved.parent.parent)
    assert file_hashes(saved) == before


def test_truth_errors_independently_recomputed_from_saved_profiles(saved):
    profiles = rows(saved / "profiles.csv")
    errors = rows(saved / "truth_errors.csv")
    assert len(profiles) == 1200 and len(errors) == 30
    for case_id in boundary.CASE_IDS:
        means = []
        for model in boundary.MODELS:
            data = [row for row in profiles if row["case_id"] == case_id and row["boundary"] == model]
            assert [int(row["index"]) for row in data] == list(range(200))
            mean = np.array([float(row["mean"]) if row["mean"] else np.nan for row in data])
            truth = np.array([float(row["truth"]) for row in data])
            missing = np.array([row["input_valid"] == "0" for row in data])
            means.append(mean)
            for region, mask in {"all": np.ones(200, bool), "edge": (np.arange(200) < 7) | (np.arange(200) >= 193),
                                 "left_edge": np.arange(200) < 7, "right_edge": np.arange(200) >= 193, "gap": missing}.items():
                row, = [row for row in errors if row["case_id"] == case_id and row["method"] == boundary.METHODS[model]
                        and row["region"] == region]
                accepted = mask & np.isfinite(mean)
                assert int(row["n_considered"]) == mask.sum()
                assert int(row["n_compared"]) == accepted.sum()
                assert int(row["n_undefined"]) == (mask & ~np.isfinite(mean)).sum()
                assert row["n_gap_filled"] == row["n_reference_undefined"] == "0"
                assert row["reference_method"] == "known_clean_signal"
                if accepted.any():
                    error = mean[accepted] - truth[accepted]
                    assert float(row["rmse"]) == pytest.approx(np.sqrt(np.mean(error**2)), abs=1e-14)
                    assert float(row["max_abs_error"]) == pytest.approx(np.max(np.abs(error)), abs=1e-14)
                else:
                    assert row["rmse"] == row["max_abs_error"] == ""
                if region == "gap" and "middle_gap" in case_id:
                    assert (row["n_considered"], row["n_compared"], row["n_undefined"]) == ("40", "14", "26")
            for row in data:
                assert int(row["accepted"]) == int(bool(row["mean"]))
                assert float(row["rho"]) == np.clip(float(row["raw_mask_response"]), 0, 1)
                assert float(row["N_eq"]) == pytest.approx(15 * float(row["rho"]), abs=5e-13)
        np.testing.assert_array_equal(np.isfinite(means[0]), np.isfinite(means[1]))
        np.testing.assert_array_equal(np.isfinite(means[0]), np.isfinite(means[2]))
    no_values = boundary.truth_error_row("test", "case", "periodic", "all", np.ones(3, bool),
                                         np.full(3, np.nan), np.arange(3.))
    assert no_values["n_considered"] == no_values["n_undefined"] == 3
    assert no_values["n_compared"] == 0 and no_values["rmse"] == no_values["max_abs_error"] == ""


def test_interior_discrepancies_include_every_defined_cell_and_zero_variance_windows(saved, prepared):
    profiles = rows(saved / "profiles.csv")
    interior = rows(saved / "interior_agreement.csv")
    assert len(interior) == 20 and all(row["status"] == "pass" for row in interior)
    for case_id in boundary.CASE_IDS:
        mask = (np.arange(200) >= 7) & (np.arange(200) < 193)
        for model in boundary.BOUNDARIES:
            a = [row for row in profiles if row["case_id"] == case_id and row["boundary"] == model]
            b = [row for row in profiles if row["case_id"] == case_id and row["boundary"] == "truncated"]
            for statistic in ("mean", "variance", "std", "coverage", "support"):
                key = {"coverage": "rho", "support": "N_eq"}.get(statistic, statistic)
                actual = np.array([float(row[key]) if row[key] else np.nan for row in a])
                expected = np.array([float(row[key]) if row[key] else np.nan for row in b])
                defined = mask & np.isfinite(actual) & np.isfinite(expected)
                row, = [row for row in interior if row["case_id"] == case_id
                         and row["method"] == boundary.METHODS[model] and row["statistic"] == statistic]
                assert row["reference_method"] == "numpy_truncated_windows" and row["region"] == "interior"
                assert int(row["n_considered"]) == 186
                expected_count = 160 if "middle_gap" in case_id and statistic in ("mean", "variance", "std") else 186
                assert int(row["n_compared"]) == defined.sum() == expected_count
                assert int(row["n_method_undefined"]) == int(row["n_reference_undefined"]) == 186 - expected_count
                delta = actual[defined] - expected[defined]
                assert float(row["max_abs_difference"]) == pytest.approx(np.max(np.abs(delta)), abs=1e-14)
                assert float(row["rmse"]) == pytest.approx(np.sqrt(np.mean(delta**2)), abs=1e-14)
                atol, rtol = boundary.tolerance(statistic)
                assert float(row["atol"]) == atol and float(row["rtol"]) == rtol
    reference = prepared[1]["outputs"]["truncated"]
    one = (reference["n_observed"] == 1) & mask
    np.testing.assert_array_equal(np.flatnonzero(one), [86, 113])
    np.testing.assert_array_equal(reference["variance"][one], [0, 0])
    for model in boundary.BOUNDARIES:
        result = prepared[1]["outputs"][model]
        assert np.all((result["std"][one] - reference["std"][one])**2
                      <= np.abs(result["variance"][one] - reference["variance"][one]) + 1e-14)


@pytest.mark.parametrize("filename,match,key,value", [
    ("settings.csv", {"setting": "input_sha256"}, "value", "bad"),
    ("settings.csv", {"setting": "periodic_backend"}, "value", "cartesian"),
    ("settings.csv", {"setting": "toolkit_source_sha256.polar.py"}, "value", "bad"),
    ("settings.csv", {"setting": "variance_atol"}, "value", "0.1"),
    ("comparison.csv", {"method": "toolkit_rfft_periodic", "region": "all", "statistic": "mean"}, "atol", "0.1"),
    ("comparison.csv", {"method": "toolkit_rfft_periodic", "region": "edge", "statistic": "variance"}, "rmse", "0.1"),
    ("truth_errors.csv", {"method": "toolkit_rfft_periodic", "region": "all"}, "rmse", "0.1"),
    ("truth_errors.csv", {"method": "toolkit_rfft_periodic", "region": "gap"}, "rmse", "0"),
    ("profiles.csv", {"boundary": "periodic", "index": "0"}, "mean", "0"),
    ("profiles.csv", {"boundary": "truncated", "index": "0"}, "mean", "0"),
    ("interior_agreement.csv", {"method": "toolkit_dct_reflective", "statistic": "mean"}, "n_compared", "185"),
    ("interior_agreement.csv", {"method": "toolkit_rfft_periodic", "statistic": "std"}, "rmse", "0.1"),
])
def test_figure_only_checks_saved_new_evidence_before_plotting(prepared, saved, tmp_path, filename, match, key, value):
    copied = tmp_path / saved.name
    shutil.copytree(saved, copied)
    amend_csv(copied / filename, match, key, value)
    before = file_hashes(copied)
    with pytest.raises(ValueError, match="Saved"):
        boundary.verify_saved(prepared, copied)
    assert file_hashes(copied) == before


def test_artist_data_mapping_light_original_style_unclipped_curves_and_unbridged_gaps(drawing, prepared):
    from matplotlib.path import Path as PlotPath
    from matplotlib.text import Text
    from matplotlib.colors import to_rgba
    assert len(drawing.axes) == 6 and len(drawing.legends) == 1
    assert tuple(drawing.get_size_inches()) == (6.5, 7.1)
    texts = [text for text in drawing.findobj(Text) if text.get_visible() and text.get_text().strip()]
    assert min(text.get_fontsize() for text in texts) >= 10.4
    assert all(text.get_fontweight() == "normal" for text in texts)
    for i, ax in enumerate(drawing.axes):
        assert ax.get_title(loc="left").startswith(f"({chr(97 + i)})")
        assert not any(spine.get_visible() for spine in ax.spines.values())
        assert not any(line.get_visible() for line in (*ax.get_xgridlines(), *ax.get_ygridlines()))
        arrows = [text for text in ax.texts if text.get_gid() == "light_axis_arrow"]
        assert len(arrows) == 2
        assert all(arrow.arrow_patch.get_edgecolor() == to_rgba("#888888") for arrow in arrows)
        assert all(arrow.arrow_patch.get_linewidth() == 0.8 for arrow in arrows)
        case = prepared[1] if i in (1, 3, 5) else prepared[0]
        t = case["t"]
        mask = (t >= ax.get_xlim()[0]) & (t <= ax.get_xlim()[1])
        lines = {line.get_gid(): line for line in ax.lines}
        statistic = "std" if i == 3 else ("rho" if i == 5 else "mean")
        for model in boundary.MODELS:
            line = lines[f"{model}_{statistic}"]
            result = case["outputs"][model]
            values = np.clip(result["denominator"], 0, 1) if i == 5 else result[statistic]
            np.testing.assert_array_equal(line.get_xdata(), t[mask])
            np.testing.assert_array_equal(line.get_ydata(), values[mask])
            assert line.get_color() == boundary.COLORS[model]
            moves = sum(code == PlotPath.MOVETO for _, code in line.get_path().iter_segments())
            assert moves == (2 if i in (1, 3) else 1)
        if i not in (3, 5):
            np.testing.assert_array_equal(lines["observed"].get_ydata(), case["data"][mask])
            np.testing.assert_array_equal(lines["truth"].get_ydata(), case["truth"][mask])
            assert lines["observed"].get_color() == "#bbbbbb" and lines["observed"].get_alpha() == 0.45
            assert lines["observed"].get_linestyle() == "None"
            assert lines["truth"].get_color() == "#333333" and lines["truth"].is_dashed()
        # Check every actual displayed point/curve, not just the axis settings.
        for line in ax.lines:
            y = line.get_ydata()
            assert np.nanmin(y) >= ax.get_ylim()[0] and np.nanmax(y) <= ax.get_ylim()[1]
        if i in (1, 3, 5):
            patch, = [patch for patch in ax.patches if patch.get_gid() == "original_missing_interval"]
            assert patch.get_x() == pytest.approx((t[79] + t[80]) / 2)
            assert patch.get_x() + patch.get_width() == pytest.approx((t[119] + t[120]) / 2)
        assert not ax.images and not ax.collections
    zoom_boxes = drawing.axes[0].patches
    assert len(zoom_boxes) == 2
    for box, (xlim, ylim) in zip(zoom_boxes, boundary.ZOOMS):
        assert box.get_gid() == "boundary_zoom_box" and box.get_linestyle() == "--"
        assert box.get_xy() == (xlim[0], ylim[0])
        assert (box.get_width(), box.get_height()) == (xlim[1] - xlim[0], ylim[1] - ylim[0])
        assert box.get_edgecolor() == to_rgba("#999999")
    coverage = {line.get_gid(): line for line in drawing.axes[5].lines}
    assert coverage["reflective_rho"].is_dashed()  # Exposes the identical coverage beneath.
    assert [text.get_text() for text in drawing.legends[0].texts] == [
        "Noisy observations", "True signal", "Periodic (FFT)", "Reflective (DCT)", "Truncated (NumPy)",
    ]
    assert not any(artist.get_rasterized() for artist in drawing.findobj())


def test_single_truncated_numpy_curve_replaces_matched_reference_markers(drawing, prepared):
    for i, ax in enumerate(drawing.axes):
        lines = {line.get_gid(): line for line in ax.lines}
        statistic = "std" if i == 3 else ("rho" if i == 5 else "mean")
        assert not any(gid.startswith("numpy_") for gid in lines)
        curve = lines[f"truncated_{statistic}"]
        assert curve.get_marker() == "None" and curve.is_dashed()
        if i in (1, 3):
            t = curve.get_xdata()
            undefined = (t >= prepared[1]["t"][87]) & (t <= prepared[1]["t"][112])
            assert np.isnan(curve.get_ydata()[undefined]).all()


def test_figure_only_exports_only_assets_no_numeric_writes(prepared, saved, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Figure-only must not create runs, write numerical CSVs or collect a new environment")

    for name in ("create_run_directory", "save_evidence", "write_csv", "environment_settings"):
        monkeypatch.setattr(boundary, name, forbidden)
    frozen_before, new_before = file_hashes(boundary.FROZEN_RUN), file_hashes(saved)
    output = tmp_path / "assets"
    assert boundary.main(["--figure-only", "--saved-run", str(saved), "--figure-output", str(output)]) == 0
    assert file_hashes(boundary.FROZEN_RUN) == frozen_before and file_hashes(saved) == new_before
    verify_assets(output, saved)
    assert sorted(file_hashes(output)) == ["figure_settings.csv", "pdf/boxcar_boundary_comparison.pdf", "png/boxcar_boundary_comparison.png"]


def verify_assets(output, saved_run):
    from PIL import Image
    values = {row["setting"]: row["value"] for row in rows(output / "figure_settings.csv")}
    assert values["selected_run"] == saved_run.name
    assert json.loads(values["selected_cases"]) == list(boundary.CASE_IDS)
    assert values["figure_width_in"] == "6.5" and values["figure_height_in"] == "7.1"
    assert float(values["min_text_size_pt"]) >= 10.4
    assert values["completion"] == values["restoration"] == "False"
    assert values["script_sha256"] == digest(boundary.__file__)
    assert values["original_plotting_source_sha256"] == digest(SCRIPTS / "generate_boundary_figure.py")
    assert values["inherited_script_sha256"] == digest(SCRIPTS / "generate_1d_boxcar_validation.py")
    assert values["checks"].startswith("PASS:")
    assert "no exterior padding" in values["numpy_display"]
    assert values["numpy_label"] == "Truncated (NumPy)"
    assert json.loads(values["displayed_models"]) == list(boundary.MODELS)
    assert "available in-domain" in values["numpy_coverage"]
    for name, value in _RELEASE_HASHES.items():
        assert values[f"toolkit_source_sha256.{name}"] == value
    for name in ("settings.csv", "profiles.csv", "truth_errors.csv", "comparison.csv", "interior_agreement.csv"):
        assert values[f"saved_{name}_sha256"] == digest(saved_run / name)
    for name in ("settings.csv", "coverage.csv", "comparison.csv"):
        assert values[f"frozen_{name}_sha256"] == digest(boundary.FROZEN_RUN / name)
    for kind in ("pdf", "png"):
        assert values[f"{kind}_sha256"] == digest(output / values[f"{kind}_path"])
    with Image.open(output / values["png_path"]) as image:
        assert image.size == (1950, 2130)
        assert all(abs(value - 300) < 0.01 for value in image.info["dpi"])
        assert image.getpixel((0, 0)) == (255, 255, 255, 255)
    pdf = (output / values["pdf_path"]).read_bytes()
    assert b"/FontFile2" in pdf and b"/Subtype /Type3" not in pdf
    assert b"/Subtype /Image" not in pdf


def test_selected_saved_evidence_and_assets(prepared):
    boundary.verify_saved(prepared, boundary.SELECTED_RUN)
    verify_assets(boundary.FIGURE_DIRECTORY, boundary.SELECTED_RUN)


@pytest.mark.parametrize("destination", (boundary.FROZEN_RUN, boundary.SELECTED_RUN,
                                         boundary.ROOT / "manuscript/figures/pdf",
                                         boundary.ROOT / "experiments/figures"))
def test_cannot_write_figure_into_saved_results_or_old_assets(destination, monkeypatch):
    monkeypatch.setattr(boundary, "prepare_comparison", lambda: pytest.fail("Reject before calculating"))
    with pytest.raises(ValueError, match="protected|saved experiment results"):
        boundary.main(["--figure-only", "--figure-output", str(destination)])


def test_custom_saved_results_and_unknown_assets_are_protected(saved, prepared, tmp_path):
    with pytest.raises(ValueError, match="saved experiment results"):
        boundary.export_comparison(prepared, saved, saved / "figures")
    target = tmp_path / "pdf" / "boxcar_boundary_comparison.pdf"
    target.parent.mkdir()
    target.write_bytes(b"not this figure")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        boundary.check_figure_output(tmp_path)
    assert target.read_bytes() == b"not this figure"


@pytest.mark.parametrize("args", (["--figure-only", "--run-id", "bad"],
                                 ["--figure-only", "--output-root", "unused"],
                                 ["--saved-run", "unused"]))
def test_cannot_mix_new_run_and_figure_only_options(args, monkeypatch):
    monkeypatch.setattr(boundary, "prepare_comparison", lambda: pytest.fail("Reject before calculating"))
    with pytest.raises(SystemExit) as error:
        boundary.main(args)
    assert error.value.code == 2


def test_introduction_revision_preserves_manuscript_outside_introduction():
    current = (boundary.ROOT / "manuscript/main.tex").read_text()
    original = subprocess.check_output(["git", "show", "90faea4:manuscript/main.tex"], cwd=boundary.ROOT).decode()
    start, end = "\\section{Introduction}", "\\section{Spectral filtering on Cartesian and polar grids}"
    sections = []
    for text in (current, original):
        assert text.count(start) == text.count(end) == 1
        sections.append(text[:text.index(start)] + text[text.index(end):])
    assert sections[0] == sections[1]
    intro = current[current.index(start):current.index(end)]
    for text in ("local mean, variance, and standard deviation", "Cartesian and polar grids",
                 "validity mask", "reflective or periodic boundary independently",
                 "range-dependent angular widths", "\\citet{Valdivia2026RadarDCT}",
                 "\\texttt{dct\\_toolkit}", "rather than separate calls to the released statistics functions",
                 "does not establish recovery of missing values"):
        assert text in intro
    assert intro.count("\\label{fig:statistics_pipeline}") == 1
    assert "figures/round1/pipeline/pdf/masked_statistics_pipeline.pdf" in intro
    assert "default completion is not used" in intro


def test_manuscript_changes_preserve_tables_math_and_unrelated_text():
    current = (boundary.ROOT / "manuscript/main.tex").read_text()
    original = subprocess.check_output(["git", "show", "25ff036:manuscript/main.tex"], cwd=boundary.ROOT).decode()
    start = "\\subsection{One-dimensional boxcar"
    end = "\\subsection{Gaussian statistics on a real two-dimensional radar sweep}"
    # Overview/protocol/interpretation paragraphs support the 1D comparison.
    # Separate authorizations cover the radar callout/caption, the explanation
    # of signed Gaussian weights, the wind display/Discussion revision, and the
    # author's abstract/Introduction rewrites plus the separate matched radar benchmark.
    approved_paragraphs = (
        "The selected v1.0.0 numerical comparisons and benchmarks use only direct ratios",
        "The experiments examine numerical agreement and local statistics on incomplete grids",
        "Agreement with matched references isolates numerical consistency",
        "Figure~\\ref{fig:radar_statistics} displays the natural-mask",
        "Figure~\\ref{fig:wind_statistics}",
        "The componentwise wind flags illustrate",
        "For the $720\\times1180$ natural-mask radar sweep",
        "The matched 3D runtime and process-RAM results",
        "Write the complete filter as",
    )
    explanation_prefix = "The second raw moment $M_2$ can be smaller than"
    assert current.count(explanation_prefix) == 1 and explanation_prefix not in original
    explanation = next(line for line in current.splitlines() if line.startswith(explanation_prefix))
    for phrase in ("Gaussian frequency response", "negative spatial weights", "even in exact arithmetic",
                   "does not restore the usual weighted-variance interpretation",
                   "a clamped zero should not be interpreted as an absence of local variability"):
        assert phrase in explanation
    robust_prefix = "One possible extension is to exclude or downweight suspect observations"
    assert current.count(robust_prefix) == 1 and robust_prefix not in original
    radar_start = "\\subsection{Matched radar-statistics costs}"
    radar_end = "\\subsection{Separate mean-only costs}"
    assert current.count(radar_start) == 1 and radar_start not in original
    outside_sections = []
    for text in (current, original):
        outside = text[:text.index(start)] + text[text.index(end):]
        if radar_start in outside:
            outside = outside[:outside.index(radar_start)] + outside[outside.index(radar_end):]
        abstract = r"\\begin\{abstract\}.*?\\end\{abstract\}"
        assert len(re.findall(abstract, outside, re.S)) == 1
        outside = re.sub(abstract, "ABSTRACT", outside, flags=re.S)
        intro_start = "\\section{Introduction}"
        intro_end = "\\section{Spectral filtering on Cartesian and polar grids}"
        assert outside.count(intro_start) == outside.count(intro_end) == 1
        outside = outside[:outside.index(intro_start)] + "INTRODUCTION\n" + outside[outside.index(intro_end):]
        for prefix in approved_paragraphs:
            assert outside.count(prefix) == 1
            outside = re.sub(r"^" + re.escape(prefix) + r"[^\n]*$", prefix, outside, flags=re.M)
        outside = re.sub(r"\n" + re.escape(explanation_prefix) + r"[^\n]*\n", "", outside)
        outside = re.sub(r"\n" + re.escape(robust_prefix) + r"[^\n]*\n", "", outside)
        radar_caption = r"\t\\caption\{Natural-mask COW sweep 12,[^\n]*\}"
        assert len(re.findall(radar_caption, outside)) == 1
        outside = re.sub(radar_caption, "RADAR_CAPTION", outside)
        wind_caption = r"\t\\caption\{(?:Wind-component statistics on zero-based plane 9|Observed wind and local statistics)[^\n]*\}"
        assert len(re.findall(wind_caption, outside)) == 1
        outside = re.sub(wind_caption, "WIND_CAPTION", outside)
        outside_sections.append(outside)
    assert outside_sections[0] == outside_sections[1]
    current_tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", current, re.S)
    added_tables = [table for table in current_tables if "\\label{tab:radar_performance}" in table]
    assert len(added_tables) == 1
    assert [table for table in current_tables if table not in added_tables] == re.findall(
        r"\\begin\{table\}.*?\\end\{table\}", original, re.S)
    assert re.findall(r"\\begin\{(equation\*?|align\*?)\}(.*?)\\end\{\1\}", current, re.S) == re.findall(
        r"\\begin\{(equation\*?|align\*?)\}(.*?)\\end\{\1\}", original, re.S)
    for text in (current, original):
        assert text.count("\\label{fig:boxcar_missingness}") == 1
    table_start, table_end = "\\begin{table}[pos=hbp]", "\\end{table}"
    new_section, old_section = current[current.index(start):current.index(end)], original[original.index(start):original.index(end)]
    assert new_section[new_section.index(table_start):new_section.index(table_end)] == old_section[old_section.index(table_start):old_section.index(table_end)]
    assert "figures/round1/1d_boundary_comparison/pdf/boxcar_boundary_comparison.pdf" in new_section
