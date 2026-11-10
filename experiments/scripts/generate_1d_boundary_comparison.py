#!/usr/bin/env python3
"""Compare two boundary assumptions on the approved complete/gapped realization.

This is new width-15 boxcar evidence, not a reproduction of the historical
Gaussian experiment. Normal execution verifies two inputs and both released
backends before creating a new, nonoverwriting run and the new figure assets.
--figure-only recomputes those same two cases against a selected saved NEW run;
it writes figure assets only. Neither mode runs the 75-case validation or timings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

import generate_1d_boxcar_validation as inherited
from _review_common import (
    COMPARISON_FIELDS, SETTINGS_FIELDS, TOOLKIT_COMMIT, _RELEASE_HASHES,
    _csv_value, create_run_directory, environment_settings, load_toolkit, write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN_RUN = ROOT / "experiments/results/round1/1d_boxcar/lead-review-1d-20261027"
SELECTED_RUN = ROOT / "experiments/results/round1/1d_boundary_comparison/boundary-truncated-20261030"
FIGURE_DIRECTORY = ROOT / "manuscript/figures/round1/1d_boundary_comparison"
CASE_IDS = ("trend_sine_seed0_complete_n200_m15", "trend_sine_seed0_middle_gap_n200_m15")
BOUNDARIES = ("periodic", "reflective")
MODELS = (*BOUNDARIES, "truncated")
METHODS = {"periodic": "toolkit_rfft_periodic", "reflective": "toolkit_dct_reflective",
           "truncated": "numpy_truncated_windows"}
REFERENCES = {"periodic": "numpy_wrapped_windows", "reflective": "numpy_symmetric_windows"}
N, WIDTH = inherited.N, inherited.WIDTH
FLOOR, MASTER_SEED = inherited.DENOMINATOR_FLOOR, inherited.MASTER_SEED
TOLERANCES = inherited.TOLERANCES
TRUTH_FIELDS = [
    "run_id", "case_id", "method", "reference_method", "component", "statistic",
    "region", "estimate_type", "n_considered", "n_compared", "n_undefined",
    "n_reference_undefined", "n_gap_filled", "max_abs_error", "rmse", "units", "notes",
]
PROFILE_FIELDS = [
    "run_id", "case_id", "boundary", "method", "index", "t", "input_valid",
    "observed", "truth", "accepted", "mean", "variance", "std", "raw_mask_response",
    "rho", "N_eq",
]
FIGURE_STEM = "boxcar_boundary_comparison"
FIGURE_SIZE = (6.5, 7.1)
NUMPY_LABEL = "Truncated (NumPy)"
COLORS = {"observed": "#bbbbbb", "truth": "#333333", "periodic": "#1b9e77",
          "reflective": "#d95f02", "truncated": "#377eb8", "axes": "#888888", "zoom": "#999999"}
ZOOMS = (((0.0, 1.2), (-1.4, 8.0)), ((8.8, 10.0), (6.5, 17.2)))
PANEL_LIMITS = (((0, 10), (-2, 18)), ((0, 10), (-2, 18)), ZOOMS[0],
                ((0, 10), (-0.25, 7.2)), ZOOMS[1], ((0, 10), (-0.06, 1.09)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError(f"Unexpected CSV schema: {path}")
        return list(reader)


def periodic_options():
    # Cartesian is reflective-only. Explicit geometry selects the actual RFFT
    # pass in released polar.py, even though this single axis is linear, not polar.
    return dict(axis_types=["linear"], width_units="grid", boundary=["periodic"],
                filter_axes=[0], kernel_type="boxcar_discrete")


def periodic_moments(data, width, toolkit):
    valid = np.isfinite(data)
    clean = np.where(valid, data, np.nan)
    geometry = periodic_options()
    controls = dict(coordinates="polar", restore_input_nan=False,
                    _stable_fallback=False, _denominator_floor=FLOOR, **geometry)
    result = {name: function(clean, width, **controls) for name, function in (
        ("mean", toolkit.dct_mean), ("variance", toolkit.dct_variance), ("std", toolkit.dct_std),
    )}
    result["denominator"] = toolkit.smooth_polar(valid.astype(float), width, **geometry)
    result["support"] = toolkit.dct_count(
        valid, width, coordinates="polar", restore_input_nan=False, **geometry,
    )
    return result


def wrapped_numpy_moments(data, width):
    """Explicit modulo-index windows; no toolkit/spectral weights or padding."""
    if width < 1 or width % 2 != 1:
        raise ValueError("The discrete reference requires a positive odd width.")
    indices = (np.arange(data.size)[:, None] + np.arange(-(width // 2), width // 2 + 1)) % data.size
    windows = data[indices]
    valid = np.isfinite(windows)
    safe = np.where(valid, windows, 0.0)
    count = valid.sum(axis=1)
    denominator = count / width
    accepted = np.isfinite(denominator) & (denominator > FLOOR)
    mean = np.full(data.shape, np.nan)
    np.divide(safe.sum(axis=1), count, out=mean, where=accepted)
    residual = np.where(valid, safe - mean[:, None], 0.0)
    variance = np.full(data.shape, np.nan)
    np.divide((residual**2).sum(axis=1), count, out=variance, where=accepted)
    return dict(mean=mean, variance=variance, std=np.sqrt(variance),
                denominator=denominator, support=count.astype(float))


def truncated_numpy_moments(data, width):
    """Average finite values in clipped windows, without exterior padding.

    Coverage is the valid fraction of the in-domain window. Its size shrinks
    near an outer boundary; nominal equivalent count remains width * coverage,
    not the number of distinct observations in that shorter window.
    """
    if data.ndim != 1 or not data.size or width < 1 or width % 2 != 1:
        raise ValueError("The truncated reference needs a nonempty 1D input and positive odd width.")
    mean, variance = np.full(data.shape, np.nan), np.full(data.shape, np.nan)
    available, observed = np.zeros(data.size, dtype=int), np.zeros(data.size, dtype=int)
    denominator = np.empty(data.size)
    for i in range(data.size):
        window = data[max(0, i - width // 2):min(data.size, i + width // 2 + 1)]
        values = window[np.isfinite(window)]
        available[i], observed[i] = window.size, values.size
        denominator[i] = values.size / window.size
        if denominator[i] > FLOOR:
            mean[i] = np.mean(values)
            variance[i] = np.mean((values - mean[i])**2)
    return dict(mean=mean, variance=variance, std=np.sqrt(variance), denominator=denominator,
                support=width * np.clip(denominator, 0, 1),
                n_available=available, n_observed=observed)


def regions(data):
    left = np.arange(data.size) < WIDTH // 2
    right = np.arange(data.size) >= data.size - WIDTH // 2
    return {"all": np.ones(data.size, dtype=bool), "edge": left | right,
            "left_edge": left, "right_edge": right, "gap": ~np.isfinite(data)}


def tolerance(statistic):
    if statistic in ("mask_response", "coverage"):
        return TOLERANCES["support"][0] / WIDTH, TOLERANCES["support"][1]
    return TOLERANCES[statistic]


def comparison_row(run_id, case_id, boundary, statistic, region, selected, actual, expected):
    a, b = actual[selected], expected[selected]
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    joint = finite_a & finite_b
    difference = a[joint] - b[joint]
    atol, rtol = tolerance(statistic)
    passed = (np.array_equal(finite_a, finite_b) and not np.isinf(a).any()
              and not np.isinf(b).any()
              and np.all(np.abs(difference) <= atol + rtol * np.abs(b[joint])))
    return dict(
        run_id=run_id, case_id=case_id, method=METHODS[boundary], reference_method=REFERENCES[boundary],
        component="trend_sine", statistic=statistic, region=region,
        estimate_type={"support": "equivalent_data_count", "mask_response": "raw_mask_response",
                       "coverage": "local_data_coverage"}.get(
            statistic, "direct_normalized_moments"), kernel_match="exact_discrete_boxcar",
        n_considered=int(selected.sum()), n_compared=int(joint.sum()),
        n_method_undefined=int((~finite_a).sum()), n_reference_undefined=int((~finite_b).sum()),
        n_method_gap_filled=0, n_reference_gap_filled=0,
        max_abs_difference=float(np.max(np.abs(difference))) if joint.any() else "",
        rmse=float(np.sqrt(np.mean(difference**2))) if joint.any() else "",
        atol=atol, rtol=rtol, status=("pass" if joint.any() else "no_comparable") if passed else "fail",
        units="1" if statistic in ("mask_response", "coverage") else inherited.UNITS[statistic],
        notes=f"Matched {boundary} windows on identical input; centered population variance reference. "
              "No completion/restoration; defined masks must match. Numerical disagreement, not truth error.",
    )


def truth_error_row(run_id, case_id, boundary, region, selected, mean, truth):
    joint = selected & np.isfinite(mean) & np.isfinite(truth)
    error = mean[joint] - truth[joint]
    return dict(
        run_id=run_id, case_id=case_id, method=METHODS[boundary], reference_method="known_clean_signal",
        component="trend_sine", statistic="mean", region=region,
        estimate_type="direct_mean_vs_clean_signal", n_considered=int(selected.sum()),
        n_compared=int(joint.sum()), n_undefined=int((selected & ~np.isfinite(mean)).sum()),
        n_reference_undefined=int((selected & ~np.isfinite(truth)).sum()), n_gap_filled=0,
        max_abs_error=float(np.max(np.abs(error))) if joint.any() else "",
        rmse=float(np.sqrt(np.mean(error**2))) if joint.any() else "", units="signal_unit",
        notes="One fixed replicate, not replicate inference. All three means use matching accepted sets; "
              "unsupported positions excluded, not filled. Gap means originally missing inputs. "
              "Truth error is separate from numerical disagreement against matched windows.",
    )


def check_result(result, reference, label):
    accepted = np.isfinite(result["denominator"]) & (result["denominator"] > FLOOR)
    for statistic in ("mean", "variance", "std"):
        if not np.array_equal(np.isfinite(result[statistic]), accepted) or np.isinf(result[statistic]).any():
            raise ValueError(f"Undefined/strict-floor moment policy changed: {label}, {statistic}")
    for statistic in (*TOLERANCES, "mask_response"):
        field = "denominator" if statistic == "mask_response" else statistic
        atol, rtol = tolerance(statistic)
        if not np.allclose(result[field], reference[field], atol=atol, rtol=rtol, equal_nan=True):
            raise ValueError(f"Fixed-tolerance agreement failed: {label}, {statistic}")
    if not np.isfinite(result["denominator"]).all():
        raise ValueError(f"Nonfinite mask response: {label}")
    atol, rtol = TOLERANCES["support"]
    if not np.allclose(result["support"], WIDTH * np.clip(result["denominator"], 0, 1), atol=atol, rtol=rtol):
        raise ValueError(f"Coverage/equivalent-count mapping changed: {label}")
    return accepted


def verify_frozen(case, settings_rows, coverage_rows, comparison_rows):
    """Keep the old reflective evidence read-only; verify every generation setting."""
    case_id = case["case_id"]
    rows = [row for row in settings_rows if row["case_id"] == case_id]
    frozen = {row["setting"]: row["value"] for row in rows}
    if len(frozen) != len(rows):
        raise ValueError("Duplicate frozen setting.")
    for name, (value, unit) in case["inherited_settings"].items():
        match = [row for row in rows if row["setting"] == name]
        if len(match) != 1 or match[0]["value"] != str(value) or match[0]["unit"] != unit:
            raise ValueError(f"Frozen generation setting mismatch: {case_id}, {name}")
    if frozen.get("dct_toolkit_commit") != TOOLKIT_COMMIT or frozen.get("dct_toolkit_version") != "1.0.0":
        raise ValueError("Frozen toolkit identity mismatch.")
    for method, result in (("toolkit_dct_direct", case["outputs"]["reflective"]),
                           (inherited.REFERENCE, case["references"]["reflective"])):
        for region, selected in inherited.regions(case["data"]).items():
            actual = inherited.coverage_row(FROZEN_RUN.name, case_id, method, region, selected, case["data"], result)
            matches = [row for row in coverage_rows if row["case_id"] == case_id
                       and row["method"] == method and row["region"] == region]
            if len(matches) != 1:
                raise ValueError("Missing frozen coverage row.")
            for key in inherited.COVERAGE_FIELDS:
                if key == "notes":
                    continue
                value = actual[key]
                old = matches[0][key]
                if isinstance(value, float):
                    atol, rtol = TOLERANCES["variance" if key == "min_raw_moment_variance" else "support"]
                    same = np.isclose(float(old), value, atol=atol, rtol=rtol)
                else:
                    same = old == str(value)
                if not same:
                    raise ValueError(f"Frozen coverage mismatch: {case_id}, {method}, {region}, {key}")
    for region, selected in inherited.regions(case["data"]).items():
        for statistic, (atol, rtol) in TOLERANCES.items():
            actual = inherited.comparison_row(
                FROZEN_RUN.name, case_id, "trend_sine", "toolkit_dct_direct", statistic,
                region, selected, case["outputs"]["reflective"][statistic],
                case["references"]["reflective"][statistic],
            )
            matches = [row for row in comparison_rows if row["case_id"] == case_id
                       and row["method"] == "toolkit_dct_direct" and row["region"] == region
                       and row["statistic"] == statistic]
            if len(matches) != 1:
                raise ValueError("Missing frozen comparison row.")
            for key, value in actual.items():
                old = matches[0][key]
                same = (np.isclose(float(old), value, atol=atol, rtol=rtol)
                        if key in ("max_abs_difference", "rmse") and value != "" else old == str(value))
                if not same or actual["status"] == "fail":
                    raise ValueError(f"Frozen numerical comparison mismatch: {case_id}, {region}, {statistic}, {key}")
    case["frozen_settings"] = frozen


def prepare_comparison():
    """Calculate only the approved illustrative pair, with all checks before writes."""
    toolkit = load_toolkit()  # All nine source hashes, exact release and imported path.
    frozen_settings = read_rows(FROZEN_RUN / "settings.csv", SETTINGS_FIELDS)
    frozen_coverage = read_rows(FROZEN_RUN / "coverage.csv", inherited.COVERAGE_FIELDS)
    frozen_comparison = read_rows(FROZEN_RUN / "comparison.csv", COMPARISON_FIELDS)
    for rows in (frozen_settings, frozen_coverage, frozen_comparison):
        if any(row["run_id"] != FROZEN_RUN.name for row in rows):
            raise ValueError("Unexpected frozen run identity.")
    t = np.linspace(0, 10, N)
    truth = 1.2 * t + 2.5 * np.sin(1.5 * t)
    cases = []
    for case_id, _, data, settings in inherited.make_cases("full", MASTER_SEED):
        if case_id not in CASE_IDS:
            continue
        case = dict(case_id=case_id, data=data, t=t, truth=truth, inherited_settings=settings)
        reflective = inherited.toolkit_moments(data, WIDTH, toolkit)
        case["outputs"] = dict(periodic=periodic_moments(data, WIDTH, toolkit), reflective=reflective,
                               truncated=truncated_numpy_moments(data, WIDTH))
        case["references"] = dict(periodic=wrapped_numpy_moments(data, WIDTH), reflective=inherited.numpy_moments(data, WIDTH))
        masks = [check_result(case["outputs"][boundary], case["references"][boundary], f"{case_id}, {boundary}")
                 for boundary in BOUNDARIES]
        truncated = case["outputs"]["truncated"]
        accepted = np.isfinite(truncated["denominator"]) & (truncated["denominator"] > FLOOR)
        if any(not np.array_equal(np.isfinite(truncated[statistic]), accepted)
               for statistic in ("mean", "variance", "std")):
            raise ValueError("Truncated-window strict-floor/undefined policy changed.")
        masks.append(accepted)
        expected = np.ones(N, dtype=bool)
        if "middle_gap" in case_id:
            expected[87:113] = False
        if any(not np.array_equal(mask, expected) for mask in masks):
            raise ValueError("All three treatments must have matching approved accepted sets (200 / 174).")
        verify_frozen(case, frozen_settings, frozen_coverage, frozen_comparison)
        cases.append(case)
        if len(cases) == 2:
            break  # Stop within replicate zero: no other profile/replicate calculation.
    if tuple(case["case_id"] for case in cases) != CASE_IDS:
        raise ValueError("The two approved inputs were not reproduced.")
    retained = np.isfinite(cases[1]["data"])
    if not np.array_equal(cases[0]["data"][retained], cases[1]["data"][retained]):
        raise ValueError("The complete and gapped inputs must share the exact same noise.")
    return cases


def evidence_rows(cases, run_id):
    comparisons, errors, profiles, interior = [], [], [], []
    for case in cases:
        for boundary in BOUNDARIES:
            result, reference = case["outputs"][boundary], case["references"][boundary]
            for region, selected in regions(case["data"]).items():
                for statistic in (*TOLERANCES, "mask_response"):
                    field = "denominator" if statistic == "mask_response" else statistic
                    comparisons.append(comparison_row(run_id, case["case_id"], boundary, statistic, region,
                                                       selected, result[field], reference[field]))
            selected = (np.arange(N) >= WIDTH // 2) & (np.arange(N) < N - WIDTH // 2)
            baseline = case["outputs"]["truncated"]
            for statistic in ("mean", "variance", "std", "coverage", "support"):
                actual = np.clip(result["denominator"], 0, 1) if statistic == "coverage" else result[statistic]
                expected = np.clip(baseline["denominator"], 0, 1) if statistic == "coverage" else baseline[statistic]
                row = comparison_row(run_id, case["case_id"], boundary, statistic, "interior", selected,
                                     actual, expected)
                row.update(reference_method=METHODS["truncated"],
                           notes="Full width-15 window inside domain: indices [7,193). Identical finite observations "
                                 "and weights; undefined sets must match. All defined interior cells are included, "
                                 "including one-observation windows. Std near zero can amplify variance roundoff. "
                                 "This is numerical disagreement, not clean-signal estimation error.")
                interior.append(row)
        for boundary in MODELS:
            result = case["outputs"][boundary]
            for region, selected in regions(case["data"]).items():
                errors.append(truth_error_row(run_id, case["case_id"], boundary, region, selected, result["mean"], case["truth"]))
            for i in range(N):
                profiles.append(dict(
                    run_id=run_id, case_id=case["case_id"], boundary=boundary, method=METHODS[boundary],
                    index=i, t=case["t"][i], input_valid=int(np.isfinite(case["data"][i])),
                    observed=case["data"][i], truth=case["truth"][i], accepted=int(np.isfinite(result["mean"][i])),
                    mean=result["mean"][i], variance=result["variance"][i], std=result["std"][i],
                    raw_mask_response=result["denominator"][i], rho=np.clip(result["denominator"][i], 0, 1),
                    N_eq=result["support"][i],
                ))
    if any(row["status"] == "fail" for row in comparisons + interior):
        raise ValueError("Numerical comparison failed; no evidence may be saved.")
    return comparisons, errors, profiles, interior


def source_settings():
    paths = {"script": Path(__file__), "inherited_script": Path(inherited.__file__),
             "common_helper": Path(__file__).with_name("_review_common.py"),
             "original_plotting_source": ROOT / "experiments/scripts/generate_boundary_figure.py",
             "original_figure": ROOT / "manuscript/figures/pdf/boundary_sanity_1d.pdf"}
    settings = {f"{name}_sha256": sha256(path) for name, path in paths.items()}
    settings.update({f"toolkit_source_sha256.{name}": digest for name, digest in _RELEASE_HASHES.items()})
    settings.update({f"frozen_{name}_sha256": sha256(FROZEN_RUN / name)
                     for name in ("settings.csv", "comparison.csv", "coverage.csv")})
    return settings


def calculation_settings(case):
    settings = dict(case["inherited_settings"])
    settings.update({key: (value, "") for key, value in source_settings().items()})
    settings.update({
        "script": (str(Path(__file__).resolve()), ""),
        "inherited_run": (str(FROZEN_RUN), ""),
        "inherited_case": (case["case_id"], ""),
        "inherited_saved_script_sha256": (case["frozen_settings"]["script_sha256"], ""),
        "boundary": (list(BOUNDARIES), ""),
        "displayed_models": (list(MODELS), ""),
        "reflective_backend": ("released Cartesian DCT-II; half-sample symmetric", ""),
        "periodic_backend": ("released explicit smooth_polar RFFT; coordinates=polar; axis_types=['linear']; "
                             "boundary=['periodic']; width_units=grid; filter_axes=[0]", ""),
        "references": (REFERENCES, ""),
        "truncated_baseline": ("NumPy: clip each centered window to the input domain, retain finite observations, "
                               "renormalize by their count; centered population variance; no exterior padding", ""),
        "truncated_coverage": ("finite observations / in-domain window positions; not valid / nominal width", ""),
        "truncated_equivalent_count": ("nominal width * coverage; not actual count near shortened outer windows", ""),
        "interior_comparison": ("spectral results vs truncated NumPy on indices [7,193), full windows only; "
                                 "undefined cells excluded but counted; std near zero may amplify variance roundoff", ""),
        "truth_identity": ("1.2*t+2.5*sin(1.5*t); t=np.linspace(0,10,200)", ""),
        "truth_sha256": (hashlib.sha256(case["truth"].tobytes()).hexdigest(), ""),
        "truth_target": ("direct mean vs known clean signal; same accepted sets for both boundaries", ""),
        "comparison_target": ("numerical disagreement vs independent matching-boundary spatial windows", ""),
        "regions": ("all; edge=first/last7; left_edge=first7; right_edge=last7; gap=originally missing", ""),
        "reference_variance": ("centered valid-window population residuals; ddof=0", ""),
        "completion": (False, ""), "restoration": (False, ""),
        "replicate_scope": ("only trend_sine replicate0 complete and middle_gap; no replicate inference", ""),
        "mask_response_atol": (tolerance("mask_response")[0], "1"),
        "mask_response_rtol": (tolerance("mask_response")[1], "1"),
        "checks": ("PASS: exact inherited settings/input/mask/shared noise; 24 frozen reflective regional "
                   "comparisons and 12 coverage rows; all9 released sources; fixed-tolerance mean/variance/std/"
                    "raw mask/count; matching accepted sets for three treatments; 20 full-window interior "
                    "comparisons against truncated NumPy; no filling", ""),
    })
    return settings


def save_evidence(cases, run_id=None, output_root=None):
    # Assemble and check all arrays/rows/settings before creating the unique run.
    comparisons, errors, profiles, interior = evidence_rows(cases, "pending")
    common = environment_settings()
    settings = {case["case_id"]: {**common, **calculation_settings(case)} for case in cases}
    run_id, output = create_run_directory("1d_boundary_comparison", run_id, output_root)
    for rows in (comparisons, errors, profiles, interior):
        for row in rows:
            row["run_id"] = run_id
    setting_rows = [dict(run_id=run_id, case_id=case_id, setting=name,
                         value=value, unit=unit) for case_id, case in settings.items()
                    for name, (value, unit) in sorted(case.items())]
    for filename, fields, rows in (("comparison.csv", COMPARISON_FIELDS, comparisons),
                                   ("truth_errors.csv", TRUTH_FIELDS, errors),
                                   ("profiles.csv", PROFILE_FIELDS, profiles),
                                   ("interior_agreement.csv", COMPARISON_FIELDS, interior),
                                  ("settings.csv", SETTINGS_FIELDS, setting_rows)):
        write_csv(output / filename, fields, rows)
    return output


def verify_saved(cases, saved_run):
    """Recompute all NEW evidence rows, including truth targets and plotted data."""
    saved_run = Path(saved_run).expanduser().resolve()
    expected = evidence_rows(cases, saved_run.name)
    for filename, fields, actual_rows in zip(
        ("comparison.csv", "truth_errors.csv", "profiles.csv", "interior_agreement.csv"),
        (COMPARISON_FIELDS, TRUTH_FIELDS, PROFILE_FIELDS, COMPARISON_FIELDS), expected,
    ):
        old_rows = read_rows(saved_run / filename, fields)
        if len(old_rows) != len(actual_rows):
            raise ValueError(f"Saved evidence row count changed: {filename}")
        for old, actual in zip(old_rows, actual_rows):
            for key, value in actual.items():
                serialized = str(_csv_value(value))
                if key in ("max_abs_difference", "rmse", "max_abs_error") and serialized:
                    statistic = actual["statistic"]
                    atol, rtol = tolerance(statistic)
                    same = bool(old[key]) and np.isclose(float(old[key]), value, atol=atol, rtol=rtol)
                elif filename == "profiles.csv" and key in ("mean", "variance", "std", "raw_mask_response", "rho", "N_eq") and serialized:
                    statistic = {"raw_mask_response": "mask_response", "rho": "mask_response", "N_eq": "support"}.get(key, key)
                    atol, rtol = tolerance(statistic)
                    same = bool(old[key]) and np.isclose(float(old[key]), value, atol=atol, rtol=rtol)
                else:
                    same = old[key] == serialized
                if not same:
                    raise ValueError(f"Saved evidence mismatch: {filename}, {actual['case_id']}, {key}")
    rows = read_rows(saved_run / "settings.csv", SETTINGS_FIELDS)
    if any(row["run_id"] != saved_run.name or row["case_id"] not in CASE_IDS for row in rows):
        raise ValueError("Saved settings run/case identity mismatch.")
    for case in cases:
        selected = [row for row in rows if row["case_id"] == case["case_id"]]
        saved = {row["setting"]: row for row in selected}
        if len(saved) != len(selected):
            raise ValueError("Duplicate saved setting.")
        for name, (value, unit) in calculation_settings(case).items():
            # A later figure-only style edit has a new current script hash. It
            # must still reproduce the saved numerical evidence in full.
            if name == "script_sha256":
                continue
            if (name not in saved or saved[name]["value"] != str(_csv_value(value))
                    or saved[name]["unit"] != unit):
                raise ValueError(f"Saved calculation setting mismatch: {case['case_id']}, {name}")
        for name, value in (("dct_toolkit_version", "1.0.0"), ("dct_toolkit_commit", TOOLKIT_COMMIT)):
            if saved.get(name, {}).get("value") != value:
                raise ValueError("Saved release identity mismatch.")
    return saved_run


def figure_metadata(cases, saved_run):
    metadata = source_settings()
    metadata.update(dict(
        purpose="Width-15 boxcar boundary assumptions and local statistics for one nonperiodic signal",
        figure_kind="new approved two-case comparison with clipped-window NumPy; not historical Gaussian reproduction",
        selected_run=saved_run.name, selected_run_directory=str(saved_run), selected_cases=list(CASE_IDS),
        inherited_run=str(FROZEN_RUN), master_seed=MASTER_SEED, replicate=0, array_size=N,
        truth_identity="1.2*t+2.5*sin(1.5*t); t=np.linspace(0,10,200)",
        truth_sha256=hashlib.sha256(cases[0]["truth"].tobytes()).hexdigest(),
        kernel="centered odd discrete boxcar", requested_width=WIDTH, discrete_width=WIDTH,
        boundaries=list(BOUNDARIES), periodic_backend=periodic_options(), reflective_backend="Cartesian DCT-II",
        displayed_models=list(MODELS),
        denominator_floor=FLOOR, denominator_rule="finite raw d strictly > floor; no additive epsilon",
        completion=False, restoration=False, dct_toolkit_commit=TOOLKIT_COMMIT, dct_toolkit_version="1.0.0",
        reference="independent NumPy wrapped/reflected windows; centered population variance ddof=0",
        numerical_target="matched-boundary validation in comparison.csv; full-window truncated-NumPy agreement in interior_agreement.csv",
        truth_target="direct mean vs known clean signal on matching accepted sets; truth_errors.csv",
        regions="all; first/last7 combined and separately; originally missing inputs (14/40 comparable)",
        panel_order="(a) complete mean; (b) gapped mean; (c) left zoom of a; (d) gapped std; (e) right zoom of a; (f) gapped rho",
        colors=COLORS, panel_limits=PANEL_LIMITS, zoom_boxes=ZOOMS,
        observed_display="faint #bbbbbb points, alpha=0.45; no connecting line",
        truth_display="dashed #333333, not noisy observations", axes_display="thin gray arrow axes; no frame/grid; regular text",
        model_display="green periodic / orange reflective; blue dash-dot NumPy truncated; no reference markers",
        numpy_display="single clipped-window NumPy baseline; no exterior padding; finite-observation renormalization",
        numpy_label=NUMPY_LABEL,
        numpy_coverage="valid / available in-domain positions; shortened outer windows are not missing observations",
        coverage_display="coincident green solid / orange dashed / blue dotted curves",
        local_std_interpretation="population spread includes signal structure; not mean uncertainty or noise variance",
        missing_shading="original [80,120) input cells; midpoint edges; only b,d,f",
        undefined_display="NaN curve segments unbridged; 26 unsupported positions excluded from truth RMSE",
        checks="PASS: exact old generation settings/input/mask/shared noise; frozen reflective numerics/counts; "
               "all9 released source hashes; fixed-tolerance wrapped/reflected reference checks including raw mask; "
                "matching three-model accepted sets; 100 matched-boundary comparisons, 20 interior comparisons, "
                "30 truth rows, 1200 profile rows recomputed against selected new evidence; no completion; no benchmark",
    ))
    for statistic in (*TOLERANCES, "mask_response"):
        metadata[f"{statistic}_atol"], metadata[f"{statistic}_rtol"] = tolerance(statistic)
    saved_settings = read_rows(saved_run / "settings.csv", SETTINGS_FIELDS)
    for filename in ("settings.csv", "comparison.csv", "truth_errors.csv", "profiles.csv", "interior_agreement.csv"):
        metadata[f"saved_{filename}_sha256"] = sha256(saved_run / filename)
    for case in cases:
        case_id = case["case_id"]
        for key in ("input_sha256", "mask_sha256", "seed_sequence_entropy", "noise_spawn_key", "mask_spawn_key"):
            metadata[f"{case_id}.{key}"] = case["inherited_settings"][key][0]
        metadata[f"{case_id}.inherited_saved_script_sha256"] = case["frozen_settings"]["script_sha256"]
        metadata[f"{case_id}.new_saved_script_sha256"] = next(
            row["value"] for row in saved_settings if row["case_id"] == case_id and row["setting"] == "script_sha256")
        metadata[f"{case_id}.interior_single_observation_windows"] = int(np.sum(
            (case["outputs"]["truncated"]["n_observed"] == 1)
            & (np.arange(N) >= WIDTH // 2) & (np.arange(N) < N - WIDTH // 2)))
        for boundary in MODELS:
            accepted = np.isfinite(case["outputs"][boundary]["mean"])
            prefix = f"{case_id}.{boundary}"
            metadata[f"{prefix}.n_defined"] = int(accepted.sum())
            metadata[f"{prefix}.n_undefined"] = int((~accepted).sum())
            metadata[f"{prefix}.n_defined_at_missing"] = int((accepted & ~np.isfinite(case["data"])).sum())
            for region, selected in regions(case["data"]).items():
                row = truth_error_row(saved_run.name, case_id, boundary, region, selected,
                                      case["outputs"][boundary]["mean"], case["truth"])
                for field in ("n_considered", "n_compared", "n_undefined", "rmse"):
                    metadata[f"{prefix}.{region}.truth_{field}"] = row[field]
    for row in evidence_rows(cases, saved_run.name)[3]:
        prefix = f"{row['case_id']}.{row['method']}.interior.{row['statistic']}"
        for key in ("n_considered", "n_compared", "n_method_undefined", "max_abs_difference", "rmse"):
            metadata[f"{prefix}.{key}"] = row[key]
    return metadata


def create_figure(cases):
    # Local styling only: no changes to the shared helper or prior figure styles.
    from _review_figures import configure_plotting
    configure_plotting(use_agg=True)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    mpl.rcParams.update({"font.size": 10.4, "axes.labelsize": 10.4, "axes.titlesize": 10.4,
                         "xtick.labelsize": 10.4, "ytick.labelsize": 10.4, "legend.fontsize": 10.4,
                         "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal"})
    fig, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.065, top=0.855, hspace=0.62, wspace=0.39)
    titles = ("(a) Complete input", "(b) Middle gap", "(c) Left boundary",
              "(d) Local standard deviation", "(e) Right boundary", "(f) Local data coverage")
    t = cases[0]["t"]
    edges = np.r_[0, (t[:-1] + t[1:]) / 2, 10]
    for i, ax in enumerate(axes.flat):
        (xlim, ylim) = PANEL_LIMITS[i]
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_title(titles[i], loc="left", pad=8, fontweight="normal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.tick_params(length=3, width=0.5, color=COLORS["axes"], labelcolor="#555555", pad=2)
        for xy in ((1.025, 0), (0, 1.015)):
            ax.annotate("", xy=xy, xytext=(0, 0), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=COLORS["axes"], lw=0.8),
                        annotation_clip=False, gid="light_axis_arrow")
        ax.set_xlabel(r"$t$", loc="right", labelpad=1, color="#555555")
        ax.set_ylabel("Std (a.u.)" if i == 3 else (r"Coverage $\rho$" if i == 5 else "Signal (a.u.)"),
                      color="#555555", labelpad=4)
        if i in (1, 3, 5):
            ax.axvspan(edges[80], edges[120], color="#eeeeee", linewidth=0,
                       zorder=0, gid="original_missing_interval")
        if i in (0, 1, 2, 4):
            case = cases[1] if i == 1 else cases[0]
            selected = (t >= xlim[0]) & (t <= xlim[1])
            ax.plot(t[selected], case["data"][selected], linestyle="none", marker="o", markersize=2.5, markeredgewidth=0,
                    color=COLORS["observed"], alpha=0.45, zorder=1, gid="observed")
            ax.plot(t[selected], case["truth"][selected], color=COLORS["truth"], linestyle=(0, (4, 4)),
                    linewidth=1.0, zorder=2, gid="truth")
            statistic = "mean"
        else:
            case, selected = cases[1], np.ones(N, dtype=bool)
            statistic = "std" if i == 3 else "rho"
        for boundary in MODELS:
            result = case["outputs"][boundary]
            values = np.clip(result["denominator"], 0, 1) if statistic == "rho" else result[statistic]
            linestyle = ((0, (1, 2)) if statistic == "rho" else (0, (4, 2, 1, 2))) if boundary == "truncated" else (
                (0, (4, 3)) if statistic == "rho" and boundary == "reflective" else "-")
            ax.plot(t[selected], values[selected], color=COLORS[boundary],
                    linewidth={"periodic": 1.8, "reflective": 1.3, "truncated": 1.2}[boundary],
                    linestyle=linestyle, zorder={"periodic": 3, "reflective": 4, "truncated": 5}[boundary],
                    gid=f"{boundary}_{statistic}")
        if i in (0, 1):
            ax.set(xticks=[0, 5, 10], yticks=[0, 5, 10, 15])
        elif i == 2:
            ax.set(xticks=[0, 0.4, 0.8, 1.2], yticks=[0, 3, 6])
        elif i == 4:
            ax.set(xticks=[9, 9.5, 10], yticks=[8, 12, 16])
        elif i == 3:
            ax.set(xticks=[0, 5, 10], yticks=[0, 3, 6])
        else:
            ax.set(xticks=[0, 5, 10], yticks=[0, 0.5, 1])
    for xlim, ylim in ZOOMS:
        axes[0, 0].add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                                       edgecolor=COLORS["zoom"], facecolor="none", linestyle="--",
                                       linewidth=0.8, zorder=5, gid="boundary_zoom_box"))
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS["observed"],
                      alpha=0.45, markersize=4),
               Line2D([], [], color=COLORS["truth"], linestyle=(0, (4, 4)), linewidth=1),
               Line2D([], [], color=COLORS["periodic"], linewidth=1.8),
               Line2D([], [], color=COLORS["reflective"], linewidth=1.3),
               Line2D([], [], color=COLORS["truncated"], linestyle=(0, (4, 2, 1, 2)), linewidth=1.2)]
    fig.legend(handles, ["Noisy observations", "True signal", "Periodic (FFT)", "Reflective (DCT)",
                         NUMPY_LABEL],
               loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=3,
               columnspacing=1.4, handlelength=1.8, handletextpad=0.6, labelspacing=0.65)
    for text in fig.findobj(Text):
        if text.get_fontsize() < 10.4:
            text.set_fontsize(10.4)
    return fig


def check_figure_output(output):
    output = Path(output).expanduser().resolve()
    if output.is_relative_to(ROOT / "experiments/results"):
        raise ValueError("Figure output must not be inside saved experiment results.")
    if output.is_relative_to(ROOT) and not output.is_relative_to(FIGURE_DIRECTORY):
        raise ValueError("Historical and other workspace assets are protected.")
    files = [output / kind / f"{FIGURE_STEM}.{kind}" for kind in ("pdf", "png")]
    sidecar = output / "figure_settings.csv"
    if any(path.exists() for path in (*files, sidecar)):
        if not sidecar.is_file() or {row["figure"] for row in read_rows(sidecar, ["figure", "setting", "value"])} != {FIGURE_STEM}:
            raise ValueError("Refusing to overwrite assets without this new figure's own sidecar.")
    return output


def export_comparison(cases, saved_run, output=FIGURE_DIRECTORY):
    from _review_figures import export_figure
    output = check_figure_output(output)
    if output.is_relative_to(Path(saved_run).expanduser().resolve()):
        raise ValueError("Figure output must not be inside saved experiment results.")
    saved_run = verify_saved(cases, saved_run)
    metadata = figure_metadata(cases, saved_run)
    fig = create_figure(cases)
    import matplotlib.pyplot as plt
    try:
        return export_figure(fig, output, FIGURE_STEM, metadata)
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="new unique run name; refuses overwrite")
    parser.add_argument("--output-root", type=Path, help="round root for a new 1d_boundary_comparison run")
    parser.add_argument("--figure-only", action="store_true", help="verify saved new evidence; write assets only")
    parser.add_argument("--saved-run", type=Path, help="new evidence run for --figure-only (default: selected run)")
    parser.add_argument("--figure-output", type=Path, default=FIGURE_DIRECTORY)
    args = parser.parse_args(argv)
    if args.figure_only and (args.run_id is not None or args.output_root is not None):
        parser.error("--figure-only cannot create a numerical run")
    if args.saved_run is not None and not args.figure_only:
        parser.error("--saved-run requires --figure-only")
    check_figure_output(args.figure_output)  # Reject protected destinations before any calculation.
    cases = prepare_comparison()
    if args.figure_only:
        saved_run = args.saved_run or SELECTED_RUN
    else:
        saved_run = save_evidence(cases, args.run_id, args.output_root)
    paths = export_comparison(cases, saved_run, args.figure_output)
    print(f"Verified two cases; 100 matched-boundary and 20 interior comparisons; 30 separate truth-error rows: {saved_run}")
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
