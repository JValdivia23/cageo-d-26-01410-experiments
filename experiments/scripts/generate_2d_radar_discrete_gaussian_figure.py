#!/usr/bin/env python3
"""Reproduce the selected Gaussian radar maps without changing saved results.

The numerical driver and all ten saved evidence files are checked before the
natural-mask W15 spectral calculation is reproduced. All nine returned arrays
must match the saved worker hashes. The existing radar layout is reused without
its analytical-filter negative-variance overlay. No reference or timing is run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

import benchmark_2d_radar_discrete_gaussian as experiment


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "radar-discrete-gaussian-20261110"
RUN_DIRECTORY = ROOT / "experiments/results/round1/2d_radar_discrete_gaussian" / RUN_ID
CASE_ID = "natural_W15"
METHOD = "discrete_spectral"
STEM = "radar_gaussian_statistics"
FIGURE_DIRECTORY = ROOT / "manuscript/figures/round1/2d_radar_discrete_gaussian"
SAVED_SHA256 = {
    "agreement_summary.csv": "6d087be3818f53ff6329cb288649fd4b75c093431f63bf57a485d8cb5d11310f",
    "benchmark_phases.csv": "8af29461d09304b9c00a17800aeab70251422756cac85b69a7df036cc043924e",
    "benchmark_runs.csv": "3db2569101e40b8d3be477cc23c2824b4a1d41416cd9f6cc061b52df7912dbad",
    "benchmark_summary.csv": "bc27de8188f78328f53df51203601bada8715243f08e67d9f6a077c458fc3f69",
    "comparison.csv": "bc5e722b19bf7758cbe5aa8d2bf1fb49ecda6874fa394854e5811974d5b153a3",
    "kernel_checks.csv": "6c75c45ead0475ad176de24636b265856ccaa7082603b940d1de8fc8566814af",
    "kernel_comparison.csv": "eb3ed955f51ac64690beece95d8d7909e8222ba7eacd663f1381099a00389066",
    "settings.csv": "d6bde42ce270e1bda6c381fcb0c07579950334f81b2ddd4bd3679375d45b0afc",
    "timing_summary.csv": "2d449b21124f438c2e688d6cbe397eb170c7092532d5c9fe402ce6b2762f93b8",
    "variance_diagnostics.csv": "f2705564acc684c0ec69fcd318f5b5a714b7ae88c4596a1c5e7bd07fa29d1bb4",
}


def saved_evidence(directory=RUN_DIRECTORY):
    evidence = {}
    for name, expected in SAVED_SHA256.items():
        path = directory / name
        if experiment.base.radar.sha256_file(path) != expected:
            raise ValueError(f"Saved Gaussian evidence SHA-256 mismatch: {name}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if (not rows or any(None in row or None in row.values() or row["run_id"] != RUN_ID for row in rows)):
            raise ValueError(f"Invalid selected Gaussian evidence: {name}")
        evidence[name] = rows
    frozen = {row["setting"]: row["value"] for row in evidence["settings.csv"] if row["case_id"] == CASE_ID}
    expected = {
        "verified": "True", "shape": "[720, 1180]", "width": "15.0", "az_res_deg": "0.5",
        "input_sha256": experiment.base.DECODED_SHA256,
        "documented_input_sha256": experiment.base.radar.DOCUMENTED_SHA256,
        "kernel_radius": "ceil(4*sigma)", "kernel_truncate": "4.0",
        "gaussian_sigma": "converted_grid_width/sqrt(12)",
        "denominator_floor": "0.001", "completion": "False", "restoration": "False",
        "dct_toolkit_version": "1.0.0", "dct_toolkit_commit": experiment.base.TOOLKIT_COMMIT,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"Selected Gaussian setting mismatch: {key}")
    if json.loads(frozen["source_sha256"]) != experiment.source_hashes():
        raise ValueError("Selected Gaussian numerical sources changed")
    if (any(row["status"] != "pass" for name in ("comparison.csv", "kernel_checks.csv") for row in evidence[name])
            or any(row["status"] != "success" for row in evidence["benchmark_runs.csv"])):
        raise ValueError("Selected Gaussian evidence is not verified")
    return frozen, evidence


def prepare_figure(saved_run=RUN_DIRECTORY, radar_path=None):
    from _review_common import _RELEASE_HASHES

    radar = experiment.base.radar
    frozen, evidence = saved_evidence(saved_run)
    colormap = radar._figure_reflectivity_colormap()  # Reject missing required palette before radar I/O.
    path = Path(radar_path or os.environ.get("REAL_RADAR_PATH", str(radar.DEFAULT_RADAR))).expanduser().resolve()
    data, ranges, az_res, metadata = radar.read_radar(path, quick=False)
    if (metadata["input_sha256"][0] != radar.DOCUMENTED_SHA256
            or experiment.base.array_sha256(data) != frozen["input_sha256"]
            or data.shape != experiment.SHAPE or az_res != float(frozen["az_res_deg"])):
        raise ValueError("Figure radar file, decoded input or geometry differs from saved evidence")
    toolkit = experiment.base.load_toolkit()
    audit = {}
    fields = experiment.calculate(METHOD, data, 15., az_res, toolkit, audit)
    fingerprints = experiment.base.fingerprints(fields)
    if (fingerprints != json.loads(frozen["output_sha256"])[METHOD]
            or audit != experiment.expected_audit(METHOD, data.shape)):
        raise ValueError("Figure outputs or filter counts differ from saved Gaussian workers")
    accepted = np.isfinite(fields["raw_mask_response"]) & (fields["raw_mask_response"] > experiment.FLOOR)
    negative = accepted & (fields["raw_variance"] < 0)
    for key in ("mean", "second_moment", "raw_variance", "variance", "std"):
        if not np.array_equal(np.isfinite(fields[key]), accepted):
            raise ValueError(f"Figure undefined mask differs: {key}")
    central = ranges <= 10000.
    selections = {"all": np.ones(data.shape, bool), "centre": np.broadcast_to(central, data.shape)}
    for region, selection in selections.items():
        row = next(row for row in evidence["variance_diagnostics.csv"]
                   if row["case_id"] == CASE_ID and row["method"] == METHOD and row["region"] == region)
        if (int(row["n_defined"]) != int(accepted[selection].sum())
                or int(row["n_negative_raw_variance"]) != int(negative[selection].sum())
                or float(row["min_raw_variance"]) != float(np.nanmin(fields["raw_variance"][selection]))):
            raise ValueError(f"Figure regional diagnostics differ: {region}")
    if np.nanmin(fields["raw_variance"]) < -experiment.base.TOLERANCES["variance"][0]:
        raise ValueError("Figure has a negative raw variance beyond the fixed tolerance")
    azimuth = radar._figure_selected_azimuth(path, metadata)
    x, y, angular_edges, radial_edges = radar._figure_polar_mesh(azimuth, ranges)
    cx = np.sin(np.deg2rad(azimuth))[:, None] * ranges[None, :] / 1000
    cy = np.cos(np.deg2rad(azimuth))[:, None] * ranges[None, :] / 1000
    zoom = (np.abs(cx) <= 10) & (np.abs(cy) <= 10)
    enclosing_radius = np.sqrt(2) * 10 / np.cos(np.diff(angular_edges).max() / 2)
    count = int(np.searchsorted(radial_edges, enclosing_radius))
    if count >= radial_edges.size:
        raise ValueError("Native mesh does not enclose all four square-zoom corners")
    settings = {
        "purpose": "COW DBZHC Gaussian mean, standard deviation and local data coverage",
        "figure_kind": "reproduction of selected natural-mask W15 discrete_spectral arrays; no new numerical run or timing",
        "selected_run": RUN_ID, "selected_case": CASE_ID, "selected_method": METHOD,
        "numerical_reference": experiment.REFERENCE,
        "input_filename": path.name, "input_sha256": radar.DOCUMENTED_SHA256,
        "decoded_input_sha256": frozen["input_sha256"],
        "script_sha256": radar.sha256_file(Path(__file__)),
        "numerical_source_sha256": experiment.source_hashes(),
        "output_sha256": fingerprints, "filter_audit": audit,
        "dct_toolkit_commit": experiment.base.TOOLKIT_COMMIT, "dct_toolkit_version": toolkit.__version__,
        "implementation_scope": "experiment-local sampled Gaussian response; not the released Gaussian API",
        "kernel": frozen["kernel_discretization"], "gaussian_sigma": frozen["gaussian_sigma"],
        "kernel_radius": frozen["kernel_radius"], "kernel_truncate": 4.,
        "width": 15., "width_convention": frozen["width_convention"],
        "boundaries": frozen["boundaries"], "operator_order": frozen["operator_order"],
        "denominator_floor": experiment.FLOOR, "denominator_rule": "finite raw d > 0.001, strict",
        "completion": False, "restoration": False,
        "new_timing_performed": False, "new_spatial_reference_performed": False,
        "n_input_observed": int(np.isfinite(data).sum()), "n_accepted": int(accepted.sum()),
        "n_negative_raw_variance": int(negative.sum()),
        "negative_overlay": "none; numerical raw-variance diagnostics retained here, not highlighted on the maps",
        "variance_reporting": "max(M2-mu^2,0); std is its square root",
        "coverage_definition": "rho=clip(raw d,0,1), including cells with undefined moments",
        "central_definition": "range <= 10000 m, all azimuths; distinct from square display",
        "central_n_accepted": int(accepted[:, central].sum()),
        "central_n_negative_raw_variance": int(negative[:, central].sum()),
        "central_gate_count": int(central.sum()),
        "central_last_gate_center_m": float(ranges[central][-1]),
        "zoom_definition": "-10<=x<=10 km and -10<=y<=10 km",
        "zoom_processing": "viewport after full 720x1180 filtering; no cropped filtering or resampling",
        "zoom_count_definition": "native cell centers inside square; boundary polygons clipped by viewport",
        "zoom_gate_count": count, "zoom_outer_cell_corner_m": float(radial_edges[count] * 1000),
        "zoom_n_cells": int(zoom.sum()), "zoom_n_observed": int(np.isfinite(data)[zoom].sum()),
        "zoom_n_accepted": int(accepted[zoom].sum()), "zoom_n_undefined": int((~accepted)[zoom].sum()),
        "zoom_n_negative_raw_variance": int(negative[zoom].sum()),
        "plot_coordinates": "x=slant_range*sin(actual_azimuth), y=slant_range*cos(actual_azimuth), km; not surface/geodesic",
        "native_mesh": "native midpoint corners and exact circular seam closure; no resampling or transposition",
        "azimuth_coordinate_sha256": experiment.base.array_sha256(azimuth),
        "range_coordinate_sha256": experiment.base.array_sha256(ranges),
        "reflectivity_colormap": colormap.name,
        "reflectivity_colormap_source": "cmweather.cm_colorblind.ChaseSpectral; required, no fallback",
        "reflectivity_color_limits_dbz": [0, 40], "reflectivity_colorbar_extend": "both",
        "std_colormap": "cividis", "std_color_limits_dbz": [0, 10], "std_colorbar_extend": "max",
        "coverage_colormap": "YlGnBu", "coverage_color_limits": [0, 1],
        "full_axis_limits_km": [-90, 90], "zoom_axis_limits_km": [-10, 10],
        "panel_order": "observed, mean, full std, full rho, square-zoom std, square-zoom rho",
        "figure_layout": "inherited 6.5x7.4 inch canvas, 1.95 inch square panels; no clamping key",
        "data_panel_size_in": [1.95, 1.95],
        "dense_mesh_rendering": "rasterized native cells; labels, axes and zoom squares remain vector",
        "checks": "PASS: ten immutable evidence files; numerical source hashes; exact raw and decoded radar input; nine released source files; all nine output hashes and filter audit; defined masks; regional counts and raw minima; native mesh and post-filter square zoom",
    }
    for name, value in SAVED_SHA256.items():
        settings[f"saved_{name}_sha256"] = value
    for name, value in _RELEASE_HASHES.items():
        settings[f"toolkit_source_sha256.{name}"] = value
    for name in ("numpy", "scipy", "astropy", "xarray", "netCDF4", "cmweather"):
        settings[f"{name}_version"] = importlib.metadata.version(name)
    for key, values, limits in (("observed", data, (0, 40)), ("mean", fields["mean"], (0, 40)),
                                ("std", fields["std"], (0, 10)), ("raw_variance", fields["raw_variance"], None)):
        finite = values[np.isfinite(values)]
        settings[f"{key}_finite_min"] = float(finite.min())
        settings[f"{key}_finite_max"] = float(finite.max())
        if limits is not None:
            settings[f"{key}_below_color_limit_count"] = int(np.sum(finite < limits[0]))
            settings[f"{key}_above_color_limit_count"] = int(np.sum(finite > limits[1]))
    case = dict(data=data, ranges=ranges, azimuth=azimuth, x=x, y=y, angular_edges=angular_edges,
                radial_edges=radial_edges, fields=fields, accepted=accepted, negative=negative,
                central=central, zoom=zoom, zoom_gate_count=count)
    return case, settings


def create_figure(case):
    """Reuse the approved geometry and style, omitting analytical-filter overlays."""
    fig = experiment.base.radar.create_radar_figure(case)
    for ax in fig.axes[:6]:
        for artist in list(ax.collections):
            if artist.get_gid() == "negative_raw_variance":
                artist.remove()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.axes[-2]._colorbar.set_label("Std (dBZ)", labelpad=5)
    return fig


def export(output=FIGURE_DIRECTORY, saved_run=RUN_DIRECTORY, radar_path=None):
    from _review_figures import export_figure

    output = Path(output).expanduser().resolve()
    if output.is_relative_to((ROOT / "experiments/results").resolve()):
        raise ValueError("Figure output must not be inside saved numerical results")
    case, settings = prepare_figure(saved_run, radar_path)
    fig = create_figure(case)
    import matplotlib.pyplot as plt

    try:
        return export_figure(fig, output, STEM, settings)
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--saved-run", type=Path, default=RUN_DIRECTORY)
    parser.add_argument("--radar-path", type=Path)
    args = parser.parse_args(argv)
    for kind, path in export(args.output, args.saved_run, args.radar_path).items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
