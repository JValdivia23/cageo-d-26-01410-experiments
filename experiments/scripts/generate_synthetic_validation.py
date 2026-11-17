#!/usr/bin/env python3
"""Generate multi-seed synthetic validation metrics and figures for arXiv."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def save_figure_outputs(
    fig: plt.Figure,
    figures_dir: Path,
    stem: str,
    *,
    legacy_png: bool = True,
    legacy_png_dpi: int = 220,
    trim_top: bool = False,
) -> dict[str, str]:
    """Save paired exports, optionally trimming only the unused top margin."""
    png_dir = figures_dir / "png"
    pdf_dir = figures_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / f"{stem}.png"
    pdf_path = pdf_dir / f"{stem}.pdf"
    bbox_inches = None
    if trim_top:
        fig.canvas.draw()
        content = fig.get_tightbbox(fig.canvas.get_renderer())
        bbox_inches = matplotlib.transforms.Bbox.from_extents(
            0.0,
            0.0,
            fig.get_figwidth(),
            min(fig.get_figheight(), content.y1 + 0.08),
        )
    fig.savefig(png_path, dpi=300, bbox_inches=bbox_inches)
    fig.savefig(pdf_path, bbox_inches=bbox_inches)

    outputs = {"png_300dpi": str(png_path), "pdf_vector": str(pdf_path)}
    if legacy_png:
        legacy_path = figures_dir / f"{stem}.png"
        fig.savefig(legacy_path, dpi=legacy_png_dpi, bbox_inches=bbox_inches)
        outputs["legacy_png"] = str(legacy_path)
    return outputs


def _ensure_dct_toolkit_import() -> None:
    """Ensure dct_toolkit is importable."""
    env_path = os.environ.get("DCT_TOOLKIT_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.exists():
            sys.path.insert(0, str(candidate))

    try:
        import dct_toolkit
    except ImportError as exc:
        raise ImportError(
            "dct_toolkit is not importable. Install it in the active Python "
            "environment or set DCT_TOOLKIT_PATH to the repository root "
            "that contains the dct_toolkit package."
        ) from exc

    version_raw = str(getattr(dct_toolkit, "__version__", ""))
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_raw)
    if not match:
        raise ImportError(
            "Unable to verify dct_toolkit version. Expected >= 0.5.0; "
            "set DCT_TOOLKIT_PATH to a compatible repository checkout."
        )
    version_tuple = tuple(int(part) for part in match.groups())
    if version_tuple < (0, 5, 0):
        raise ImportError(
            f"dct_toolkit version {version_raw} is too old; expected >= 0.5.0. "
            "Set DCT_TOOLKIT_PATH to a compatible repository checkout."
        )


def _safe_git_head(repo_root: Path | None) -> str:
    if repo_root is None:
        return "unknown"
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _dct_toolkit_provenance() -> dict[str, str]:
    import dct_toolkit  # pylint: disable=import-outside-toplevel

    module_file = getattr(dct_toolkit, "__file__", None)
    module_path = Path(module_file).resolve() if module_file else None
    package_root = module_path.parent if module_path else None
    repo_root = package_root.parent if package_root else None
    return {
        "dct_toolkit_version": str(getattr(dct_toolkit, "__version__", "unknown")),
        "dct_toolkit_module_path": str(module_path) if module_path else "unknown",
        "dct_toolkit_repo_root": str(repo_root) if repo_root else "unknown",
        "dct_toolkit_git_commit": _safe_git_head(repo_root),
    }


_ensure_dct_toolkit_import()

from dct_toolkit import dct_count, dct_mean, dct_smooth, dct_std, dct_variance


def synthetic_field(shape: tuple[int, int]) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, shape[0])
    x = np.linspace(-1.0, 1.0, shape[1])
    yy, xx = np.meshgrid(y, x, indexing="ij")
    base = 0.8 * np.sin(2.5 * np.pi * xx) + 0.6 * np.cos(2.0 * np.pi * yy)
    bump = 1.3 * np.exp(-((xx - 0.35) ** 2 + (yy + 0.15) ** 2) / 0.08)
    slope = 0.35 * xx - 0.25 * yy
    return base + bump + slope


def polar_truth(shape: tuple[int, int]) -> np.ndarray:
    n_az, n_r = shape
    theta = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
    r = np.linspace(1.0, float(n_r), n_r)
    th, rr = np.meshgrid(theta, r, indexing="ij")
    periodic = np.sin(3.0 * th) + 0.4 * np.cos(5.0 * th + 0.07 * rr)
    radial = 0.003 * rr + 0.25 * np.exp(-((rr - 70.0) ** 2) / 900.0)
    return periodic + radial


def synthetic_cyclone_wind_3d(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth idealized 3D cyclone-like wind field (u, v, w)."""
    nz, ny, nx = shape
    z = np.linspace(-1.0, 1.0, nz)
    y = np.linspace(-1.0, 1.0, ny)
    x = np.linspace(-1.0, 1.0, nx)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    r = np.sqrt(xx**2 + yy**2) + 1e-6
    r0 = 0.45
    vtan = 1.15 * (r / r0) * np.exp(1.0 - (r / r0))
    shear = 1.0 + 0.30 * np.sin(np.pi * zz)

    u_tan = -vtan * (yy / r) * shear
    v_tan = vtan * (xx / r) * shear

    w_core = 0.24 * np.exp(-((r / 0.35) ** 2)) * np.cos(0.5 * np.pi * zz)

    u_bg = 0.05 * np.sin(2.0 * np.pi * zz) + 0.03 * xx
    v_bg = -0.04 * np.cos(1.5 * np.pi * zz) - 0.02 * yy
    w_bg = 0.04 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)

    u = u_tan + u_bg
    v = v_tan + v_bg
    w = w_core + w_bg
    return u, v, w


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_mcar_mask(
    shape: tuple[int, int], miss_rate: float, rng: np.random.Generator
) -> np.ndarray:
    return rng.random(shape) >= miss_rate


def make_block_mask(
    shape: tuple[int, int], gap_frac: float, rng: np.random.Generator
) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    total = shape[0] * shape[1]
    block_cells = int(total * gap_frac)
    h = max(4, int(np.sqrt(block_cells)))
    w = max(4, int(np.sqrt(block_cells)))
    h = min(h, shape[0] - 1)
    w = min(w, shape[1] - 1)
    y0 = int(rng.integers(0, shape[0] - h))
    x0 = int(rng.integers(0, shape[1] - w))
    mask[y0 : y0 + h, x0 : x0 + w] = False
    return mask


def summarize_runs(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(arr))
    ci95_half_width = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci95_half_width,
        "ci95_high": mean + ci95_half_width,
    }


def run_missingness_experiment(
    truth: np.ndarray,
    width: tuple[float, float],
    rng: np.random.Generator,
) -> dict[str, list[dict[str, float]]]:
    noise_std = 0.25
    levels = [0.1, 0.2, 0.35, 0.5, 0.65]
    out = {"mcar": [], "block": []}

    for level in levels:
        noisy = truth + rng.normal(0.0, noise_std, size=truth.shape)

        mask_mcar = make_mcar_mask(truth.shape, level, rng)
        obs_mcar = noisy.copy()
        obs_mcar[~mask_mcar] = np.nan
        pred_mcar = dct_mean(
            obs_mcar,
            width=width,
            coordinates="cartesian",
            restore_input_nan=False,
        )
        count_mcar = dct_count(
            mask_mcar,
            width=width,
            coordinates="cartesian",
            restore_input_nan=False,
        )
        out["mcar"].append(
            {
                "missing_fraction": float(level),
                "rmse": rmse(pred_mcar, truth),
                "effective_count_mean": float(np.nanmean(count_mcar)),
            }
        )

        mask_block = make_block_mask(truth.shape, level, rng)
        obs_block = noisy.copy()
        obs_block[~mask_block] = np.nan
        pred_block = dct_mean(
            obs_block,
            width=width,
            coordinates="cartesian",
            restore_input_nan=False,
        )
        count_block = dct_count(
            mask_block,
            width=width,
            coordinates="cartesian",
            restore_input_nan=False,
        )
        out["block"].append(
            {
                "missing_fraction": float(level),
                "rmse": rmse(pred_block, truth),
                "effective_count_mean": float(np.nanmean(count_block)),
            }
        )

    return out


def periodic_fft_gaussian_smooth(
    field: np.ndarray, width: tuple[float, float]
) -> np.ndarray:
    """Periodic Gaussian smoothing via FFT for boundary sanity comparison."""
    n0, n1 = field.shape
    sigma0 = float(width[0]) / np.sqrt(12.0)
    sigma1 = float(width[1]) / np.sqrt(12.0)
    omega0 = 2.0 * np.pi * np.fft.fftfreq(n0)
    omega1 = 2.0 * np.pi * np.fft.rfftfreq(n1)
    h0 = np.exp(-0.5 * (omega0 * sigma0) ** 2)
    h1 = np.exp(-0.5 * (omega1 * sigma1) ** 2)
    h2d = h0[:, None] * h1[None, :]
    spec = np.fft.rfftn(field, axes=(0, 1))
    smoothed = np.fft.irfftn(spec * h2d, s=field.shape, axes=(0, 1))
    return np.asarray(smoothed, dtype=float)


def run_boundary_experiment(rng: np.random.Generator) -> dict[str, float]:
    shape = (160, 160)
    truth = synthetic_field(shape)
    noise = rng.normal(0.0, 0.25, size=shape)
    obs = truth + noise

    width = (9.0, 9.0)
    pred_ref = dct_smooth(obs, width=width, coordinates="cartesian", prefill_max_iter=0)
    pred_per = periodic_fft_gaussian_smooth(obs, width=width)

    edge = np.zeros(shape, dtype=bool)
    edge_w = 12
    edge[:edge_w, :] = True
    edge[-edge_w:, :] = True
    edge[:, :edge_w] = True
    edge[:, -edge_w:] = True

    return {
        "rmse_reflective_all": rmse(pred_ref, truth),
        "rmse_periodic_all": rmse(pred_per, truth),
        "rmse_reflective_edge": rmse(pred_ref[edge], truth[edge]),
        "rmse_periodic_edge": rmse(pred_per[edge], truth[edge]),
    }


def run_variance_calibration(
    truth: np.ndarray,
    width: tuple[float, float],
    rng: np.random.Generator,
) -> dict[str, object]:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, truth.shape[0]),
        np.linspace(-1.0, 1.0, truth.shape[1]),
        indexing="ij",
    )
    sigma_map = 0.12 + 0.15 * (xx + 1.0) / 2.0 + 0.1 * np.exp(-((yy - 0.2) ** 2) / 0.2)
    true_var_map = sigma_map**2

    n_rep = 30
    sq_err_accum = np.zeros_like(truth)
    est_var_accum = np.zeros_like(truth)

    for _ in range(n_rep):
        mask = make_mcar_mask(truth.shape, 0.2, rng)
        noise = rng.normal(0.0, 1.0, size=truth.shape) * sigma_map
        obs = truth + noise
        obs[~mask] = np.nan

        pred = dct_smooth(obs, width=width, coordinates="cartesian", prefill_max_iter=3)
        var_est = dct_variance(
            obs,
            width=width,
            coordinates="cartesian",
            restore_input_nan=False,
        )

        sq_err_accum += (pred - truth) ** 2
        est_var_accum += var_est

    mse_map = sq_err_accum / n_rep
    est_var_mean = est_var_accum / n_rep

    valid = np.isfinite(est_var_mean) & np.isfinite(true_var_map)
    x = true_var_map[valid].ravel()
    y = est_var_mean[valid].ravel()
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]

    mse_global = float(np.nanmean(mse_map))

    return {
        "slope_est_vs_true_var": float(slope),
        "intercept_est_vs_true_var": float(intercept),
        "corr_est_vs_true_var": float(corr),
        "global_mse_of_smoothed_mean": mse_global,
        "true_var_mean": float(np.mean(true_var_map)),
        "est_var_mean": float(np.mean(est_var_mean)),
        "scatter_true_var": x[::200].tolist(),
        "scatter_est_var": y[::200].tolist(),
    }


def run_timing_summary(
    truth: np.ndarray,
    width: tuple[float, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    mask = make_mcar_mask(truth.shape, 0.25, rng)
    obs = truth + rng.normal(0.0, 0.2, size=truth.shape)
    obs[~mask] = np.nan

    runtimes_ms: dict[str, list[float]] = {
        "dct_mean_cartesian": [],
        "dct_variance_cartesian": [],
        "dct_count_cartesian": [],
        "dct_smooth_cartesian": [],
        "dct_mean_polar_periodic": [],
    }

    polar_obs = polar_truth((360, 120)) + rng.normal(0.0, 0.2, size=(360, 120))
    polar_mask = make_mcar_mask((360, 120), 0.2, rng)
    polar_obs[~polar_mask] = np.nan

    repeats = 6
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = dct_mean(obs, width=width, coordinates="cartesian", restore_input_nan=False)
        runtimes_ms["dct_mean_cartesian"].append(1000.0 * (time.perf_counter() - t0))

        t0 = time.perf_counter()
        _ = dct_variance(
            obs,
            width=width,
            coordinates="cartesian",
            restore_input_nan=False,
        )
        runtimes_ms["dct_variance_cartesian"].append(
            1000.0 * (time.perf_counter() - t0)
        )

        t0 = time.perf_counter()
        _ = dct_count(
            mask,
            width=width,
            coordinates="cartesian",
            restore_input_nan=False,
        )
        runtimes_ms["dct_count_cartesian"].append(1000.0 * (time.perf_counter() - t0))

        t0 = time.perf_counter()
        _ = dct_smooth(obs, width=width, coordinates="cartesian", prefill_max_iter=3)
        runtimes_ms["dct_smooth_cartesian"].append(1000.0 * (time.perf_counter() - t0))

        t0 = time.perf_counter()
        _ = dct_mean(
            polar_obs,
            width=(8.0, 7.0),
            coordinates="polar",
            az_res_deg=1.0,
            az_boundary="periodic",
            restore_input_nan=False,
        )
        runtimes_ms["dct_mean_polar_periodic"].append(
            1000.0 * (time.perf_counter() - t0)
        )

    return {k: float(np.mean(v)) for k, v in runtimes_ms.items()}


def run_boundary_sanity_assets(rng: np.random.Generator) -> dict[str, object]:
    """Generate boundary-sanity arrays for edge-wrap visualization."""
    shape = (160, 160)
    truth = synthetic_field(shape)
    obs = truth + rng.normal(0.0, 0.25, size=shape)

    width = (9.0, 9.0)
    pred_ref = dct_smooth(obs, width=width, coordinates="cartesian", prefill_max_iter=0)
    pred_per = periodic_fft_gaussian_smooth(obs, width=width)

    err_ref = np.abs(pred_ref - truth)
    err_per = np.abs(pred_per - truth)

    edge = np.zeros(shape, dtype=bool)
    edge_w = 12
    edge[:edge_w, :] = True
    edge[-edge_w:, :] = True
    edge[:, :edge_w] = True
    edge[:, -edge_w:] = True

    return {
        "truth": truth,
        "pred_ref": pred_ref,
        "pred_per": pred_per,
        "err_ref": err_ref,
        "err_per": err_per,
        "edge_mask": edge.astype(float),
        "edge_width": edge_w,
        "metrics": {
            "rmse_reflective_all": rmse(pred_ref, truth),
            "rmse_periodic_all": rmse(pred_per, truth),
            "rmse_reflective_edge": rmse(pred_ref[edge], truth[edge]),
            "rmse_periodic_edge": rmse(pred_per[edge], truth[edge]),
        },
    }


def run_outlier_identification_experiment(
    truth_wind_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    width_3d: tuple[float, float, float],
    thresholds: list[float],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Run 3D vector-wind outlier-identification protocol with threshold sweep."""
    u_truth, v_truth, w_truth = truth_wind_3d
    noise_std = 0.14
    outlier_fraction = 0.003

    u_noisy = u_truth + rng.normal(0.0, noise_std, size=u_truth.shape)
    v_noisy = v_truth + rng.normal(0.0, noise_std, size=v_truth.shape)
    w_noisy = w_truth + rng.normal(0.0, noise_std, size=w_truth.shape)

    injected_mask = rng.random(u_truth.shape) < outlier_fraction
    phi = rng.uniform(0.0, 2.0 * np.pi, size=u_truth.shape)
    cos_theta = rng.uniform(-1.0, 1.0, size=u_truth.shape)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    dir_x = sin_theta * np.cos(phi)
    dir_y = sin_theta * np.sin(phi)
    dir_z = cos_theta

    outlier_amp = rng.uniform(1.25, 2.25, size=u_truth.shape)
    u_obs = u_noisy + injected_mask * outlier_amp * dir_x
    v_obs = v_noisy + injected_mask * outlier_amp * dir_y
    w_obs = w_noisy + injected_mask * outlier_amp * dir_z

    speed_obs = np.sqrt(u_obs**2 + v_obs**2 + w_obs**2)
    speed_mean = dct_mean(
        speed_obs,
        width=width_3d,
        coordinates="cartesian",
        restore_input_nan=False,
    )
    speed_std = dct_std(
        speed_obs,
        width=width_3d,
        coordinates="cartesian",
        restore_input_nan=False,
    )
    sigma_local = np.maximum(speed_std, 1e-6)
    residual = np.abs(speed_obs - speed_mean)

    u_mean = dct_mean(
        u_obs,
        width=width_3d,
        coordinates="cartesian",
        restore_input_nan=False,
    )
    v_mean = dct_mean(
        v_obs,
        width=width_3d,
        coordinates="cartesian",
        restore_input_nan=False,
    )
    w_mean = dct_mean(
        w_obs,
        width=width_3d,
        coordinates="cartesian",
        restore_input_nan=False,
    )

    threshold_metrics: list[dict[str, float]] = []
    best_f1 = -1.0
    best_threshold = thresholds[0]
    best_detected = None

    injected = injected_mask.astype(bool)
    n_injected = int(np.sum(injected))

    for thr in thresholds:
        detected = residual > (thr * sigma_local)
        tp = int(np.sum(detected & injected))
        fp = int(np.sum(detected & ~injected))
        fn = int(np.sum(~detected & injected))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        threshold_metrics.append(
            {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
            best_detected = detected

    if best_detected is None:
        raise RuntimeError("Outlier sweep produced no detected mask.")

    per_slice_injected = np.sum(injected, axis=(1, 2))
    if np.any(per_slice_injected > 0):
        z_idx = int(np.argmax(per_slice_injected))
    else:
        z_idx = int(u_truth.shape[0] // 2)

    tp = int(np.sum(best_detected & injected))
    fp = int(np.sum(best_detected & ~injected))
    fn = int(np.sum(~best_detected & injected))

    representative = {
        "slice_index": z_idx,
        "selected_threshold": best_threshold,
        "u_truth": u_truth.astype(float),
        "v_truth": v_truth.astype(float),
        "w_truth": w_truth.astype(float),
        "u_observed": u_obs.astype(float),
        "v_observed": v_obs.astype(float),
        "w_observed": w_obs.astype(float),
        "u_mean": u_mean.astype(float),
        "v_mean": v_mean.astype(float),
        "w_mean": w_mean.astype(float),
        "injected_mask": injected.astype(bool),
        "detected_mask": best_detected.astype(bool),
        "n_injected_total": n_injected,
        "n_detected_total": int(np.sum(best_detected)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1_at_selected_threshold": float(best_f1),
    }

    return {
        "threshold_metrics": threshold_metrics,
        "representative": representative,
        "sigma_local_mean": float(np.mean(sigma_local)),
        "noise_std": float(noise_std),
        "outlier_fraction": float(outlier_fraction),
    }


def aggregate_outlier_threshold_sweep(
    all_runs: list[list[dict[str, float]]], thresholds: list[float]
) -> list[dict[str, object]]:
    """Aggregate threshold-sweep metrics over seeds."""
    summary: list[dict[str, object]] = []
    for idx, thr in enumerate(thresholds):
        row: dict[str, object] = {"threshold": float(thr)}
        for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
            vals = [run[idx][key] for run in all_runs]
            row[key] = summarize_runs(vals)
        summary.append(row)
    return summary


def _import_cflib_from_notebook_path() -> tuple[object, str]:
    """Import cflib using the same local path pattern as the notebook."""
    module_candidates = [
        Path("~/onedrive/code/research").expanduser(),
        Path("~/Library/CloudStorage/OneDrive-UCB-O365/code/research").expanduser(),
    ]

    used_path: Path | None = None
    for module_dir in module_candidates:
        if module_dir.exists():
            if str(module_dir) not in sys.path:
                sys.path.insert(0, str(module_dir))
            used_path = module_dir
            break

    if used_path is None:
        raise ImportError(
            "Unable to locate cflib path. Expected one of: "
            f"{', '.join(str(p) for p in module_candidates)}"
        )

    import cflib  # pylint: disable=import-outside-toplevel

    return cflib, str(used_path)


def _resolve_chasespectral_cmap() -> tuple[str, str]:
    """Register ChaseSpectral when cmweather is available."""
    try:
        import cmweather  # pylint: disable=unused-import,import-outside-toplevel

        return "ChaseSpectral", "available"
    except Exception as exc:  # pragma: no cover
        return "turbo", f"missing ({exc})"


def _format_radar_acquisition_date(dataset, data_path: Path) -> str:
    """Infer acquisition date from dataset time coordinate or file name."""
    try:
        if "time" in dataset.coords and int(dataset["time"].size) > 0:
            t0 = np.datetime64(dataset["time"].values[0])
            return np.datetime_as_string(t0, unit="D")
    except Exception:
        pass

    match = re.search(r"(20\d{2})(\d{2})(\d{2})", str(data_path))
    if match:
        yyyy, mm, dd = match.groups()
        return f"{yyyy}-{mm}-{dd}"
    return "date unavailable"


def run_optional_real_radar_figure(
    fig_dir: Path, dct_toolkit_repo_root: str, seed: int = 20260415
) -> dict[str, object]:
    """Create optional 6-panel real-radar figure using notebook workflow."""
    status: dict[str, object] = {
        "enabled": True,
        "generated": False,
        "status": "not_attempted",
        "path_used": "",
        "dependency_xarray": "unknown",
        "dependency_cflib": "unknown",
        "dependency_cmweather": "unknown",
        "message": "",
    }

    try:
        import xarray as xr  # pylint: disable=import-outside-toplevel

        status["dependency_xarray"] = "available"
    except Exception as exc:  # pragma: no cover
        status["dependency_xarray"] = "missing"
        status["status"] = "dependency_missing"
        status["message"] = f"xarray unavailable: {exc}"
        return status

    candidates: list[Path] = []
    env_path = os.environ.get("REAL_RADAR_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    default_cow1 = Path(
        "~/onedrive/code/research/smoothing/"
        "cfrad_QCed_20220223_023027.051_to_20220223_023410.833_COW1high_SUR.nc"
    ).expanduser()
    candidates.append(default_cow1)

    if dct_toolkit_repo_root and dct_toolkit_repo_root != "unknown":
        repo = Path(dct_toolkit_repo_root)
        candidates.extend(
            [
                repo / "notebooks" / "data" / "sample_radar.nc",
                repo / "notebooks" / "data" / "radar_sample.nc",
                repo / "examples" / "data" / "sample_radar.nc",
            ]
        )

    existing = [p for p in candidates if p.exists()]
    status["candidate_paths"] = [str(p) for p in candidates]

    if not existing:
        status["status"] = "missing_data_path"
        status["message"] = "No real-radar file found; set REAL_RADAR_PATH to enable."
        return status

    data_path = existing[0]
    status["path_used"] = str(data_path)

    data = None
    try:
        cflib, cflib_path = _import_cflib_from_notebook_path()
        status["dependency_cflib"] = "available"
        status["cflib_path"] = cflib_path

        cmap_name, cmweather_status = _resolve_chasespectral_cmap()
        status["dependency_cmweather"] = cmweather_status

        data = xr.open_dataset(data_path, decode_timedelta=False)

        preferred = ["DBZHC", "DBZHCC", "DBZH", "reflectivity", "ZH", "z", "data"]
        raw_field = "DBZHC" if "DBZHC" in data.data_vars else None
        if raw_field is None:
            for key in preferred:
                if key in data.data_vars and np.issubdtype(data[key].dtype, np.number):
                    raw_field = key
                    break
        if raw_field is None:
            status["status"] = "error"
            status["message"] = "Dataset has no numeric variable for plotting."
            return status

        da = data[raw_field]
        if da.ndim < 2:
            status["status"] = "error"
            status["message"] = (
                f"Variable '{raw_field}' has ndim={da.ndim}; expected >=2."
            )
            return status

        # Notebook-matched workflow: smooth full field values, then render
        # with cflib.plot_cfradial(..., elv=2).
        az_res_deg = 0.5
        narrow_field = f"{raw_field}_gaussian5"
        broad_field = f"{raw_field}_gaussian15"

        data[narrow_field] = data[raw_field].copy()
        data[narrow_field].values = dct_smooth(
            data[narrow_field].values,
            width=5,
            coordinates="polar",
            az_res_deg=az_res_deg,
            kernel_type="gaussian",
            az_boundary="periodic",
        )

        data[broad_field] = data[raw_field].copy()
        data[broad_field].values = dct_smooth(
            data[broad_field].values,
            width=15,
            coordinates="polar",
            az_res_deg=az_res_deg,
            kernel_type="gaussian",
            az_boundary="periodic",
        )

        zoom_meta = make_real_radar_6panel_figure(
            fig_dir=fig_dir,
            dataset=data,
            cflib_module=cflib,
            raw_field=raw_field,
            smooth_narrow_field=narrow_field,
            smooth_broad_field=broad_field,
            cmap_name=cmap_name,
            elv_deg=2.0,
            zoom_km=20.0,
            data_context_note=(
                "C-band radar (COW) from WINTRE-MIX field campaign, "
                f"{_format_radar_acquisition_date(data, data_path)}"
            ),
        )

        status.update(
            {
                "generated": True,
                "status": "ok",
                "message": "Generated optional real-radar 6-panel figure.",
                "variable_used": raw_field,
                "fields_generated": [narrow_field, broad_field],
                "az_res_deg": float(az_res_deg),
                "shape_2d": [int(da.shape[-2]), int(da.shape[-1])],
                "finite_fraction_raw": float(np.mean(np.isfinite(da.values))),
                "colormap_used": cmap_name,
                **zoom_meta,
            }
        )
        return status

    except Exception as exc:  # pragma: no cover
        status["status"] = "error"
        status["message"] = f"Real-radar figure failed: {exc}"
        return status
    finally:
        if data is not None:
            try:
                data.close()
            except Exception:
                pass


def aggregate_missingness(
    all_missingness: list[dict[str, list[dict[str, float]]]],
) -> dict[str, list[dict[str, float]]]:
    levels = [0.1, 0.2, 0.35, 0.5, 0.65]
    summary = {"mcar": [], "block": []}
    for family in ("mcar", "block"):
        for idx, level in enumerate(levels):
            rmse_values = [run[family][idx]["rmse"] for run in all_missingness]
            count_values = [
                run[family][idx]["effective_count_mean"] for run in all_missingness
            ]
            rmse_stats = summarize_runs(rmse_values)
            count_stats = summarize_runs(count_values)
            summary[family].append(
                {
                    "missing_fraction": level,
                    "rmse": rmse_stats,
                    "effective_count_mean": count_stats,
                }
            )
    return summary


def aggregate_flat_metrics(
    all_metrics: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    keys = sorted(all_metrics[0].keys())
    return {k: summarize_runs([d[k] for d in all_metrics]) for k in keys}


def make_pipeline_figure(fig_dir: Path, n_seeds: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")

    boxes = [
        (0.04, 0.55, 0.2, 0.28, "Synthetic truth\n+ deterministic field"),
        (0.29, 0.55, 0.2, 0.28, "Masking + noise\n+ MCAR / block gaps"),
        (0.54, 0.55, 0.2, 0.28, "DCT estimators\n+ mean/variance/count"),
        (0.79, 0.55, 0.17, 0.28, "Metrics +\n+ figures with uncertainty"),
        (0.29, 0.12, 0.2, 0.24, "Boundary test\n+ reflective vs periodic"),
        (0.54, 0.12, 0.2, 0.24, "Variance proxy\n+ calibration fit"),
    ]

    for x0, y0, w, h, label in boxes:
        rect = plt.Rectangle(
            (x0, y0), w, h, facecolor="#f1f5f9", edgecolor="#0f172a", linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.24, 0.69), (0.29, 0.69)),
        ((0.49, 0.69), (0.54, 0.69)),
        ((0.74, 0.69), (0.79, 0.69)),
        ((0.39, 0.55), (0.39, 0.36)),
        ((0.64, 0.55), (0.64, 0.36)),
        ((0.49, 0.24), (0.54, 0.24)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#334155"),
        )

    ax.text(
        0.04,
        0.92,
        f"Synthetic Validation Workflow (n={n_seeds} seeds)",
        fontsize=13,
        weight="bold",
        color="#0f172a",
    )
    fig.tight_layout()
    save_figure_outputs(
        fig, fig_dir, "method_pipeline_schematic", legacy_png=True, legacy_png_dpi=220
    )
    plt.close(fig)


def make_missingness_figure(
    fig_dir: Path, missingness_summary: dict[str, list[dict[str, float]]]
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x1 = [100.0 * d["missing_fraction"] for d in missingness_summary["mcar"]]
    y1 = [d["rmse"]["mean"] for d in missingness_summary["mcar"]]
    e1 = [d["rmse"]["std"] for d in missingness_summary["mcar"]]

    x2 = [100.0 * d["missing_fraction"] for d in missingness_summary["block"]]
    y2 = [d["rmse"]["mean"] for d in missingness_summary["block"]]
    e2 = [d["rmse"]["std"] for d in missingness_summary["block"]]

    ax.errorbar(x1, y1, yerr=e1, marker="o", linewidth=2.0, capsize=3, label="MCAR")
    ax.errorbar(
        x2,
        y2,
        yerr=e2,
        marker="s",
        linewidth=2.0,
        capsize=3,
        label="Block gaps",
    )
    ax.set_xlabel("Missing fraction (%)")
    ax.set_ylabel("RMSE vs truth (mean ± std)")
    ax.set_title("Missingness sensitivity (multi-seed)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure_outputs(
        fig, fig_dir, "missingness_vs_error", legacy_png=True, legacy_png_dpi=220
    )
    plt.close(fig)


def make_boundary_figure(
    fig_dir: Path, boundary_summary: dict[str, dict[str, float]]
) -> None:
    labels = ["All bins", "Azimuth edge bins"]
    ref_means = [
        boundary_summary["rmse_reflective_all"]["mean"],
        boundary_summary["rmse_reflective_edge"]["mean"],
    ]
    per_means = [
        boundary_summary["rmse_periodic_all"]["mean"],
        boundary_summary["rmse_periodic_edge"]["mean"],
    ]
    ref_err = [
        boundary_summary["rmse_reflective_all"]["std"],
        boundary_summary["rmse_reflective_edge"]["std"],
    ]
    per_err = [
        boundary_summary["rmse_periodic_all"]["std"],
        boundary_summary["rmse_periodic_edge"]["std"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, ref_means, yerr=ref_err, capsize=4, width=w, label="Reflective")
    ax.bar(x + w / 2, per_means, yerr=per_err, capsize=4, width=w, label="Periodic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE vs truth (mean ± std)")
    ax.set_title("Polar boundary comparison (multi-seed)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure_outputs(
        fig, fig_dir, "boundary_effect_plot", legacy_png=True, legacy_png_dpi=220
    )
    plt.close(fig)


def make_variance_figure(
    fig_dir: Path, var_cal: dict[str, dict[str, float] | list[float]]
) -> None:
    x = np.asarray(var_cal["scatter_true_var"], dtype=float)
    y = np.asarray(var_cal["scatter_est_var"], dtype=float)
    slope = float(var_cal["slope_est_vs_true_var"]["mean"])
    intercept = float(var_cal["intercept_est_vs_true_var"]["mean"])
    corr = float(var_cal["corr_est_vs_true_var"]["mean"])

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(x, y, s=12, alpha=0.35, label="Grid samples (seed 0 subset)")
    xfit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    yfit = slope * xfit + intercept
    ax.plot(xfit, yfit, linewidth=2.0, label="Linear fit (mean coefficients)")
    ax.set_xlabel("Reference local noise variance")
    ax.set_ylabel("Estimated local variance")
    ax.set_title("Variance calibration proxy")
    ax.text(
        0.04,
        0.95,
        f"slope={slope:.3f}, corr={corr:.3f}",
        transform=ax.transAxes,
        va="top",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure_outputs(
        fig, fig_dir, "variance_calibration_plot", legacy_png=True, legacy_png_dpi=220
    )
    plt.close(fig)


def _subsample_vector_field(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    stride: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sz, sy, sx = stride
    z_axis = np.linspace(-1.0, 1.0, u.shape[0])[::sz]
    y_axis = np.linspace(-1.0, 1.0, u.shape[1])[::sy]
    x_axis = np.linspace(-1.0, 1.0, u.shape[2])[::sx]
    zz, yy, xx = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    uu = u[::sz, ::sy, ::sx]
    vv = v[::sz, ::sy, ::sx]
    ww = w[::sz, ::sy, ::sx]
    return (
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
        uu.ravel(),
        vv.ravel(),
        ww.ravel(),
    )


def _sample_masked_vectors(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mask: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty, empty, empty

    rng = np.random.default_rng(seed)
    if idx.shape[0] > max_points:
        keep = rng.choice(idx.shape[0], size=max_points, replace=False)
        idx = idx[keep]

    z_idx, y_idx, x_idx = idx[:, 0], idx[:, 1], idx[:, 2]
    z_axis = np.linspace(-1.0, 1.0, u.shape[0])
    y_axis = np.linspace(-1.0, 1.0, u.shape[1])
    x_axis = np.linspace(-1.0, 1.0, u.shape[2])
    xx = x_axis[x_idx]
    yy = y_axis[y_idx]
    zz = z_axis[z_idx]
    uu = u[z_idx, y_idx, x_idx]
    vv = v[z_idx, y_idx, x_idx]
    ww = w[z_idx, y_idx, x_idx]
    return xx, yy, zz, uu, vv, ww


def _quiver_with_speed_colors(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    *,
    norm: matplotlib.colors.Normalize,
    cmap_name: str = "viridis",
    alpha: float = 0.7,
    length: float = 0.09,
    linewidth: float = 0.6,
) -> None:
    if x.size == 0:
        return
    speed = np.sqrt(u**2 + v**2 + w**2)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm(speed))
    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        colors=colors,
        length=length,
        normalize=True,
        linewidths=linewidth,
        arrow_length_ratio=0.28,
        alpha=alpha,
    )


def _style_wind_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xlabel("x", fontsize=8, labelpad=-3)
    ax.set_ylabel("y", fontsize=8, labelpad=-3)
    ax.set_zlabel("z", fontsize=8, labelpad=-3)
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_zticks([-1.0, 0.0, 1.0])
    ax.tick_params(labelsize=7, pad=1)
    ax.view_init(elev=22, azim=35)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.75))
    except Exception:
        pass


def make_outlier_slices_figure(
    fig_dir: Path, representative: dict[str, object]
) -> None:
    """Create 3D sparse-quiver visualization for injected and detected outliers."""
    u_obs = np.asarray(representative["u_observed"], dtype=float)
    v_obs = np.asarray(representative["v_observed"], dtype=float)
    w_obs = np.asarray(representative["w_observed"], dtype=float)
    u_mean = np.asarray(representative["u_mean"], dtype=float)
    v_mean = np.asarray(representative["v_mean"], dtype=float)
    w_mean = np.asarray(representative["w_mean"], dtype=float)
    injected_mask = np.asarray(representative["injected_mask"], dtype=bool)
    detected_mask = np.asarray(representative["detected_mask"], dtype=bool)

    tp_mask = detected_mask & injected_mask
    fp_mask = detected_mask & ~injected_mask
    fn_mask = (~detected_mask) & injected_mask

    x_bg, y_bg, z_bg, u_bg, v_bg, w_bg = _subsample_vector_field(
        u_obs, v_obs, w_obs, stride=(3, 8, 8)
    )
    x_mu, y_mu, z_mu, u_mu, v_mu, w_mu = _subsample_vector_field(
        u_mean, v_mean, w_mean, stride=(3, 8, 8)
    )

    speed_ref = np.sqrt(
        np.r_[u_bg, u_mu] ** 2 + np.r_[v_bg, v_mu] ** 2 + np.r_[w_bg, w_mu] ** 2
    )
    vmin = float(np.percentile(speed_ref, 5.0))
    vmax = float(np.percentile(speed_ref, 95.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    x_inj, y_inj, z_inj, u_inj, v_inj, w_inj = _sample_masked_vectors(
        u_obs,
        v_obs,
        w_obs,
        injected_mask,
        max_points=180,
        seed=101,
    )
    x_tp, y_tp, z_tp, u_tp, v_tp, w_tp = _sample_masked_vectors(
        u_obs,
        v_obs,
        w_obs,
        tp_mask,
        max_points=180,
        seed=103,
    )
    x_fp, y_fp, z_fp, u_fp, v_fp, w_fp = _sample_masked_vectors(
        u_obs,
        v_obs,
        w_obs,
        fp_mask,
        max_points=120,
        seed=107,
    )
    x_fn, y_fn, z_fn, _, _, _ = _sample_masked_vectors(
        u_obs,
        v_obs,
        w_obs,
        fn_mask,
        max_points=100,
        seed=109,
    )

    fig = plt.figure(figsize=(10.8, 9.2))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.04,
        right=0.90,
        bottom=0.04,
        top=0.96,
        hspace=0.18,
        wspace=0.08,
    )
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    ax2 = fig.add_subplot(gs[1, :], projection="3d")

    # Center panel (c) under panels (a) and (b) while keeping panel sizes large.
    pos0 = ax0.get_position()
    pos2 = ax2.get_position()
    ax2.set_position([0.5 - 0.5 * pos0.width, pos2.y0 + 0.005, pos0.width, pos2.height])

    _quiver_with_speed_colors(
        ax0,
        x_bg,
        y_bg,
        z_bg,
        u_bg,
        v_bg,
        w_bg,
        norm=norm,
        cmap_name="viridis",
        alpha=0.55,
        length=0.09,
        linewidth=0.55,
    )
    if x_inj.size > 0:
        ax0.quiver(
            x_inj,
            y_inj,
            z_inj,
            u_inj,
            v_inj,
            w_inj,
            color="#d62728",
            length=0.13,
            normalize=True,
            linewidths=1.0,
            arrow_length_ratio=0.35,
            alpha=0.95,
        )
    _style_wind_axis(ax0, "(a) Observed wind + injected outliers")
    # Place a red arrow inline with the panel-(a) title text.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = ax0.title.get_window_extent(renderer=renderer)
    title_y_mid = 0.5 * (title_bbox.y0 + title_bbox.y1)
    arrow_x, arrow_y = ax0.transAxes.inverted().transform(
        (title_bbox.x1 + 4.0, title_y_mid)
    )
    ax0.text2D(
        arrow_x,
        arrow_y,
        "\u279c",
        transform=ax0.transAxes,
        fontsize=12,
        ha="left",
        va="center",
        fontweight="bold",
        color="#d62728",
        clip_on=False,
    )

    _quiver_with_speed_colors(
        ax1,
        x_mu,
        y_mu,
        z_mu,
        u_mu,
        v_mu,
        w_mu,
        norm=norm,
        cmap_name="viridis",
        alpha=0.85,
        length=0.09,
        linewidth=0.6,
    )
    _style_wind_axis(ax1, "(b) Local-mean wind (DCT normalized stats)")

    ax2.quiver(
        x_mu,
        y_mu,
        z_mu,
        u_mu,
        v_mu,
        w_mu,
        color="#b0b0b0",
        length=0.08,
        normalize=True,
        linewidths=0.45,
        arrow_length_ratio=0.28,
        alpha=0.22,
    )
    if x_tp.size > 0:
        ax2.quiver(
            x_tp,
            y_tp,
            z_tp,
            u_tp,
            v_tp,
            w_tp,
            color="#d62728",
            length=0.13,
            normalize=True,
            linewidths=1.1,
            arrow_length_ratio=0.35,
            alpha=0.95,
        )
    if x_fp.size > 0:
        ax2.quiver(
            x_fp,
            y_fp,
            z_fp,
            u_fp,
            v_fp,
            w_fp,
            color="#ff7f0e",
            length=0.12,
            normalize=True,
            linewidths=0.95,
            arrow_length_ratio=0.34,
            alpha=0.9,
        )
    if x_fn.size > 0:
        ax2.scatter(x_fn, y_fn, z_fn, s=10, color="#1f77b4", alpha=0.75)

    k_sel = float(representative["selected_threshold"])
    _style_wind_axis(
        ax2,
        f"(c) Detected outliers (k={k_sel:.2f})",
    )
    legend_handles = [
        Line2D([0], [0], color="#d62728", lw=2, label="Detected TP"),
        Line2D([0], [0], color="#ff7f0e", lw=2, label="Detected FP"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#1f77b4",
            markeredgecolor="#1f77b4",
            markersize=5,
            label="Missed injected (FN)",
        ),
    ]
    ax2.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.10, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Wind speed magnitude (a.u.)")

    save_figure_outputs(
        fig, fig_dir, "outlier_3d_injected_vs_detected", legacy_png=False
    )
    plt.close(fig)


def make_outlier_threshold_curve_figure(
    fig_dir: Path, outlier_summary: list[dict[str, object]]
) -> None:
    """Create threshold sweep precision/recall/F1 figure."""
    x = np.asarray([d["threshold"] for d in outlier_summary], dtype=float)

    def _mean_std(metric: str) -> tuple[np.ndarray, np.ndarray]:
        means = np.asarray([d[metric]["mean"] for d in outlier_summary], dtype=float)
        stds = np.asarray([d[metric]["std"] for d in outlier_summary], dtype=float)
        return means, stds

    p_mean, p_std = _mean_std("precision")
    r_mean, r_std = _mean_std("recall")
    f_mean, f_std = _mean_std("f1")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, p_mean, marker="o", linewidth=2.0, label="Precision")
    ax.fill_between(x, p_mean - p_std, p_mean + p_std, alpha=0.15)
    ax.plot(x, r_mean, marker="s", linewidth=2.0, label="Recall")
    ax.fill_between(x, r_mean - r_std, r_mean + r_std, alpha=0.15)
    ax.plot(x, f_mean, marker="^", linewidth=2.2, label="F1")
    ax.fill_between(x, f_mean - f_std, f_mean + f_std, alpha=0.15)

    ax.set_xlabel("Threshold k in $|\\|\\mathbf{v}\\|-\\mu| > k\\sigma$")
    ax.set_ylabel("Score (mean ± std)")
    ax.set_title("3D cyclone-wind outlier-identification threshold sweep")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure_outputs(
        fig, fig_dir, "outlier_3d_threshold_performance", legacy_png=False
    )
    plt.close(fig)


def make_boundary_sanity_figure(fig_dir: Path, sanity: dict[str, object]) -> None:
    """Visualize reflective vs periodic edge behavior for sanity check."""
    truth = np.asarray(sanity["truth"], dtype=float)
    pred_ref = np.asarray(sanity["pred_ref"], dtype=float)
    pred_per = np.asarray(sanity["pred_per"], dtype=float)
    err_ref = np.asarray(sanity["err_ref"], dtype=float)
    err_per = np.asarray(sanity["err_per"], dtype=float)
    edge_mask = np.asarray(sanity["edge_mask"], dtype=float)

    vmin = float(np.nanpercentile(truth, 2.0))
    vmax = float(np.nanpercentile(truth, 98.0))
    emax = float(np.nanpercentile(np.r_[err_ref.ravel(), err_per.ravel()], 99.0))

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    panels = [
        (truth, "Truth", "turbo", vmin, vmax),
        (pred_ref, "Reflective (DCT)", "turbo", vmin, vmax),
        (pred_per, "Periodic (FFT)", "turbo", vmin, vmax),
        (err_ref, "|Error| Reflective", "magma", 0.0, emax),
        (err_per, "|Error| Periodic", "magma", 0.0, emax),
        (edge_mask, "Edge mask", "gray_r", 0.0, 1.0),
    ]

    for ax, (arr, title, cmap, lo, hi) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    save_figure_outputs(fig, fig_dir, "boundary_sanity_edge_views", legacy_png=False)
    plt.close(fig)


def make_real_radar_6panel_figure(
    fig_dir: Path,
    dataset,
    cflib_module,
    raw_field: str,
    smooth_narrow_field: str,
    smooth_broad_field: str,
    cmap_name: str,
    elv_deg: float = 2.0,
    zoom_km: float = 20.0,
    data_context_note: str = "",
) -> dict[str, object]:
    """Create required 6-panel real-radar figure with notebook plotting style."""
    # Keep zoom centered at radar origin and let cflib handle sweep selection
    # from the requested elevation angle, matching the notebook workflow.
    x_center = 0.0
    y_center = 0.0

    half = float(zoom_km) / 2.0
    xlim = (x_center - half, x_center + half)
    ylim = (y_center - half, y_center + half)

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.3))
    panel_fields = [raw_field, smooth_narrow_field, smooth_broad_field]
    top_titles = ["(a) Raw", "(b) Smoothed W=5", "(c) Smoothed W=15"]
    bottom_titles = [
        "(d) Raw (zoom 20 km x 20 km)",
        "(e) Smoothed W=5 (zoom 20 km x 20 km)",
        "(f) Smoothed W=15 (zoom 20 km x 20 km)",
    ]

    for col, field_name in enumerate(panel_fields):
        ax_top = axes[0, col]
        cflib_module.plot_cfradial(
            dataset,
            field_name,
            elv=elv_deg,
            cmap=cmap_name,
            vmax=40,
            vmin=0,
            ax=ax_top,
            rasterized=True,
        )
        top_xlim = ax_top.get_xlim()
        top_ylim = ax_top.get_ylim()
        ax_top.set_title(top_titles[col], fontsize=10)
        zoom_box = plt.Rectangle(
            (xlim[0], ylim[0]),
            float(zoom_km),
            float(zoom_km),
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            linewidth=1.0,
        )
        ax_top.add_patch(zoom_box)

        ax_bottom = axes[1, col]
        cflib_module.plot_cfradial(
            dataset,
            field_name,
            elv=elv_deg,
            cmap=cmap_name,
            vmax=40,
            vmin=0,
            ax=ax_bottom,
            rasterized=True,
        )
        ax_top.set_xlim(top_xlim)
        ax_top.set_ylim(top_ylim)
        ax_bottom.set_xlim(xlim)
        ax_bottom.set_ylim(ylim)
        ax_bottom.set_title(bottom_titles[col], fontsize=10)

    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes.ravel():
        ax.tick_params(labelsize=8)

    norm = plt.Normalize(vmin=0, vmax=40)
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.17, 0.018, 0.65])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Reflectivity (dBZ)")

    if data_context_note:
        meta_handle = Line2D([], [], linestyle="", label=data_context_note)
        fig.legend(
            handles=[meta_handle],
            loc="upper left",
            bbox_to_anchor=(0.05, 0.935),
            frameon=True,
            handlelength=0.0,
            handletextpad=0.0,
            borderpad=0.35,
            fontsize=8.5,
        )

    fig.subplots_adjust(
        left=0.05,
        right=0.90,
        bottom=0.07,
        top=0.88,
        wspace=0.16,
        hspace=0.20,
    )
    save_figure_outputs(
        fig, fig_dir, "real_radar_6panel", legacy_png=False, trim_top=True
    )
    plt.close(fig)

    return {
        "elv_deg": float(elv_deg),
        "zoom_km": float(zoom_km),
        "zoom_center_km": [float(x_center), float(y_center)],
        "zoom_bounds_km": [
            float(xlim[0]),
            float(xlim[1]),
            float(ylim[0]),
            float(ylim[1]),
        ],
    }


def write_csv(results_dir: Path, metrics: dict[str, object]) -> None:
    csv_path = results_dir / "synthetic_metrics.csv"
    rows: list[dict[str, object]] = []

    for family in ("mcar", "block"):
        for item in metrics["missingness"][family]:
            rows.append(
                {
                    "experiment": "missingness",
                    "scenario": family,
                    "key": "rmse",
                    "x_value": item["missing_fraction"],
                    "n": item["rmse"]["n"],
                    "mean": item["rmse"]["mean"],
                    "std": item["rmse"]["std"],
                    "ci95_low": item["rmse"]["ci95_low"],
                    "ci95_high": item["rmse"]["ci95_high"],
                    "aux_key": "effective_count_mean",
                    "aux_mean": item["effective_count_mean"]["mean"],
                    "aux_std": item["effective_count_mean"]["std"],
                }
            )

    for key, stats in metrics["boundary"].items():
        rows.append(
            {
                "experiment": "boundary",
                "scenario": "polar",
                "key": key,
                "x_value": "",
                "n": stats["n"],
                "mean": stats["mean"],
                "std": stats["std"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "aux_key": "",
                "aux_mean": "",
                "aux_std": "",
            }
        )

    for key, value in metrics["variance_calibration"].items():
        if isinstance(value, dict) and "mean" in value:
            rows.append(
                {
                    "experiment": "variance_calibration",
                    "scenario": "proxy",
                    "key": key,
                    "x_value": "",
                    "n": value["n"],
                    "mean": value["mean"],
                    "std": value["std"],
                    "ci95_low": value["ci95_low"],
                    "ci95_high": value["ci95_high"],
                    "aux_key": "",
                    "aux_mean": "",
                    "aux_std": "",
                }
            )

    for key, stats in metrics["runtime_ms"].items():
        rows.append(
            {
                "experiment": "runtime",
                "scenario": "timing",
                "key": key,
                "x_value": "",
                "n": stats["n"],
                "mean": stats["mean"],
                "std": stats["std"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "aux_key": "relative_timing",
                "aux_mean": "1",
                "aux_std": "",
            }
        )

    outlier_block = metrics.get("outlier_3d", {})
    sweep = outlier_block.get("threshold_sweep", [])
    for item in sweep:
        thr = item["threshold"]
        for key in ("precision", "recall", "f1"):
            stats = item[key]
            rows.append(
                {
                    "experiment": "outlier_threshold",
                    "scenario": "3d",
                    "key": key,
                    "x_value": thr,
                    "n": stats["n"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "ci95_low": stats["ci95_low"],
                    "ci95_high": stats["ci95_high"],
                    "aux_key": "threshold",
                    "aux_mean": thr,
                    "aux_std": "",
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "scenario",
                "key",
                "x_value",
                "n",
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
                "aux_key",
                "aux_mean",
                "aux_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "png").mkdir(parents=True, exist_ok=True)
    (figures_dir / "pdf").mkdir(parents=True, exist_ok=True)

    seed_list = [20260414, 20260415, 20260416, 20260417, 20260418]
    truth = synthetic_field((160, 160))
    width = (9.0, 9.0)

    truth_wind_3d = synthetic_cyclone_wind_3d((18, 96, 96))
    width_3d = (3.0, 7.0, 7.0)
    threshold_grid = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    missingness_runs: list[dict[str, list[dict[str, float]]]] = []
    boundary_runs: list[dict[str, float]] = []
    variance_runs: list[dict[str, object]] = []
    runtime_runs: list[dict[str, float]] = []
    outlier_threshold_runs: list[list[dict[str, float]]] = []
    representative_outlier: dict[str, object] | None = None

    boundary_sanity_assets = run_boundary_sanity_assets(
        np.random.default_rng(seed_list[0])
    )

    for idx, seed in enumerate(seed_list):
        rng = np.random.default_rng(seed)
        missingness_runs.append(run_missingness_experiment(truth, width, rng))
        boundary_runs.append(run_boundary_experiment(rng))
        variance_runs.append(run_variance_calibration(truth, width, rng))
        runtime_runs.append(run_timing_summary(truth, width, rng))

        outlier_res = run_outlier_identification_experiment(
            truth_wind_3d=truth_wind_3d,
            width_3d=width_3d,
            thresholds=threshold_grid,
            rng=rng,
        )
        outlier_threshold_runs.append(outlier_res["threshold_metrics"])
        if idx == 0:
            representative_outlier = outlier_res["representative"]

    missingness_summary = aggregate_missingness(missingness_runs)
    boundary_summary = aggregate_flat_metrics(boundary_runs)
    runtime_summary = aggregate_flat_metrics(runtime_runs)
    outlier_summary = aggregate_outlier_threshold_sweep(
        outlier_threshold_runs, threshold_grid
    )

    variance_numeric: list[dict[str, float]] = []
    for v in variance_runs:
        variance_numeric.append(
            {
                "slope_est_vs_true_var": float(v["slope_est_vs_true_var"]),
                "intercept_est_vs_true_var": float(v["intercept_est_vs_true_var"]),
                "corr_est_vs_true_var": float(v["corr_est_vs_true_var"]),
                "global_mse_of_smoothed_mean": float(v["global_mse_of_smoothed_mean"]),
                "true_var_mean": float(v["true_var_mean"]),
                "est_var_mean": float(v["est_var_mean"]),
            }
        )
    variance_summary = aggregate_flat_metrics(variance_numeric)
    variance_summary["scatter_true_var"] = variance_runs[0]["scatter_true_var"]
    variance_summary["scatter_est_var"] = variance_runs[0]["scatter_est_var"]

    environment = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        **_dct_toolkit_provenance(),
        "timing_note": "Runtime values are relative and hardware-dependent.",
    }

    real_radar_status = run_optional_real_radar_figure(
        fig_dir=figures_dir,
        dct_toolkit_repo_root=environment.get("dct_toolkit_repo_root", "unknown"),
        seed=20260415,
    )

    metrics = {
        "seeds": seed_list,
        "n_seeds": len(seed_list),
        "environment": environment,
        "missingness": missingness_summary,
        "boundary": boundary_summary,
        "boundary_sanity": {
            "edge_width": int(boundary_sanity_assets["edge_width"]),
            "metrics": boundary_sanity_assets["metrics"],
        },
        "variance_calibration": variance_summary,
        "runtime_ms": runtime_summary,
        "outlier_3d": {
            "shape": [18, 96, 96],
            "width": [3.0, 7.0, 7.0],
            "threshold_grid": threshold_grid,
            "threshold_sweep": outlier_summary,
            "representative_seed": seed_list[0],
            "representative": {
                "slice_index": int(representative_outlier["slice_index"])
                if representative_outlier
                else -1,
                "selected_threshold": float(
                    representative_outlier["selected_threshold"]
                )
                if representative_outlier
                else float("nan"),
                "n_injected_total": int(representative_outlier["n_injected_total"])
                if representative_outlier
                else 0,
                "n_detected_total": int(representative_outlier["n_detected_total"])
                if representative_outlier
                else 0,
                "f1_at_selected_threshold": float(
                    representative_outlier["f1_at_selected_threshold"]
                )
                if representative_outlier
                else 0.0,
            },
        },
        "real_radar": real_radar_status,
        "status": {
            "synthetic_outputs": "ok",
            "optional_real_radar_generated": bool(
                real_radar_status.get("generated", False)
            ),
        },
        "figure_outputs": {
            "png_dir": str(figures_dir / "png"),
            "pdf_dir": str(figures_dir / "pdf"),
            "legacy_root_png_kept": True,
        },
    }

    with (results_dir / "synthetic_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_csv(results_dir, metrics)

    make_pipeline_figure(figures_dir, len(seed_list))
    make_missingness_figure(figures_dir, missingness_summary)
    make_boundary_figure(figures_dir, boundary_summary)
    make_boundary_sanity_figure(figures_dir, boundary_sanity_assets)
    make_variance_figure(figures_dir, variance_summary)
    if representative_outlier is not None:
        make_outlier_slices_figure(figures_dir, representative_outlier)
    make_outlier_threshold_curve_figure(figures_dir, outlier_summary)

    print("Generated synthetic validation outputs.")
    print(f"Metrics JSON: {results_dir / 'synthetic_metrics.json'}")
    print(f"Metrics CSV:  {results_dir / 'synthetic_metrics.csv'}")
    print(f"Figures png:  {figures_dir / 'png'}")
    print(f"Figures pdf:  {figures_dir / 'pdf'}")


if __name__ == "__main__":
    main()
