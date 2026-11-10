#!/usr/bin/env python3
"""Redraw the submitted visual teaser for the direct-ratio validation path.

The synthetic field and display style follow generate_teaser_v13.py in the
historical paper workspace. Outputs are newly calculated with the checked v1.0.0
Cartesian filter; no saved experiment or historical figure is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.fft import dctn

from _review_common import TOOLKIT_COMMIT, load_toolkit
from _review_figures import DEFAULT_FIGURE_ROOT, configure_plotting, export_figure


FIGSIZE = (6.5, 3.75)
STEM = "visual_statistics_pipeline"
WIDTH = 7.0
FLOOR = 1e-3


def calculate():
    """Recreate the old field, then use the released complete Gaussian filter."""
    toolkit = load_toolkit()
    from dct_toolkit.cartesian import smooth_cartesian
    from dct_toolkit.core import get_dct_transfer_function

    # RandomState reproduces the legacy np.random.seed(42) stream.
    rng = np.random.RandomState(42)
    n = 80
    x = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, x)
    field = (np.sin(X * 2) * np.cos(Y * 2.5)
             + 0.5 * np.sin(np.sqrt(X**2 + Y**2) * 5)
             + 0.3 * np.sin(X * 8) * np.cos(Y * 8))
    field += rng.normal(0, 0.30, (n, n))
    mask = np.ones_like(field)
    theta = np.arctan2(Y, X)
    mask[(theta > 0.3) & (theta < 1.0) & (np.sqrt(X**2 + Y**2) < 2.5)] = 0
    mask.flat[rng.choice(n * n, size=int(0.04 * n * n), replace=False)] = 0
    mask[:n // 2, :8] = 0
    mask[n // 2:, n - 8:] = 0
    mask[:8, :n // 2] = 1
    mask[:8, n // 2:] = 0
    mask[n - 8:, :n // 2] = 0
    mask[n - 8:, n // 2:] = 1

    valid = mask.astype(bool)
    values = np.where(valid, field, 0.0)
    a1, a2, d = [smooth_cartesian(a, WIDTH, kernel_type="gaussian")
                 for a in (values, values**2, mask)]
    accepted = np.isfinite(d) & (d > FLOOR)
    mean = np.full_like(d, np.nan)
    moment2 = np.full_like(d, np.nan)
    np.divide(a1, d, out=mean, where=accepted)
    np.divide(a2, d, out=moment2, where=accepted)
    std = np.sqrt(np.maximum(0, moment2 - mean**2))
    transfer_1d = get_dct_transfer_function(n, "gaussian", WIDTH)
    return dict(field=field, mask=mask, values=values, d=d, a1=a1, a2=a2,
                mean=mean, std=std, coverage=np.clip(d, 0, 1),
                spectrum=dctn(values, type=2, norm="ortho"),
                transfer=np.outer(transfer_1d, transfer_1d), version=toolkit.__version__)


def create_figure(data):
    configure_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    ink = "#1a1a1a"
    gray = "#777777"
    valid = data["mask"].astype(bool)
    field = np.ma.masked_where(~valid, data["field"])
    vmin, vmax = np.percentile(data["field"][valid], [1, 99])
    field_cmap = plt.get_cmap("RdBu_r").copy()
    field_cmap.set_bad("#d1d1d1")
    moment_cmap = field_cmap.copy()
    std_cmap = plt.get_cmap("YlOrRd").copy()
    std_cmap.set_bad("#d1d1d1")

    def panel(rect, title, image, cmap, lo=None, hi=None):
        ax = fig.add_axes(rect)
        im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4, color=ink)
        for spine in ax.spines.values():
            spine.set_color(gray)
            spine.set_linewidth(0.7)
        return ax, im

    # The upper row retains the original input / spectrum / response / mean story.
    panel([0.08, 0.48, 0.20, 0.38], "Input field", field, field_cmap, vmin, vmax)
    spec = np.flipud(np.log10(np.abs(data["spectrum"][:45, :45]) + 1e-12))
    lo, hi = np.percentile(spec, [1, 99])
    panel([0.37, 0.525, 0.115, 0.23], "Spectrum", np.clip(spec, lo, hi), "magma")
    panel([0.495, 0.525, 0.115, 0.23], r"$H_g$", np.flipud(data["transfer"][:45, :45]),
          "Greens", 0, 1)
    ax_mean, _ = panel([0.715, 0.48, 0.20, 0.38], "Local mean", data["mean"],
                       moment_cmap, vmin, vmax)
    ax_mean.add_patch(Rectangle((25.5, 25.5), 28, 28, linewidth=1.5,
                                edgecolor="#e6a700", facecolor="none"))
    fig.text(0.49, 0.87, "Spectral domain", ha="center", va="center",
             fontsize=9, fontweight="bold", color=ink)
    fig.text(0.49, 0.815, "DCT-II (reflective)", ha="center", va="center",
             fontsize=8, color="#555555")
    fig.text(0.322, 0.72, r"$\mathcal{T}$", ha="center", fontsize=10, color=ink)
    fig.text(0.66, 0.72, r"$\times H_g\;\mathcal{T}^{-1}$", ha="center",
             fontsize=9, color=ink)
    fig.text(0.66, 0.54, "normalize", ha="center", va="center",
             fontsize=8, color="#555555")
    for start, end in (((0.29, 0.64), (0.355, 0.64)), ((0.62, 0.64), (0.70, 0.64))):
        fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                       arrowstyle="->", mutation_scale=12, color="#555555", linewidth=1.2))

    # The caption describes the three filtered inputs; the maps share one crop.
    fig.text(0.5, 0.425, "Enlarged yellow-box region", ha="center", va="center",
             fontsize=8.5, color="#555555")

    crop = np.s_[26:54, 26:54]
    std_max = float(np.nanpercentile(data["std"][crop], 98))
    maps = [(r"Mean $\mu$", data["mean"][crop], moment_cmap, vmin, vmax),
            (r"Std $\sigma$", data["std"][crop], std_cmap, 0, std_max),
            (r"Coverage $\rho$", data["coverage"][crop], "Greens", 0, 1)]
    for i, (title, image, cmap, low, high) in enumerate(maps):
        left = 0.17 + i * 0.225
        _, im = panel([left, 0.06, 0.16, 0.2773], title, image, cmap, low, high)
        bar = fig.colorbar(im, cax=fig.add_axes([left + 0.163, 0.06, 0.010, 0.2773]),
                           ticks=[low, high], extend="max" if i == 1 else "neither")
        bar.ax.tick_params(labelsize=7, length=2, pad=1)
        bar.ax.set_yticklabels([f"{low:.1f}", f"{high:.1f}"])
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    args = parser.parse_args(argv)
    data = calculate()
    fig = create_figure(data)
    settings = {
        "purpose": "Illustrative direct normalized statistics on a synthetic 2D Cartesian field",
        "historical_style_and_field": "DCT_paper2/02_arXiv_Paper/figure_outputs/generate_teaser_v13.py; seed 42, 80 x 80",
        "toolkit_version": data["version"], "toolkit_commit": TOOLKIT_COMMIT,
        "operator": "v1.0.0 smooth_cartesian; Gaussian width 7 grid cells; reflective on both axes",
        "input_rule": "original validity mask; invalid data and squared data explicitly zero",
        "denominator": "raw filtered original mask; finite d > 0.001; no completion or mask restoration",
        "outputs": "mean=a1/d; std=sqrt(max(0,a2/d-mean**2)); coverage=clip(d,0,1)",
        "visualization": "same central [26:54,26:54] crop; gray missing or undefined; std colorbar extended at 98th percentile",
        "spectrum": "log10 magnitude of DCT-II of zero-filled data; displayed first 45 frequencies per axis",
        "accepted_cells": int(np.count_nonzero(np.isfinite(data["mean"]))),
        "undefined_cells": int(np.count_nonzero(~np.isfinite(data["mean"]))),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    paths = export_figure(fig, args.output_root / "visual_pipeline", STEM, settings)
    import matplotlib.pyplot as plt
    plt.close(fig)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
