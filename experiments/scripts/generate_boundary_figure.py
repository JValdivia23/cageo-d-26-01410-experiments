import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIG_PDF_DIR = Path(__file__).resolve().parents[1] / "figures" / "pdf"
FIG_PNG_DIR = Path(__file__).resolve().parents[1] / "figures" / "png"

FIG_PDF_DIR.mkdir(parents=True, exist_ok=True)
FIG_PNG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    json_path = RESULTS_DIR / "boundary_1d_data.json"
    with open(json_path) as f:
        dataset = json.load(f)
    return dataset


def _strip_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_axis_arrows(
    ax, xlabel="x", ylabel="Signal", fontsize=9, head_len=0.012, head_width=0.006
):
    ax.annotate(
        "",
        xy=(1.0, 0),
        xycoords="axes fraction",
        xytext=(0.0, 0),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )
    ax.annotate(
        "",
        xy=(0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 0.0),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )
    ax.text(
        1.02,
        -0.02,
        xlabel,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="left",
        va="top",
        color="#555555",
    )
    ax.text(
        -0.02,
        1.02,
        ylabel,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="right",
        va="bottom",
        color="#555555",
        rotation=0,
    )
    ax.tick_params(
        axis="both",
        which="both",
        length=3,
        width=0.5,
        color="#888888",
        labelsize=8,
        pad=2,
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def _draw_zoom_dashed_rect(ax_main, xlim, ylim):
    y_lo = max(ylim[0], ax_main.get_ylim()[0])
    y_hi = min(ylim[1], ax_main.get_ylim()[1])
    rect = plt.Rectangle(
        (xlim[0], y_lo),
        xlim[1] - xlim[0],
        y_hi - y_lo,
        edgecolor="#999999",
        facecolor="none",
        linestyle="--",
        linewidth=0.8,
        zorder=5,
    )
    ax_main.add_patch(rect)


def make_figure(dataset):
    data = dataset["data"]
    x = np.array([d["x"] for d in data])
    truth = np.array([d["truth"] for d in data])
    noisy = np.array([d["noisy"] for d in data])
    dct_smoothed = np.array([d["dct_smoothed"] for d in data])
    fft_smoothed = np.array([d["fft_smoothed"] for d in data])

    color_dct = "#d95f02"
    color_fft = "#1b9e77"
    color_truth = "#333333"
    color_noisy = "#bbbbbb"

    fig = plt.figure(figsize=(10, 3.8))

    ax_main = fig.add_axes([0.06, 0.18, 0.54, 0.75])
    ax_left = fig.add_axes([0.68, 0.55, 0.30, 0.38])
    ax_right = fig.add_axes([0.68, 0.08, 0.30, 0.38])

    # ---- Main plot ----
    ax_main.scatter(x, noisy, s=6, color=color_noisy, alpha=0.45, zorder=1)
    ax_main.plot(
        x,
        truth,
        color=color_truth,
        linewidth=1.0,
        linestyle="--",
        dashes=(4, 4),
        zorder=2,
    )
    ax_main.plot(x, fft_smoothed, color=color_fft, linewidth=2.0, zorder=3)
    ax_main.plot(x, dct_smoothed, color=color_dct, linewidth=2.0, zorder=4)

    _strip_spines(ax_main)
    _draw_axis_arrows(ax_main)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_noisy,
            markersize=4,
            label="Noisy",
        ),
        Line2D(
            [0],
            [0],
            color=color_truth,
            linewidth=1.0,
            linestyle="--",
            dashes=(4, 4),
            label="Truth",
        ),
        Line2D([0], [0], color=color_fft, linewidth=2.0, label="Periodic (FFT)"),
        Line2D([0], [0], color=color_dct, linewidth=2.0, label="Reflective (DCT)"),
    ]
    ax_main.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=7,
        framealpha=0.85,
        edgecolor="none",
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.3,
    )

    left_xlim, left_ylim = [0, 1.2], [0, 5]
    right_xlim, right_ylim = [8.8, 10.0], [10, 17]

    _draw_zoom_dashed_rect(ax_main, left_xlim, left_ylim)
    _draw_zoom_dashed_rect(ax_main, right_xlim, right_ylim)

    # ---- Left zoom ----
    _plot_zoom(
        ax_left,
        x,
        truth,
        noisy,
        dct_smoothed,
        fft_smoothed,
        left_xlim,
        left_ylim,
        "Left boundary",
        color_dct,
        color_fft,
        color_truth,
        color_noisy,
    )

    # ---- Right zoom ----
    _plot_zoom(
        ax_right,
        x,
        truth,
        noisy,
        dct_smoothed,
        fft_smoothed,
        right_xlim,
        right_ylim,
        "Right boundary",
        color_dct,
        color_fft,
        color_truth,
        color_noisy,
    )

    return fig


def _plot_zoom(
    ax,
    x,
    truth,
    noisy,
    dct_smoothed,
    fft_smoothed,
    xlim,
    ylim,
    title,
    color_dct,
    color_fft,
    color_truth,
    color_noisy,
):
    mask = (x >= xlim[0]) & (x <= xlim[1])
    ax.scatter(x[mask], noisy[mask], s=8, color=color_noisy, alpha=0.45, zorder=1)
    ax.plot(
        x[mask],
        truth[mask],
        color=color_truth,
        linewidth=1.0,
        linestyle="--",
        dashes=(4, 4),
        zorder=2,
    )
    ax.plot(x[mask], fft_smoothed[mask], color=color_fft, linewidth=1.8, zorder=3)
    ax.plot(x[mask], dct_smoothed[mask], color=color_dct, linewidth=1.8, zorder=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=4, loc="left")

    _strip_spines(ax)
    _draw_axis_arrows(ax, fontsize=7)


def main():
    dataset = load_data()
    fig = make_figure(dataset)

    pdf_path = FIG_PDF_DIR / "boundary_sanity_1d.pdf"
    fig.savefig(pdf_path, dpi=None, bbox_inches="tight")
    print(f"Saved {pdf_path}")

    png_path = FIG_PNG_DIR / "boundary_sanity_1d.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
