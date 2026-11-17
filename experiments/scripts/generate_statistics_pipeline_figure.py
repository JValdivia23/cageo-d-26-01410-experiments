#!/usr/bin/env python3
"""Draw the corrected conceptual pipeline; no experiment or toolkit computation.

Run with the dct-publish Python. Defaults to manuscript/figures/round1/pipeline;
--output-root changes the round1 root, retaining the pipeline/pdf and png layout.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _review_figures import DEFAULT_FIGURE_ROOT, configure_plotting, export_figure


FIGSIZE = (6.5, 5.6)
STEM = "masked_statistics_pipeline"


def create_figure():
    """Return a fixed-size, all-vector diagram of the selected direct-ratio path."""
    configure_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path as DrawPath

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, FIGSIZE[0]), ylim=(0, FIGSIZE[1]))
    ax.set_axis_off()
    ink, edge = "#263544", "#6B7B8A"

    def box(x, y, width, height, *, fill="#F7F9FB", gid):
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.055",
            linewidth=0.85, edgecolor=edge, facecolor=fill, zorder=2,
        )
        patch.set_gid(gid)
        ax.add_patch(patch)

    def text(x, y, value, *, size=10, bold=False):
        return ax.text(x, y, value, ha="center", va="center", fontsize=size,
                       fontweight="bold" if bold else "normal", color=ink,
                       linespacing=1.2, zorder=3)

    def arrow(points, gid):
        path = DrawPath(points, [DrawPath.MOVETO] + [DrawPath.LINETO] * (len(points) - 1))
        patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=10,
                                linewidth=0.9, color=edge, zorder=1)
        patch.set_gid(gid)
        ax.add_patch(patch)

    box(0.20, 4.98, 6.10, 0.49, gid="original_inputs")
    text(3.25, 5.30, r"Original observations $x$ and validity mask $m$", size=10.5, bold=True)
    text(3.25, 5.08,
         r"$m=1$: retained finite observation; $m=0$: unavailable or excluded")

    lanes = [
        (1.17, "data", r"$x_m$", r"Valid $x$; zero elsewhere", r"$a_1=\mathcal{S}_g\{x_m\}$"),
        (3.25, "squares", r"$x_m^{(2)}$", r"Valid $x^2$; zero elsewhere",
         r"$a_2=\mathcal{S}_g\{x_m^{(2)}\}$"),
        (5.33, "mask", r"$m$", "Original validity mask", r"$d=\mathcal{S}_g\{m\}$"),
    ]
    for center, name, symbol, description, response in lanes:
        box(center - 0.93, 4.07, 1.86, 0.59, gid=f"prepared_{name}")
        text(center, 4.46, symbol, size=11)
        text(center, 4.22, description)
        arrow([(center, 4.98), (center, 4.68)], f"original_to_{name}")

        box(center - 0.93, 3.23, 1.86, 0.56, fill="#E8F0F7", gid=f"filter_{name}")
        text(center, 3.62, "Same complete")
        text(center, 3.40, r"$\mathcal{S}_g$", size=11.5)
        arrow([(center, 4.06), (center, 3.81)], f"{name}_to_complete_filter")

        box(center - 0.93, 2.55, 1.86, 0.42, gid=f"response_{name}")
        text(center, 2.76, response, size=11)
        arrow([(center, 3.22), (center, 2.99)], f"complete_filter_to_{name}_response")

    box(0.20, 0.65, 3.93, 1.55, fill="#EEF3F8", gid="accepted_moments")
    text(2.165, 2.04, "Moment estimates", bold=True)
    text(2.165, 1.77, r"Only if $d$ is finite and $d>\epsilon=10^{-3}$")
    text(2.165, 1.44, r"$\mu=a_1/d\qquad M_2=a_2/d$", size=11)
    text(2.165, 1.12,
         r"$\sigma^2=\max(0,M_2-\mu^2),\quad \sigma=\sqrt{\sigma^2}$", size=11)
    text(2.165, 0.84, "Otherwise: moment outputs undefined")
    for center, name in ((1.17, "data"), (3.25, "squares")):
        arrow([(center, 2.54), (center, 2.22)], f"{name}_response_to_moments")
    arrow([(5.33, 2.37), (4.00, 2.37), (4.00, 2.22)], "mask_response_to_moments")

    # The original mask response branches BEFORE moment acceptance. Neither
    # coverage nor equivalent count uses a completed mask or a ratio-gated field.
    box(4.43, 0.65, 1.87, 1.55, fill="#EFF5EE", gid="coverage_and_count")
    text(5.365, 2.04, "Local data coverage", bold=True)
    text(5.365, 1.77, r"$\rho=\mathrm{clip}(d,0,1)$", size=11)
    text(5.365, 1.44, "Equivalent data count", bold=True)
    text(5.365, 1.19, r"$N_{\mathrm{eq}}=A\rho$", size=11)
    text(5.365, 0.84, "$A$: nominal grid factor\nfrom requested widths")
    arrow([(5.33, 2.54), (5.33, 2.22)], "mask_response_to_diagnostics")
    ax.add_patch(Circle((5.33, 2.37), 0.022, facecolor=edge, edgecolor="none", zorder=3))

    text(3.25, 0.28,
         "Per-axis kernels, widths and boundary modes, plus geometry\n"
         r"and stage order, define the complete operator $\mathcal{S}_g$.")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    args = parser.parse_args(argv)
    fig = create_figure()
    settings = {
        "purpose": "Conceptual masked local-statistics pipeline for the selected direct-ratio validation",
        "figure_kind": "conceptual diagram; no new experiment or numerical results",
        "definition_source": "manuscript/main.tex, Sections 2.1 and 3.1-3.3",
        "input_rule": "m=1 for retained finite observations; x_m and x_m^(2) explicitly zero elsewhere",
        "operator": "same complete prescribed S_g applied separately to x_m, x_m^(2), and original m",
        "operator_choices": "per-axis kernels, widths, boundaries; geometry and prescribed stage order",
        "moment_acceptance": "finite d and d > epsilon; epsilon=1e-3; strict test, not additive",
        "moments": "mu=a1/d; M2=a2/d; variance=max(0,M2-mu^2); std=sqrt(variance)",
        "rejected_moments": "undefined; completion and input-mask restoration disabled in selected validation",
        "coverage_and_count": "rho=clip(d,0,1); N_eq=A*rho; independent of moment acceptance",
        "nominal_factor": "A uses converted requested grid widths, not rounded footprint; N_eq is not ESS",
        "interpretation": "population-moment and coverage-fraction interpretations require nonnegative spatial weights",
        "completion_scope": "default API completion described separately in Section 3.3; not validated here",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    paths = export_figure(fig, args.output_root / "pipeline", STEM, settings)
    import matplotlib.pyplot as plt

    plt.close(fig)
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return paths


if __name__ == "__main__":
    main()
