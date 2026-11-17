"""Small, figure-only style/export helpers for the round-1 manuscript graphics.

Import this module lazily from figure-only branches, not benchmark paths. Merely
importing it loads no plotting libraries and performs no calculations or I/O.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import platform
import re
from typing import Any, Mapping


DEFAULT_FIGURE_ROOT = (
    Path(__file__).resolve().parents[2] / "manuscript" / "figures" / "round1"
)
EXPORT_DPI = 300


def configure_plotting(*, use_agg: bool = True) -> None:
    """Select Agg before pyplot and set a readable, embedded-font house style."""
    import matplotlib as mpl

    if use_agg:
        mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "text.usetex": False,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "savefig.dpi": EXPORT_DPI,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _check_text_bounds(fig) -> list:
    """Reject labels outside the fixed canvas rather than silently cropping them."""
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [text for text in fig.findobj(Text)
             if text.get_visible() and text.get_text().strip()]
    for text in texts:
        bounds = text.get_window_extent(renderer)
        if (bounds.x0 < -0.5 or bounds.y0 < -0.5
                or bounds.x1 > fig.bbox.width + 0.5
                or bounds.y1 > fig.bbox.height + 0.5):
            raise ValueError(f"Text outside figure canvas: {text.get_text()!r}")
    return texts


def export_figure(
    fig, case_dir: str | Path, stem: str, settings: Mapping[str, Any],
) -> dict[str, Path]:
    """Save the same fixed canvas as PDF/300-dpi PNG plus figure_settings.csv.

    ``settings`` must include a nonempty ``purpose``; add the actual source/case
    choices there, not invented numerical results. Output dimensions, style,
    versions and SHA-256 hashes are recorded automatically. The returned mapping
    has ``pdf``, ``png`` and ``settings`` paths. One case directory describes one
    figure; rerunning intentionally replaces that figure and its metadata only.
    """
    import matplotlib as mpl

    from _review_common import write_csv

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stem):
        raise ValueError(f"Invalid figure stem: {stem!r}")
    if not str(settings.get("purpose", "")).strip():
        raise ValueError("Figure settings must state a purpose.")
    texts = _check_text_bounds(fig)
    directory = Path(case_dir).expanduser().resolve()
    paths = {kind: directory / kind / f"{stem}.{kind}" for kind in ("pdf", "png")}
    paths["settings"] = directory / "figure_settings.csv"
    for kind in ("pdf", "png"):
        paths[kind].parent.mkdir(parents=True, exist_ok=True)

    # No tight bounding box: both exports retain the requested physical canvas.
    # Omit PDF timestamps so unchanged sources/settings give stable assets.
    with mpl.rc_context({"savefig.bbox": None}):
        for kind in ("pdf", "png"):
            metadata = ({"Title": str(settings["purpose"]), "CreationDate": None,
                         "ModDate": None} if kind == "pdf" else None)
            fig.savefig(paths[kind], format=kind, dpi=EXPORT_DPI, bbox_inches=None,
                        facecolor="white", edgecolor="white", transparent=False,
                        metadata=metadata)

    actual = {
        "figure_width_in": float(fig.get_figwidth()),
        "figure_height_in": float(fig.get_figheight()),
        "export_dpi": EXPORT_DPI,
        "background": "white, opaque",
        "canvas_cropping": "none (fixed figsize)",
        "font_family": list(mpl.rcParams["font.family"]),
        "base_font_size_pt": mpl.rcParams["font.size"],
        "min_text_size_pt": min((text.get_fontsize() for text in texts), default=""),
        "max_text_size_pt": max((text.get_fontsize() for text in texts), default=""),
        "mathtext_fontset": mpl.rcParams["mathtext.fontset"],
        "pdf_fonttype": mpl.rcParams["pdf.fonttype"],
        "ps_fonttype": mpl.rcParams["ps.fonttype"],
        "matplotlib_version": mpl.__version__,
        "backend": mpl.get_backend(),
        "python_version": platform.python_version(),
        "figure_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for kind in ("pdf", "png"):
        actual[f"{kind}_path"] = str(paths[kind].relative_to(directory))
        actual[f"{kind}_sha256"] = hashlib.sha256(paths[kind].read_bytes()).hexdigest()
    write_csv(paths["settings"], ["figure", "setting", "value"], [
        {"figure": stem, "setting": key, "value": value}
        for key, value in sorted({**settings, **actual}.items())
    ])
    return paths
