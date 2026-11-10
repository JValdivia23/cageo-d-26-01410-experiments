"""Figure-only checks; no scientific experiments or benchmarks are executed."""

import csv
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import _review_figures as figures
import generate_statistics_pipeline_figure as pipeline


@pytest.fixture
def drawing():
    fig = pipeline.create_figure()
    yield fig
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_imports_do_not_load_figure_or_numerical_dependencies():
    code = (
        "import _review_figures, generate_statistics_pipeline_figure, sys; "
        "assert not any(name in sys.modules for name in "
        "('matplotlib', 'numpy', 'scipy', 'dct_toolkit', 'PIL'))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=SCRIPTS, check=True)


def test_default_output_root():
    assert figures.DEFAULT_FIGURE_ROOT == SCRIPTS.parents[1] / "manuscript/figures/round1"


def test_pipeline_labels_and_independent_diagnostic_branch(drawing):
    from matplotlib.text import Text

    labels = [text.get_text() for text in drawing.findobj(Text) if text.get_text()]
    assert labels.count("Same complete") == labels.count(r"$\mathcal{S}_g$") == 3
    for label in (
        r"$x_m$", r"$x_m^{(2)}$", "Original validity mask",
        r"$a_1=\mathcal{S}_g\{x_m\}$", r"$a_2=\mathcal{S}_g\{x_m^{(2)}\}$",
        r"$d=\mathcal{S}_g\{m\}$", r"Only if $d$ is finite and $d>\epsilon=10^{-3}$",
        r"$\mu=a_1/d\qquad M_2=a_2/d$",
        r"$\sigma^2=\max(0,M_2-\mu^2),\quad \sigma=\sqrt{\sigma^2}$",
        "Otherwise: moment outputs undefined", r"$\rho=\mathrm{clip}(d,0,1)$",
        r"$N_{\mathrm{eq}}=A\rho$",
    ):
        assert label in labels
    assert not any("+\\epsilon" in text or r"N_{\mathrm{eq}}=d" in text for text in labels)
    branch_ids = {patch.get_gid() for patch in drawing.axes[0].patches
                  if patch.get_gid() and "_to_" in patch.get_gid()}
    assert "mask_response_to_moments" in branch_ids
    assert "mask_response_to_diagnostics" in branch_ids
    assert not any("moments_to_diagnostics" in gid for gid in branch_ids)
    assert not drawing.axes[0].images
    assert not any(artist.get_rasterized() for artist in drawing.findobj())


def test_fixed_canvas_fonts_png_dpi_and_export_hashes(drawing, tmp_path):
    import matplotlib as mpl
    from PIL import Image

    paths = figures.export_figure(drawing, tmp_path / "pipeline", pipeline.STEM, {
        "purpose": "Test conceptual diagram export", "figure_width_in": "not actual",
        "source_case": "no experiment",
    })
    assert tuple(drawing.get_size_inches()) == pipeline.FIGSIZE
    with Image.open(paths["png"]) as image:
        assert image.size == (1950, 1680)
        assert all(abs(value - 300) < 0.01 for value in image.info["dpi"])
        assert image.getpixel((0, 0)) == (255, 255, 255, 255)
    with paths["settings"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["figure", "setting", "value"]
        rows = list(reader)
    assert all(row["figure"] == pipeline.STEM for row in rows)
    values = {row["setting"]: row["value"] for row in rows}
    assert len(values) == len(rows)
    assert values["figure_width_in"] == "6.5"
    assert values["figure_height_in"] == "5.6"
    assert float(values["min_text_size_pt"]) >= 10
    assert values["source_case"] == "no experiment"
    assert values["pdf_fonttype"] == values["ps_fonttype"] == "42"
    assert mpl.get_backend().lower() == "agg"
    for kind in ("pdf", "png"):
        assert values[f"{kind}_sha256"] == hashlib.sha256(paths[kind].read_bytes()).hexdigest()
        assert paths[kind].relative_to(tmp_path / "pipeline") == Path(values[f"{kind}_path"])


def test_exports_are_reproducible_and_ignore_tight_bbox(drawing, tmp_path):
    import matplotlib as mpl

    with mpl.rc_context({"savefig.bbox": "tight"}):
        first = figures.export_figure(drawing, tmp_path / "a", "pipeline", {"purpose": "Test"})
        second = figures.export_figure(drawing, tmp_path / "b", "pipeline", {"purpose": "Test"})
    for kind in ("pdf", "png", "settings"):
        assert first[kind].read_bytes() == second[kind].read_bytes()


@pytest.mark.parametrize("stem", ["../outside", "/absolute", "x/y", ""])
def test_export_rejects_unsafe_filenames(drawing, tmp_path, stem):
    with pytest.raises(ValueError, match="Invalid figure stem"):
        figures.export_figure(drawing, tmp_path, stem, {"purpose": "Test"})


def test_export_requires_purpose(drawing, tmp_path):
    with pytest.raises(ValueError, match="purpose"):
        figures.export_figure(drawing, tmp_path, "pipeline", {})


def test_export_rejects_clipped_text(drawing, tmp_path):
    drawing.text(-0.1, 0.5, "Outside the canvas")
    with pytest.raises(ValueError, match="Text outside figure canvas"):
        figures.export_figure(drawing, tmp_path, "pipeline", {"purpose": "Test"})
