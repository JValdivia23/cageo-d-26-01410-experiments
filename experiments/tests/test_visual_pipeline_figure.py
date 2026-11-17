"""Checks for the synthetic introductory illustration, not a validation study."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import dctn, idctn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import _review_figures as figures
import generate_visual_pipeline_figure as visual


@pytest.fixture(scope="module")
def data():
    return visual.calculate()


def test_filter_and_direct_ratios_against_independent_dct(data):
    n = 80
    omega = np.pi * np.arange(n) / n
    h = np.exp(-0.5 * omega**2 * (visual.WIDTH / np.sqrt(12))**2)
    transfer = np.outer(h, h)

    def reference(a):
        return idctn(dctn(a, type=2, norm="ortho") * transfer, type=2, norm="ortho")

    for key, array in (("a1", data["values"]), ("a2", data["values"]**2),
                       ("d", data["mask"])):
        np.testing.assert_allclose(data[key], reference(array), atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(data["transfer"], transfer, atol=2e-15, rtol=2e-14)
    accept = np.isfinite(data["d"]) & (data["d"] > visual.FLOOR)
    assert np.any(accept) and np.any(~accept)
    assert np.array_equal(np.isfinite(data["mean"]), accept)
    assert np.array_equal(np.isfinite(data["std"]), accept)
    np.testing.assert_allclose(data["mean"][accept], data["a1"][accept] / data["d"][accept])
    expected_std = np.sqrt(np.maximum(0, data["a2"][accept] / data["d"][accept]
                                      - data["mean"][accept]**2))
    np.testing.assert_allclose(data["std"][accept], expected_std)
    np.testing.assert_array_equal(data["coverage"], np.clip(data["d"], 0, 1))
    assert np.all(data["values"][data["mask"] == 0] == 0)


def test_matches_release_direct_api(data):
    toolkit = visual.load_toolkit()
    field = np.where(data["mask"].astype(bool), data["field"], np.nan)
    options = dict(kernel_type="gaussian", restore_input_nan=False,
                   _stable_fallback=False, _denominator_floor=visual.FLOOR)
    np.testing.assert_allclose(data["mean"], toolkit.dct_mean(field, visual.WIDTH, **options),
                               atol=1e-12, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(data["std"], toolkit.dct_std(field, visual.WIDTH, **options),
                               atol=1e-10, rtol=1e-10, equal_nan=True)


def test_figure_labels_and_reproducible_exports(data, tmp_path):
    from matplotlib.text import Text
    from PIL import Image
    import matplotlib.pyplot as plt

    fig = visual.create_figure(data)
    labels = [text.get_text() for text in fig.findobj(Text) if text.get_text()]
    for label in ("Input field", "Spectral domain", "DCT-II (reflective)",
                  "Local mean", "normalize", r"Coverage $\rho$"):
        assert label in labels
    assert "Same filter applied to" not in labels
    assert "Normalize with the filtered mask" not in labels
    assert not any("N_{\\mathrm{eff}}" in label for label in labels)
    # The tick labels must not collide with the next enlarged map or its title.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    heading = next(text for text in fig.texts if text.get_text() == "Enlarged yellow-box region")
    heading_bottom = heading.get_window_extent(renderer).y0
    assert all(heading_bottom > fig.axes[index].title.get_window_extent(renderer).y1 + 4
               for index in (4, 6, 8))
    for image_ax, bar_ax, next_ax in ((fig.axes[4], fig.axes[5], fig.axes[6]),
                                      (fig.axes[6], fig.axes[7], fig.axes[8])):
        assert bar_ax.get_position().x0 > image_ax.get_position().x1
        tick_right = max(t.get_window_extent(renderer).x1 for t in bar_ax.get_yticklabels())
        assert tick_right + 4 < next_ax.get_window_extent(renderer).x0
    first = figures.export_figure(fig, tmp_path / "a", visual.STEM, {"purpose": "test"})
    second = figures.export_figure(fig, tmp_path / "b", visual.STEM, {"purpose": "test"})
    for kind in ("png", "pdf", "settings"):
        assert first[kind].read_bytes() == second[kind].read_bytes()
    with Image.open(first["png"]) as image:
        assert image.size == (1950, 1125)
        assert all(abs(x - 300) < 0.01 for x in image.info["dpi"])
    plt.close(fig)
