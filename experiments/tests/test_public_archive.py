"""Public packaging boundary and saved numerical results, without article text."""

import csv
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_public_archive_excludes_article_reviews_and_private_input():
    paths = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    assert any(path.startswith("experiments/scripts/") for path in paths)
    for name in filter(None, paths):
        path = Path(name)
        assert path.suffix.lower() not in {
            ".tex", ".bib", ".docx", ".nc", ".h5", ".npy", ".npz", ".bundle",
        }, name
        assert not any(part.startswith("comments") for part in path.parts), name
        if path.parts[0] == "manuscript":
            assert path.parts[1] == "figures", name


def test_current_cartesian_ratios_from_saved_repetitions():
    directory = (
        ROOT / "experiments/results/round1/v2.0.0/3d_wind_boxcar"
        / "release-v2-20261111"
    )
    raw = rows(directory / "benchmark_runs.csv")
    assert len(raw) == 120
    assert all(row["status"] == "success" for row in raw)
    medians = {}
    for case, method in dict.fromkeys(
        (row["case_id"], row["method"]) for row in raw
    ):
        selected = [
            row for row in raw
            if (row["case_id"], row["method"]) == (case, method)
        ]
        assert len(selected) == 3
        medians[case, method] = np.median(
            [float(row["elapsed_s"]) for row in selected]
        )
    cases = {case for case, method in medians if method == "dct_shared_moments"}
    for method, expected in [
        ("numpy_windows", 47), ("astropy_direct", 40), ("astropy_fft", 5),
    ]:
        ratios = {
            case: medians[case, method] / medians[case, "dct_shared_moments"]
            for case in cases
        }
        assert max(ratios, key=ratios.get) == "broad20"
        assert round(max(ratios.values())) == expected
    for case in cases:
        assert medians[case, "scipy_uniform"] < medians[case, "dct_shared_moments"]
