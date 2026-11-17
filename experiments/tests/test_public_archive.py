"""Public packaging boundary and saved numerical results, without article text."""

import csv
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_public_archive_excludes_article_reviews_and_binary_observations():
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


def test_radar_release_instructions_preserve_the_original_input_identity():
    filename = (
        "cfrad_QCed_20220223_023027.051_to_20220223_023410.833_COW1high_SUR.nc"
    )
    checksum = "f1e45c26da3002c31cfea1e62ece5a40c18a668d374193205daf793f41a4fae0"
    data = ROOT / "experiments/data"
    assert (data / "SHA256SUMS").read_text() == f"{checksum}  {filename}\n"
    instructions = (data / "README.md").read_text()
    for phrase in (
        f"releases/download/radar-data-v1/{filename}",
        "releases/download/radar-data-v1/SHA256SUMS",
        checksum, "766,632,010", "REAL_RADAR_PATH",
        "https://doi.org/10.48514/WZ46-W047",
        "https://data.eol.ucar.edu/dataset/612.037",
        "MIT software license does not relicense them",
        "does not fully document",
    ):
        assert phrase in instructions
    source = (
        ROOT / "experiments/scripts/generate_2d_radar_gaussian_validation.py"
    ).read_text()
    assert f'DOCUMENTED_SHA256 = "{checksum}"' in source
    assert "releases/tag/radar-data-v1" in (ROOT / "README.md").read_text()


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
