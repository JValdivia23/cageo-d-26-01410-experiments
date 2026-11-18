"""Check the native Astropy padding variant without timing or writing results."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / "scripts"))
import benchmark_3d_wind_fft_padding as padding
import generate_3d_wind_boxcar_validation as wind


RUN_ROOT = EXPERIMENTS / "results/round1/v2.0.0/3d_wind_boxcar"


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def convolve_fft():
    return pytest.importorskip("astropy.convolution").convolve_fft


@pytest.mark.parametrize("shape,width", [((3, 4, 5), (1, 3, 5)), ((1, 5, 2), (5, 3, 7))])
def test_halo_crop_preserves_corner_impulse(shape, width, convolve_fft):
    field = np.zeros(shape)
    field[0, 0, -1] = 1
    calls = []

    def native(values, kernel, **kwargs):
        calls.append((values.shape, kernel.shape, kwargs))
        return convolve_fft(values, kernel, **kwargs)

    result = padding.prepare_halo_fft(shape, width, native)(field)
    expected = wind.prepare_filter("numpy_windows", shape, width, None, 1)(field)
    assert result.shape == shape
    np.testing.assert_allclose(result, expected, rtol=wind.RTOL, atol=wind.ATOL["mean"])
    (fft_shape, kernel_shape, options), = calls
    assert fft_shape == tuple(n + w - 1 for n, w in zip(shape, width))
    assert kernel_shape == width
    assert options["boundary"] == "wrap"
    assert not options["psf_pad"] and not options["fft_pad"]
    assert "fftn" not in options and "ifftn" not in options


@pytest.fixture(scope="module")
def frozen_inputs():
    return {
        row["case_id"]: row["value"]
        for row in read_rows(RUN_ROOT / "release-v2-20261111/settings.csv")
        if row["setting"] == "input_sha256"
    }


@pytest.mark.parametrize("case_id", tuple(dict.fromkeys((*padding.REPORTED_CASES, *wind.CONTROLS))))
def test_halo_statistics_match_reference_and_old_fft(case_id, convolve_fft, frozen_inputs, monkeypatch):
    case = wind.CASES[case_id]
    observed, _, settings = wind.generate_input(case)
    assert settings["input_sha256"] == frozen_inputs[case_id]
    reference = wind.calculate("numpy_windows", observed, case.width, None)
    old_fft = wind.calculate(padding.PADDED_FFT, observed, case.width, convolve_fft)
    prepare_filter = wind.prepare_filter

    def prepare(method, shape, width, backend, chunk_planes):
        if method == padding.HALO_FFT:
            return padding.prepare_halo_fft(shape, width, backend)
        return prepare_filter(method, shape, width, backend, chunk_planes)

    monkeypatch.setattr(wind, "prepare_filter", prepare)
    audit = {}
    result = wind.calculate(padding.HALO_FFT, observed, case.width, convolve_fft, audit=audit)
    assert audit == {"full_nd_filter_calls": 7, "filter_preparations": 1}
    for component in wind.COMPONENTS:
        for statistic in wind.STATISTICS:
            assert result[component][statistic].shape == case.shape
    for comparison in (reference, old_fft):
        rows = wind.compare_outputs("test", case_id, padding.HALO_FFT, result, comparison, observed)
        assert all(row["status"] in ("pass", "no_comparable") for row in rows), [
            row for row in rows if row["status"] == "fail"
        ]


def test_saved_padding_timings_use_one_matched_campaign(frozen_inputs):
    directory = RUN_ROOT / "astropy-fft-halo-20261118"
    if not directory.exists():
        pytest.skip("the separate padding benchmark has not been saved here")
    settings, timings = defaultdict(dict), defaultdict(list)
    for row in read_rows(directory / "settings.csv"):
        settings[row["case_id"]][row["setting"]] = row["value"]
    source_hashes = padding.source_hashes()
    assert set(settings) == set(padding.REPORTED_CASES) | set(wind.CONTROLS)
    for case, values in settings.items():
        assert values["numerical_status"] == "pass"
        assert values["input_sha256"] == frozen_inputs[case]
        assert json.loads(values["review_sources_sha256"]) == source_hashes
    comparisons = read_rows(directory / "comparison.csv")
    assert all(row["status"] in ("pass", "no_comparable") for row in comparisons)
    assert {row["case_id"] for row in comparisons if row["method"] == padding.HALO_FFT} == set(settings)
    for row in read_rows(directory / "benchmark_runs.csv"):
        case, method = row["case_id"], row["method"]
        assert row["status"] == "success"
        worker = json.loads(settings[case][f"worker_{method}_{row['repetition']}"])
        assert worker["input_sha256"] == settings[case]["input_sha256"] == frozen_inputs[case]
        assert worker["full_nd_filter_calls"] == 7 and worker["filter_preparations"] == 1
        assert worker["n_outputs"] == 12
        assert worker["v2_review_sources_sha256"] == source_hashes
        timings[case, method].append(row)
    summaries = {
        (row["case_id"], row["method"]): row
        for row in read_rows(directory / "benchmark_summary.csv")
    }
    assert set(timings) == set(summaries) == {
        (case, method)
        for case in padding.REPORTED_CASES
        for method in (*wind.METHODS, padding.HALO_FFT)
    }
    for key, rows in timings.items():
        summary = summaries[key]
        assert {row["repetition"] for row in rows} == {"1", "2", "3"}
        assert int(summary["n_success"]) == 3 and int(summary["n_failure"]) == 0
        assert float(summary["median_time_s"]) == np.median([float(row["elapsed_s"]) for row in rows])
        assert float(summary["max_peak_ram_mib"]) == max(float(row["peak_ram_mib"]) for row in rows)
