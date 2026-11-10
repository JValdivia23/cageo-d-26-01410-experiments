import csv
import json
from pathlib import Path

import numpy as np
import scipy.fft
from dct_toolkit.core import dct_convolve_1d, get_dct_transfer_function

N = 200
W = 15
NOISE_STD = 1.0
FIGURE_SEED = 42
METRIC_SEEDS = [0, 1, 2, 3, 4]


def build_truth(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 10.0, n)
    truth = 1.2 * x + 2.5 * np.sin(1.5 * x)
    return x, truth


def smooth_reflective_and_periodic(signal: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    n = signal.size

    h_dct = get_dct_transfer_function(n=n, kernel_type="gaussian", width=width)
    dct_smoothed = dct_convolve_1d(signal, H=h_dct)

    sigma = width / np.sqrt(12.0)
    omega = scipy.fft.fftfreq(n, d=1.0) * 2.0 * np.pi
    h_fft = np.exp(-0.5 * (omega * sigma) ** 2)
    fft_smoothed = scipy.fft.ifft(scipy.fft.fft(signal) * h_fft).real

    return dct_smoothed, fft_smoothed


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std}


def generate_outputs() -> None:
    x, truth = build_truth(N)

    # Figure dataset (single seed for reproducible visual)
    np.random.seed(FIGURE_SEED)
    noisy = truth + np.random.normal(0.0, NOISE_STD, N)
    dct_smoothed, fft_smoothed = smooth_reflective_and_periodic(noisy, W)

    # Metrics across seeds used in manuscript table
    edge_idx = np.r_[0:W, N - W : N]
    all_dct, all_fft, edge_dct, edge_fft = [], [], [], []

    for seed in METRIC_SEEDS:
        np.random.seed(seed)
        noisy_seed = truth + np.random.normal(0.0, NOISE_STD, N)
        dct_seed, fft_seed = smooth_reflective_and_periodic(noisy_seed, W)

        all_dct.append(rmse(dct_seed, truth))
        all_fft.append(rmse(fft_seed, truth))
        edge_dct.append(rmse(dct_seed[edge_idx], truth[edge_idx]))
        edge_fft.append(rmse(fft_seed[edge_idx], truth[edge_idx]))

    rmse_summary = {
        "all_domain": {
            "reflective": summarize(all_dct),
            "periodic": summarize(all_fft),
        },
        "edge_only": {
            "reflective": summarize(edge_dct),
            "periodic": summarize(edge_fft),
        },
    }

    all_rel = (rmse_summary["all_domain"]["periodic"]["mean"] / rmse_summary["all_domain"]["reflective"]["mean"] - 1.0) * 100.0
    edge_rel = (rmse_summary["edge_only"]["periodic"]["mean"] / rmse_summary["edge_only"]["reflective"]["mean"] - 1.0) * 100.0
    rmse_summary["all_domain"]["relative_increase_periodic_percent"] = all_rel
    rmse_summary["edge_only"]["relative_increase_periodic_percent"] = edge_rel

    payload = {
        "metadata": {
            "N": N,
            "W": W,
            "kernel": "gaussian",
            "noise_std": NOISE_STD,
            "figure_seed": FIGURE_SEED,
            "metric_seeds": METRIC_SEEDS,
            "edge_definition": "first and last W samples",
            "description": "1D boundary condition sanity check: reflective DCT vs periodic FFT",
        },
        "boundary_rmse_summary": rmse_summary,
        "data": [
            {
                "index": int(i),
                "x": float(x[i]),
                "truth": float(truth[i]),
                "noisy": float(noisy[i]),
                "dct_smoothed": float(dct_smoothed[i]),
                "fft_smoothed": float(fft_smoothed[i]),
            }
            for i in range(N)
        ],
    }

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "boundary_1d_data.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = results_dir / "boundary_1d_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "reflective_mean", "reflective_std", "periodic_mean", "periodic_std", "relative_increase_periodic_percent"])
        writer.writerow(
            [
                "all_domain_rmse",
                rmse_summary["all_domain"]["reflective"]["mean"],
                rmse_summary["all_domain"]["reflective"]["std"],
                rmse_summary["all_domain"]["periodic"]["mean"],
                rmse_summary["all_domain"]["periodic"]["std"],
                rmse_summary["all_domain"]["relative_increase_periodic_percent"],
            ]
        )
        writer.writerow(
            [
                "edge_only_rmse",
                rmse_summary["edge_only"]["reflective"]["mean"],
                rmse_summary["edge_only"]["reflective"]["std"],
                rmse_summary["edge_only"]["periodic"]["mean"],
                rmse_summary["edge_only"]["periodic"]["std"],
                rmse_summary["edge_only"]["relative_increase_periodic_percent"],
            ]
        )

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    generate_outputs()
