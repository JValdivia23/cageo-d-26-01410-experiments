# CAGEO-D-26-01410 experiments

This repository contains the experiment code, recorded settings, numerical
results, and reproducible figure assets accompanying CAGEO-D-26-01410.
It does **not** distribute the article text, bibliography, reviewer responses,
or private author-workspace history. The exact radar input is distributed
separately as a release attachment, not as a Git-tracked file.

## Contents

- `experiments/scripts/`: numerical calculations, benchmark drivers, and figure
  exporters. Existing numerical sources are preserved byte-for-byte.
- `experiments/results/`: frozen measurements, numerical comparisons, input and
  output hashes, settings, and software provenance.
- `experiments/tests/`: numerical, saved-result, source-identity, and figure
  checks. Article-wording and article-table cross-checks belong to the private
  author workspace and are not included here.
- `manuscript/figures/`: figure exports and their metadata **only**. This
  historical path is retained because the frozen source files and figure
  checks use it; it contains no article source or bibliography.
- `experiments/README.md`: experiment definitions and reproduction commands.

The latest [wind FFT padding comparison](experiments/README.md#wind-fft-padding-comparison)
includes 90 measurements for six implementations with matched kernels, boundaries,
and output grids. The earlier wind size, width, and missingness results remain
archived separately.

No old repository history is required or included. Historical comparisons that
require private author-workspace commits are explicitly skipped by the
standalone compatibility layer; numerical checks are not relaxed.

## Versions and environment

The current calculations use `dct-toolkit` **v2.0.0**, commit
`7ff29e998dd1102d3fef2a84050402e86dc7ef67`. The radar benchmark deliberately
matches a four-standard-deviation sampled Gaussian across all implementations;
it does not measure the full Gaussian default.

Archived calculations retain **v1.0.0**, commit
`9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c`, or the submitted **v0.5.0**
provenance recorded in their own settings. Do not substitute a new toolkit
release beneath an old measurement.

The recorded current environment is Python 3.11.13, NumPy 2.4.4, SciPy 1.17.1,
and Astropy 8.0.2. Figure exports additionally use Matplotlib 3.10.8, Pillow
12.2.0, and cmweather 0.3.2. Radar input uses xarray and netCDF4. Benchmarks were
measured on an Apple M4 Pro with x86_64 Python under Rosetta 2 and requested
single-threaded numerical libraries. Runtime and total-process RSS are
implementation- and environment-specific.

Install dependencies in an isolated environment:

```bash
python -m pip install -r requirements.txt
```

Extract both exact toolkit sources from a local clone of
<https://github.com/JValdivia23/dct-toolkit>, without changing that checkout:

```bash
export TOOLKIT_REPO=/path/to/dct-toolkit
export RUNTIME=/path/to/new-runtime-directory
mkdir -p "$RUNTIME/v1" "$RUNTIME/v2"
git -C "$TOOLKIT_REPO" archive 9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c | tar -xf - -C "$RUNTIME/v1"
git -C "$TOOLKIT_REPO" archive 7ff29e998dd1102d3fef2a84050402e86dc7ef67 | tar -xf - -C "$RUNTIME/v2"
export DCT_TOOLKIT_SOURCE="$RUNTIME/v1"
export DCT_TOOLKIT_V2_SOURCE="$RUNTIME/v2"
export PYTHONPATH="$DCT_TOOLKIT_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The source-checked bridge runs v2.0.0 in separate child interpreters. Legacy
tests use v1.0.0, so the two packages are not mixed in a process.

## Verification

From the repository root:

```bash
python -B -m pytest -q -p no:cacheprovider experiments/tests
```

The tests use small controls and existing saved files; they do not launch a
new full benchmark campaign. Optional archived-result checks are enabled by:

```bash
export REVIEW_RESULT_DIRS=experiments/results/round1/1d_boxcar/lead-review-1d-20261027:experiments/results/round1/2d_radar_gaussian/lead-review-radar-exact-only-20261027:experiments/results/round1/3d_wind_boxcar/lead-review-3d-matched-20261027
export LARGE_MEAN_RESULT_DIR=experiments/results/round1/3d_large_boxcar_mean/quick-large-mean-1gib-20261027
export RADAR_BENCHMARK_RESULT_DIR=experiments/results/round1/2d_radar_statistics_benchmark/radar-matched-statistics-20261102
export RADAR_LOOP_RESULT_DIR=experiments/results/round1/2d_radar_range_loops/radar-range-loops-20261103
export DISCRETE_GAUSSIAN_RESULT_DIR=experiments/results/round1/2d_radar_discrete_gaussian/radar-discrete-gaussian-20261110
export GAUSSIAN_DOUBLE_LOOP_RESULT_DIR=experiments/results/round1/2d_radar_gaussian_double_loop/radar-gaussian-double-loop-20261111
python -B -m pytest -q -p no:cacheprovider experiments/tests
```

Tests needing unavailable private Git history, the original absolute checkout
path, or the exact radar input give explicit skip reasons. These limitations
are separate from the article-text checks excluded from this public package.

Release-documentation verification on 17 November 2026 reports **430 passed, 62 skipped**
and one existing radar-dependency binary-compatibility warning. All 123 frozen
numerical source, result, and figure files match the original archive
byte-for-byte. No full experiment or benchmark campaign was rerun.

## Data access

The exact processed COW `DBZHC` input is available in the
[radar-data-v1 release](https://github.com/JValdivia23/cageo-d-26-01410-experiments/releases/tag/radar-data-v1).
The unchanged NetCDF file is 766,632,010 bytes (731 MiB). See
[`experiments/data/README.md`](experiments/data/README.md) for download commands,
the SHA-256 checksum, and attribution to Wurman and Kosiba's
[WINTRE-MIX FARM data](https://doi.org/10.48514/WZ46-W047).
After downloading and verifying the file:

```bash
export REAL_RADAR_PATH=/path/to/cfrad_QCed_20220223_023027.051_to_20220223_023410.833_COW1high_SUR.nc
```

Full radar reproduction needs that exact file, which is not included in a Git
clone or the source-code ZIP. The upstream quality-control procedure is not
fully documented in the supplied file; publishing it does not establish the
cause of missing observations.

## Publication boundary

Only experiment files and necessary numerical figure assets belong in Git.
Do not copy a manuscript checkout, bibliography, review correspondence, private
input, or old Git history into this repository. The author-approved public
radar file is a release asset and remains excluded from Git. The public-archive
test checks the tracked-file boundary. Original numerical source hashes and
saved measurements remain unchanged; no benchmark is rerun merely to package
them.

## License

The software is MIT-licensed; see `LICENSE`. Citation metadata is provided in
`CITATION.cff`. The radar observations retain their source data terms and
attribution; the software license does not relicense them.
