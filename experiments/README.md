# Experiment definitions and reproduction

All paths below are relative to the repository root. Use the exact toolkit
sources and environment described in the top-level README. Existing result
directories are frozen: choose a new `--run-id` for any rerun, and never replace
recorded measurements or change their source identity.

## Current v2.0.0 calculations

`experiments/scripts/run_review_v2.py` binds the unchanged numerical functions
to source-checked `dct-toolkit` v2.0.0, commit
`7ff29e998dd1102d3fef2a84050402e86dc7ef67`. It records both bridge and inherited
source hashes. The four original release campaigns are stored under
`experiments/results/round1/v2.0.0/`, each with run ID `release-v2-20261111`.

| Experiment directory | Definition | Recorded scope |
|---|---|---|
| `1d_boxcar` | Width-15 half-sample reflective boxcar statistics against NumPy symmetric windows, SciPy uniform filtering, and extended FFT convolution | Three profiles, five replicates, five missing-data patterns; 75 cases |
| `1d_boundary_comparison` | Complete and middle-gap observations with reflective DCT, periodic FFT, and clipped in-domain NumPy windows | The same two inputs; separate interior agreement and clean-signal errors |
| `2d_radar_gaussian_double_loop` | Range-dependent, azimuth-first/range-second Gaussian statistics; periodic azimuth and reflective range | One 720 × 1180 COW sweep, W5/W15, matched four-sigma kernels; spectral, explicit Python double-loop, and matched SciPy spatial calculations |
| `3d_wind_boxcar` | Componentwise u/v/w statistics over reflective spatial boxcars, with one common validity mask | Eight grid/window/missingness configurations; 120 timing repetitions across five implementations |

The 1D signals are sampled on 200 points. The profiles are a trend with sine
variation, curved endpoints, and an abrupt step. The complete observations are
shared across complete, random-removal, middle-gap, boundary-gap, and all-missing
patterns. The master seed is 20261027; each signal/replicate pair has independent
noise and missingness child streams. Exact formulas and all generated inputs
are defined in the unchanged scripts and saved settings.

The radar moments use available dBZ observations directly. Every method applies
three complete ordered filters to the finite mask, zero-filled observations,
and their squares, then normalizes once. The independent reference uses explicit
centered-residual sums. The Python double-loop calculation visits each output
cell in each axis stage and uses NumPy neighbourhood dot products.

All direct moments require finite raw mask response greater than 1e-3.
Completion and restoration are disabled. Variance is checked before clamping;
near-zero standard deviation uses the recorded variance-error bound. Numerical
agreement on incomplete input is not a recovery or reliability guarantee.

The four-sigma cutoff is specific to the matched radar comparison. The toolkit's
full sampled Gaussian default has no fixed cutoff, and its end-to-end runtime
is not measured by these benchmark files.

Example reruns:

```bash
python -B experiments/scripts/run_review_v2.py boxcar1d --run-id my-new-run
python -B experiments/scripts/run_review_v2.py boundary --run-id my-new-run
python -B experiments/scripts/run_review_v2.py wind --mode all --run-id my-new-run
python -B experiments/scripts/run_review_v2.py radar --run-id my-new-run --scratch-root /path/to/scratch
```

The radar command requires the exact released input identified in
`experiments/data/README.md`. Run benchmark campaigns sequentially, not alongside
other CPU-intensive work.

### Wind FFT padding comparison

[`astropy-fft-halo-20261118`](results/round1/v2.0.0/3d_wind_boxcar/astropy-fft-halo-20261118/)
contains the latest five-case wind comparison: six implementations at three grid
sizes and three boxcar widths, with a shared baseline and approximately 20%
missingness. The run records three fresh-process repetitions per method and case
(90 total) and the numerical comparisons, settings, and source hashes. It does
not replace the earlier eight-case run, which retains the additional missingness
timings.

Both native Astropy `convolve_fft` settings add a symmetric border and crop to
the original grid. One adds extra zero padding; the other uses circular
convolution on the reflected array, which gives the same retained values within
tolerance because the retained windows do not cross its outer edge. Both calls
recompute the kernel FFT for each field. Removing the extra padding reduces
median time and peak process memory in four cases but increases both on the
largest grid. SciPy's uniform filter has the lowest median runtime in all five
cases.

Use the versions and thread settings in the top-level README. The inputs are
synthetic and no radar file is needed:

```bash
python -B -m pytest -q -p no:cacheprovider experiments/tests/test_wind_fft_padding.py
python -B experiments/scripts/benchmark_3d_wind_fft_padding.py --mode all --run-id my-new-wind-run
```

## Measurement interpretation

The radar and wind benchmarks include filter preparation, padding, allocation,
all filtering passes, normalization, variance, and square roots. Import, input
reading, plotting, and numerical verification are outside the timer. Each
repetition uses a separate Python process. Radar has five repetitions per
method/width; the wind study has three per method/configuration.

RSS is maximum total process resident set size, not algorithm-only memory.
Measurements use matching inputs and filtering workloads but do not establish
a universal ranking or an asymptotic crossover. The specialized SciPy running
sum is faster in every recorded 3D full-statistics case. The radar loop
comparison is specific to the explicit Python implementation.

## Frozen historical results

Earlier studies remain byte-for-byte with their original source hashes:

- `round1/1d_boxcar/lead-review-1d-20261027/`: v1.0.0 75-case boxcar comparison.
- `round1/1d_boundary_comparison/boundary-truncated-20261030/`: original boundary
  comparison, profiles, and separate truth-error diagnostics.
- `round1/2d_radar_discrete_gaussian/radar-discrete-gaussian-20261110/`: sampled
  Gaussian control and the saved Gaussian radar figure arrays.
- `round1/2d_radar_gaussian_double_loop/radar-gaussian-double-loop-20261111/`:
  original v1.0.0 explicit-loop Gaussian comparison.
- `round1/2d_radar_gaussian/lead-review-radar-exact-only-20261027/`: analytical
  response, controlled-gap and floor-sensitivity evidence. The 30 low-floor
  tolerance exceedances remain recorded; they are not sampled-Gaussian tests.
- `round1/2d_radar_statistics_benchmark/radar-matched-statistics-20261102/` and
  `round1/2d_radar_range_loops/radar-range-loops-20261103/`: matched analytical
  response comparisons, not timings of the current sampled Gaussian.
- `round1/3d_wind_boxcar/lead-review-3d-matched-20261027/`: original v1.0.0 wind
  statistics and 120 measurements.
- `round1/3d_large_boxcar_mean/quick-large-mean-1gib-20261027/`: a separate 1-GiB
  scalar mean-only check, two in-place filters and two repetitions per case,
  with sampled output checks. It is not the seven-filter wind workload.

The top-level `boundary_1d_*` and `synthetic_metrics.*` files preserve submitted
v0.5.0 results. Their original generators overwrite those historical outputs,
so do not run them over the archive.

The old drivers require the exact v1.0.0 extraction; use `run_review_v2.py` for
the current calculations rather than changing the old version or source hash.

## Figure reproduction

The saved vector PDFs, 300-dpi PNGs, and `figure_settings.csv` files are under
`manuscript/figures/`. There is no article text in that directory. Frozen
exporters retain this directory name so their source hashes stay valid.

Each metadata file records source/input/output identities, display settings,
and export hashes. Current numerical calculations revalidate the retained
boundary, radar, and wind arrays without relabeling the original figure
provenance. The full-default Gaussian pipeline has a separate v2.0.0 export.

Use a new output directory rather than replacing the saved assets:

```bash
python -B experiments/scripts/run_review_v2.py visual --output-root /path/to/new-pipeline
python -B experiments/scripts/generate_1d_boundary_comparison.py --figure-only --figure-output /path/to/new-boundary-figure
python -B experiments/scripts/generate_2d_radar_discrete_gaussian_figure.py --help
python -B experiments/scripts/generate_3d_wind_boxcar_validation.py --figure-only --figure-output /path/to/new-wind-figure
```

Legacy figure exporters use the v1.0.0 runtime and may require private
author-workspace Git history to verify older source identities. In a standalone
copy, checks requiring that history are explicitly skipped, not treated as
passed. Saved assets remain usable and their hashes independently verifiable.
Radar figure reproduction also requires the exact released radar input.
