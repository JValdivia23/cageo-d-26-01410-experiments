# Experiments

## Retained submitted experiments

Copied from `../../DCT_paper2/02_arXiv_Paper/`:

- `scripts/generate_1d_boundary_data.py`
- `scripts/generate_boundary_figure.py`
- `scripts/generate_synthetic_validation.py`
- `results/boundary_1d_data.json`, `results/boundary_1d_metrics.csv`
- `results/synthetic_metrics.json`, `results/synthetic_metrics.csv`

These are the existing scripts and stored results, not new review computations.
Script changes are limited to the author-requested presentation corrections
(2026-10-27): `DOW6` to `COW`, `X-band` to `C-band`, removal of the unnecessary
workflow heading, and trimming the unused top margin. Top trimming is enabled
only for the radar export; other figures retain their existing export behavior.
The calculations and stored results are unchanged. The manuscript's vector
radar figure has the same presentation corrections, with a matching 300-dpi PNG;
its plotted data were not recomputed.
`synthetic_metrics.json` records toolkit v0.5.0, commit
`9d3b60ed408b5be819ae0eee0276c56840b712a7`, seeds, and environment metadata.
Original path handling and optional radar plotting dependencies are preserved;
the radar workflow still uses the external `cflib` helper.

See [data/README.md](data/README.md) for the QCed input. Stored results were copied,
not regenerated. Update code only for the selected review task, using toolkit
v1.0.0 (`9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c`) for new computations and
keeping these historical results distinct. The original generators overwrite
their `results/` outputs when run; do not rerun them over this copied baseline.

## How to run

From `experiments/`, with the `dct-publish` VS Code kernel selected
(`dct-publish` conda environment, Python 3.11.13):

```bash
python scripts/generate_1d_boundary_data.py
python scripts/generate_boundary_figure.py
python scripts/generate_synthetic_validation.py
```

Run the 1D data generator before the boundary figure, which reads
`results/boundary_1d_data.json`. The validation script writes fresh
`results/` metrics and `figures/` outputs, overwriting the stored files above.

- New review work uses `dct-publish` with `dct-toolkit` 1.0.0 (NumPy 2.4.4,
  SciPy 1.17.1, Matplotlib 3.10.8, xarray/netCDF4 available, as installed
  on 2026-10-26). No Jupyter notebook is required; these are plain scripts.
- The stored results instead record toolkit v0.5.0 (`9d3b60e`), Python 3.12.4,
  NumPy 1.26.4, Matplotlib 3.8.4, and seeds `20260414`–`20260418`. That
  environment is not reconstructed here; do not present another environment
  as reproducing it without verification.
- `dct-build` is currently an empty Python 3.11.13 environment (no NumPy) and
  is not a review-run environment.
- Radar input is optional: set `REAL_RADAR_PATH` to the QCed file documented
  in [data/README.md](data/README.md). Without it, the validation script falls
  back to a hardcoded private path and then to toolkit sample files. The radar
  figure additionally needs the private `cflib` helper on the plotting path
  (`cmweather` only selects the colormap).

## Round-1 comparison scripts

The new examples run independently and do not overwrite the submitted scripts,
results, or figures:

- `scripts/generate_1d_boxcar_validation.py`: agreement of local statistics with
  spatial boxcar calculations, including missing values and boundary cases;
  no timing study.
- `scripts/generate_2d_radar_gaussian_validation.py`: one COW sweep, Gaussian
  local statistics with an exactly matched spatial numerical reference, plus
  spectral-only mean costs. No ordinary sampled SciPy Gaussian comparison.
- `scripts/generate_3d_wind_boxcar_validation.py`: separate boxcar statistics
  for `u`, `v`, and `w`, with a common seven-filter calculation for prepared
  DCT, NumPy, SciPy, and Astropy time/process-RAM comparisons. The released
  statistics API is verified separately, not timed.
- `scripts/benchmark_2d_radar_statistics.py`: the separately authorized matched
  full-statistics radar benchmark. It preserves the earlier radar script and
  results, and does not regenerate Figure 3. See the protocol below.

New files go under `results/round1/<experiment>/<run_id>/`. Each run has
`comparison.csv` and `settings.csv`; timed runs also have `benchmark_runs.csv`
and `benchmark_summary.csv`. The latter is calculated from the individual
measurements. Missing measurements are blank, not zero. Times are seconds and
3D process RAM is MiB (1 MiB = 1,048,576 bytes). The exact calculations, settings,
source hashes, and limitations are saved with each run. The plan's
[measurement definitions](../comments1/plan.md#csv-files-and-what-they-measure)
explain which operations are included. No benchmark plots are generated.

Supporting CSVs include `coverage.csv` (1D), `support_diagnostics.csv` and
`support_by_range.csv` (radar), and `application_counts.csv` (3D). The ordinary
sampled SciPy Gaussian comparison was removed at the author's request because
its discrete weights differed. New radar runs do not write
`gaussian_approximation.csv` or spatial-reference timings. The full-weight
spatial calculation uses exactly matching weights and remains a numerical
reference only, not a practical spatial performance baseline. Old outputs are
preserved in their original run directories but are not selected for the paper.

### Publication figures

The author-approved figures use explicit export modes, separate from the normal
validation and timing paths. After preparing the exact toolkit source and
environment [below](#use-the-exact-toolkit-source), run these commands from the
CAGEO root:

```bash
"$PYTHON" -B experiments/scripts/generate_visual_pipeline_figure.py
"$PYTHON" -B experiments/scripts/generate_1d_boundary_comparison.py --figure-only
"$PYTHON" -B experiments/scripts/generate_2d_radar_gaussian_validation.py --figure-only
"$PYTHON" -B experiments/scripts/generate_3d_wind_boxcar_validation.py --figure-only
```

These create paired PDF and 300-dpi PNG assets in
`manuscript/figures/round1/{visual_pipeline,1d_boundary_comparison,2d_radar_gaussian,3d_wind_boxcar}/`,
each with `pdf/`, `png/`, and a `figure_settings.csv` sidecar. PDFs retain embedded
vector text and axes; dense map cells are rasterized. The shared figure helper
checks text bounds and records actual dimensions, plotting versions, display
choices, and output hashes. Regeneration replaces only these new-version assets;
all historical figures and saved numerical results remain preserved.

The visual pipeline restores the submitted teaser's seed-42 synthetic field,
mask, and colors, but recalculates the three filtered inputs using the verified
v1.0.0 Cartesian Gaussian filter, width 7, and reflective boundaries. The raw
mask response must be finite and strictly greater than `1e-3` to normalize the
moments; otherwise mean and standard deviation are undefined. Coverage is the
clipped original-mask response, without moment gating. The spectrum shows the
zero-filled input; all three enlarged maps show the same central crop. The std
color scale is capped at its 98th percentile and marked as extended. This is an
illustration, not a new validation or accuracy experiment. The previous
conceptual diagram remains in `round1/pipeline/` with its original generator.
The revised 1D display compares reflective and periodic
width-15 boxcars with a single unpadded NumPy baseline on the same trend-plus-sine
replicate-zero realization, first complete and then with the middle gap. NumPy
clips each window to the input domain and renormalizes over its finite values;
it neither wraps nor repeats exterior observations. The legend calls this
treatment **Truncated (NumPy)**; boundary outputs are retained, not discarded.
Its coverage is the valid
fraction of the available in-domain positions, so a complete shortened edge
window has coverage one. Nominal equivalent count is still `15*rho`, not the
actual observation count in that shorter window. True signal and complete-input
boundary zooms restore the original illustration's message; local standard
deviation and coverage show the missing-data extension. Radar reproduces the natural-mask W15
case on the full selected sweep before extracting central display cells, marking
accepted negative raw variances separately from undefined moments. It requires
the exact documented NetCDF file; `REAL_RADAR_PATH` may select its location, but
a different input hash or sample-data substitution is rejected. Wind reproduces
only `baseline20`, filters the complete 3D components, then draws observed/mean
3D views and a mean/std section on zero-based plane 9. Arrow decimation is a
display choice, not another filter. Each export
verifies its input identity, fixed settings, defined-output masks, and applicable
saved comparisons/diagnostics before writing assets. No benchmark or full-case
validation campaign is run by `--figure-only`.

The radar presentation restores **ChaseSpectral from cmweather 0.3.2** and the
original **0–40 dBZ** scale for both observations and normalized mean. The palette
is required; missing cmweather raises an error rather than substituting another
colormap. Colorbar extensions and the metadata's below/above-limit counts retain
out-of-range values without changing the numerical arrays. The bottom std and
coverage panels are square `[-10,10] km` viewports, marked by matching squares
in the full maps. Their 188-gate mesh includes all four corners and is selected
after full-sweep filtering, without resampling. This square is larger than the
unchanged `R<=10 km` diagnostic region in Table 6: `zoom_*` metadata records its
cell-center counts separately from the frozen radial `central_*` diagnostics.
The square contains 99,973 accepted estimates and no negative raw variances;
all 4,593 affected cells remain marked in the full std map. Only panel (c) has a
red **Clamped** key, beside its shortened **Std** title and clear of the data.
There is no global legend: the caption identifies gray missing observations in
(a), gray undefined estimates in (b,c,e), and the red clamping condition
$M_2<\mu^2$ in (c). The key identifies the unchanged clamp, not negative std or
evidence of absent variability. Section 3.1 explains this limitation; the sidecar
records the key location and the warning about zeros. Removing 0.4 inches of the
top legend band gives a 6.5-by-7.4-inch export (1950-by-2220 pixels at 300 dpi),
while retaining the 1.95-inch-square panels, physical spacing, and text sizes.
Full-map and zoom std panels share **0–10 dBZ** limits with `extend="max"`;
the 118 values above 10 dBZ remain unchanged (maximum 10.6708 dBZ), with
saturation counts and actual extrema recorded in the sidecar.
Historical figures and all saved numerical results are unchanged.

The four-panel wind figure restores the original 22°/35° 3D perspective and
viridis vector-magnitude colors, using a common `(z,y,x)` lattice with strides
`(3,8,8)` and offsets `(1,4,4)`. The 3D arrows have fixed length `0.11` in
normalized coordinates; color encodes the observed vector magnitude or the
magnitude of the componentwise mean, not mean wind speed. Gray points mark
sampled missing inputs; 100 deterministically sampled observed injections are
red, not classified as detections. Dashed outlines locate the section at
`z=0.0588235294`. Its scalar maps show only `mu_w` (−0.3 to 0.3) and `sigma_w`
(0 to 0.4), with horizontal mean arrows and a length key in the former. Coverage
and the other two std maps are omitted only from the display: all full-volume
component arrays remain byte-identical to the preceding figure preparation.

Section 6 retains TP/FP/FN as separate component diagnostics, not another
performance figure or a combined vector detector. The figure-only sidecar
records their counts and mask hashes under `diagnostic.*`, using the existing
eligibility gates and fixed `k=3`: `(286,92,120)` for `u`, `(270,87,136)` for `v`,
and `(290,218,116)` for `w`. All 406 observed injections are eligible in this
case; the other 115 injections are missing and are not false negatives. This is
a new aggregation of the reproduced current case, not a change to frozen
`application_counts.csv` or reuse of the historical wind-speed experiment.
The Discussion cites Garcia (2010) and Vitale (2021) for robust-smoothing and
instrument-despiking context. No iterative rejection/reweighting, threshold
calibration, independent detector evaluation, benchmark, or new field case was
run. The paired exports are 6.5 by 7 inches (1950 by 2100 pixels at 300 dpi),
with vector arrows/text and rasterized native section cells.

The 2026-10-30 boundary comparison adds a small, separate numerical run rather
than modifying the original 75-case reflective validation. Its new
`comparison.csv` checks each boundary model against its matched spatial
reference. `interior_agreement.csv` compares both spectral calculations with the
truncated NumPy baseline at indices `[7,193)`, where the full window is in-domain.
It includes all defined interior cells and separately records undefined counts;
no one-observation windows are removed to improve agreement. `truth_errors.csv`
instead records mean errors against the known clean signal, including the
evaluation domain, number compared, and undefined counts. `profiles.csv` retains
the inputs, truth, and all three displayed outputs; `settings.csv` records their
exact provenance. These are two illustrative inputs
from one noise realization, not a new benchmark or a multi-replicate accuracy
study. Unsupported gap interiors are excluded from error summaries, not filled
or silently scored as correct. Local standard deviation describes neighborhood
variation, not uncertainty in the mean or the injected noise alone.

The selected run is
[`boundary-truncated-20261030`](results/round1/1d_boundary_comparison/boundary-truncated-20261030/):
100 matched-boundary check rows, 20 interior-agreement rows, 30 truth-error rows,
and 1,200 profile rows. The three earlier boundary/truth runs are preserved but
are not selected for the current figure. Interior maximum mean, variance, and
coverage discrepancies are `3.51e-14`, `3.13e-13`, and `4.44e-16`. The gapped std
discrepancy reaches `5.59e-7` where the centered reference variance is zero;
square-root amplification is explicitly retained, not described as a
machine-epsilon std result. Whole-domain and boundary truth errors are separate
measurements, not numerical disagreement or multi-replicate accuracy claims.

The figure-only command above rechecks this selected run without rewriting its
CSVs. To save a separately authorized reproduction as a new run, use a previously
unused name, for example:

```bash
"$PYTHON" -B experiments/scripts/generate_1d_boundary_comparison.py --run-id boundary-comparison-new
```

Normal execution calculates only the selected two inputs and refuses to overwrite
an existing run. `--saved-run <path>` selects another compatible run for
`--figure-only`; `--figure-output <path>` changes the asset destination, subject
to the script's historical-asset/result protection checks.

The original Gaussian figure/results and the first reviewed boxcar missingness
figure remain preserved. The latter can still be regenerated with
`generate_1d_boxcar_validation.py --figure-only` under
`manuscript/figures/round1/1d_boxcar/`, but it is no longer the selected Figure 2.
Neither its existing export mode nor the original numerical/default-run code
was changed for the boundary comparison.

The three experiment scripts retain their original numerical definitions and
ordinary CLI behavior. Plotting dependencies are imported only in the explicit
figure branch. The saved CSVs retain their original script hashes; the figure
sidecars additionally record the current extended script and source-CSV hashes.
Saved-output tests accept the historical source only when its recorded SHA
matches immutable pre-figure commit
`c66106488a4e17bfb45260f4764106c5682254de` and the executable numerical/default-run
AST remains unchanged apart from narrowly checked figure-only additions. The
tests reject modified formulas, thresholds, ordinary CLI behavior, imports, and
import-time side effects rather than silently accepting a stale source hash.
This historical verification requires the manuscript's Git history.

Figure generation used Matplotlib 3.10.8 and Pillow 12.2.0 in `dct-publish`,
with cmweather 0.3.2 required for the radar reflectivity palette;
toolkit, NumPy, SciPy, and radar-reading versions are recorded with each export.
The earlier 64/79-test counts below describe the experiment-only stages. The
first four-figure checkpoint passed 225 tests. With
`REVIEW_RESULT_DIRS` set to the three selected runs, `LARGE_MEAN_RESULT_DIR` set
to the selected large-mean run, the exact-source environment, and the real radar
file available, `RADAR_BENCHMARK_RESULT_DIR` set to the earlier FFT/direct run,
and `RADAR_LOOP_RESULT_DIR` set to the new loop run below, the expanded
numerical/provenance/figure suite passes **390 tests**, including 83
boundary-comparison tests, 53 wind-figure/diagnostic tests, 47 earlier radar
benchmark checks, 25 range-loop/abstract checks, and four added radar-presentation
checks (square domain/corners, exact shared palette/limits, no fallback, and
panel-local key placement without overlap or panel shrinkage).
Its figure tests use temporary
output directories and do not alter the frozen CSVs or generate new timing
measurements. The known nonfatal NumPy/netCDF compatibility warning remains.

### Matched radar-statistics benchmark

The 2026-11-02 comparison below is retained unchanged as earlier FFT/direct
evidence. The current main comparison adds a NumPy range-gate loop in a separate
script and run; see **Range-adaptive loop comparison** below. Do not combine a
new NumPy time with an older prepared-spectral median.

The separate `benchmark_2d_radar_statistics.py` uses the same documented
`720x1180` sweep, natural mask, middle-quarter controlled removal, and nominal
Gaussian widths 5/15. Each method constructs one filter and applies it three
times: to the finite mask, zero-filled data, and squared zero-filled data.
Common code then calculates mean, second raw moment, raw/clamped variance,
std, coverage, and equivalent count with the same strict raw `d>1e-3` rule.
Completion and restoration remain disabled.

The prepared DCT/RFFT method follows the pinned release's operator and analytic
responses, with vectorized azimuth-response construction and shared preparation;
it does not time separate released statistics-API calls. The SciPy FFT baseline
uses `scipy.fft.rfft/irfft`, circular azimuth filtering, an even range extension
of length `2R`, range filtering, and cropping. The SciPy direct baseline uses
`scipy.ndimage.convolve1d` with
wrap/reflect boundaries and every finite-grid weight, including signed tails;
inverse FFTs construct these weights inside its setup cost. It is explicitly a
**full-support direct** implementation, not ordinary sampled/truncated Gaussian
filtering. The slower independent cosine-sum construction remains an untimed
numerical oracle, never a headline timing baseline.

Before timing, all outputs and raw/accepted masks must agree on the full sweep
and centre/seam/outer regions at the original fixed tolerances, and the new
weights must agree with the original cosine sums. Released API outputs are
also checked but not timed. Each of five repetitions then uses a fresh worker,
with matching imports, requested one-thread limits, no filter warm-up, rotated
method order, and no simultaneous measurements. Masking, zero replacement,
preparation, three full filters, normalization, and allocations are timed;
reading, imports, validation, and hashing are not. Each worker's full output
fingerprints must match its previously verified implementation before its
timing enters successful summaries. Total RSS uses the same 1-ms sampling and
new-lifetime-high-water-mark convention as the 3D comparison.

After preparing the exact-source environment below, run from the CAGEO root:

```bash
"$PYTHON" -B experiments/scripts/benchmark_2d_radar_statistics.py --run-id radar-statistics-new
```

Use a new run ID: existing directories cannot be overwritten. `--validate-only`
performs the numerical checks without timing. Results go to
`results/round1/2d_radar_statistics_benchmark/<run_id>/` as `comparison.csv`,
`benchmark_runs.csv`, `benchmark_summary.csv`, and `settings.csv`. All weights,
inputs, settings and original result files remain distinct from the earlier
mean-only measurements. Set `RADAR_BENCHMARK_RESULT_DIR` to the selected run
when testing its saved evidence; `REVIEW_RESULT_DIRS` still selects only the
original three comparison runs. Unit tests use tiny controls, not new radar
timings. No broad size-scaling, new dataset, or benchmark figure is included.

The selected run is
[`radar-matched-statistics-20261102`](results/round1/2d_radar_statistics_benchmark/radar-matched-statistics-20261102/):
580 full-array/weight comparisons pass at unchanged tolerances, including the
untimed released API, followed by 60 successful worker measurements. For the
natural mask, median DCT/RFFT costs are `0.0554806` and `0.0569323` s for widths
5/15; corresponding SciPy FFT costs are `0.0828262` and `0.0893512` s and SciPy
full-support direct costs are `2.5542280` and `2.6014751` s. The prepared
calculation is **45.7–46.0 times faster than equivalent SciPy direct convolution**
for this mask and 45.7–49.2 times faster across all four cases. Against the SciPy
FFT implementation, the speedups are **1.49–1.57** and 1.49–1.75, respectively.
Speedup means the baseline's median time divided by the prepared implementation's
median, using unrounded measurements. These factors apply only to the same full
finite-grid weights, not ordinary sampled/truncated Gaussian smoothing or all
NumPy/SciPy functions. These were the earlier Table 11 results; current
manuscript Table 11 uses the newly measured NumPy/SciPy loop comparison below.
Individual measurements, quartiles, worker input/output hashes, filter counts,
source hashes, and environment settings remain in this preserved run's four CSVs.
At that checkpoint, the abstract reported the natural-mask speedups against
both SciPy alternatives, not an untimed-oracle speedup. The saved method IDs
remain `symmetric_fft` and
`full_support_direct`; their numerical code, recorded source hashes, and all
four result CSVs are unchanged. That reporting clarification did not include a
NumPy baseline; the separately authorized new loop comparison is documented
below. These earlier speedups describe the prepared calculation, not the
released statistics API.
The original radar generator, four scientific figures, previous 129 result
files, ten original numerical tables, and displayed equations are unchanged.

To include this saved run in the checks without rerunning timings:

```bash
export RADAR_BENCHMARK_RESULT_DIR=experiments/results/round1/2d_radar_statistics_benchmark/radar-matched-statistics-20261102
"$PYTHON" -B -m pytest -q -p no:cacheprovider experiments/tests
```

### Range-adaptive loop comparison

`scripts/benchmark_2d_radar_range_loops.py` compares **our prepared spectral
calculation versus these range-adaptive loop implementations** on the same
`720x1180` radar sweep, widths 5/15, and natural/controlled-removal masks.
`benchmark_2d_radar_statistics.py` and all its earlier CSVs remain unchanged;
the new script imports its verified spectral calculation and common statistics
construction rather than changing the measured earlier implementation.

Both direct methods use a different azimuth kernel at every range gate, with
the same legacy mapping `W_az(j)=W/((j+1)*az_res_radians)` as the spectral method.
This does not use actual physical gate ranges or introduce a fixed-width
azimuth approximation. Both prepare the same full finite-grid weights once
using the independent analytic responses and efficient inverse FFTs, inside
the timed interval. The slower independent cosine-sum weights remain an
untimed numerical oracle. Signed tails and all offsets are retained; these
are not ordinary sampled/truncated spatial Gaussian kernels.

- `numpy_range_loop` loops over range gates with `np.pad(..., mode="wrap")`
  and `np.convolve(..., mode="valid")`. Its subsequent range filtering loops
  over rays with `np.pad(..., mode="symmetric")` and `np.convolve`. NumPy's
  `symmetric`, not its `reflect`, matches SciPy's half-sample `reflect` boundary.
- `scipy_range_loop` uses the same range-gate loop with
  `scipy.ndimage.convolve1d(..., mode="wrap")`, followed by its compiled
  axis-1 `convolve1d(..., mode="reflect")`. It is not forced into an unnecessary
  Python ray loop where SciPy already supports an axis pass.
- `prepared_dct_rfft` uses the unchanged experiment-local prepared implementation
  from the earlier benchmark, not separate released statistics-API calls.

Every method applies three complete filters to the finite mask, zero-filled
observations, and their squares, then uses identical raw-denominator acceptance,
moments, variance clamp, coverage, and equivalent count. Normalization follows
both axes. Full-array/region comparisons, weights, accepted masks, and the
released API must pass the unchanged tolerances before timing. Workers verify
the four participating source hashes, input identity, all nine output hashes,
one preparation, and three complete filters. Each direct worker also counts
all 3,540 azimuth-gate convolutions; NumPy counts 2,160 range-row convolutions
and SciPy counts three compiled range passes.

Five repetitions per method/case use fresh, sequential interpreters, matching
imports and one-thread settings, rotated method order, and no filter warm-up.
Preparation, padding, filtering, masking, normalization, moments, and allocation
are included. Reading, imports, source checks, numerical verification, and
fingerprints are excluded. RSS uses 1-ms in-call sampling and any newly exceeded
lifetime high-water mark, including inputs/libraries rather than just allocations.

After preparing the exact-source environment below, run from the CAGEO root:

```bash
"$PYTHON" -B experiments/scripts/benchmark_2d_radar_range_loops.py --run-id radar-loops-new
```

Existing run directories cannot be overwritten. `--validate-only` performs
correctness checks without timings. Results use the four existing CSV schemas
under `results/round1/2d_radar_range_loops/<run_id>/`. Set `RADAR_LOOP_RESULT_DIR`
to the selected loop run for its saved-evidence and current-manuscript checks;
keep `RADAR_BENCHMARK_RESULT_DIR` set to the earlier run to test its preserved
source/results. Unit tests use tiny controls and never launch a radar campaign.

The selected new run is
[`radar-range-loops-20261103`](results/round1/2d_radar_range_loops/radar-range-loops-20261103/).
All **580 weight/full-array comparisons** pass at the unchanged tolerances,
including the untimed released API, followed by **60 successful fresh-worker
measurements** (five per method/case). The maximum mean/std discrepancies are
`1.38e-9`/`2.91e-8` dBZ and all accepted masks agree exactly. Across the four
cases, prepared medians are `0.0530–0.0544` s, NumPy-loop medians
`0.8495–0.8580` s, and SciPy-loop medians `2.6412–2.6573` s. Prepared spectral
calculation is **15.7–16.2 times faster than the NumPy loop** and **48.6–49.9
times faster than the SciPy loop**. The natural-mask subset gives 15.7–15.8 and
48.6–49.4, respectively; the revised Abstract summarizes these as approximately
16 and 49 times faster for prepared computation of all three statistics.
Table 11 gives each case's speedup from unrounded medians, time, and RSS;
Section 5.3 retains the smaller earlier SciPy FFT
speedups separately, without mixing the two timing campaigns. These findings
concern the tested range-adaptive loops with full finite-grid weights, not all
NumPy/SciPy algorithms or ordinary sampled/truncated Gaussian smoothing.

To test both frozen radar runs and the current manuscript without rerunning
timings, add these to the existing selected-run environment:

```bash
export RADAR_BENCHMARK_RESULT_DIR=experiments/results/round1/2d_radar_statistics_benchmark/radar-matched-statistics-20261102
export RADAR_LOOP_RESULT_DIR=experiments/results/round1/2d_radar_range_loops/radar-range-loops-20261103
"$PYTHON" -B -m pytest -q -p no:cacheprovider experiments/tests
```

### Matched 3D calculation

Every benchmark method filters the common finite mask once, then filters each
component and its square: **seven full-volume filters in total**. The same
code constructs the mean, population variance, std, and support and applies the
same denominator threshold for every method. Kernels/transfer arrays are
prepared once per calculation, inside the timed interval, then reused. The
worker records and checks the actual filter count and output count.

`dct_shared_moments` uses the verified release's `get_dct_transfer_function`
and its N-D DCT/multiply/inverse-DCT sequence, with transfer preparation moved
before the seven calls. This is an experiment-local prepared implementation,
**not performance of the released statistics API**. Numerical comparisons
also verify the released API as `dct_toolkit_api_validation_only`; that
19-filter workflow is never timed. Spatial references do not use toolkit
filters or spectral weights. Astropy reuses its dense spatial kernel, but
`convolve_fft`'s internal kernel FFT/padding costs remain inside each call;
the benchmark does not claim those internal transforms are cached.

All fresh timing workers preload the same toolkit/SciPy/Astropy libraries,
outside the timer and without filter warm-ups. Before/peak process RAM remains
reported separately and includes the common library footprint; neither the
peak nor peak minus baseline is an exact count of filter allocations. Window
widths are in grid cells: methods match within each case, but changing volume
resolution does not preserve the physical footprint of a fixed-width window.

The Abstract's Cartesian maxima use the saved full-statistics run
`lead-review-3d-matched-20261027`, not the separate large-array mean-only check.
The widest-window case (`broad20`) gives speedups of 50.426 over NumPy window
reductions, 43.605 over Astropy direct convolution, and 5.818 over Astropy FFT
convolution, rounded in the Abstract to 50, 44, and six. Saved-output checks
verify these ratios from unrounded medians after checking all raw repetitions.
The specialized SciPy boxcar comparison remains in Section 5 rather than the
Abstract; no overall performance equivalence or universal speed advantage is
claimed. No new timing campaign was run for this wording change.

### Use the exact toolkit source

An editable installation can report version 1.0.0 while importing a newer
checkout. `_review_common.py` checks the SHA-256 hashes of all nine toolkit
Python files against commit `9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c` before
allowing a calculation. It rejects different code rather than recording a
misleading version string.

For the current machine, prepare an isolated copy and dependencies as follows.
This does not modify the external toolkit checkout or the conda environment.
Run these commands from the CAGEO root:

```bash
PYTHON=/Users/java1127/anaconda3/envs/dct-publish/bin/python
TOOLKIT_REPO=/Users/java1127/Library/CloudStorage/OneDrive-UCB-O365/Projects/DCT_interp_func/dct_toolkit
REVIEW_RUNTIME=$(mktemp -d /private/var/folders/nt/th92gg516gqbtqd8y3fqclv00000gp/T/opencode/cageo-review.XXXXXX)
mkdir -p "$REVIEW_RUNTIME/toolkit" "$REVIEW_RUNTIME/deps"
git -C "$TOOLKIT_REPO" archive 9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c | tar -xf - -C "$REVIEW_RUNTIME/toolkit"
"$PYTHON" -m pip install --no-deps --target "$REVIEW_RUNTIME/deps" astropy==8.0.2 astropy-iers-data==0.2026.10.19.0.52.1 pyerfa==2.0.1.5 PyYAML==6.0.3 packaging==26.3
export DCT_TOOLKIT_SOURCE="$REVIEW_RUNTIME/toolkit"
export PYTHONPATH="$REVIEW_RUNTIME/toolkit:$REVIEW_RUNTIME/deps"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Use each script's `--help` for its supported cases and run controls. Final
benchmarks must run one at a time, after numerical checks pass; do not measure
them while other agents or experiment processes compete for the CPU.

With that environment set, the agreed cases can be rerun from the CAGEO root
as follows. Omit `--run-id` to create a new timestamped folder, or supply a new
identifier; existing runs are never overwritten. Run these commands in order,
not concurrently:

```bash
"$PYTHON" -B experiments/scripts/generate_1d_boxcar_validation.py --mode full
"$PYTHON" -B experiments/scripts/generate_2d_radar_gaussian_validation.py --controlled-gap --repeats 3 --warmups 1
"$PYTHON" -B experiments/scripts/generate_3d_wind_boxcar_validation.py --mode all --cases small20,baseline_complete,baseline20,baseline50,larger20,narrow20,broad20,plane_control --repeats 3
```

The numerical comparisons and benchmarks disable gap filling for every
method. Estimates with an inadequate denominator stay undefined; the CSVs
separately count where the released default would be eligible to fill gaps.
Those counts are not measurements of the default gap-filling algorithm.
The 3D application uses a fixed illustrative component threshold, not an
optimized outlier detector. Near-zero std differences must be read alongside
the much smaller variance differences because taking a square root magnifies
roundoff near zero.

For tests of both the code and selected saved outputs, set `REVIEW_RESULT_DIRS`
to the selected run directories separated by `:` on macOS/Linux, then run:

```bash
"$PYTHON" -B -m pytest -q -p no:cacheprovider experiments/tests
```

Without `REVIEW_RESULT_DIRS`, saved-output checks are skipped. Earlier agent
smoke runs and deliberate failure tests are retained but are not automatically
selected as paper measurements.

### Selected corrected runs

- [1D boxcar](results/round1/1d_boxcar/lead-review-1d-20261027/): 75 cases and
  2,700 numerical comparison rows; no failed checks.
- [Radar exact-reference-only run](results/round1/2d_radar_gaussian/lead-review-radar-exact-only-20261027/):
  246 primary comparison rows pass or correctly have no comparable values;
  six successful spectral-only mean timings. No sampled Gaussian comparison
  file or spatial performance baseline is generated.
- [3D matched seven-filter run](results/round1/3d_wind_boxcar/lead-review-3d-matched-20261027/):
  1,920 passing numerical rows and 390 correctly non-comparable rows, including
  the untimed API validation; 120 successful isolated time/RAM measurements.

All **64 review tests passed** on these selected runs. In addition to numerical
agreement, they check seven actual filter calls, one preparation per method,
DCT agreement with the released filter/statistics API, removal of the radar
sampled-Gaussian comparison, source hashes, worker inputs/import lists, and
summaries recomputed from the individual measurements. Workers with mismatched
inputs, outputs, or filter/preparation counts are rejected. The new radar
numerical and diagnostic CSVs are identical to the earlier run apart from run
IDs; removing the approximation did not alter the exact checks or tolerances.
No existing figure or manuscript text was changed.

Radar also saves `threshold_sensitivity.csv`: its 648 nondefault-floor checks
retain the same 30 fixed-tolerance exceedances at `1e-5`. The correction did not
change those calculations or tolerances. Execution requires
default-floor checks and mask-response/acceptance checks to pass before timing.
Use `--strict-sensitivity` to make any nondefault stress failure produce a
nonzero exit as well. This is a direct-ratio precision/coverage test, not a test
of default gap completion. The earlier failed full run is preserved separately.

Measurements used the specified x86_64 conda Python under Rosetta 2 on an Apple
M4 Pro with 24 GiB RAM, not native ARM Python. Thread limits were requested as
one and final benchmarks ran sequentially. Total process RAM includes library
footprints. Do not infer a universal speed/RAM advantage or label shared-moment
results as released statistics-API timings. Read the saved settings before
generalizing to other implementations, machines, or default gap-filling workloads.

### Quick large-array mean-only check

`scripts/benchmark_large_3d_boxcar_mean.py` is a separately authorized exploratory
check, not a rerun of the full three-component statistics benchmark. It uses a
single `512³` float64 input (1 GiB), seeded 20% NaNs, and boxcars of side 3/31.
SciPy uniform and prepared DCT each filter the mask and data, then normalize.
Both methods reuse/overwrite the same input/mask buffers; kernel/transfer
preparation is included. Two repetitions run in fresh processes with identical
libraries and requested single-thread settings. No large arrays or plots are
saved. With the runtime above, run from the CAGEO root:

```bash
"$PYTHON" -B experiments/scripts/benchmark_large_3d_boxcar_mean.py --input-gib 1 --repeats 2
```

Outputs use the normal CSV formats under `results/round1/3d_large_boxcar_mean/`.
Small-array comparisons cover every output cell; large-array comparisons use
2,173 fixed edge/random samples, explicitly not a full-array correctness check.
Native macOS `vm_stat` page sizes are used for resource guards because Rosetta
can make `psutil.virtual_memory()` page-derived quantities inaccurate. Swap
activity during timing invalidates the measurement and stops the campaign.
A 2-GiB input is optional (`--input-gib 2`) and is refused without sufficient
headroom. Results must not be mixed with full-statistics timings.

The selected [1-GiB quick run](results/round1/3d_large_boxcar_mean/quick-large-mean-1gib-20261027/)
completed all eight measurements with identical input hashes and no native
swap-counter increases during timing. Median seconds:

| Boxcar | SciPy uniform, in place | Prepared DCT, in place |
|---|---:|---:|
| `3³` | 13.980 | 15.583 |
| `31³` | 13.853 | 15.510 |

Maximum total process RSS was approximately 2.34 GiB. All eight sampled
comparison rows pass; maximum sampled mean discrepancy was `2.50e-15`.
The host remained under pre-existing memory pressure, and two repetitions are
exploratory. There is no measured 1-to-2-GiB scaling ratio: post-run conservative
headroom was 4.82 GiB versus the 6.10-GiB safety requirement, so no 2-GiB input
was allocated. Both methods show little kernel-width sensitivity in this case;
this does not establish a universal speed ranking or a dataset-size crossover.

Set `LARGE_MEAN_RESULT_DIR` to this run's absolute directory alongside the three
`REVIEW_RESULT_DIRS` paths above to check all saved results. The full suite then
passes **79 tests**: the earlier 64 plus 15 large-array source/resource/output
checks. The source and saved outputs of the three main experiments are unchanged.

### Superseded initial comparison runs

The [initial radar run](results/round1/2d_radar_gaussian/lead-review-radar-final-20261027/)
and [initial 3D benchmark](results/round1/3d_wind_boxcar/lead-review-3d-benchmark-20261027/)
are preserved unchanged. The radar run includes a different-kernel sampled
Gaussian comparison; the 3D timings compare 19-filter API calls with seven-filter
references and have different imported-library footprints. Neither is selected
as equivalent-method performance evidence for the paper.

The check of the frozen release ran its 786 bundled tests successfully. This
checks the released software; it is not evidence of accuracy or speed on the
new experimental inputs.
