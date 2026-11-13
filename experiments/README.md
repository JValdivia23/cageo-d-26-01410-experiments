# Experiments

The selected v2.0.0 runs and figure assets listed below are included in this
archive. Dated verification checkpoints retain their original selections, test
counts, and publication status; later publication does not change the recorded
scientific files or measurement provenance.

## Current release: v2.0.0

New review computations use released **v2.0.0**, commit
`7ff29e998dd1102d3fef2a84050402e86dc7ef67`. The release changes the default
`gaussian` to the full normalized sampled Gaussian, evaluated to float64
precision without a fixed spatial cutoff. The explicit `gaussian_spectral`
option reproduces the former analytical response. PyPI and GitHub availability
are verified; the conda-forge update is submitted, with build/review pending.

The radar comparison intentionally keeps **four sigma in both implementations**
to compare equivalent discrete filters. Standard spatial convolution means
Python azimuth/range double loops in each ordered axis stage, with a NumPy dot
product per neighbourhood. The same samples, repeated boundary contributions,
strict finite raw `d>1e-3`, and three complete filters remain unchanged. These
are experiment-local prepared implementations, **not** timings of the release's
full-Gaussian statistics API. The full Gaussian needs no larger transforms or
additional filtering passes on a fixed grid, but its response-preparation costs
can differ; its end-to-end runtime is not measured. Auxiliary composed SciPy
code and every old CSV row remain preserved, not selected in the radar tables.
Unrelated Cartesian SciPy comparisons remain selected.

### Version-specific execution

`scripts/run_review_v2.py` reuses the frozen numerical functions through explicit
bindings for release loading, environment/provenance recording, output-directory
creation, and untimed fresh-worker launching. It does not edit the v1.0.0
drivers or `_review_common.py`, or change cases, numerical tolerances, timers,
preparation/filter counts, or missing-data policies. The wrapper rejects changed
frozen source hashes. Each new worker verifies the ten released Python files
and records both adapter and inherited-source hashes. The radar entry point
does not execute the inherited analytical-Gaussian branches; do not run those
old branches against the new default. Reproducing pre-2.0 Gaussian results
requires the original runtime or explicit `gaussian_spectral`, not a version
string substitution.

New runs use `results/round1/v2.0.0/<experiment>/release-v2-20261111/`, separate
from all older measurements:

| Experiment | Experiment directory | Scope |
|---|---|---|
| 1D boxcar | `1d_boxcar` | Same 75 cases and original tolerances; no timing |
| Boundary comparison | `1d_boundary_comparison` | Same two inputs; three treatments; Figure 2 profiles rechecked without export |
| Radar Gaussian | `2d_radar_gaussian_double_loop` | W5/W15; four-sigma spectral/loop pair selected; SciPy retained as auxiliary evidence; five repetitions |
| 3D wind | `3d_wind_boxcar` | Same eight timing cases and mandatory numerical controls; five methods; three repetitions |

All four campaigns completed. The 1D run has 2,115 passing and 585
no-comparable rows; the boundary pair has 90 passing and ten no-comparable
rows, plus the original interior/truth/profile checks. The 3D run has 1,920
passing and 390 no-comparable rows and 120 successful measurements. Radar has
108 passing centered-reference comparisons, 36 passing spectral/loop comparisons,
2,362 passing kernel checks, and 30 successful measurements. All numerical CSVs
listed above match their predecessors exactly except for run IDs. Figure 2's
profiles, Figure 3's nine spectral arrays, and Figure 4's full-volume and section
array hashes reproduce; their existing exports remain unchanged.

Fresh radar spectral/loop medians are **0.0798/3.8404 s at W5** and
**0.0841/3.7916 s at W15**, with unrounded ratios
`48.138432685166315` and `45.078972687668326`. Their maximum process RSS is
248.9/252.4 and 235.2/250.1 MiB, respectively. Spectral preparation medians are
22.70/23.22 ms; loop preparation is 6.34/6.46 ms, already included in totals.
Auxiliary SciPy medians are 0.0666/0.0854 s, retained in CSVs rather than the
selected radar tables. The new 3D maximum ratios are 46.6543 over NumPy,
40.1756 over Astropy direct, and 5.4226 over Astropy FFT, all in `broad20`.
Specialized SciPy is faster than DCT in all eight cases. Narrow-window Astropy
direct and DCT now both round to 0.0315 s, with overlapping interquartile ranges;
the manuscript no longer carries forward the older faster-Astropy statement.
These are fresh host measurements, not evidence that v2.0.0 caused a speedup;
the old measurements are not pooled or relabeled.

Use an isolated extraction; do not change the external toolkit checkout:

```bash
PYTHON=/path/to/python3.11  # NumPy 2.4.4, SciPy 1.17.1, Astropy 8.0.2
TOOLKIT_REPO=/path/to/dct_toolkit
REVIEW_V2=$(mktemp -d)
mkdir -p "$REVIEW_V2/toolkit"
git -C "$TOOLKIT_REPO" archive 7ff29e998dd1102d3fef2a84050402e86dc7ef67 | tar -xf - -C "$REVIEW_V2/toolkit"
export DCT_TOOLKIT_V2_SOURCE="$REVIEW_V2/toolkit"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Run from the repository root; use a new run ID. Benchmarks must be sequential.
"$PYTHON" -B experiments/scripts/run_review_v2.py boxcar1d --run-id my-v2-run
"$PYTHON" -B experiments/scripts/run_review_v2.py boundary --run-id my-v2-run
"$PYTHON" -B experiments/scripts/run_review_v2.py wind --mode all --run-id my-v2-run
"$PYTHON" -B experiments/scripts/run_review_v2.py radar --run-id my-v2-run --scratch-root "$REVIEW_V2" # exact private radar input required
"$PYTHON" -B -m pytest -q -p no:cacheprovider experiments/tests/test_review_v2.py
```

Figure 1 is refreshed separately with the full default, using the same field,
mask, layout, palette, and crop. Its raw filters are checked against independent
precision-level spatial sums and its moments against the released API:

```bash
"$PYTHON" -B experiments/scripts/run_review_v2.py visual --output-root /path/to/new-figure-directory
```

The selected export is `manuscript/figures/round1/v2.0.0/visual_pipeline/`, with
vector PDF, 300-dpi PNG, and source/array hashes. Existing figure assets are never
overwritten. Figures 2–4 retain their original export provenance; calculations
are revalidated, rather than relabeling the original sidecars as v2.0.0 exports.
The 1-GiB mean-only check and analytical-response radar studies retain explicit
v1.0.0 provenance and are not remeasured or included in the new medians. No raw
radar input is copied into either repository. The local standalone checkout is
prepared separately; neither repository is committed or pushed by this work.

Verification completed on 12 November 2026. The exact tagged toolkit passes
all **988 package tests** independently. The source full-suite checkpoint has
**499 passes and two pre-existing obsolete manuscript-preservation failures**;
the two final table/asset checks added afterward also pass. The final standalone
full suite has **497 passes, the same two failures, four expected history/path
skips**, and the known radar-dependency warning. All **21 v2-specific checks**
pass there. The manuscript builds to 33 pages with resolved citations/references
and only the existing title-box warning; changed pages were inspected. All 203
pre-refresh scientific files, original comments, and Personal notes are preserved.
The two obsolete tests were not rewritten or waived, so the full suite is not
claimed green. No new independent-agent review was performed for this refresh.

## Historical v1.0.0 protocol and checkpoints

The sections below preserve the earlier commands, selections, findings, and
verification records. They require their original v1.0.0 runtime and are not
instructions to relabel those results or to run old Gaussian calls with v2.0.0.
The current release selection above supersedes their present-tense version and
timing selections, not their scientific evidence.

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
- `scripts/benchmark_2d_radar_discrete_gaussian.py`: a separate sampled-Gaussian
  equivalence control, with an independent centered-residual variance reference
  and a three-method benchmark. It does not replace the released analytical
  Gaussian or any earlier saved result. See the protocol at the end of this file.
- `scripts/benchmark_2d_radar_gaussian_double_loop.py`: the author-requested
  explicit polar spatial baseline, with nested azimuth/range cell loops for
  each ordered axis stage. The frozen campaign measures spectral filtering,
  these loops, and composed SciPy convolutions together. The main manuscript
  selects only spectral filtering and standard spatial convolution (the loops);
  the equivalent SciPy implementation remains auxiliary evidence. No timed
  source, earlier result, or Figure 3 is changed by that reporting selection.

New files go under `results/round1/<experiment>/<run_id>/`. Each run has
`comparison.csv` and `settings.csv`; timed runs also have `benchmark_runs.csv`
and `benchmark_summary.csv`. The latter is calculated from the individual
measurements. Missing measurements are blank, not zero. Times are seconds and
3D process RAM is MiB (1 MiB = 1,048,576 bytes). The exact calculations, settings,
source hashes, and limitations are saved with each run. The plan's
[measurement definitions](../comments1/plan.md#csv-files-and-what-they-measure)
explain which operations are included. No benchmark plots are generated.

Supporting CSVs include `coverage.csv` (1D), `support_diagnostics.csv` and
`support_by_range.csv` (archived analytical-response radar), and
`application_counts.csv` (3D). The earlier analytical-response campaign removed
an unmatched sampled SciPy Gaussian comparison because its weights differed.
The Gaussian-convolution experiments instead match the sampled kernels
exactly, as described [below](#sampled-gaussian-equivalence-control). Their
centered-residual reference remains untimed; practical spectral and spatial
filtering are measured together. All earlier outputs are preserved in their
original run directories without relabeling.

### Publication figures

The author-approved figures use explicit export modes, separate from the normal
validation and timing paths. After preparing the exact toolkit source and
environment [below](#use-the-exact-toolkit-source), run these commands from the
CAGEO root:

```bash
"$PYTHON" -B experiments/scripts/generate_visual_pipeline_figure.py
"$PYTHON" -B experiments/scripts/generate_1d_boundary_comparison.py --figure-only
"$PYTHON" -B experiments/scripts/generate_2d_radar_discrete_gaussian_figure.py
"$PYTHON" -B experiments/scripts/generate_3d_wind_boxcar_validation.py --figure-only
```

These create paired PDF and 300-dpi PNG assets in
`manuscript/figures/round1/{visual_pipeline,1d_boundary_comparison,2d_radar_discrete_gaussian,3d_wind_boxcar}/`,
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
deviation and coverage show the missing-data extension. Radar reproduces the
selected Gaussian-convolution natural-mask W15 case on the full sweep before
extracting central display cells, without the old analytical-filter red overlay.
It requires
the exact documented NetCDF file; `REAL_RADAR_PATH` may select its location, but
a different input hash or sample-data substitution is rejected. Wind reproduces
only `baseline20`, filters the complete 3D components, then draws observed/mean
3D views and a mean/std section on zero-based plane 9. Arrow decimation is a
display choice, not another filter. Each export
verifies its input identity, fixed settings, defined-output masks, and applicable
saved comparisons/diagnostics before writing assets. No benchmark or full-case
validation campaign is run by these figure exports.

The radar presentation restores **ChaseSpectral from cmweather 0.3.2** and the
original **0–40 dBZ** scale for both observations and normalized mean. The palette
is required; missing cmweather raises an error rather than substituting another
colormap. Colorbar extensions and the metadata's below/above-limit counts retain
out-of-range values without changing the numerical arrays. The bottom std and
coverage panels are square `[-10,10] km` viewports, marked by matching squares
in the full maps. Their 188-gate mesh includes all four corners and is selected
after full-sweep filtering, without resampling. This square is larger than the
`R<=10 km` diagnostic region in Table 6: `zoom_*` metadata records its cell-center
counts separately from the radial `central_*` diagnostics. With the new Gaussian
calculation, the square has 99,961 accepted estimates and the radial region has
90,140. There is no red overlay or **Clamped** key: the caption identifies gray
missing observations in (a) and undefined estimates in (b,c,e). Raw-variance
diagnostics, including 66 tiny negative values over the full sweep and none in
either central region, remain in the metadata rather than being emphasized on
the maps. The 6.5-by-7.4-inch canvas (1950-by-2220 pixels at 300 dpi) retains the
1.95-inch-square panels, physical spacing, and text sizes.
Full-map and zoom std panels share **0–10 dBZ** limits with `extend="max"`;
the new calculation has 118 values above 10 dBZ (maximum 10.6708 dBZ), with
saturation counts and actual extrema recorded in the sidecar.
The generator verifies all ten selected result-file hashes, the frozen numerical
source hashes, and all nine W15 returned-array hashes before drawing. It never
changes result CSVs, reruns the centered reference, or repeats timings. The
previous analytical-response figure and generator remain unchanged in
`round1/2d_radar_gaussian/` and `generate_2d_radar_gaussian_validation.py`; the
legacy `--figure-only` command reproduces that archived version, not Figure 3.

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

### Archived analytical-response radar-statistics benchmark

The 2026-11-02 comparison below is retained unchanged as analytical-response
FFT/direct evidence. The later NumPy range-gate loop comparison is also archived
below. Current manuscript Tables 6 and 11 instead select the sampled-Gaussian
run described at the end of this file. Do not combine medians across these runs
or attribute analytical-response speedups to the current Gaussian convolution.

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
NumPy/SciPy functions. These were earlier Table 11 results; neither this run nor
the subsequent analytical-response loop comparison supplies the current table.
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

### Archived analytical-response range-adaptive loop comparison

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
to the selected loop run for saved-evidence and archived-manuscript checks;
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
48.6–49.4, respectively; at checkpoint `c4924a4`, the Abstract summarized these
as approximately 16 and 49 times faster for prepared computation of all three
statistics. That checkpoint's Table 11 gave each case's speedup, time, and RSS;
the performance section retained the smaller earlier SciPy FFT
speedups separately, without mixing the two timing campaigns. These findings
concern the tested range-adaptive loops with full finite-grid weights, not all
NumPy/SciPy algorithms or ordinary sampled/truncated Gaussian smoothing.

To test both frozen radar runs and their archived manuscript reporting without rerunning
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

### Sampled Gaussian equivalence control

The author approved this separate control on 2026-11-10 to distinguish exact
discrete-kernel equivalence from agreement between two implementations of the
analytical Gaussian response. It uses only the original `720 x 1180` COW sweep
and `W=5,15`; no new controlled-gap or denominator-sensitivity cases are added.
The released toolkit, earlier radar scripts and measurements, and archived
analytical-response figure remain unchanged.

For every range gate, the angular standard deviation is
`sigma = W / ((j+1) * azimuth_spacing_radians * sqrt(12))`; the range standard
deviation is `W/sqrt(12)`. The new discrete kernel samples
`exp(-0.5*(offset/sigma)**2)` at integer offsets from `-ceil(4*sigma)` through
`+ceil(4*sigma)`, then normalizes their sum to one. This explicitly chosen
four-sigma support is used identically by both discrete methods; it is not a
claim of equality to the continuous analytical response. Astropy 8.0.2
`Gaussian1DKernel` with the same explicit size and `mode='center'` independently
checks every kernel outside the timed calculation.

The discrete spectral response is the finite cosine sum of these weights.
The implementation constructs it efficiently by accumulating all offsets on a
periodic grid and transforming that grid. Periodic azimuth uses period `N_az`;
half-sample reflective range uses period `2*N_range` and its first `N_range`
cosine modes. Kernels wider than an axis retain all repeated boundary
contributions; no kernel-width cap or minimum-one width is applied.

The untimed reference calculates centered angular variances from nonnegative
squared residuals, then combines within- and between-angular-mean variation
along range. The angular mask responses remain in those range weights, so this
is the same complete azimuth-first/range-second estimator, not sequentially
normalized smoothing. It checks every output cell without FFTs, DCTs,
convolution calls or subtraction of raw moments. Small tests also enumerate
each original donor independently, including repeated wrap/reflection, odd and
even axis lengths, singleton axes, missing values, zeros and a single donor.

Correctness requires the original radar tolerances and identical defined-output
masks for the discrete spectral and spatial methods. Raw variance is checked
before clamping, including its minimum and negative-value counts. Near zero,
the square root magnifies tiny variance differences: as in the existing 3D
checks, std uses the square-root variance bound only where the reference variance
is at most the fixed variance absolute tolerance. Nonzero-variance std checks,
all variance checks and analytical released-API checks keep their original
tolerances. Analytical-versus-sampled results are saved separately as
`different_operator`, not misclassified as failed numerical implementations.

The three timed methods are `analytical_spectral` (the unchanged prepared
released analytical response), `discrete_spectral` (the exact sampled-kernel
response), and `discrete_spatial` (SciPy `convolve1d`, wrap in azimuth and reflect
in range). Each prepares once and filters the original mask, zero-filled data,
and squared data before normalization. Every timing includes kernel/response
preparation and all common statistics/allocations. Preparation is also recorded
as a subset of total time, not added to it. Imports, input reading, independent
verification and output fingerprints are excluded. Five fresh sequential
workers per method/width use identical imports and requested one-thread limits;
method order rotates without warm-up. Worker checks enforce source/input/output
hashes and actual preparation/filter counts before a timing is accepted.
Process RSS follows the earlier 1-ms plus newly exceeded lifetime-peak protocol.

After setting up the pinned runtime as above, run from the CAGEO root:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON" -B -m pytest experiments/tests/test_radar_discrete_gaussian.py -q
"$PYTHON" -B experiments/scripts/benchmark_2d_radar_discrete_gaussian.py \
  --run-id radar-discrete-gaussian-20261110
```

`--validate-only` skips timing. Every run creates a new directory under
`results/round1/2d_radar_discrete_gaussian/` and refuses to overwrite an existing
run. Compact outputs are `agreement_summary.csv` (four full-sweep/near-radar
rows) and `timing_summary.csv` (two width rows). Detailed correctness checks,
operator differences, raw-variance diagnostics, kernel checks, individual
timings/preparation times, RSS and provenance remain in the accompanying files.
Set `DISCRETE_GAUSSIAN_RESULT_DIR` to a completed run to enable the saved-result
tests. This retained control run and its findings follow below; the later
explicit-loop run supplies the current tables and timing comparison.

#### Retained Gaussian control run and findings (2026-11-10)

The retained control run is
[`radar-discrete-gaussian-20261110`](results/round1/2d_radar_discrete_gaussian/radar-discrete-gaussian-20261110/).
All **90 correctness comparisons**, **2,362 kernel checks** and **30 fresh-worker
measurements** pass. These are new measurements, not medians reused from an
earlier run. The host is an Apple M4 Pro (`Mac16,7`) with 24 GiB RAM; Python is
x86_64 under Rosetta 2, with the requested one-thread settings described above.

The four-row agreement summary below pools the maximum mean and **unclamped**
variance differences of both discrete methods against the independent centered
reference. Minimum raw variance and negative counts refer to the discrete
spectral method. Mean differences have dBZ units; variance quantities have
dBZ-squared units. Near radar means range at most 10 km, evaluated after the
complete sweep was filtered.

| W | Region | Defined | Max mean difference | Max variance difference | Minimum raw variance | Negative raw count |
|---|---|---:|---:|---:|---:|---:|
| 5 | Full sweep | 574592 | 2.11e-11 | 8.69e-10 | -7.22e-11 | 52 |
| 5 | Near radar | 77889 | 1.78e-11 | 7.88e-10 | 0.742 | 0 |
| 15 | Full sweep | 640682 | 1.76e-11 | 4.39e-10 | -1.81e-10 | 66 |
| 15 | Near radar | 90140 | 3.17e-12 | 1.80e-10 | 3.67 | 0 |

The centered reference is nonnegative by construction. The largest discrete
spectral raw negative has magnitude about `1.81e-10` dBZ-squared; the direct spatial
raw negatives are at most `5.68e-14`. Neither discrete method has a negative
value beyond the fixed variance comparison tolerance. This is not a claim of
exactly zero floating-point negatives: the reported std remains nonnegative
because the common reporting rule clamps raw negative moment differences.

Median total times in milliseconds, over five repetitions including preparation:

| W | Analytical spectral | Discrete spectral | Discrete spatial |
|---|---:|---:|---:|
| 5 | 65.1 | 89.1 | 67.8 |
| 15 | 63.8 | 87.9 | 87.7 |

Median preparation times are 2.72/2.78 ms for the analytical response,
23.75/24.55 ms for the discrete spectral method, and 6.44/6.67 ms for the
discrete spatial method (`W=5/15`). The analytical response has the lowest
observed total medians, but its W5 timing interquartile interval overlaps the
discrete spatial interval. Five repetitions and two widths do not establish a
general ranking. The exact discrete spectral method is slower than the direct
spatial method at W5 and has a similar median at W15.

The analytical response is **not the same discrete estimator**. It has 13,322
and 4,593 negative raw variances, with minima `-1771.898` and `-3645.103`
dBZ-squared for W5 and W15. A separate test reproduces these large negative
values with independently constructed cosine-sum spatial weights and an
explicit centered-residual sum at the worst location for each width. They are
not merely raw-moment cancellation. Analytical-versus-discrete timing ratios
must therefore not be described as speedups for an identical Gaussian kernel.
`kernel_comparison.csv` keeps their numerical and output-coverage differences
separate from the passing discrete-kernel equivalence checks.

At the experiment-only checkpoint, all 41 new tests passed with selected-run
checks enabled. That broader suite reported **409 passed, 15 skipped, two previously identified stale
manuscript-preservation failures**, and one binary-compatibility runtime warning
while importing the radar input dependencies. The numerical driver and saved
timing source hashes remain unchanged during the later manuscript integration.

#### Earlier manuscript integration and retained Figure 3 (2026-11-10)

At this checkpoint, the author approved using Gaussian convolution as the main
radar example. Section 2.2 now defines the sampled kernel and its cosine
response, and Section 4.1 distinguishes the experiment-specific implementation
from released v1.0.0. That checkpoint's Table 6 used the four pooled
centered-reference agreement rows, and Table 11 used spectral/spatial Gaussian
convolution plus the different analytical response. The 2026-11-11 explicit-loop
comparison below replaces those table/timing selections, not the Gaussian
definition or Figure 3. Earlier analytical-response gap/floor checks are archived,
not relabeled as tests of the current kernel.

`generate_2d_radar_discrete_gaussian_figure.py` reproduces the new W15 spectral
outputs for Figure 3 after verifying every evidence/source/output hash. Its
PDF, 300-dpi PNG, and metadata are under
`manuscript/figures/round1/2d_radar_discrete_gaussian/`. The six maps retain the
previous palette, scales, panel sizes, and square zooms, without the analytical
filter's red overlay or clamping key. The old figure and its generator remain
unchanged. To check the selected numerical evidence and new manuscript/figure:

```bash
export DISCRETE_GAUSSIAN_RESULT_DIR=experiments/results/round1/2d_radar_discrete_gaussian/radar-discrete-gaussian-20261110
"$PYTHON" -B -m pytest -q experiments/tests/test_radar_discrete_gaussian.py \
  experiments/tests/test_radar_discrete_gaussian_manuscript.py
```

At this checkpoint, all 63 focused tests passed, including byte-identical
reproduction of the new PDF, PNG, and sidecar. With all selected-result
environments enabled, the full suite reported **454 passed**, the same two obsolete manuscript-preservation
failures, and one radar-dependency runtime warning. The prescribed `latexmk`
build has 33 pages, resolved citations/references, and only the pre-existing
title-box warning. Table 6, Figure 3, and Table 11 are on pages 20, 21, and 27.

The requested independent scientific/writing reviewer reconstructed full-array
centered-residual references independently of the experiment's reference
routine, verified all nine output hashes for every method/case, reaggregated
both tables and preparation costs, reproduced the exports, and inspected the
figure and compiled pages. Its three refinements—release-specific signed-weight
qualification, explicit variance tolerances, and an overflow fix—are implemented;
the follow-up found no blocker. No numerical source, saved result, earlier
figure, or timing measurement changed during integration. The new sources,
results, and figure had not yet been copied or pushed to the separate public
experiment repository at that checkpoint.

### Explicit polar Gaussian double-loop comparison

The main radar presentation selects spectral filtering versus **standard spatial
convolution**, defined for this experiment as the explicit Python double-loop
implementation with NumPy neighbourhood dot products. This two-method selection
simplifies the main tables and summaries. The composed SciPy implementation was
verified to compute equivalent statistics and remains archived auxiliary evidence;
its omission does not imply that SciPy cannot implement the polar operator or
that an explicit double loop is the only spatial algorithm. Reported speedups
apply specifically to the stated Python implementation, not optimized spatial
filtering generally. Unrelated 1D/3D SciPy comparisons remain unchanged.

This decision changes only reporting. The frozen driver, its source identity,
and every CSV row remain intact, including all three methods' checks and all
30 worker measurements. Table 11 selects the 20 spectral/double-loop measurements
from that campaign; no experiment is rerun and no measurements are deleted or
combined across runs. Table 6's values and Figure 3 remain unchanged. Table 1
and Section 2.2 additionally define `ell` as the integer kernel-center offset
along the filtered axis, in grid cells; this notation clarification changes no
kernel or response.

The author clarified that the main polar comparison should use an explicit
spatial implementation, not describe composed one-dimensional SciPy routines
as a ready-made polar filter. The new
`benchmark_2d_radar_gaussian_double_loop.py` keeps the original sweep, W5/W15,
sampled Gaussian kernels, width conversion, boundaries, strict raw `d>1e-3`,
three-filter workload, and output policies unchanged.

The explicit implementation prepares the Gaussian weights once. For each
complete filter, it wrap-pads each range gate's azimuth column and visits every
`(azimuth, range)` output in nested Python loops. Each cell is a `numpy.dot`
over its gate-specific one-dimensional Gaussian neighbourhood. It then pads
the angular result along range using NumPy `symmetric` padding (half-sample
reflection) and again visits every cell in nested loops for the range dot
product. Neither stage calls convolution routines or spectral transforms.
This preserves the angular-first operator; it is not a stationary 2D kernel
using the target range's angular width for all contributing gates. Boundary
support wider than an axis retains every repeated contribution.

The complete filter acts separately on the mask, zero-filled observations, and
their squares, with normalization only after both stages. Runtime cell-update
counters verify `3*N_az*N_range` assignments in each stage. Preparation,
padding, all three filters and returned statistics/allocations are timed.
The timed loops use NumPy dot products, not scalar Python loops over kernel
weights; this is an implementation-specific comparison, not the intrinsic
cost of every spatial algorithm.

Both the explicit loops and spectral filtering are checked over the full sweep
against the independent centered-residual reference. The main four-row summary
uses **direct spectral-minus-double-loop discrepancies** before variance
clamping; `comparison.csv` separately keeps all three methods' centered-reference
checks, including the now-auxiliary SciPy implementation using exactly the same
kernels. No claim about the absence of equivalent library implementations is
needed. Every spectral array must also match the previous Gaussian run's saved
hashes, so Figure 3 is not regenerated for this baseline change.

After setting up the pinned runtime and one-thread limits above:

```bash
"$PYTHON" -B experiments/scripts/benchmark_2d_radar_gaussian_double_loop.py \
  --run-id radar-gaussian-double-loop-reproduction
```

Only a new run directory under `results/round1/2d_radar_gaussian_double_loop/`
is written; existing runs are never overwritten. Five fresh sequential workers
per method/width rotate method order without warm-up, preload identical
libraries, and verify participating sources, input bytes, all nine output
arrays, and actual filter/cell counts. Individual times, preparation times,
total process RSS, and comparison settings remain in the run. New spectral
medians are never combined with an older loop or SciPy median. The numerical
driver does not publish or synchronize repositories.

#### Selected explicit-loop run and findings (2026-11-11)

The selected run is
[`radar-gaussian-double-loop-20261111`](results/round1/2d_radar_gaussian_double_loop/radar-gaussian-double-loop-20261111/).
All **108 centered-reference comparisons**, **36 direct spectral/loop
comparisons**, **2,362 kernel checks**, and **30 fresh-worker measurements**
pass. The measured grid is the original `720x1180` sweep with its natural mask,
at W5/W15 and strict finite raw `d>1e-3`, without completion or mask restoration.
No new gap, floor-sensitivity, scaling, or recovery study was run.

The current Table 6 uses the following direct spectral-minus-loop differences,
not the preceding control's pooled centered-reference differences. Mean has
dBZ units and raw variance has dBZ-squared units. Full sweep and near radar
contain 849,600 and 95,760 cells, respectively; near radar means range at most
10 km after full-sweep filtering.

| W | Region | Defined | Max mean difference | Max raw-variance difference |
|---|---|---:|---:|---:|
| 5 | Full sweep | 574592 | 2.12e-11 | 8.69e-10 |
| 5 | Near radar | 77889 | 1.78e-11 | 7.88e-10 |
| 15 | Full sweep | 640682 | 1.76e-11 | 4.39e-10 |
| 15 | Near radar | 90140 | 3.18e-12 | 1.81e-10 |

Separate centered-reference maxima are `2.11e-11` dBZ for spectral mean and
`8.69e-10` dBZ-squared for spectral raw variance; explicit-loop raw variance
differs by at most `1.61e-12` dBZ-squared. The spectral raw minima remain
`-7.2248e-11` and `-1.8116e-10` dBZ-squared at W5/W15, with no negatives beyond
the variance tolerance. The largest spectral std discrepancy against the
centered reference is `1.8444e-5` dBZ near zero variance, assessed with the
previously documented square-root bound; raw-variance tolerances are unchanged.
These numerical negatives remain distinct from the substantial analytical-
response negatives preserved in the preceding control.

The current Table 11 selects the following medians and maximum total process
RSS from the frozen run, each from five fresh-worker repetitions including
preparation and padding. Standard spatial convolution is the explicit Python
double-loop implementation:

| W | Spectral (s) | Standard spatial (s) | Spectral RSS (MiB) | Standard spatial RSS (MiB) |
|---|---:|---:|---:|---:|
| 5 | 0.0932 | 3.8925 | 257.9 | 252.2 |
| 15 | 0.0876 | 3.9330 | 251.9 | 246.2 |

Loop/spectral ratios from unrounded medians are `41.764725652689826` and
`44.894399362975115`, reported as 41.8 and 44.9. They describe this explicit
Python-cell-loop implementation with NumPy dot products, not an intrinsic
speedup over all spatial convolution. Median preparation times in
spectral/standard-spatial order are `23.61/6.76` ms at W5 and `23.67/7.12` ms
at W15; these are subsets
of total time, not extra costs. Per-filter padding is included in the total,
not in the preparation submeasurement. Hardware, Rosetta, RSS-sampling, and
five-repetition limitations remain as documented above.

#### Archived auxiliary SciPy evidence

The same frozen run retains `discrete_spatial`, the composed SciPy implementation,
in `comparison.csv`, `benchmark_runs.csv`, `benchmark_summary.csv`,
`benchmark_phases.csv`, `timing_summary.csv`, and `settings.csv`. It is numerically
equivalent to the selected polar operator: all of its centered-reference checks
pass, with maximum mean and raw-variance discrepancies of `4.62e-14` dBZ and
`2.06e-12` dBZ-squared. Its kernel/worker/output identities and every recorded
time, preparation cost, and RSS value remain testable. The saved summary files
still describe the original three-method campaign, not a rewritten two-method
campaign. This material is auxiliary rather than selected main-table evidence;
it must not be deleted or relabeled as invalid to justify the reporting choice.

#### Verification and standalone preparation

Enable both Gaussian saved-result suites when checking the current numerical,
manuscript, and retained-figure evidence:

```bash
export DISCRETE_GAUSSIAN_RESULT_DIR=experiments/results/round1/2d_radar_discrete_gaussian/radar-discrete-gaussian-20261110
export GAUSSIAN_DOUBLE_LOOP_RESULT_DIR=experiments/results/round1/2d_radar_gaussian_double_loop/radar-gaussian-double-loop-20261111
"$PYTHON" -B -m pytest -q experiments/tests/test_radar_discrete_gaussian.py \
  experiments/tests/test_radar_discrete_gaussian_manuscript.py \
  experiments/tests/test_radar_gaussian_double_loop.py
```

All 88 focused tests pass after the reporting/notation update. The full
selected-result source suite reports **479 passed**, the same two obsolete
manuscript-preservation failures, and the
known radar-dependency runtime warning. A full-suite interaction in the new
in-process worker unit tests was corrected with a test-only memory allowance
and deliberate below-RSS rejection checks; neither the numerical driver nor
the actual fresh-worker 1024-MiB limit or saved measurements changed.

The prescribed 33-page build has resolved citations/references and only the
existing title-box warning; Tables 6/11 and Figure 3 are on pages 20/27 and 21.
The direct repository URL is retained in Section 5.1, with an explicit statement
that the new Gaussian sources/results are not yet in the public archive.
The preceding independent numerical review reproduced all six method/width calculations and
their nine output hashes, all centered-reference/direct comparisons, and the
actual full-sweep loop cell counts. It verified kernel/source/input identities,
distinct matched workers, all raw table/preparation/RSS aggregations, unchanged
Figure 3, and the qualified writing. Its availability-wording and URL-layout
corrections are implemented; final actual-page inspection confirmed the full
clickable URL within the margins and no remaining scoped blocker. No timing
campaign was repeated during that review.

The two-method reporting selection and `ell` definitions add no numerical
calculation or timing campaign. The rebuilt manuscript still has 33 pages,
with Table 1 on page 4, Eq. (5) on page 5, Table 6 on page 20, Figure 3 on
page 21, and Table 11 on page 27. The independent reviewer confirmed the updated
terminology, notation, two-method selection, preserved non-radar content, and
actual layout, and independently reaggregated the selected time/RSS/preparation
values and 41.8/44.9-times ratios. Its documentation review confirmed the rationale
and preservation of equivalent SciPy evidence. The sole response-location
refinement was implemented: zero-based sweep, fill-value decoding, and generator
mask/QC details are attributed to the selected run's `settings.csv`, not to prose
that Section 5.3 does not contain. Upstream radar QC remains unverified. No scoped
scientific or layout blocker remains. Five final reporting/saved-result checks
also pass in both checkouts after the optional wording refinement.

The standalone checkout at `/Users/java1127/projects/cageo-d-26-01410-experiments`
contains byte-identical copies of the 32 selected source/test/result/manuscript/
figure files. Its existing compatibility `conftest.py` is preserved, with only
three additional missing-`c4924a4` history checks handled separately. Its full
suite reports **475 passed, two unchanged obsolete manuscript-test failures,
four history/path skips**, and one warning. A separate portability run with no
CAGEO history available reports **414 passed, 67 documented history/path skips**,
and one warning. All new Gaussian numerical, saved-result, current-table, and
figure-reproduction checks run in that portability test, apart from the one
explicitly history-dependent old-figure preservation check. No raw radar data
was copied. Both workspaces remain uncommitted and unpushed; public publication
of the new Gaussian sources/results remains pending.
