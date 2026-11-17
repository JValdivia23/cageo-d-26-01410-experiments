# Radar input and download

The exact processed COW input is available as an attachment to the
[radar-data-v1 release](https://github.com/JValdivia23/cageo-d-26-01410-experiments/releases/tag/radar-data-v1).
It is the unchanged original volume used by the archived experiments, not a
new extraction or a reprocessed substitute. The NetCDF file stays outside Git
history and is not included in a clone or source-code ZIP.

## Source and attribution

The observations are from the WINTRE-MIX campaign. Please cite:

> Wurman, Joshua, and Karen Kosiba (2022). *WINTRE-MIX FARM data*.
> Flexible Array of Radars and Mesonets.
> https://doi.org/10.48514/WZ46-W047

The [UCAR EOL COW dataset listing](https://data.eol.ucar.edu/dataset/612.037)
and [WINTRE-MIX data catalog](https://data.eol.ucar.edu/master_lists/generated/wintre-mix/)
provide the original access context. These sources are also identified in the
published Data availability statement of
[Valdivia et al. (2026)](https://doi.org/10.5194/amt-19-5211-2026).
The observations retain their source data terms and attribution; this
repository's MIT software license does not relicense them.

## Download and verify

Set `DATA_DIR` to a local directory with sufficient space, then run:

```bash
export DATA_DIR="$HOME/radar-data"
mkdir -p "$DATA_DIR"
curl --fail --location --output "$DATA_DIR/cfrad_QCed_20220223_023027.051_to_20220223_023410.833_COW1high_SUR.nc" \
  "https://github.com/JValdivia23/cageo-d-26-01410-experiments/releases/download/radar-data-v1/cfrad_QCed_20220223_023027.051_to_20220223_023410.833_COW1high_SUR.nc"
curl --fail --location --output "$DATA_DIR/SHA256SUMS" \
  "https://github.com/JValdivia23/cageo-d-26-01410-experiments/releases/download/radar-data-v1/SHA256SUMS"
(cd "$DATA_DIR" && shasum -a 256 -c SHA256SUMS)
export REAL_RADAR_PATH="$DATA_DIR/cfrad_QCed_20220223_023027.051_to_20220223_023410.833_COW1high_SUR.nc"
```

Use the data only after the checksum reports `OK`. The experiment drivers
accept this location through `REAL_RADAR_PATH` (or `--radar-path` where supported).
No numerical source, saved measurement, or figure is changed by downloading it.

## File identity and interpretation

The original identity was recorded on 26 October 2026 and reverified before
publication on 17 November 2026:

- Filename: `cfrad_QCed_20220223_023027.051_to_20220223_023410.833_COW1high_SUR.nc`.
- Size: **766,632,010 bytes** (731 MiB).
- SHA-256: `f1e45c26da3002c31cfea1e62ece5a40c18a668d374193205daf793f41a4fae0`.
- Instrument: `COW1high`, the C-band COW radar.
- Field: `DBZHC` (dBZ); stored fill value approximately `-9e33` decodes to NaN.
- Zero-based sweep 12: fixed angle 1.9927734°, 720 × 1180 cells, approximately
  0.5° azimuth spacing, first gate 95 m, nominal gate spacing 74.9414 m.

The reader retains all finite observations, including zeros, without additional
reflectivity thresholds or quality-flag exclusions. The file records DORADE to
CF-Radial conversion and a first-gate adjustment, but does not fully document
the upstream quality-control procedure. The `QCed` filename alone does not
establish that procedure or explain the missing sectors. Public access should
not be confused with verification of measurement quality.
