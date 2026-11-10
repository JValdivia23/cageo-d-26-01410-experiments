# QCed radar input

The existing workflow uses this external file; it is not copied into Git:

```text
~/onedrive/code/research/smoothing/cfrad_QCed_20220223_023027.051_to_20220223_023410.833_COW1high_SUR.nc
```

`generate_synthetic_validation.py` already supports `REAL_RADAR_PATH` to select
another location for the file.

Verified locally on 2026-10-26:

- Size: **766,632,010 bytes**.
- SHA-256: `f1e45c26da3002c31cfea1e62ece5a40c18a668d374193205daf793f41a4fae0`.
- Field: `DBZHC` (dBZ); stored fill value approximately `-9e33` decodes to NaN.
- Zero-based sweep 12: fixed angle 1.9927734°, 720 × 1180 cells, approximately
  0.5° azimuth spacing, first gate 95 m, nominal gate spacing 74.9414 m.

The file metadata identify `COW1high`. On 2026-10-27, the author confirmed that
the radar is COW (C-band) and the former `DOW6` figure label was a mistake. This labeling
correction does not change the source data or stored results. The upstream QC
procedure, public access route for this exact file, and redistribution permission
remain unverified. The filename alone does not establish the QC procedure;
obtain the file/access details from the author.
