"""Release checks and CSV output shared by the round-1 experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from datetime import datetime, timezone
from numbers import Real
from typing import Any


TOOLKIT_COMMIT = "9cdf40f6ae02a48aa4c6f1eca07de0df6c72935c"
_RELEASE_HASHES = {
    "__init__.py": "8f8f79946d29e487473b6e2896295c5bb12641b806208f34c1de7a9f6dd6a1c7",
    "_geometry.py": "6c362987e910c90c5d47f4a6888deff624afbba2eb53af0a769cb54304561e68",
    "_kernels.py": "ffb9036f7ee39b1a7ada336e518c3639f1391897ef581c73ca8f572f4f1f295d",
    "_widths.py": "908ea06bafadbd4b208d948cb13909ec22652a444a3f5582dbd4fe828f2e211c",
    "cartesian.py": "b9a016be0de4d01d9ac96075ddf4feb3e0828a56ce290033d37f4bce6da73716",
    "core.py": "b387d52af62879ac7ce5759cd501835eea511c79cb493f71bcc2db691c63a76d",
    "gap_filling.py": "58b45f3c26ae63d8d665e349fc3fdc46b2dfad41f83227d5b1b43814413eebb3",
    "polar.py": "68c702812e952cc0e1566477236ecc25597411f9d872b4c2f8fde9047881bb67",
    "stats.py": "8c29113bbec5b81a9262fa301e19783e515495a457e732a090a89b4436159146",
}

COMPARISON_FIELDS = [
    "run_id", "case_id", "method", "reference_method", "component", "statistic",
    "region", "estimate_type", "kernel_match", "n_considered", "n_compared",
    "n_method_undefined", "n_reference_undefined", "n_method_gap_filled",
    "n_reference_gap_filled", "max_abs_difference", "rmse", "atol", "rtol",
    "status", "units", "notes",
]
BENCHMARK_FIELDS = [
    "run_id", "case_id", "method", "operation", "repetition", "status",
    "elapsed_s", "ram_before_mib", "peak_ram_mib", "reason",
]
SUMMARY_FIELDS = [
    "run_id", "case_id", "method", "operation", "n_success", "n_failure",
    "median_time_s", "q25_time_s", "q75_time_s", "max_peak_ram_mib",
]
SETTINGS_FIELDS = ["run_id", "case_id", "setting", "value", "unit"]


def load_toolkit():
    """Import only the exact publication release, checking every source file.

    DCT_TOOLKIT_SOURCE may point to a read-only checkout or an extracted
    ``git archive`` containing dct_toolkit/. This avoids trusting an editable
    installation's version string when its checkout has advanced.
    """
    source = os.environ.get("DCT_TOOLKIT_SOURCE")
    if source:
        source_path = Path(source).expanduser().resolve()
        package = source_path / "dct_toolkit"
        if not package.is_dir():
            raise RuntimeError(f"DCT_TOOLKIT_SOURCE has no dct_toolkit/: {source_path}")
        sys.path.insert(0, str(source_path))
    else:
        spec = importlib.util.find_spec("dct_toolkit")
        if spec is None or spec.origin is None:
            raise RuntimeError("dct_toolkit is not installed; set DCT_TOOLKIT_SOURCE.")
        package = Path(spec.origin).resolve().parent

    actual_files = {str(p.relative_to(package)) for p in package.rglob("*.py")}
    if actual_files != set(_RELEASE_HASHES):
        raise RuntimeError("Toolkit source files do not match the approved v1.0.0 release.")
    for filename, expected in _RELEASE_HASHES.items():
        path = package / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(
                f"Toolkit source differs from {TOOLKIT_COMMIT}: {path}. "
                "Set DCT_TOOLKIT_SOURCE to an extraction of that commit."
            )
    toolkit = importlib.import_module("dct_toolkit")
    if Path(toolkit.__file__).resolve().parent != package.resolve():
        raise RuntimeError("A different toolkit was imported before the release check.")
    if toolkit.__version__ != "1.0.0":
        raise RuntimeError("The approved toolkit version is 1.0.0.")
    return toolkit


def create_run_directory(
    experiment: str, run_id: str | None = None, output_root: str | Path | None = None,
) -> tuple[str, Path]:
    """Create output_root/experiment/run_id without overwriting an earlier run."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for value in (experiment, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(f"Invalid experiment/run identifier: {value!r}")
    root = Path(output_root) if output_root is not None else (
        Path(__file__).resolve().parents[1] / "results" / "round1"
    )
    directory = root.expanduser().resolve() / experiment / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return run_id, directory


def environment_settings() -> dict[str, tuple[Any, str]]:
    """Record the actual libraries and checked source, not a nominal environment."""
    toolkit = load_toolkit()
    settings: dict[str, tuple[Any, str]] = {
        "python_executable": (sys.executable, ""),
        "python_version": (platform.python_version(), ""),
        "platform": (platform.platform(), ""),
        "cpu": (platform.processor(), ""),
        "logical_cpus": (os.cpu_count(), ""),
        "dct_toolkit_version": (toolkit.__version__, ""),
        "dct_toolkit_commit": (TOOLKIT_COMMIT, ""),
        "dct_toolkit_source": (str(Path(toolkit.__file__).resolve().parent), ""),
        "toolkit_source_verification": ("SHA-256 of all nine Python source files", ""),
        "command": (sys.argv, ""),
        "working_directory": (str(Path.cwd()), ""),
        "script_sha256": (hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest()
                          if Path(sys.argv[0]).is_file() else "", ""),
        "common_helper_sha256": (hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), ""),
    }
    for name in ("numpy", "scipy", "astropy", "xarray", "netCDF4", "psutil"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        settings[f"{name}_version"] = (version, "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        settings[name] = (os.environ.get(name, "not set"), "")
    return settings


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except ValueError:
            value = value.tolist()
    if isinstance(value, Real) and not math.isfinite(value):
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(path: str | Path, fieldnames: list[str], rows) -> None:
    """Write plain measured numbers; absent results are blank, never fake zeros."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def write_settings(path: str | Path, run_id: str, cases: dict[str, dict]) -> None:
    """Merge each case's settings with verified environment information.

    A setting may be a plain value or a (value, unit) tuple. Use a list for
    sequence-valued settings such as widths or boundary modes.
    """
    common = environment_settings()
    rows = []
    for case_id, case in cases.items():
        for name, value in sorted({**case, **common}.items()):
            if isinstance(value, tuple) and len(value) == 2:
                value, unit = value
            else:
                unit = ""
            rows.append(dict(run_id=run_id, case_id=case_id, setting=name,
                             value=value, unit=unit))
    write_csv(path, SETTINGS_FIELDS, rows)
