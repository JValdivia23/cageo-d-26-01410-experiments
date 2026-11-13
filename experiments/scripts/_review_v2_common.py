"""Strict v2.0.0 runtime, separate from the frozen v1.0.0 review helper."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys

from _review_common import SETTINGS_FIELDS, create_run_directory as _create, write_csv


TOOLKIT_VERSION = "2.0.0"
TOOLKIT_COMMIT = "7ff29e998dd1102d3fef2a84050402e86dc7ef67"
RELEASE_HASHES = {
    "__init__.py": "a4454641a8c93d555301501f137e1e6fd0b78ab4fc66cd86f6d2cdbd539a6472",
    "_geometry.py": "6c362987e910c90c5d47f4a6888deff624afbba2eb53af0a769cb54304561e68",
    "_kernels.py": "78a2148715ec21f8b9b7e777d722072278eb91d1fd11a7a43b368c60a5e9593b",
    "_widths.py": "908ea06bafadbd4b208d948cb13909ec22652a444a3f5582dbd4fe828f2e211c",
    "_gaussian.py": "cb5fdab4a190ab3bbd951d8c1b8a5185da33529739cd8d9feec9a8f2cbeb9bab",
    "cartesian.py": "7925aab9a38634c2168803e27087d8c548cf4209dd06284dd9b5a90bcf07ef39",
    "core.py": "6b6499c5e6661afbb9393a86070716eeca85722fc789e1203c165d24ac999831",
    "gap_filling.py": "58b45f3c26ae63d8d665e349fc3fdc46b2dfad41f83227d5b1b43814413eebb3",
    "polar.py": "9a4ebc4eb7b4b3b80363d921870e7787ef9da425259f70ea036cabc43d2fd4be",
    "stats.py": "e6bc3d21739b6725256f0300b397d4528061ae1bea911cc476b64b7c7476273c",
}
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "experiments/results/round1/v2.0.0"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_toolkit():
    """Check the tag's ten source files, version, and actual imported path."""
    source = os.environ.get("DCT_TOOLKIT_V2_SOURCE")
    if source:
        source = Path(source).expanduser().resolve()
        package = source / "dct_toolkit"
        sys.path.insert(0, str(source))
    else:
        spec = importlib.util.find_spec("dct_toolkit")
        if spec is None or spec.origin is None:
            raise RuntimeError("Set DCT_TOOLKIT_V2_SOURCE to the v2.0.0 extraction.")
        package = Path(spec.origin).resolve().parent
    actual = {str(p.relative_to(package)) for p in package.rglob("*.py")}
    if actual != set(RELEASE_HASHES):
        raise RuntimeError("Toolkit source files do not match v2.0.0.")
    for name, digest in RELEASE_HASHES.items():
        if sha256(package / name) != digest:
            raise RuntimeError(f"Toolkit source differs from {TOOLKIT_COMMIT}: {name}")
    toolkit = importlib.import_module("dct_toolkit")
    if (Path(toolkit.__file__).resolve().parent != package
            or toolkit.__version__ != TOOLKIT_VERSION):
        raise RuntimeError("A different toolkit was imported before the v2.0.0 check.")
    return toolkit


def create_run_directory(experiment, run_id=None, output_root=None):
    return _create(experiment, run_id, OUTPUT_ROOT if output_root is None else output_root)


def environment_settings():
    toolkit = load_toolkit()
    entrypoint = Path(__file__).with_name("run_review_v2.py")
    settings = {
        "python_executable": (sys.executable, ""),
        "python_version": (platform.python_version(), ""),
        "platform": (platform.platform(), ""),
        "cpu": (platform.processor(), ""),
        "logical_cpus": (os.cpu_count(), ""),
        "dct_toolkit_version": (toolkit.__version__, ""),
        "dct_toolkit_commit": (TOOLKIT_COMMIT, ""),
        "dct_toolkit_source": (str(Path(toolkit.__file__).resolve().parent), ""),
        "toolkit_source_verification": ("SHA-256 of all ten v2.0.0 Python source files", ""),
        "toolkit_source_sha256": (RELEASE_HASHES, ""),
        "command": (sys.argv, ""),
        "working_directory": (str(Path.cwd()), ""),
        "script": (str(entrypoint), ""),
        "script_sha256": (sha256(entrypoint), ""),
        "common_helper_sha256": (sha256(__file__), ""),
        "version_bridge": ("explicit runtime bindings; frozen numerical functions unchanged", ""),
    }
    for name in ("numpy", "scipy", "astropy", "matplotlib", "xarray", "netCDF4", "psutil"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        settings[f"{name}_version"] = (version, "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        settings[name] = (os.environ.get(name, "not set"), "")
    return settings


def write_settings(path, run_id, cases):
    common = environment_settings()
    rows = []
    for case_id, case in cases.items():
        for name, value in sorted({**case, **common}.items()):
            value, unit = value if isinstance(value, tuple) and len(value) == 2 else (value, "")
            rows.append(dict(run_id=run_id, case_id=case_id, setting=name, value=value, unit=unit))
    write_csv(path, SETTINGS_FIELDS, rows)
