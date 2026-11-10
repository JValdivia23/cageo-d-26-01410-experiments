"""Checks for source provenance and saved round-1 measurements."""

import csv
import os
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _review_common as common


@pytest.fixture(autouse=True)
def restore_import_path():
    original = list(sys.path)
    yield
    sys.path[:] = original


def test_run_directories_do_not_overwrite(tmp_path):
    run_id, directory = common.create_run_directory("example", "test-run", tmp_path)
    assert run_id == "test-run"
    assert directory == tmp_path / "example" / "test-run"
    with pytest.raises(FileExistsError):
        common.create_run_directory("example", "test-run", tmp_path)
    for unsafe in ("../escape", "/absolute", "x/y", ""):
        with pytest.raises(ValueError):
            common.create_run_directory(unsafe, "test-run", tmp_path)


def test_csv_preserves_numbers_and_leaves_absent_results_blank(tmp_path):
    path = tmp_path / "values.csv"
    value = 1.2345678901234567
    common.write_csv(path, ["value", "missing", "nonfinite", "count"], [
        {"value": value, "missing": None, "nonfinite": float("nan"), "count": 0},
        {"value": float("inf"), "missing": "", "nonfinite": float("-inf"), "count": 7},
    ])
    rows = list(csv.DictReader(path.open()))
    assert float(rows[0]["value"]) == value
    assert rows[0]["count"] == "0"
    assert rows[0]["missing"] == rows[0]["nonfinite"] == ""
    assert rows[1]["value"] == rows[1]["nonfinite"] == ""


def test_csv_rejects_unrecorded_columns(tmp_path):
    with pytest.raises(ValueError):
        common.write_csv(tmp_path / "bad.csv", ["known"], [{"unknown": 1}])


def test_release_check_rejects_unrelated_source(tmp_path, monkeypatch):
    (tmp_path / "dct_toolkit").mkdir()
    (tmp_path / "dct_toolkit" / "__init__.py").write_text('__version__ = "1.0.0"\n')
    monkeypatch.setenv("DCT_TOOLKIT_SOURCE", str(tmp_path))
    with pytest.raises(RuntimeError, match="do not match"):
        common.load_toolkit()


@pytest.mark.skipif(not os.environ.get("DCT_TOOLKIT_SOURCE"), reason="set exact release source")
def test_release_check_rejects_changed_code(tmp_path, monkeypatch):
    source = Path(os.environ["DCT_TOOLKIT_SOURCE"])
    shutil.copytree(source / "dct_toolkit", tmp_path / "dct_toolkit")
    with (tmp_path / "dct_toolkit" / "stats.py").open("a") as stream:
        stream.write("\n# Changed source must not be reported as the release.\n")
    monkeypatch.setenv("DCT_TOOLKIT_SOURCE", str(tmp_path))
    with pytest.raises(RuntimeError, match="differs from"):
        common.load_toolkit()


@pytest.mark.skipif(not os.environ.get("DCT_TOOLKIT_SOURCE"), reason="set exact release source")
def test_settings_keep_verified_commit(tmp_path):
    path = tmp_path / "settings.csv"
    common.write_settings(path, "run", {"case": {
        "width": ([3, 7, 7], "grid cells"),
        "dct_toolkit_commit": "incorrect caller-supplied commit",
    }})
    rows = list(csv.DictReader(path.open()))
    values = {row["setting"]: row["value"] for row in rows}
    assert values["dct_toolkit_commit"] == common.TOOLKIT_COMMIT
    assert values["width"] == "[3, 7, 7]"
    assert all(row["run_id"] == "run" and row["case_id"] == "case" for row in rows)
    assert values["common_helper_sha256"]
