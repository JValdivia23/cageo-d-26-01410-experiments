"""Standalone-clone compatibility for publication-history provenance tests.

Two provenance checks compare the current scripts and figure settings
against the immutable CAGEO publication-history commits
``c66106488a4e17bfb45260f4764106c5682254de`` (pre-figure source tree) and
``f27468f21c5e1ab931736dce4239a0dcf9374083`` (wind figure settings).
Fresh clones of this experiments-only repository do not contain those
commits, so without handling the dependent tests would error inside the
``git show`` calls. When the commits are absent, the dependent tests are
skipped (with a pointer to the repo README) instead of erroring; all other
tests run normally. No source file is modified and no check is weakened
where the history is present.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_COMMITS = (
    "c66106488a4e17bfb45260f4764106c5682254de",
    "f27468f21c5e1ab931736dce4239a0dcf9374083",
)

# Tests in test_review_outputs.py that call the history-backed
# historical_script_source() / assert_recorded_script_source() guard.
HISTORY_TESTS_IN_TEST_REVIEW_OUTPUTS = (
    "test_recorded_source_accepts_original_and_current_figure_only_extension",
    "test_recorded_source_does_not_accept_an_unverified_historical_hash",
    "test_historical_source_guard_rejects_numerical_cli_and_import_side_effect_mutations",
    "test_figure_only_contract_supports_separate_parsers_without_changing_default_paths",
    "test_saved_comparisons_and_settings",
    "test_saved_benchmark_summaries",
    "test_radar_sensitivity_and_removed_comparison",
    "test_3d_workers_used_the_verified_inputs",
)
# Fixtures in test_wind_figure.py whose evidence chain calls git show on the
# history commits via _figure_saved_evidence()/prepare_wind_figure().
GHOST_WIND_FIXTURES = ("evidence", "prepared", "drawing")


def history_available() -> bool:
    """Return True when all CAGEO history commits are present locally."""
    try:
        for commit in HISTORY_COMMITS:
            result = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=REPO_ROOT, capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                return False
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def pytest_collection_modifyitems(config, items):
    if history_available():
        return
    marker = pytest.mark.skip(
        reason="standalone clone lacks the CAGEO publication-history commits "
        "(see repo README, 'How to verify'); run in the CAGEO checkout for full provenance checks"
    )
    for item in items:
        nodeid = item.nodeid
        if "test_review_outputs.py" in nodeid and any(
            nodeid.rsplit("::", 1)[-1].startswith(name) for name in HISTORY_TESTS_IN_TEST_REVIEW_OUTPUTS
        ):
            item.add_marker(marker)
        elif "test_wind_figure.py" in nodeid:
            fixtures = set(getattr(item, "fixturenames", ()) or ())
            if fixtures & set(GHOST_WIND_FIXTURES):
                item.add_marker(marker)
