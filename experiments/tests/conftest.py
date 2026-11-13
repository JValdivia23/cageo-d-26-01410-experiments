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

A further test, ``test_selected_saved_evidence_and_assets``, compares the
saved ``script`` setting against the recomputed absolute checkout path;
that equality only holds in the original workspace location, so the test is
skipped (with a location reason) whenever the checkout has been relocated.

Three further archived-evidence checks use CAGEO commit ``c4924a4`` after
the Gaussian manuscript replacement. Their own availability check skips only
those three tests when that commit is absent; all new Gaussian numerical,
saved-worker, current-manuscript, and figure-reproduction checks still run.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_COMMITS = (
    "c66106488a4e17bfb45260f4764106c5682254de",
    "f27468f21c5e1ab931736dce4239a0dcf9374083",
    "90faea45a4ee488679499a117be5e73470553e9e",
    "25ff03628d4c6feb02664223cd8fb6d7fb38cf1b",
)
GAUSSIAN_HISTORY_COMMIT = "c4924a4f0fd5f29edcbc36b2031bb62ad967a009"
GAUSSIAN_HISTORY_TESTS = {
    "test_radar_range_loops.py": (
        "test_archived_analytical_manuscript_table_matches_times_speedups_and_rss",
        "test_archived_analytical_manuscript_speedups_and_range_adaptive_loop_framing",
    ),
    "test_radar_discrete_gaussian_manuscript.py": (
        "test_prior_analytical_figure_and_numerical_evidence_remain_unchanged",
    ),
}

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
# Manuscript-revision tests in test_1d_boundary_comparison.py that diff
# main.tex against publication-history originals (git show 90faea4 / 25ff036).
HISTORY_TESTS_IN_TEST_BOUNDARY_COMPARISON = (
    "test_introduction_revision_preserves_manuscript_outside_introduction",
    "test_manuscript_changes_preserve_tables_math_and_unrelated_text",
)
# Fixtures in test_wind_figure.py whose evidence chain calls git show on the
# history commits via _figure_saved_evidence()/prepare_wind_figure().
GHOST_WIND_FIXTURES = ("evidence", "prepared", "drawing")


def history_available(commits=HISTORY_COMMITS) -> bool:
    """Return True when all requested CAGEO history commits are present locally."""
    try:
        for commit in commits:
            result = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=REPO_ROOT, capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                return False
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _recorded_boundary_script_path() -> str | None:
    """Absolute script path recorded in the saved boundary run, if readable."""
    settings = (
        REPO_ROOT / "experiments" / "results" / "round1" / "1d_boundary_comparison"
        / "boundary-truncated-20261030" / "settings.csv"
    )
    try:
        with settings.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("setting") == "script":
                    return row.get("value") or None
    except OSError:
        return None
    return None


def pytest_collection_modifyitems(config, items):
    if not history_available():
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
            elif "test_1d_boundary_comparison.py" in nodeid and any(
                nodeid.rsplit("::", 1)[-1].startswith(name) for name in HISTORY_TESTS_IN_TEST_BOUNDARY_COMPARISON
            ):
                item.add_marker(marker)
            elif "test_wind_figure.py" in nodeid:
                fixtures = set(getattr(item, "fixturenames", ()) or ())
                if fixtures & set(GHOST_WIND_FIXTURES):
                    item.add_marker(marker)
    if not history_available((GAUSSIAN_HISTORY_COMMIT,)):
        marker = pytest.mark.skip(
            reason="standalone clone lacks CAGEO archived-evidence commit c4924a4 "
            "(see repo README, 'How to verify'); run in CAGEO for the history comparison"
        )
        for item in items:
            filename = item.nodeid.split("::", 1)[0].rsplit("/", 1)[-1]
            test_name = item.nodeid.rsplit("::", 1)[-1]
            if test_name in GAUSSIAN_HISTORY_TESTS.get(filename, ()):
                item.add_marker(marker)
    # test_selected_saved_evidence_and_assets asserts the saved absolute
    # `script` path equals the current checkout path; it passes in the
    # original workspace and can only fail after relocation.
    recorded = _recorded_boundary_script_path()
    current = str(REPO_ROOT / "experiments" / "scripts" / "generate_1d_boundary_comparison.py")
    if recorded is not None and recorded != current:
        for item in items:
            if item.nodeid.endswith("test_1d_boundary_comparison.py::test_selected_saved_evidence_and_assets"):
                item.add_marker(pytest.mark.skip(
                    reason=f"saved evidence records the original workspace path {recorded!r}; "
                    f"this checkout is {current!r}. Run in the original CAGEO checkout for this check."
                ))
