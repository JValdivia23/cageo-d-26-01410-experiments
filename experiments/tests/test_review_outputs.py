"""Independently check the saved CSV files selected for author review.

Set REVIEW_RESULT_DIRS to the run directories, separated by the platform's
path separator. This checks the saved files, not an agent's printed summary.
"""

import ast
import copy
import csv
from functools import cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _review_common import (
    BENCHMARK_FIELDS, COMPARISON_FIELDS, SETTINGS_FIELDS, SUMMARY_FIELDS,
    TOOLKIT_COMMIT,
)


def read_rows(path, fields):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == fields, path
        rows = list(reader)
    assert rows, path
    assert all(None not in row and None not in row.values() for row in rows), path
    return rows


RUNS = [Path(p) for p in os.environ.get("REVIEW_RESULT_DIRS", "").split(os.pathsep) if p]
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT_NAMES = {
    "1d_boxcar": "generate_1d_boxcar_validation.py",
    "2d_radar_gaussian": "generate_2d_radar_gaussian_validation.py",
    "3d_wind_boxcar": "generate_3d_wind_boxcar_validation.py",
}
# Immutable pre-figure tree: all three scripts match the selected runs here.
RECORDED_SOURCE_COMMIT = "c66106488a4e17bfb45260f4764106c5682254de"


@cache
def historical_script_source(name):
    assert name in SCRIPT_NAMES.values(), name
    return subprocess.run(
        ["git", "show", f"{RECORDED_SOURCE_COMMIT}:experiments/scripts/{name}"],
        cwd=SCRIPTS.parents[1], check=True, capture_output=True,
    ).stdout


def ast_identity(node):
    return ast.dump(node, include_attributes=False)


def expression(source):
    return ast.parse(source, mode="eval").body


def figure_constant(value):
    """Only inert literals/names and the existing read-only source-path idiom."""
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, ast.Name):
        return value.id.isupper()
    if isinstance(value, (ast.List, ast.Tuple)):
        return all(figure_constant(item) for item in value.elts)
    if isinstance(value, ast.Dict):
        return all(key is not None and figure_constant(key) and figure_constant(item)
                   for key, item in zip(value.keys, value.values))
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
        return figure_constant(value.operand)
    if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        return figure_constant(value.left) and figure_constant(value.right)
    if isinstance(value, ast.Subscript):
        return figure_constant(value.value) and figure_constant(value.slice)
    return ast_identity(value) in {
        ast_identity(expression("Path(__file__).resolve()")),
        ast_identity(expression("Path(__file__).resolve().parents")),
    }


def parser_rejection(node):
    """A read-only option predicate followed only by parser.error(literal)."""
    if not isinstance(node, ast.If) or node.orelse or len(node.body) != 1:
        return False
    call = node.body[0]
    if not (isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)):
        return False
    call = call.value
    if (ast_identity(call.func) != ast_identity(expression("parser.error"))
            or call.keywords or len(call.args) != 1
            or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str)):
        return False
    allowed = (ast.BoolOp, ast.UnaryOp, ast.Compare, ast.Attribute, ast.Name, ast.Constant,
               ast.Tuple, ast.List, ast.Load, ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq,
               ast.Is, ast.IsNot, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)
    return all(isinstance(part, allowed)
               and (not isinstance(part, ast.Name) or part.id == "args" or part.id.isupper())
               and (not isinstance(part, ast.Attribute)
                    or isinstance(part.value, ast.Name) and part.value.id == "args")
               for part in ast.walk(node.test))


def strip_figure_cli(original, current, exporters, events):
    """Strip only explicit opt-in flags, their guard, and an early export/return.

    Never strip arbitrary statements, changed signatures, existing validation,
    or an else branch. Separate parse_args() functions use the same contract.
    """
    parsed = next(i for i, node in enumerate(original.body)
                  if isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "args")
    first_work = parsed + 1
    if original.name == "parse_args":
        first_work = len(original.body) - 1  # Original return args must survive.
    else:
        while first_work < len(original.body) and parser_rejection(original.body[first_work]):
            first_work += 1
    output_guard = ast.parse(
        'if args.figure_output is not None and not args.figure_only:\n'
        '    parser.error("--figure-output requires --figure-only")'
    ).body[0]
    kept = []
    for node in current.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast_identity(node.value.func) == ast_identity(expression("parser.add_argument"))
                and node.value.args and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value in ("--figure-only", "--figure-output")):
            call, flag = node.value, node.value.args[0].value
            assert len(kept) <= parsed and len(call.args) == 1, "Figure flags must precede argument parsing"
            keywords = {item.arg: item.value for item in call.keywords}
            kind, expected = (("action", expression('"store_true"')) if flag == "--figure-only"
                              else ("type", expression("Path")))
            assert len(keywords) == len(call.keywords) == 2 and set(keywords) == {kind, "help"}
            assert ast_identity(keywords[kind]) == ast_identity(expected)
            assert isinstance(keywords["help"], ast.Constant) and isinstance(keywords["help"].value, str)
            events.append(flag)
        elif ast_identity(node) == ast_identity(output_guard):
            assert parsed < len(kept) <= first_work, "Figure guard must precede ordinary work"
            events.append("output_guard")
        elif isinstance(node, ast.If) and ast_identity(node.test) == ast_identity(expression("args.figure_only")):
            assert parsed < len(kept) <= first_work and not node.orelse, "Figure dispatch must be early and have no else"
            if original.name == "parse_args":
                assert node.body and all(parser_rejection(item) for item in node.body)
                events.append("figure_validation")
            else:
                assert all(parser_rejection(item) for item in node.body[:-2])
                dispatches = [ast.parse(
                    f'for kind, path in {name}(args.figure_output).items():\n'
                    '    print(f"{kind}: {path}")\nreturn 0'
                ).body for name in exporters]
                assert any([ast_identity(item) for item in node.body[-2:]]
                           == [ast_identity(item) for item in dispatch] for dispatch in dispatches), "Unexpected figure dispatch"
                events.append("dispatch")
        else:
            kept.append(node)
    result = copy.deepcopy(current)
    result.body = kept
    assert ast_identity(result) == ast_identity(original), f"Original {original.name} CLI/default body changed"
    return result


def assert_figure_only_extension(original_source, current_source):
    """Require every original executable module node, in order, unchanged.

    Only the module docstring, inert FIGURE_* constants, lazy undecorated
    _figure_* / prepare_*_figure / create_*_figure / export_*_figure functions,
    and the explicit opt-in CLI grammar above may be added. Imports, numerical
    definitions, ordinary CLI, signatures and validation remain exact ASTs.
    """
    original, current = ast.parse(original_source), ast.parse(current_source)
    for tree in (original, current):
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            tree.body.pop(0)
    original_functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
    added_functions = {node.name for node in current.body
                       if isinstance(node, ast.FunctionDef) and node.name not in original_functions}
    exporters = {name for name in added_functions if name.startswith("export_") and name.endswith("_figure")}
    kept, events, added_names = [], [], set()
    for node in current.body:
        if isinstance(node, ast.FunctionDef) and node.name in added_functions:
            assert re.fullmatch(r"_figure_[a-z0-9_]+|(?:prepare|create|export)_[a-z0-9_]+_figure", node.name), node.name
            assert not node.decorator_list and node.name not in added_names, "Figure definitions must be lazy and unique"
            assert all(isinstance(value, ast.Constant)
                       or isinstance(value, ast.Name) and value.id.startswith("FIGURE_")
                       for value in node.args.defaults + [v for v in node.args.kw_defaults if v is not None]), "Figure defaults must be inert"
            added_names.add(node.name)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith("FIGURE_")):
            name = node.targets[0].id
            assert name not in added_names and figure_constant(node.value), "Figure constants must be inert and unique"
            added_names.add(name)
        else:
            if isinstance(node, ast.FunctionDef) and node.name in ("main", "parse_args"):
                node = strip_figure_cli(original_functions[node.name], node, exporters, events)
            kept.append(node)
    if events:
        assert sorted(item for item in events if item != "figure_validation") == sorted([
            "--figure-only", "--figure-output", "output_guard", "dispatch",
        ]), "Incomplete or duplicate figure-only CLI"
        assert events.count("figure_validation") <= 1
    current.body = kept
    assert ast_identity(current) == ast_identity(original), "Original imports/constants/numerical definitions or entry point changed"


def assert_recorded_script_source(path, recorded_hash, current_source=None):
    """A direct current hash still passes, including newly recorded future runs."""
    current = path.read_bytes() if current_source is None else current_source
    if hashlib.sha256(current).hexdigest() == recorded_hash:
        return
    historical = historical_script_source(path.name)
    assert hashlib.sha256(historical).hexdigest() == recorded_hash, "Recorded script hash is not the immutable pre-figure source"
    assert_figure_only_extension(historical, current)


@pytest.mark.parametrize("name", SCRIPT_NAMES.values())
def test_recorded_source_accepts_original_and_current_figure_only_extension(name):
    historical = historical_script_source(name)
    recorded_hash = hashlib.sha256(historical).hexdigest()
    assert_figure_only_extension(historical, historical)
    assert_recorded_script_source(SCRIPTS / name, recorded_hash, historical)
    assert_recorded_script_source(SCRIPTS / name, recorded_hash)


def test_recorded_source_does_not_accept_an_unverified_historical_hash():
    with pytest.raises(AssertionError, match="immutable pre-figure source"):
        assert_recorded_script_source(SCRIPTS / SCRIPT_NAMES["1d_boxcar"], "0" * 64)


def test_direct_current_hash_remains_valid_for_future_recorded_runs():
    path = SCRIPTS / SCRIPT_NAMES["1d_boxcar"]
    current = path.read_bytes()
    assert_recorded_script_source(path, hashlib.sha256(current).hexdigest(), current)


@pytest.mark.parametrize("before,after", [
    ("DENOMINATOR_FLOOR = 1e-3", "DENOMINATOR_FLOOR = 1e-2"),
    ("raw_variance = second - mean**2", "raw_variance = second + mean**2"),
    ('default="full", help="one or five', 'default="smoke", help="one or five'),
    ("n_tests = small_tests(toolkit)", "n_tests = 1 + small_tests(toolkit)"),
    ("if args.master_seed < 0:", "if args.master_seed <= 0:"),
    ("def main(argv: list[str] | None = None)", "def main(argv: list[str] | None = ())"),
    ("from scipy.ndimage import uniform_filter", "from scipy.ndimage import uniform_filter as another_filter"),
    ('"--figure-only", action="store_true"', '"--figure-only", action="store_true", default=True'),
    ("if args.figure_only:", 'if args.figure_only or args.mode == "full":'),
    ("args.figure_output is not None and not args.figure_only", "args.figure_output is not None"),
    ("def create_missingness_figure(cases):", "@small_tests\ndef create_missingness_figure(cases):"),
    ('FIGURE_STEM = "boxcar_missingness"', 'FIGURE_STEM = np.seterr(all="ignore")'),
    ('if __name__ == "__main__":', 'N = 201\n\nif __name__ == "__main__":'),
    ('if __name__ == "__main__":', 'np.seterr(all="ignore")\n\nif __name__ == "__main__":'),
])
def test_historical_source_guard_rejects_numerical_cli_and_import_side_effect_mutations(before, after):
    path = SCRIPTS / SCRIPT_NAMES["1d_boxcar"]
    source = path.read_text()
    assert source.count(before) == 1
    changed = source.replace(before, after, 1).encode()
    recorded_hash = hashlib.sha256(historical_script_source(path.name)).hexdigest()
    with pytest.raises(AssertionError):
        assert_recorded_script_source(path, recorded_hash, changed)


@pytest.mark.parametrize("name", (SCRIPT_NAMES["2d_radar_gaussian"], SCRIPT_NAMES["3d_wind_boxcar"]))
def test_figure_only_contract_supports_separate_parsers_without_changing_default_paths(name):
    # Exercise the upcoming figure-only shape in memory, not a copied fixture
    # source or an edited experiment. The immutable originals stay in Git.
    historical = historical_script_source(name)
    tree = ast.parse(historical)
    parser = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    parsed = next(i for i, node in enumerate(parser.body)
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == "args")
    parser.body[parsed:parsed] = ast.parse(
        'parser.add_argument("--figure-only", action="store_true", help="Export only the saved figure cases")\n'
        'parser.add_argument("--figure-output", type=Path, help="Figure output directory")'
    ).body
    parser.body[parsed + 3:parsed + 3] = ast.parse(
        'if args.figure_output is not None and not args.figure_only:\n'
        '    parser.error("--figure-output requires --figure-only")\n'
        'if args.figure_only:\n'
        '    if args.run_id is not None or args.output_root is not None:\n'
        '        parser.error("Figure-only cannot write numerical results")'
    ).body
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main.body[1:1] = ast.parse(
        'if args.figure_only:\n'
        '    for kind, path in export_review_figure(args.figure_output).items():\n'
        '        print(f"{kind}: {path}")\n'
        '    return 0'
    ).body
    tree.body.insert(-1, ast.parse('def export_review_figure(output=None):\n    return {}').body[0])
    current = ast.unparse(tree).encode()
    assert_recorded_script_source(SCRIPTS / name, hashlib.sha256(historical).hexdigest(), current)
    # Moving the opt-in dispatch after any ordinary work is not figure-only.
    main.body[1], main.body[2] = main.body[2], main.body[1]
    with pytest.raises(AssertionError):
        assert_recorded_script_source(SCRIPTS / name, hashlib.sha256(historical).hexdigest(), ast.unparse(tree).encode())


@pytest.mark.skipif(not RUNS, reason="set REVIEW_RESULT_DIRS to selected saved runs")
@pytest.mark.parametrize("directory", RUNS or [Path(".")])
def test_saved_comparisons_and_settings(directory):
    settings = read_rows(directory / "settings.csv", SETTINGS_FIELDS)
    cases = {}
    for row in settings:
        key = row["case_id"]
        values = cases.setdefault(key, {})
        assert row["setting"] not in values, (directory, key, row["setting"])
        values[row["setting"]] = row["value"]
    script_path = SCRIPTS / SCRIPT_NAMES[directory.parent.name]
    helper_hash = hashlib.sha256((SCRIPTS / "_review_common.py").read_bytes()).hexdigest()
    for recorded_hash in {values["script_sha256"] for values in cases.values()}:
        assert_recorded_script_source(script_path, recorded_hash)
    for values in cases.values():
        assert values["dct_toolkit_commit"] == TOOLKIT_COMMIT
        assert values["dct_toolkit_version"] == "1.0.0"
        assert values["common_helper_sha256"] == helper_hash
    rows = read_rows(directory / "comparison.csv", COMPARISON_FIELDS)
    assert {row["statistic"] for row in rows} >= {"mean", "variance", "std", "support"}
    for row in rows:
        assert row["run_id"] == directory.name
        assert row["case_id"] in cases
        n, k = int(row["n_considered"]), int(row["n_compared"])
        assert 0 <= k <= n
        for field in ("n_method_undefined", "n_reference_undefined",
                      "n_method_gap_filled", "n_reference_gap_filled"):
            assert 0 <= int(row[field]) <= n
        assert row["status"] in ("pass", "no_comparable"), row
        assert row["n_method_undefined"] == row["n_reference_undefined"], row
        if k == 0:
            assert row["status"] == "no_comparable", row
            assert row["max_abs_difference"] == row["rmse"] == "", row
        else:
            for name in ("max_abs_difference", "rmse"):
                assert math.isfinite(float(row[name])) and float(row[name]) >= 0
            assert float(row["rmse"]) <= float(row["max_abs_difference"]) + 1e-12


@pytest.mark.skipif(not RUNS, reason="set REVIEW_RESULT_DIRS to selected saved runs")
@pytest.mark.parametrize("directory", RUNS or [Path(".")])
def test_saved_benchmark_summaries(directory):
    raw_path = directory / "benchmark_runs.csv"
    if not raw_path.exists():
        settings = read_rows(directory / "settings.csv", SETTINGS_FIELDS)
        numerical_only = any(row["setting"] == "mode" and row["value"] == "numerical"
                             for row in settings)
        assert "1d_boxcar" in str(directory) or "radar" in str(directory) or numerical_only
        return
    raw = read_rows(raw_path, BENCHMARK_FIELDS)
    summaries = read_rows(directory / "benchmark_summary.csv", SUMMARY_FIELDS)
    groups = {}
    for row in raw:
        assert row["run_id"] == directory.name
        key = row["case_id"], row["method"], row["operation"]
        groups.setdefault(key, []).append(row)
        if row["status"] in ("ok", "success", "pass"):
            assert math.isfinite(float(row["elapsed_s"])) and float(row["elapsed_s"]) > 0
            if "3d_wind_boxcar" in str(directory):
                before, peak = float(row["ram_before_mib"]), float(row["peak_ram_mib"])
                assert peak >= before > 0
            else:
                assert row["ram_before_mib"] == row["peak_ram_mib"] == ""
        else:
            assert row["reason"]
    assert len(summaries) == len(groups)
    seen = set()
    for row in summaries:
        key = row["case_id"], row["method"], row["operation"]
        assert key not in seen
        seen.add(key)
        group = groups[key]
        valid = [r for r in group if r["status"] in ("ok", "success", "pass")]
        assert int(row["n_success"]) == len(valid)
        assert int(row["n_failure"]) == len(group) - len(valid)
        if not valid:
            assert row["median_time_s"] == row["q25_time_s"] == row["q75_time_s"] == ""
            continue
        times = np.asarray([float(r["elapsed_s"]) for r in valid])
        for field, percentile in (("median_time_s", 50), ("q25_time_s", 25), ("q75_time_s", 75)):
            assert math.isclose(float(row[field]), float(np.percentile(times, percentile)),
                                rel_tol=1e-12, abs_tol=1e-12), row
        peaks = [float(r["peak_ram_mib"]) for r in valid if r["peak_ram_mib"]]
        if peaks:
            assert math.isclose(float(row["max_peak_ram_mib"]), max(peaks), rel_tol=1e-12)
        else:
            assert row["max_peak_ram_mib"] == ""

    if directory.parent.name == "3d_wind_boxcar":
        text = (SCRIPTS.parents[1] / "manuscript/main.tex").read_text()
        abstract = text.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
        cartesian = abstract.split("For three-dimensional Cartesian grids,")[1].split(".", 1)[0]
        indexed = {(row["case_id"], row["method"]): float(row["median_time_s"])
                   for row in summaries}
        cases = {case for case, method in indexed if method == "dct_shared_moments"}
        for method, label in (("numpy_windows", "NumPy window reductions"),
                              ("astropy_direct", "Astropy direct convolution"),
                              ("astropy_fft", "Astropy FFT convolution")):
            ratios = {case: indexed[case, method] / indexed[case, "dct_shared_moments"]
                      for case in cases}
            assert max(ratios, key=ratios.get) == "broad20"
            rounded = round(max(ratios.values()))
            number = {6: "six"}.get(rounded, str(rounded))
            assert f"{number} times faster than {label}" in cartesian
        assert "was up to" in cartesian and "largest gains for wider windows" in cartesian


@pytest.mark.skipif(not RUNS, reason="set REVIEW_RESULT_DIRS to selected saved runs")
@pytest.mark.parametrize("directory", RUNS or [Path(".")])
def test_radar_sensitivity_and_removed_comparison(directory):
    if "2d_radar_gaussian" not in str(directory):
        return
    assert not (directory / "gaussian_approximation.csv").exists()
    sensitivity_path = directory / "threshold_sensitivity.csv"
    if sensitivity_path.exists():
        rows = read_rows(sensitivity_path, COMPARISON_FIELDS)
        settings = read_rows(directory / "settings.csv", SETTINGS_FIELDS)
        values = {}
        for row in settings:
            values.setdefault(row["case_id"], {})[row["setting"]] = row["value"]
        for row in rows:
            assert row["status"] in ("pass", "fail", "no_comparable")
            assert float(values[row["case_id"]]["denominator_floor"]) != 1e-3
            if row["status"] == "fail":
                assert row["statistic"] not in ("raw_mask_response", "direct_acceptance", "fallback_eligibility")
        for case_id in {row["case_id"] for row in rows}:
            n_failure = sum(row["status"] == "fail" for row in rows if row["case_id"] == case_id)
            assert int(values[case_id]["strict_sensitivity_failures_this_case"]) == n_failure
            assert values[case_id]["comparison_output"] == "threshold_sensitivity.csv"
    benchmark_path = directory / "benchmark_runs.csv"
    if benchmark_path.exists():
        timed = read_rows(benchmark_path, BENCHMARK_FIELDS)
        assert {row["method"] for row in timed} == {"toolkit_v1_polar_direct"}


@pytest.mark.skipif(not RUNS, reason="set REVIEW_RESULT_DIRS to selected saved runs")
@pytest.mark.parametrize("directory", RUNS or [Path(".")])
def test_3d_workers_used_the_verified_inputs(directory):
    if "3d_wind_boxcar" not in str(directory):
        return
    rows = read_rows(directory / "settings.csv", SETTINGS_FIELDS)
    cases = {}
    for row in rows:
        cases.setdefault(row["case_id"], {})[row["setting"]] = row["value"]
    for values in cases.values():
        assert values["calculation_protocol"] == "shared_mask_seven_filters"
        assert int(values["spectral_number_full_nd_filter_calls"]) == 7
        assert int(values["reference_number_full_nd_filter_calls"]) == 7
        assert int(values["number_full_nd_filter_calls_all_benchmarked_methods"]) == 7
        for key, value in values.items():
            if not key.startswith("worker_") or not value.startswith("{"):
                continue
            worker = json.loads(value)
            if worker.get("status") == "success":
                assert worker["input_sha256"] == values["input_sha256"]
                assert worker["n_outputs"] == 12
                assert worker["worker_pid"] > 0
                assert worker["calculation_protocol"] == "shared_mask_seven_filters"
                assert worker["full_nd_filter_calls"] == 7
                assert worker["filter_preparations"] == 1
                assert set(worker["preloaded_methods"]) == {
                    "dct_shared_moments", "numpy_windows", "scipy_uniform", "astropy_direct", "astropy_fft",
                }
    benchmark_path = directory / "benchmark_runs.csv"
    if benchmark_path.exists():
        timed = read_rows(benchmark_path, BENCHMARK_FIELDS)
        assert "dct_toolkit_api_validation_only" not in {row["method"] for row in timed}
        assert "dct_shared_moments" in {row["method"] for row in timed}
