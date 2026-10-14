import json
import math
from pathlib import Path

import pytest
from tissuemc_analysis import validate

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build" / "tissuemc"
REFS = json.loads((ROOT / "configs" / "validation" / "references.json").read_text())

needs_binary = pytest.mark.skipif(not BINARY.exists(), reason="engine not built (build/tissuemc)")


@pytest.fixture(scope="module")
def v1_run():
    return validate.run_engine(
        BINARY, ROOT / "configs" / "validation" / "v1_slab.mci", photons=400_000, batches=20, seed=3
    )


@needs_binary
def test_engine_json_is_complete_and_valid(v1_run):
    run = v1_run
    assert run["engine"] == "cpu" and run["precision"] == "float64"
    assert run["photons"] == 400_000
    assert len(run["batches"]) == 20
    assert sum(b["photons"] for b in run["batches"]) == 400_000
    assert len(run["a_z"]) == run["grid"]["nz"]
    assert len(run["rd_r"]) == run["grid"]["nr"]
    assert run["anomalies"] == 0


@needs_binary
def test_python_statistics_match_the_engines_own_summary(v1_run):
    assert validate.summary_matches_engine(v1_run)
    est = validate.estimates(v1_run)
    assert est["rd"].mean == pytest.approx(v1_run["summary"]["rd"]["mean"], rel=1e-12)
    assert est["tt"].se == pytest.approx(v1_run["summary"]["tt"]["se"], rel=1e-9)


@needs_binary
def test_invariants_pass_on_a_good_run(v1_run):
    results = validate.invariant_results("v1_slab", v1_run, seed=3)
    assert all(r.comparison.passed for r in results)


@needs_binary
def test_validator_can_fail(v1_run):
    """A wrong reference value must be reported as a failure, or a green run proves nothing."""
    case = {
        "checks": [
            {"quantity": "rd", "source": "deliberately wrong", "value": 0.20, "decimals": 2},
            {"quantity": "tt", "source": "deliberately wrong", "value": 0.50, "se": 0.001},
        ]
    }
    results = validate.evaluate_case("fake", case, v1_run, seed=3)
    assert [r.comparison.passed for r in results] == [False, False]


@needs_binary
def test_validator_detects_a_single_standard_error_budget_overrun(v1_run):
    est = validate.estimates(v1_run)["rd"]
    inside = {"checks": [{"quantity": "rd", "source": "x", "value": est.mean + 2 * est.se}]}
    outside = {"checks": [{"quantity": "rd", "source": "x", "value": est.mean + 4 * est.se}]}
    assert validate.evaluate_case("c", inside, v1_run, 1)[0].comparison.passed
    assert not validate.evaluate_case("c", outside, v1_run, 1)[0].comparison.passed


@needs_binary
@pytest.mark.parametrize(
    ("precision", "seeds"), [("float64", (101, 102, 103)), ("float32", (101,))]
)
def test_engine_agrees_with_published_values_across_seeds(precision, seeds):
    """V1 and the semi-infinite oracle, several seeds, at a reduced photon count.

    Each seed is an independent draw, so a systematic error would show up in all of them while
    sampling noise would not. Seeds are fixed in advance and not chosen after the fact. The
    float32 path is the one the GPU engine runs, so it gets the same check.
    """
    cases = ["v1_slab", "o1_semi_infinite_matched", "v2_semi_infinite"]
    for name in cases:
        case = REFS["cases"][name]
        for seed in seeds:
            run = validate.run_engine(
                BINARY, ROOT / "configs" / case["config"], 1_000_000, 50, seed, precision
            )
            results = validate.evaluate_case(name, case, run, seed)
            results += validate.invariant_results(name, run, seed)
            bad = [r.line() for r in results if not r.comparison.passed]
            assert not bad, f"{name} seed {seed}: " + "; ".join(bad)


@needs_binary
def test_cli_reports_bad_input_with_a_message_and_nonzero_exit(tmp_path):
    import subprocess

    bad = tmp_path / "bad.mci"
    bad.write_text("1.0\n1\nout.mco A\n1000\n0.01 0.01\n1 1 1\n1\n1.0\n1.0 1 9 1.0 0.1\n1.0\n")
    proc = subprocess.run([str(BINARY), "run", str(bad)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "g must be in (-1, 1)" in proc.stderr
    assert "bad.mci" in proc.stderr


def test_reference_file_is_well_formed():
    for name, case in REFS["cases"].items():
        assert (ROOT / "configs" / case["config"]).exists(), name
        for chk in case["checks"]:
            assert chk["quantity"] in {"rd", "tt", "rd_plus_specular"}
            assert "oracle" in chk or "value" in chk
            if "value" in chk:
                # A published value needs a way to put an error on it.
                assert {"se", "decimals"} & chk.keys(), (name, chk["source"])
                assert not math.isnan(chk["value"])
