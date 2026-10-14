"""The README quotes numbers from the committed result files; they must not drift apart."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()


def test_scaling_table_matches_result_file():
    data = json.loads((ROOT / "results" / "cpu-scaling.json").read_text())
    for r in data["results"]:
        row = (
            f"| {r['threads']} | {r['events_per_second'] / 1e6:.1f} | {r['speedup']:.2f} "
            f"| {r['efficiency']:.0%} |"
        )
        assert row in README, f"README scaling row out of date: {row}"
    assert data["build"]["git_commit"][:7] in README
    assert data["validated_before_measuring"] is True


def test_headline_validation_rows_match_report():
    rows = json.loads((ROOT / "results" / "validation-cpu.json").read_text())
    quoted = [
        ("v1_slab", "rd", "van de Hulst"),
        ("v1_slab", "tt", "van de Hulst"),
        ("v2_semi_infinite", "rd_plus_specular", "Giovanelli"),
        ("v2_semi_infinite", "rd", "oracle (diffuse)"),
        ("o1_semi_infinite_matched", "rd", "oracle"),
        ("v3_three_layer", "rd", "MCML"),
        ("v3_three_layer", "tt", "MCML"),
    ]
    for case, quantity, source in quoted:
        matches = [
            r
            for r in rows
            if r["case"] == case and r["quantity"] == quantity and source in r["source"]
        ]
        assert matches, (case, quantity, source)
        r = matches[0]
        z = f"{r['z']:+.2f}".replace("-", "−")
        assert f"{r['ours']:.5f} ± {r['se_ours']:.5f}" in README.replace("**", ""), (case, quantity)
        assert z in README.replace("**", ""), (case, quantity, z)


def test_all_reported_checks_passed():
    rows = json.loads((ROOT / "results" / "validation-cpu.json").read_text())
    assert len(rows) == 24
    assert all(r["passed"] for r in rows)
    assert "all 24 checks pass" in README


def test_cross_check_tables_match_result_files():
    for name in (
        "mcml-crosscheck.json",
        "mcml-crosscheck-v3-extended.json",
        "mcml-crosscheck-v3-xoshiro.json",
    ):
        data = json.loads((ROOT / "results" / name).read_text())
        for c in data["comparisons"]:
            assert f"{c['ours']:.6f} ± {c['se_ours']:.6f}" in README, (
                name,
                c["case"],
                c["quantity"],
            )
            assert f"{c['mcml']:.6f} ± {c['se_mcml']:.6f}" in README, (
                name,
                c["case"],
                c["quantity"],
            )
            z = f"{c['z']:+.2f}".replace("-", "−")
            assert z in README, (name, c["case"], c["quantity"], z)


def test_cross_check_readme_states_the_failure_and_its_limit():
    """The README must not claim agreement the data do not show."""
    data = json.loads((ROOT / "results" / "mcml-crosscheck-v3-extended.json").read_text())
    tt = next(c for c in data["comparisons"] if c["quantity"] == "tt")
    assert not tt["passed"]
    assert "too weak" in README
    assert "not established" in README
    xo = json.loads((ROOT / "results" / "mcml-crosscheck-v3-xoshiro.json").read_text())
    assert xo["original_rng"] == "xoshiro"
    assert all(c["passed"] for c in xo["comparisons"])
