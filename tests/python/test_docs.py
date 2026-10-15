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


def test_gpu_r0_readme_numbers_match_the_committed_summary():
    data = json.loads((ROOT / "results" / "gpu-r0-summary.json").read_text())
    assert data["prediction_recorded_before_run"] is False
    summary = data["summary"]
    gaps = {k: v[0] for k, v in summary["cross_engine"]["gaps"].items()}
    b = summary["benchmark"]
    for text in (
        f"{b['million_events_per_s']} million events/s",
        f"{b['million_photons_per_s']} million photons/s",
        f"median kernel time {b['median_kernel_s']:.2f} s",
        f"{gaps['rd: total gap / standard error']:.4f}",
        f"{gaps['tt: total gap / standard error']:.4f}",
        f"{gaps['rd: worst per-batch relative gap']:.6f}",
        f"{gaps['tt: worst per-batch relative gap']:.6f}",
        f"{gaps['events: relative gap']:.7f}",
        f"all {summary['validation']['checks']} checks pass on the GPU engine",
    ):
        assert text in README, text
    assert summary["validation"]["failed"] == []
    assert summary["cross_engine"]["agree"] is True
    assert summary["benchmark"]["validated_first"] is True


def test_the_log_does_not_claim_a_prediction_that_was_not_made():
    log = (ROOT / "LOG.md").read_text()
    assert "none was recorded" in log
    assert "pending" not in log.lower()


def test_gpu_r1_readme_numbers_match_the_committed_summary():
    data = json.loads((ROOT / "results" / "gpu-r1-summary.json").read_text())
    assert data["prediction_recorded_before_run"] is False
    assert data["rung"] == 1
    s = data["summary"]
    assert s["validation_r1"]["failed"] == []
    assert s["cross_engine_r1"]["agree"] is True
    b = s["benchmark"]
    for run in b.values():
        assert run["validated_first"] is True
    for label, key in (("R0 (first)", "r0_first"), ("R1", "r1"), ("R0 (again)", "r0_again")):
        row = f"| {label} | {b[key]['median_kernel_s']:.3f} s | {b[key]['million_events_per_s']} |"
        assert row.replace("**", "") in README.replace("**", ""), row
    r0_mean = (b["r0_first"]["median_kernel_s"] + b["r0_again"]["median_kernel_s"]) / 2
    speedup = r0_mean / b["r1"]["median_kernel_s"]
    assert f"R1 is about {round(100 * (speedup - 1))}% faster than R0" in README.replace("**", "")
    c0, c1 = s["counters"]["r0"], s["counters"]["r1"]
    for name in (
        "Avg. Active Threads Per Warp",
        "Compute (SM) Throughput",
        "Memory Throughput",
        "L2 Cache Throughput",
        "Warp Cycles Per Issued Instruction",
        "Achieved Occupancy",
    ):
        assert f"| {c0[name]} | {c1[name]} |" in README, name


def test_gpu_r1_log_matches_the_committed_summary():
    log = (ROOT / "LOG.md").read_text().replace("**", "")
    b = json.loads((ROOT / "results" / "gpu-r1-summary.json").read_text())["summary"]["benchmark"]
    for label, key in (("R0", "r0_first"), ("R1", "r1"), ("R0 again", "r0_again")):
        run = b[key]
        text = f"{label} {run['median_kernel_s']:.3f} s ({run['million_events_per_s']}"
        assert text in log, text


def test_readme_does_not_claim_predictions_the_log_says_were_not_made():
    log = (ROOT / "LOG.md").read_text()
    if log.count("none was recorded") > 0:
        assert "writes each prediction before it is tested" not in README
        assert "none was recorded" in README or "for R0 and R1 none was recorded" in README


def test_saved_r1_notebook_outputs_match_the_committed_summary():
    """The notebook saved from Colab keeps the printed output; it must agree with the summary."""
    nb = json.loads((ROOT / "notebooks" / "colab_gpu_r1.ipynb").read_text())
    printed = "\n".join(
        "".join(o.get("text", []))
        for cell in nb["cells"]
        for o in cell.get("outputs", [])
        if o.get("output_type") == "stream"
    )
    assert not any(
        o.get("output_type") == "error" for cell in nb["cells"] for o in cell.get("outputs", [])
    )
    s = json.loads((ROOT / "results" / "gpu-r1-summary.json").read_text())["summary"]
    for key in ("r0_first", "r1", "r0_again"):
        events = s["benchmark"][key]["million_events_per_s"]
        assert f"{events} million events/s (median kernel)" in printed, key
    for name, v0 in s["counters"]["r0"].items():
        line = next(x for x in printed.splitlines() if x.strip().startswith(name))
        assert v0.replace(" cycle", "") in line, (name, line)
