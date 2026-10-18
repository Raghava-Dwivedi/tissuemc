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


def test_the_log_is_measurement_only_and_admits_the_runs_without_predictions():
    log = (ROOT / "LOG.md").read_text()
    assert "measurement-only" in log
    assert "none was recorded" in log  # the R0 and R1 entries keep their original wording
    assert "pending" not in log.lower()
    assert "not yet recorded" not in log  # no stale placeholder asking for a forecast


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


def test_public_docs_do_not_promise_predictions_that_were_not_made():
    assert "measurement-only" in README
    for name in (
        "README.md",
        "docs/DESIGN.md",
        "docs/rungs/R0.md",
        "docs/rungs/R1.md",
        "docs/rungs/R3.md",
    ):
        text = (ROOT / name).read_text()
        for phrase in (
            "writes each prediction",
            "meant to hold the author's prediction",
            "A prediction is recorded",
            "Write the prediction",
            "whether his prediction held",
        ):
            assert phrase not in text, (name, phrase)
    for name in ("colab_gpu.ipynb", "colab_gpu_r1.ipynb", "colab_gpu_r3.ipynb"):
        nb = json.loads((ROOT / "notebooks" / name).read_text())
        intro = "".join(nb["cells"][0]["source"]).lower()
        assert "prediction" not in intro, name


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


def test_gpu_r3_readme_numbers_match_the_committed_summary():
    data = json.loads((ROOT / "results" / "gpu-r3-summary.json").read_text())
    assert data["prediction_recorded_before_run"] is False
    assert data["rung"] == 3
    s = data["summary"]
    assert s["validation_r3"]["failed"] == []
    for key in ("cross_engine_r1", "cross_engine_r3"):
        assert s[key]["agree"] is True
        assert all(value <= limit for value, limit in s[key]["gaps"].values()), key
    assert "absorbed: total gap / standard error" in s["cross_engine_r1"]["gaps"]
    b = s["benchmark"]
    for run in b.values():
        assert run["validated_first"] is True
    readme = README.replace("**", "")
    for label, key in (("R1 (first)", "r1_first"), ("R3", "r3"), ("R1 (again)", "r1_again")):
        row = f"| {label} | {b[key]['median_kernel_s']:.3f} s | {b[key]['million_events_per_s']} |"
        assert row in readme, row
    r1_mean = (b["r1_first"]["median_kernel_s"] + b["r1_again"]["median_kernel_s"]) / 2
    slower = 100 * (b["r3"]["median_kernel_s"] / r1_mean - 1)
    assert slower > 0  # the README says R3 was slower; if the data change, the text must too
    assert f"it was about {slower:.1f}% slower" in readme
    r1_repeats = b["r1_first"]["kernel_seconds"] + b["r1_again"]["kernel_seconds"]
    assert max(r1_repeats) < min(b["r3"]["kernel_seconds"])  # "all ten R1 repeats beat all five R3"
    c1, c3 = s["counters"]["r1"], s["counters"]["r3"]
    rows = {
        "Threads launched": "Threads",
        "Registers per thread": "Registers Per Thread",
        "Avg. active threads per warp (of 32)": "Avg. Active Threads Per Warp",
        "Warp cycles per issued instruction": "Warp Cycles Per Issued Instruction",
        "Compute (SM) throughput": "Compute (SM) Throughput",
        "Memory throughput": "Memory Throughput",
        "L2 cache throughput": "L2 Cache Throughput",
        "Achieved occupancy": "Achieved Occupancy",
    }
    for label, name in rows.items():
        v1, v3 = c1[name], c3[name]
        if name == "Threads":
            v1, v3 = v1.split()[0], v3.split()[0]
        if name == "Registers Per Thread":
            v1, v3 = v1.split()[0], v3.split()[0]
        assert f"| {label} | {v1} | {v3} |" in readme, label
    assert c3["Threads"].split()[0] == "40,960"  # 40 multiprocessors x 1,024 threads
    # The two quoted durations and the 0.4% claim.
    d1 = int(c1["Duration"].split()[0].replace(",", ""))
    d3 = int(c3["Duration"].split()[0].replace(",", ""))
    assert abs(d3 / d1 - 1) < 0.005
    assert c1["Duration"].split()[0] in readme and c3["Duration"].split()[0] in readme


def test_gpu_r3_log_matches_the_committed_summary():
    log = (ROOT / "LOG.md").read_text().replace("**", "")
    b = json.loads((ROOT / "results" / "gpu-r3-summary.json").read_text())["summary"]["benchmark"]
    for label, key in (("R1", "r1_first"), ("R3", "r3"), ("R1 again", "r1_again")):
        run = b[key]
        text = f"{label} {run['median_kernel_s']:.3f} s ({run['million_events_per_s']}"
        assert text in log, text
    assert "did not hold for run time" in log


def _diagnosis_rows():
    """Metric -> (R1 value, R3 value) as printed, from the saved diagnostic digest."""
    import re

    data = json.loads((ROOT / "results" / "gpu-r1-r3-diagnosis.json").read_text())
    assert data["photons"] == 500000 and data["rungs"] == {"A": 1, "B": 3}
    rows = {}
    for line in data["digest"].splitlines():
        m = re.match(r"\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$", line)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(3))
    return rows


def _num(text):
    return float(text.replace(",", ""))


def test_diagnosis_readme_numbers_match_the_saved_digest():
    rows = _diagnosis_rows()
    readme = README.replace("**", "")
    for label, name in (
        (
            "Issue slots used (% of peak while active)",
            "smsp__issue_active.avg.pct_of_peak_sustained_active",
        ),
        ("Eligible warps per cycle", "smsp__warps_eligible.avg.per_cycle_active"),
        (
            "Average active lanes per instruction",
            "smsp__thread_inst_executed_per_inst_executed.ratio",
        ),
        ("L1 hit rate, local loads (%)", "l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct"),
        ("L1 hit rate, local stores (%)", "l1tex__t_sector_pipe_lsu_mem_local_op_st_hit_rate.pct"),
        ("Register-spill requests", "derived__local_spilling_requests"),
        (
            "Total wait (average warp latency, cycles)",
            "smsp__average_warp_latency_per_inst_issued.ratio",
        ),
    ):
        assert f"| {label} | {rows[name][0]} | {rows[name][1]} |" in readme, label
    pre, suf = "smsp__average_warps_issue_stalled_", "_per_issue_active.ratio"
    for reason in ("long_scoreboard", "lg_throttle"):
        a, b = rows[pre + reason + suf]
        assert f"| {a} | {b} |" in readme, reason
    for label, name in (
        ("Warp instructions executed (billions)", "smsp__inst_executed.sum"),
        ("Local-memory load instructions (billions)", "smsp__sass_inst_executed_op_local_ld.sum"),
        ("Local-memory store instructions (billions)", "smsp__sass_inst_executed_op_local_st.sum"),
        ("Local-memory sectors requested (billions)", "memory_l2_theoretical_sectors_local"),
    ):
        a, b = (_num(v) / 1e9 for v in rows[name])
        assert f"| {label} | {a:.2f} | {b:.2f} |" in readme, label


def test_diagnosis_claims_follow_from_the_saved_digest():
    rows = _diagnosis_rows()
    readme, log = README.replace("**", ""), (ROOT / "LOG.md").read_text().replace("**", "")
    pre, suf = "smsp__average_warps_issue_stalled_", "_per_issue_active.ratio"
    g = lambda n: tuple(_num(v) for v in rows[n])  # noqa: E731
    lat = g("smsp__average_warp_latency_per_inst_issued.ratio")
    ls, lg = g(pre + "long_scoreboard" + suf), g(pre + "lg_throttle" + suf)
    added = 100 * ((ls[1] - ls[0]) + (lg[1] - lg[0])) / (lat[1] - lat[0])
    assert added > 99  # "accounts for 99.5% of the increase"
    assert f"{added:.1f}% of the" in readme and f"{added:.1f}% of the increase" in log
    # The listed stall reasons add up to the reported latency, so the breakdown is complete.
    for k in (0, 1):
        total = sum(g(n)[k] for n in rows if n.startswith(pre))
        assert abs(total - lat[k]) < 0.1 * lat[k] / 10, (k, total, lat[k])
    # "Neither rung issues any global or shared loads or stores."
    for n in (
        "smsp__sass_inst_executed_op_global_ld.sum",
        "smsp__sass_inst_executed_op_global_st.sum",
        "smsp__sass_inst_executed_op_shared_ld.sum",
        "smsp__sass_inst_executed_op_shared_st.sum",
        "smsp__inst_executed_op_global_red.sum",
    ):
        assert g(n) == (0.0, 0.0), n
    # "The same number of local loads": within 5%.
    ld = g("smsp__sass_inst_executed_op_local_ld.sum")
    assert abs(ld[1] / ld[0] - 1) < 0.05
    # "About 7% more thread-instructions" and the near-equal run-time estimate.
    inst, lanes = (
        g("smsp__inst_executed.sum"),
        g("smsp__thread_inst_executed_per_inst_executed.ratio"),
    )
    work = (inst[1] * lanes[1]) / (inst[0] * lanes[0])
    assert 1.05 < work < 1.09
    assert f"{work:.2f}× the thread-instructions" in readme
    assert rows["sm__warps_active.avg.pct_of_peak_sustained_active"] == ("86.85", "99.20")
    warps = tuple(v / 100 for v in g("sm__warps_active.avg.pct_of_peak_sustained_active"))
    cycles = (inst[1] * lat[1] / warps[1]) / (inst[0] * lat[0] / warps[0])
    assert abs(cycles - 1) < 0.02
