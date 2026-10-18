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


def test_gpu_r5_r6_readme_numbers_match_the_committed_summary():
    data = json.loads((ROOT / "results" / "gpu-r5-r6-summary.json").read_text())
    assert data["prediction_recorded_before_run"] is False
    s = data["summary"]
    for key in ("validation_r5", "validation_r6"):
        assert s[key]["checks"] == 24 and s[key]["failed"] == []
    for key in ("cross_engine_r5", "cross_engine_r6"):
        assert s[key]["agree"] is True
        assert all(value <= limit for value, limit in s[key]["gaps"].values()), key
    # "R_d, T_t and event-count gaps are identical to R1's to every printed digit."
    r1_gaps = json.loads((ROOT / "results" / "gpu-r3-summary.json").read_text())["summary"][
        "cross_engine_r1"
    ]["gaps"]
    for key in ("cross_engine_r5", "cross_engine_r6"):
        for name in (
            "rd: total gap / standard error",
            "rd: worst per-batch relative gap",
            "tt: total gap / standard error",
            "tt: worst per-batch relative gap",
            "events: relative gap",
        ):
            assert s[key]["gaps"][name] == r1_gaps[name], (key, name)
    b = s["benchmark"]
    assert all(run["validated_first"] for run in b.values())
    readme = README.replace("**", "")
    labels = (
        ("R1 (first)", "r1_first"),
        ("R3", "r3"),
        ("R5", "r5"),
        ("R6", "r6"),
        ("R1 (again)", "r1_again"),
    )
    for label, key in labels:
        row = f"| {label} | {b[key]['median_kernel_s']:.3f} s | {b[key]['million_events_per_s']} |"
        assert row in readme, row
    t = {k: v["median_kernel_s"] for k, v in b.items()}
    pct = lambda a, base: 100 * (a / base - 1)  # noqa: E731
    # The claims in the text: R5 and R6 equal R1 to 0.1%, R3 slower, R6 faster than R3.
    assert abs(pct(t["r5"], t["r1_first"])) < 0.1 and abs(pct(t["r6"], t["r1_first"])) < 0.1
    assert f"R3 was {pct(t['r3'], t['r1_first']):.1f}% slower than R1" in readme
    assert f"is {pct(t['r3'], t['r6']):.1f}% faster than R3" in readme
    assert f"{pct(t['r1_again'], t['r1_first']):.1f}% slower than at the start" in readme
    for key in ("r1_first", "r5", "r6"):  # "five repeats ... span only 0.02%, 0.09% and 0.02%"
        ks = b[key]["kernel_seconds"]
        assert 100 * (max(ks) / min(ks) - 1) < 0.1, key
    spans = [
        100 * (max(b[k]["kernel_seconds"]) / min(b[k]["kernel_seconds"]) - 1)
        for k in ("r1_first", "r5", "r6")
    ]
    assert f"{spans[0]:.2f}%, {spans[1]:.2f}% and {spans[2]:.2f}%" in readme
    ak, pk = "absorbed: total gap / standard error", "absorbed: worst per-batch relative gap"
    g5, g6 = s["cross_engine_r5"]["gaps"], s["cross_engine_r6"]["gaps"]
    assert f"{g5[ak][0]:.4f} standard errors in total (R5) and {g6[ak][0]:.4f} (R6)" in readme
    assert f"{g5[pk][0]:.1e} and {g6[pk][0]:.1e}" in readme


def test_gpu_r5_r6_log_matches_the_committed_summary():
    log = (ROOT / "LOG.md").read_text().replace("**", "")
    b = json.loads((ROOT / "results" / "gpu-r5-r6-summary.json").read_text())["summary"][
        "benchmark"
    ]
    for label, key in (
        ("R1", "r1_first"),
        ("R3", "r3"),
        ("R5", "r5"),
        ("R6", "r6"),
        ("R1 again", "r1_again"),
    ):
        run = b[key]
        assert f"{label} {run['median_kernel_s']:.3f} s ({run['million_events_per_s']}" in log, (
            label
        )
    assert "Not supported as a statement about the kernel" in log
    assert "reopens a question the R1 entry closed too early" in log
    assert "no counters yet for R5 or R6" in log


def test_the_docs_do_not_claim_a_limiter_the_data_do_not_support():
    """Local memory slowed R3 but did not limit R1 (R5); the text must not say otherwise."""
    design = (ROOT / "docs" / "DESIGN.md").read_text()
    for text in (README, design):
        assert "The limit on this kernel is the latency of per-thread local memory" not in text
        assert "are each not the main limit on run time" not in text
    assert "After R5 and R6" in design and "do not identify" in design
    assert "not yet identified" in README  # the floor R1, R5 and R6 share


def _digest_rows(name):
    import re

    rows = {}
    for line in json.loads((ROOT / "results" / name).read_text())["digest"].splitlines():
        m = re.match(r"\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$", line)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(3))
    return rows


def test_r6_counters_readme_numbers_match_the_saved_digest():
    data = json.loads((ROOT / "results" / "gpu-r3-r6-digest.json").read_text())
    assert data["rungs"] == {"A": 3, "B": 6} and data["photons"] == 500000
    rows, readme = _digest_rows("gpu-r3-r6-digest.json"), README.replace("**", "")
    pre, suf = "smsp__average_warps_issue_stalled_", "_per_issue_active.ratio"
    for label, name in (
        (
            "Average warp latency per issued instruction (cycles)",
            "smsp__average_warp_latency_per_inst_issued.ratio",
        ),
        ("Waiting for room in the load/store queue (LG throttle)", pre + "lg_throttle" + suf),
        ("Waiting for a load result (long scoreboard)", pre + "long_scoreboard" + suf),
        (
            "Busiest L2 slice, throughput (% of peak)",
            "lts__throughput.max.pct_of_peak_sustained_elapsed",
        ),
        (
            "Average over L2 slices, throughput (% of peak)",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        (
            "Busiest L2 slice, atomic unit busy (%)",
            "lts__d_atomic_input_cycles_active.max.pct_of_peak_sustained_elapsed",
        ),
    ):
        assert f"| {label} | {rows[name][0]} | {rows[name][1]} |" in readme, label
    num = lambda s: float(s.replace(",", ""))  # noqa: E731
    g = lambda n: tuple(num(v) for v in rows[n])  # noqa: E731
    # The claims in the text.
    for n in (
        "smsp__sass_inst_executed_op_local_ld.sum",
        "smsp__sass_inst_executed_op_local_st.sum",
    ):
        assert g(n)[1] == 0 and g(n)[0] > 0, n  # local memory is gone in R6, present in R3
    for n in (
        "smsp__sass_inst_executed_op_global_ld.sum",
        "smsp__sass_inst_executed_op_global_st.sum",
    ):
        assert g(n) == (0.0, 0.0), n
    lat, lg, ls = (
        g("smsp__average_warp_latency_per_inst_issued.ratio"),
        g(pre + "lg_throttle" + suf),
        g(pre + "long_scoreboard" + suf),
    )
    assert lat[1] >= lat[0] * 0.99  # the wait did not shrink
    assert abs((ls[0] - ls[1]) - (lg[1] - lg[0])) < 3  # long scoreboard was replaced by LG throttle
    assert f"{100 * lg[1] / lat[1]:.0f}% of their wait" in readme
    red, atom = (
        g("smsp__inst_executed_op_global_red.sum"),
        g("smsp__inst_executed_op_generic_atom_dot_alu.sum"),
    )
    a3, a6 = (red[0] + atom[0]) / 1e9, (red[1] + atom[1]) / 1e9
    assert abs(a6 / a3 - 1) < 0.01  # "equals R3's"
    assert f"{a6:.3f} billion warp-level instructions" in readme
    mx, av = (
        g("lts__throughput.max.pct_of_peak_sustained_elapsed"),
        g("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    )
    assert abs(mx[0] - mx[1]) < 1  # the hot slice did not change when the local memory went
    assert f"({mx[1] / av[1]:.0f}×)" in readme
    cyc = g("sm__cycles_active.max"), g("sm__cycles_active.avg")
    assert f"active {100 * (cyc[0][1] / cyc[1][1] - 1):.1f}% longer than the average" in readme


def test_the_log_records_the_r6_counters_and_the_correction():
    log = (ROOT / "LOG.md").read_text()
    assert "R6 counters: the wait moved to the atomic path" in log
    assert "Suggested, not tested" in log  # the reading is not presented as a finding
    assert "This corrects the R1 entry's reading" in log


def test_ablation_readme_numbers_match_the_saved_summary():
    data = json.loads((ROOT / "results" / "gpu-r6-noprofiles-summary.json").read_text())
    assert data["diagnostic"] is True and data["prediction_recorded_before_run"] is False
    s = data["summary"]
    assert s["cross_engine_r6_noprofiles"]["agree"] is True
    assert all(v <= lim for v, lim in s["cross_engine_r6_noprofiles"]["gaps"].values())
    b = s["benchmark"]
    # The diagnostic is labelled as such in the data, and only it.
    assert (
        b["r6_noprofiles"]["profiles"] is False and b["r6_noprofiles"]["validated_first"] is False
    )
    assert all(b[k]["profiles"] is True for k in ("r6_first", "r1", "r6_again"))
    t = {k: v["median_kernel_s"] for k, v in b.items()}
    ev = {k: v["million_events_per_s"] for k, v in b.items()}
    readme = README.replace("**", "")
    for label, key in (
        ("R6 (first)", "r6_first"),
        ("R6 without the profile atomics (diagnostic)", "r6_noprofiles"),
        ("R1", "r1"),
        ("R6 (again)", "r6_again"),
    ):
        assert f"| {label} | {t[key]:.3f} s | {ev[key]} |" in readme, label
    # "The same work was done": events implied by rate times time agree to 0.1%.
    implied = {k: ev[k] * t[k] for k in t}
    assert max(implied.values()) / min(implied.values()) < 1.001
    speed = t["r6_first"] / t["r6_noprofiles"]
    share = 100 * (1 - t["r6_noprofiles"] / t["r6_first"])
    assert f"{speed:.1f} times faster" in readme and f"are {share:.0f}% of R6's run time" in readme
    # R6 steady across sessions: the previous session's value.
    prev = json.loads((ROOT / "results" / "gpu-r5-r6-summary.json").read_text())["summary"]
    assert abs(prev["benchmark"]["r6"]["median_kernel_s"] / t["r6_first"] - 1) < 0.001
    # The README must say the diagnostic is not a result, and not present it as a speedup.
    assert "It is a diagnostic, not a result" in readme
    assert "the target, not the result" in readme


def test_the_log_records_the_ablation_and_does_not_oversell_it():
    log = (ROOT / "LOG.md").read_text().replace("**", "")
    b = json.loads((ROOT / "results" / "gpu-r6-noprofiles-summary.json").read_text())["summary"][
        "benchmark"
    ]
    assert f"R6 without the profile atomics {b['r6_noprofiles']['median_kernel_s']:.3f} s" in log
    assert "Was the reading right? Yes, by intervention" in log
    assert "Not shown: that a real kernel can reach" in log
    assert (
        "the diagnostic is the target, not the result" in log.lower()
        or "target, not the result" in log
    )


def test_the_ablation_is_not_presented_as_general():
    """The 12x depends on this benchmark's 40-bin depth profile; the docs must say so."""
    config = (ROOT / "configs" / "bench" / "tissue_g09.mci").read_text()
    assert "40 50 1" in config  # nz = 40, nr = 50: the premise of the explanation
    assert "40 bins" in README and "was not measured" in README.replace("\n", " ")
    assert "coarse depth profile" in (ROOT / "docs" / "rungs" / "R7.md").read_text()
