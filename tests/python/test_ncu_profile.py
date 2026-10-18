import json
from pathlib import Path

import pytest
from tissuemc_analysis import ncu_profile

HEADER = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size",'
    '"Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"'
)


def row(kernel_id, section, metric, unit, value):
    return (
        f'"{kernel_id}","100","tissuemc","host","photon_kernel","1","7","(128, 1, 1)",'
        f'"(1563, 1, 1)","0","7.5","{section}","{metric}","{unit}","{value}"'
    )


SAMPLE = "\n".join(
    [
        "run 1: 200000 photons, 50 batches, 1.23 s (cuda, float32)",  # program output first
        "==PROF== Connected to process 100",
        HEADER,
        row(0, "GPU Speed Of Light Throughput", "Duration", "msecond", "612.31"),
        row(0, "GPU Speed Of Light Throughput", "Compute (SM) Throughput", "%", "31.20"),
        row(0, "Warp State Statistics", "Avg. Active Threads Per Warp", "", "5.40"),
        row(0, "Occupancy", "Achieved Occupancy", "%", "60.10"),
        row(1, "Occupancy", "Achieved Occupancy", "%", "99.99"),  # a second kernel: must be ignored
    ]
)


def test_parse_reads_sections_metrics_and_units():
    out = ncu_profile.parse_ncu_csv(SAMPLE)
    assert out["GPU Speed Of Light Throughput"]["Duration"] == {
        "value": "612.31",
        "unit": "msecond",
    }
    assert out["Warp State Statistics"]["Avg. Active Threads Per Warp"]["value"] == "5.40"
    assert out["Occupancy"]["Achieved Occupancy"]["value"] == "60.10"  # first kernel only


def test_parse_rejects_text_without_a_header():
    with pytest.raises(ValueError, match="header"):
        ncu_profile.parse_ncu_csv("just program output\n")


def test_the_launch_size_is_recorded():
    """The persistent kernel's lane count comes from the hardware, not from an assumption."""
    assert "LaunchStats" in ncu_profile.SECTIONS
    metrics = {metric for section, metric, _ in ncu_profile.KEY_METRICS if "Launch" in section}
    assert {"Threads", "Registers Per Thread"} <= metrics


def test_every_key_metric_names_a_requested_section_family():
    names = {section for section, _, _ in ncu_profile.KEY_METRICS}
    assert "Warp State Statistics" in names
    assert "GPU Speed Of Light Throughput" in names
    assert all(meaning for _, _, meaning in ncu_profile.KEY_METRICS)


# The raw page is WIDE: one row per kernel, one column per metric. The first versions of these
# fixtures assumed one row per metric (and then a "Section Name" column), so the tests passed
# while the real thing failed on Colab, twice. The launch columns below are copied from a real
# run; the metric names are real names from the same file.
LAUNCH = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream"]
LAUNCH += ["Block Size", "Grid Size", "Device", "CC"]
STALL = "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
WAIT = "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
INST = "smsp__inst_executed.sum"
LOCAL_LD = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"
REGS = "launch__registers_per_thread"
THREADS = "launch__thread_count"
LANES = "smsp__thread_inst_executed_per_inst_executed.ratio"
SPILL = "derived__local_spilling_requests"


def csv_line(cells):
    return ",".join(f'"{c}"' for c in cells)


def wide(metrics, units=True, second_kernel=False):
    """A raw-page CSV: header, optionally a units row, then the kernel row(s)."""
    names = list(metrics)
    lines = ["==PROF== Connected to process 100", csv_line(LAUNCH + names)]
    if units:
        lines.append(csv_line([""] * len(LAUNCH) + [metrics[n][1] for n in names]))
    ident = ["0", "100", "tissuemc", "host", "photon_kernel", "1", "7", "(128, 1, 1)"]
    ident += ["(1563, 1, 1)", "0", "7.5"]
    lines.append(csv_line(ident + [metrics[n][0] for n in names]))
    if second_kernel:
        lines.append(csv_line(["1"] + ident[1:] + ["999"] * len(names)))
    lines.append("==PROF== Disconnected from process 100")
    return "\n".join(lines)


R1_METRICS = {
    STALL: ("1.20", "warp"),
    WAIT: ("0.40", "warp"),
    INST: ("1,200,000,000", "inst"),
    LOCAL_LD: ("50,000", "sector"),
    REGS: ("48", "register/thread"),
    LANES: ("4.74", ""),
    SPILL: ("", ""),  # present in the file but empty: must be skipped, not crash
}
R3_METRICS = {
    STALL: ("6.00", "warp"),
    WAIT: ("0.40", "warp"),
    INST: ("260,000,000", "inst"),
    REGS: ("37", "register/thread"),
    THREADS: ("40,960", "thread"),  # only in B
    LANES: ("21.61", ""),
}
RAW_R1 = wide(R1_METRICS)
RAW_R3 = wide(R3_METRICS)


def test_wide_raw_pages_parse_with_units():
    metrics = ncu_profile.flatten(ncu_profile.parse_ncu_csv(RAW_R1))
    assert metrics[INST] == {"value": "1,200,000,000", "unit": "inst"}
    assert metrics[REGS]["unit"] == "register/thread"
    assert SPILL not in metrics  # empty cell
    assert "Kernel Name" not in metrics and "ID" not in metrics  # launch columns are not metrics


def test_wide_raw_pages_parse_without_a_units_row():
    metrics = ncu_profile.flatten(ncu_profile.parse_ncu_csv(wide(R1_METRICS, units=False)))
    assert metrics[INST] == {"value": "1,200,000,000", "unit": ""}
    assert metrics[STALL]["value"] == "1.20"


def test_wide_raw_pages_keep_only_the_first_kernel():
    text = wide(R1_METRICS, second_kernel=True)
    metrics = ncu_profile.flatten(ncu_profile.parse_ncu_csv(text))
    assert metrics[INST]["value"] == "1,200,000,000"


def test_a_wide_page_with_no_kernel_row_is_an_error():
    text = csv_line(LAUNCH + [INST]) + "\n"
    with pytest.raises(ValueError, match="no kernel row"):
        ncu_profile.parse_ncu_csv(text)


def test_unexpected_columns_are_reported_briefly_not_as_a_keyerror():
    text = '"Something","Else"\n"0","x"\n"ID","Not Enough"\n'
    with pytest.raises(ValueError, match="no Nsight Compute CSV header"):
        ncu_profile.parse_ncu_csv("nothing here\n")
    wide_names = [f"metric_{i}" for i in range(1000)]
    header = csv_line(["Process ID", "ID"] + wide_names)  # starts with the wrong column
    with pytest.raises(ValueError) as err:
        ncu_profile.parse_ncu_csv('"ID"' + header[len('"Process ID"') :].replace('"ID"', '"X"', 1))
    assert len(str(err.value)) < 600, "a thousand column names must not be dumped"
    assert text  # (kept so the fixture above documents the shape of a bad file)


def test_digest_puts_the_biggest_stall_first_and_gives_the_ratio():
    a = ncu_profile.flatten(ncu_profile.parse_ncu_csv(RAW_R1))
    b = ncu_profile.flatten(ncu_profile.parse_ncu_csv(RAW_R3))
    text = "\n".join(ncu_profile.digest(a, b))
    stalls = text.split("Issue and latency")[0]
    assert stalls.index("long_scoreboard") < stalls.index("_wait_")  # larger first
    long_row = next(ln for ln in text.splitlines() if "long_scoreboard" in ln)
    assert long_row.split()[-1] == "5.00"  # 6.00 / 1.20
    inst_row = next(ln for ln in text.splitlines() if INST in ln)
    assert inst_row.split()[-1] == "0.22"  # 260e6 / 1200e6, thousands separators handled
    lanes_row = next(ln for ln in text.splitlines() if LANES in ln)
    assert lanes_row.split()[-3:] == ["4.74", "21.61", "4.56"]
    local_row = next(ln for ln in text.splitlines() if "mem_local_op_ld" in ln)
    assert local_row.split()[-2:] == ["-", "-"]  # present in A only: no ratio, "-" for B
    assert any(THREADS in ln for ln in text.splitlines())  # present in B only


def test_the_digest_finds_every_group_in_real_metric_names():
    names = [
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__average_warp_latency_per_inst_issued.ratio",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__sass_inst_executed_op_local_ld.sum",
        "smsp__sass_inst_executed_op_local_st.sum",
        "smsp__inst_executed_op_branch.sum",
        "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
        "l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct",
        "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
        "derived__local_spilling_requests_pct",
        "lts__throughput.max.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__d_atomic_input_cycles_active.max.pct_of_peak_sustained_elapsed",
        "lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
        "lts__t_sector_hit_rate.pct",
        "sm__cycles_active.max",
        "sm__cycles_active.avg",
        "gpu__time_duration.sum",
        "launch__occupancy_limit_registers",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ]
    a = {n: {"value": "2", "unit": ""} for n in names}
    b = {n: {"value": "4", "unit": ""} for n in names}
    text = "\n".join(ncu_profile.digest(a, b))
    assert "no matching counters" not in text
    for n in names:
        assert n in text, n


def test_digest_falls_back_to_any_counter_that_mentions_stall():
    a = {"smsp__warp_stall_something.pct": {"value": "10", "unit": "%"}}
    b = {"smsp__warp_stall_something.pct": {"value": "30", "unit": "%"}}
    lines = ncu_profile.digest(a, b)
    assert any("warp_stall_something" in ln and ln.split()[-1] == "3.00" for ln in lines)


def test_digest_says_so_when_nothing_matches():
    assert any("no matching counters" in ln for ln in ncu_profile.digest({}, {}))


def test_digest_command_reads_two_saved_files(tmp_path, capsys):
    paths = []
    for name, text in (("a.json", RAW_R1), ("b.json", RAW_R3)):
        p = tmp_path / name
        metrics = ncu_profile.flatten(ncu_profile.parse_ncu_csv(text))
        p.write_text(json.dumps({"rung": 1, "photons": 1, "full": True, "metrics": metrics}))
        paths.append(str(p))
    assert ncu_profile.main(["--digest", *paths]) == 0
    out = capsys.readouterr().out
    assert "A = a.json" in out and "long_scoreboard" in out


def _fake_ncu(monkeypatch, text, calls):
    import subprocess

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[cmd.index("--log-file") + 1]).write_text(text)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ncu_profile, "find_ncu", lambda: "ncu")
    monkeypatch.setattr(ncu_profile.subprocess, "run", fake_run)


def test_the_raw_csv_is_kept_even_when_it_cannot_be_parsed(tmp_path, monkeypatch):
    """A full collection takes minutes; a parsing problem must not throw it away."""
    _fake_ncu(monkeypatch, '"ID","Something Else"\n"x","y"\n', [])
    keep = tmp_path / "out" / "r1.ncu.csv"
    with pytest.raises(ValueError):
        ncu_profile.profile(Path("b"), Path("c"), 1, 1000, 10, full=True, keep_csv=keep)
    assert keep.read_text().startswith('"ID","Something Else"')


def test_a_full_profile_uses_the_raw_page_and_the_full_set(monkeypatch):
    calls = []
    _fake_ncu(monkeypatch, RAW_R1, calls)
    sections = ncu_profile.profile(Path("b"), Path("c"), 3, 1000, 10, full=True)
    cmd = calls[0]
    assert cmd[cmd.index("--page") + 1] == "raw"
    assert cmd[cmd.index("--set") + 1] == "full"
    assert "--section" not in cmd
    assert cmd[-2:] == ["--rung", "3"]
    assert ncu_profile.flatten(sections)[REGS]["value"] == "48"


def test_reuse_csv_parses_the_saved_file_without_profiling_again(tmp_path, monkeypatch):
    calls = []
    _fake_ncu(monkeypatch, "SHOULD NOT BE READ", calls)
    keep = tmp_path / "r3.ncu.csv"
    keep.write_text(RAW_R3)
    sections = ncu_profile.profile(
        Path("b"), Path("c"), 3, 1000, 10, full=True, keep_csv=keep, reuse_csv=True
    )
    assert calls == []  # ncu was not run
    assert ncu_profile.flatten(sections)[THREADS]["value"] == "40,960"
    # Without the file, --reuse-csv falls back to collecting.
    keep.unlink()
    _fake_ncu(monkeypatch, RAW_R3, calls)
    ncu_profile.profile(Path("b"), Path("c"), 3, 1000, 10, full=True, keep_csv=keep, reuse_csv=True)
    assert len(calls) == 1 and keep.exists()


def test_the_full_command_writes_json_and_reuses_the_csv(tmp_path, monkeypatch, capsys):
    out = tmp_path / "r1.json"
    out.with_suffix(".ncu.csv").write_text(RAW_R1)
    monkeypatch.setattr(ncu_profile.validate, "find_root", lambda: tmp_path)
    args = ["--rung", "1", "--full", "--reuse-csv", "--photons", "10", "--output", str(out)]
    assert ncu_profile.main(args) == 0
    saved = json.loads(out.read_text())
    assert saved["full"] is True and saved["metrics"][INST]["unit"] == "inst"
    assert "counters collected" in capsys.readouterr().out


def test_the_profiler_passes_the_diagnostic_flag_through(monkeypatch):
    calls = []
    _fake_ncu(monkeypatch, RAW_R1, calls)
    ncu_profile.profile(Path("b"), Path("c"), 6, 1000, 10, full=True)
    ncu_profile.profile(Path("b"), Path("c"), 6, 1000, 10, full=True, profiles=False)
    assert "--no-profiles" not in calls[0]
    assert calls[1][-3:] == ["--rung", "6", "--no-profiles"]
