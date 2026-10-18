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


# The raw page has no "Section Name" column. The first version of these fixtures assumed it did,
# so the tests passed while the real thing failed with a KeyError on Colab.
RAW_HEADER = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size",'
    '"Grid Size","Device","CC","Metric Name","Metric Unit","Metric Value"'
)


def raw_row(kernel_id, metric, unit, value):
    return (
        f'"{kernel_id}","100","tissuemc","host","photon_kernel","1","7","(128, 1, 1)",'
        f'"(1563, 1, 1)","0","7.5","{metric}","{unit}","{value}"'
    )


STALL = "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
WAIT = "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
RAW_R1 = "\n".join(
    [
        "==PROF== Connected",
        RAW_HEADER,
        raw_row(0, STALL, "warp", "1.20"),
        raw_row(0, WAIT, "warp", "0.40"),
        raw_row(0, "smsp__inst_executed.sum", "inst", "1,200,000,000"),
        raw_row(0, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "sector", "50,000"),
        raw_row(0, "launch__registers_per_thread", "register/thread", "48"),
        raw_row(1, "smsp__inst_executed.sum", "inst", "9"),  # a second kernel: must be ignored
    ]
)
RAW_R3 = "\n".join(
    [
        RAW_HEADER,
        raw_row(0, STALL, "warp", "6.00"),
        raw_row(0, WAIT, "warp", "0.40"),
        raw_row(0, "smsp__inst_executed.sum", "inst", "260,000,000"),
        raw_row(0, "launch__registers_per_thread", "register/thread", "37"),
        raw_row(0, "launch__thread_count", "thread", "40,960"),  # only in B
    ]
)


def test_raw_pages_parse_without_section_names():
    metrics = ncu_profile.flatten(ncu_profile.parse_ncu_csv(RAW_R1))
    assert metrics["smsp__inst_executed.sum"]["value"] == "1,200,000,000"  # first kernel only
    assert metrics["launch__registers_per_thread"]["unit"] == "register/thread"


def test_digest_puts_the_biggest_stall_first_and_gives_the_ratio():
    a = ncu_profile.flatten(ncu_profile.parse_ncu_csv(RAW_R1))
    b = ncu_profile.flatten(ncu_profile.parse_ncu_csv(RAW_R3))
    text = "\n".join(ncu_profile.digest(a, b))
    stalls = text.split("Instructions executed")[0]
    assert stalls.index("long_scoreboard") < stalls.index("_wait_")  # larger first
    long_row = next(ln for ln in text.splitlines() if "long_scoreboard" in ln)
    assert long_row.split()[-1] == "5.00"  # 6.00 / 1.20
    inst_row = next(ln for ln in text.splitlines() if "smsp__inst_executed.sum" in ln)
    assert inst_row.split()[-1] == "0.22"  # 260e6 / 1200e6, thousands separators handled
    local_row = next(ln for ln in text.splitlines() if "mem_local_op_ld" in ln)
    assert local_row.split()[-2:] == ["-", "-"]  # present in A only: no ratio, "-" for B
    assert any("launch__thread_count" in ln for ln in text.splitlines())  # present in B only


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


def test_unexpected_columns_are_reported_not_a_keyerror():
    text = '"ID","Something Else"\n"0","x"\n'
    with pytest.raises(ValueError, match="unexpected Nsight Compute columns"):
        ncu_profile.parse_ncu_csv(text)


def test_the_raw_csv_is_kept_even_when_it_cannot_be_parsed(tmp_path, monkeypatch):
    """A full collection takes minutes; a parsing problem must not throw it away."""
    import subprocess

    def fake_run(cmd, **kwargs):
        log = Path(cmd[cmd.index("--log-file") + 1])
        log.write_text('"ID","Something Else"\n"0","x"\n')
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ncu_profile, "find_ncu", lambda: "ncu")
    monkeypatch.setattr(ncu_profile.subprocess, "run", fake_run)
    keep = tmp_path / "out" / "r1.ncu.csv"
    with pytest.raises(ValueError):
        ncu_profile.profile(Path("b"), Path("c"), 1, 1000, 10, full=True, keep_csv=keep)
    assert keep.read_text().startswith('"ID","Something Else"')


def test_a_full_profile_uses_the_raw_page_and_the_full_set(tmp_path, monkeypatch):
    import subprocess

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        Path(cmd[cmd.index("--log-file") + 1]).write_text(RAW_R1)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ncu_profile, "find_ncu", lambda: "ncu")
    monkeypatch.setattr(ncu_profile.subprocess, "run", fake_run)
    sections = ncu_profile.profile(Path("b"), Path("c"), 3, 1000, 10, full=True)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--page") + 1] == "raw"
    assert cmd[cmd.index("--set") + 1] == "full"
    assert "--section" not in cmd
    assert cmd[-2:] == ["--rung", "3"]
    assert ncu_profile.flatten(sections)["launch__registers_per_thread"]["value"] == "48"
