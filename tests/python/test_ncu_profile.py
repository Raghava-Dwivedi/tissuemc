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
