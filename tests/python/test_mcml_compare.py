"""Tests for the comparison tool's parsing and checks, using only synthetic data.

Nothing here needs the original MCML sources, which are not part of this repository.
"""

import math
from pathlib import Path

import pytest
from tissuemc_analysis import mcml_compare as mc

ROOT = Path(__file__).resolve().parents[2]

MCO = """A1 \t# Version number of the file format.

# User time:  1 sec = 0.00 hr.   Simulation time of this run.

InParm \t\t\t# Input parameters. cm is used.
run000.mco \tA\t\t# output file name, ASCII.
250000 \t\t\t# No. of photons
0.01\t0.01\t\t# dz, dr [cm]
40\t50\t1\t# No. of dz, dr, da.

2\t\t\t\t\t# Number of layers
#n\tmua\tmus\tg\td\t# One line for each layer
1\t\t\t\t\t# n for medium above
1.4\t1\t100\t0.9\t0.1\t# layer 1
1.4\t2\t10\t0\t0.2\t# layer 2
1\t\t\t\t\t# n for medium below

RAT #Reflectance, absorption, transmission.
0.0278\t#Specular reflectance [-]
0.25\t#Diffuse reflectance [-]
0.6\t#Absorbed fraction [-]
0.1222\t#Transmittance [-]

A_l #Absorption as a function of layer. [-]
0.3
0.3
"""

SPEC = mc.Spec(
    photons=1,
    dz=0.01,
    dr=0.01,
    nz=40,
    nr=50,
    na=1,
    n_above=1.0,
    layers=((1.4, 1.0, 100.0, 0.9, 0.1), (1.4, 2.0, 10.0, 0.0, 0.2)),
    n_below=1.0,
)


def test_parse_mco_reads_results_and_echo():
    run = mc.parse_mco(MCO)
    assert (run.rsp, run.rd, run.absorbed, run.tt) == (0.0278, 0.25, 0.6, 0.1222)
    assert run.echo["photons"] == 250000
    assert run.echo["grid"] == [40, 50, 1]
    assert run.echo["layers"][1] == [1.4, 2.0, 10.0, 0.0, 0.2]
    assert run.echo["n_below"] == 1.0


def test_check_echo_accepts_a_faithful_read():
    assert mc.check_echo(SPEC, 250000, mc.parse_mco(MCO)) == []


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("250000 \t\t\t#", "250001 \t\t\t#", "photons"),
        ("0.01\t0.01\t\t#", "0.02\t0.01\t\t#", "dz/dr"),
        ("40\t50\t1\t#", "40\t51\t1\t#", "grid"),
        ("1.4\t2\t10\t0\t0.2", "1.4\t2\t10\t0.5\t0.2", "layer parameters"),
        ("1\t\t\t\t\t# n for medium below", "1.1\t\t\t\t\t# n for medium below", "ambient"),
    ],
)
def test_check_echo_catches_each_kind_of_misread(old, new, fragment):
    assert old in MCO
    problems = mc.check_echo(SPEC, 250000, mc.parse_mco(MCO.replace(old, new)))
    assert any(fragment in p for p in problems), problems


def test_parse_mco_rejects_other_files():
    with pytest.raises(ValueError):
        mc.parse_mco("this is not an MCML output file\n")


def test_read_spec_matches_the_bundled_validation_input():
    spec = mc.read_spec(ROOT / "configs" / "validation" / "v3_three_layer.mci")
    assert spec.layers[0] == (1.37, 1.0, 100.0, 0.9, 0.1)
    assert len(spec.layers) == 3
    assert (spec.nz, spec.nr, spec.na) == (40, 50, 1)
    assert spec.n_above == spec.n_below == 1.0


def test_multirun_input_round_trips_through_our_own_reader(tmp_path):
    spec = mc.read_spec(ROOT / "configs" / "validation" / "v3_three_layer.mci")
    path = tmp_path / "multi.mci"
    mc.write_multirun_mci(spec, runs=3, photons_per_run=1234, path=path)
    again = mc.read_spec(path)
    assert again.photons == 1234
    assert again.layers == spec.layers
    # The file must also be accepted by this repository's own parser (3 runs, distinct names).
    import subprocess

    binary = ROOT / "build" / "tissuemc"
    if binary.exists():
        proc = subprocess.run(
            [str(binary), "run", str(path), "--photons", "300", "--batches", "3"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.count("run ") == 3
    names = [ln for ln in path.read_text().splitlines() if ln.endswith(".mco A")]
    assert len(names) == len(set(names)) == 3


def test_verify_sources_refuses_missing_and_changed_files(tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        mc.verify_sources(tmp_path)
    for name in mc.SOURCE_FILES:
        (tmp_path / name).write_text("not the real file\n")
    with pytest.raises(SystemExit, match="does not match"):
        mc.verify_sources(tmp_path)


def test_compare_case_skips_quantities_that_are_zero_in_both():
    run = {
        "batches": [
            {"photons": 10, "rd": 2.0, "tt": 0.0, "absorbed": [8.0]},
            {"photons": 10, "rd": 2.2, "tt": 0.0, "absorbed": [7.8]},
        ]
    }
    theirs = [mc.McmlRun(0.0, 0.2, 0.8, 0.0, {}), mc.McmlRun(0.0, 0.21, 0.79, 0.0, {})]
    out = mc.compare_case("x", run, theirs)
    assert {c.quantity for c in out} == {"rd", "absorbed"}
    assert all(math.isfinite(c.comparison.z) for c in out)


def test_xoshiro_variant_replaces_only_the_generator(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    for name in mc.SOURCE_FILES:
        (src / name).write_text(f"/* {name} */\n")
    (src / "mcmlgo.c").write_text(
        "int before(void) { return 1; }\n"
        "double RandomNum(void)\n{\n  static int first_time = 1;\n  return ran3(&idum);\n}\n"
        "int after(void) { return 2; }\n"
    )
    mc.make_xoshiro_sources(src, out)
    changed = (out / "mcmlgo.c").read_text()
    assert "ran3(&idum)" not in changed
    assert "xoshiro256++" in changed
    assert "int before(void) { return 1; }" in changed
    assert "int after(void) { return 2; }" in changed
    assert changed.count("double RandomNum(void)") == 1
    for name in mc.SOURCE_FILES:
        if name != "mcmlgo.c":
            assert (out / name).read_text() == f"/* {name} */\n"
    assert "ran3(&idum)" in (src / "mcmlgo.c").read_text()  # the input is never modified
