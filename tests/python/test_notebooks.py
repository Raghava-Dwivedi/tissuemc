"""Notebooks run on a fresh Colab machine. Guard the mistakes that only show up there."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted((ROOT / "tools").glob("*.ipynb"))


def code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_python_cells_compile(path):
    for src in code_cells(path):
        python_only = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
        )
        compile(python_only, str(path), "exec")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_cells_do_not_import_the_package_without_putting_it_on_the_path(path):
    """`pip install -e .` does not reach a Python session that was already running.

    The command-line tools it installs work (each starts a new process); an `import` in a
    notebook cell does not, unless the cell adds the source directory to sys.path first.
    """
    for src in code_cells(path):
        lines = src.splitlines()
        imports = [i for i, ln in enumerate(lines) if "tissuemc_analysis" in ln and "import" in ln]
        for i in imports:
            assert any("sys.path.insert" in ln for ln in lines[:i]), (
                f"{path.name}: a cell imports tissuemc_analysis without sys.path.insert first"
            )


def test_the_diagnostic_notebook_cannot_run_its_long_steps_after_a_failed_check():
    """Run all must not spend minutes on a run that the quick check showed is broken."""
    nb = json.loads((ROOT / "notebooks" / "colab_gpu_diagnose.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    check = next(s for s in sources if "--photons 20000" in s)
    assert ".smoke_ok" in check and "&&" in check  # the marker is written only if everything passed
    long_steps = [s for s in sources if "--photons 500000" in s]
    assert len(long_steps) == 2
    for s in long_steps:
        assert "[ -f results/.smoke_ok ]" in s and "SKIPPED" in s
        assert "--reuse-csv" in s  # a run that already collected is not repeated
    clone = next(s for s in sources if "git clone" in s)
    assert "/content/tissuemc/.git" in clone  # no second, nested clone on a re-run


def test_the_r5_notebook_gates_every_long_step_on_the_quick_check():
    nb = json.loads((ROOT / "notebooks" / "colab_gpu_r5.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    check = next(s for s in sources if "--photons 200000" in s)
    assert ".quick_ok" in check and "&&" in check  # the marker needs both rungs to agree
    for rung in (5, 6):
        assert f"--b-rung {rung}" in check
    gated = [
        s
        for s in sources
        if any(k in s for k in ("tissuemc-validate", "tissuemc-bench", "tissuemc-profile --rung"))
        or ("--b-rung" in s and "--photons 2000000" in s)
    ]
    assert len(gated) == 4  # validate, cross-engine, benchmarks, profiles
    for s in gated:
        commands = [ln for ln in s.splitlines() if ln.startswith("!")]
        assert commands and all("[ -f results/.quick_ok ]" in ln for ln in commands), s
        assert all("SKIPPED" in ln for ln in commands)
    profiles = next(s for s in sources if "--full" in s)
    assert profiles.count("--reuse-csv") == 2  # a collected profile is not repeated
    ladder = next(s for s in sources if "tissuemc-bench" in s)
    order = [
        ln.split("--rung ")[1].split()[0] for ln in ladder.splitlines() if "tissuemc-bench" in ln
    ]
    assert order == ["1", "3", "5", "6", "1"]  # R1 before and after, to show session drift


def test_the_ablation_notebook_gates_its_long_steps_and_labels_the_diagnostic():
    nb = json.loads((ROOT / "notebooks" / "colab_gpu_ablation.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    check = next(s for s in sources if "--photons 200000" in s)
    assert ".quick_ok" in check and "&&" in check and "--b-no-profiles" in check
    gated = [
        s
        for s in sources
        if any(k in s for k in ("tissuemc-bench", "tissuemc-profile --rung", "--photons 2000000"))
    ]
    assert len(gated) == 3  # the full comparison, the benchmarks, the profiles
    for s in gated:
        commands = [ln for ln in s.splitlines() if ln.startswith("!")]
        assert commands and all(
            "[ -f results/.quick_ok ]" in ln and "SKIPPED" in ln for ln in commands
        )
    benches = [
        ln
        for ln in next(s for s in sources if "tissuemc-bench" in s).splitlines()
        if "tissuemc-bench" in ln
    ]
    rungs = [ln.split("--rung ")[1].split()[0] for ln in benches]
    assert rungs == ["6", "6", "1", "6"]  # R6 before and after, to show session drift
    flagged = ["--no-profiles" in ln for ln in benches]
    assert flagged == [False, True, False, False]  # only the second is the diagnostic
    profiles = next(s for s in sources if "--full" in s)
    assert profiles.count("--reuse-csv") == 2 and profiles.count("--no-profiles") == 1
    intro = "".join(nb["cells"][0]["source"])
    assert "not a result" in intro


def test_the_r7_notebook_gates_its_long_steps_and_labels_the_diagnostic():
    nb = json.loads((ROOT / "notebooks" / "colab_gpu_r7.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    check = next(s for s in sources if "--photons 200000" in s)
    assert ".quick_ok" in check and "&&" in check and "--b-rung 7" in check
    long_keys = (
        "tissuemc-validate",
        "tissuemc-bench",
        "tissuemc-profile --rung",
        "--photons 2000000",
    )
    gated = [s for s in sources if any(k in s for k in long_keys)]
    assert len(gated) == 4  # validation, comparisons, benchmarks, profiles
    for s in gated:
        commands = [ln for ln in s.splitlines() if ln.startswith("!")]
        assert commands and all(
            "[ -f results/.quick_ok ]" in ln and "SKIPPED" in ln for ln in commands
        )
    compare = next(s for s in sources if "--photons 2000000" in s)
    assert "--b-rung 6" in compare and "--b-rung 7" in compare  # R6's profiles are checked too
    benches = [
        ln
        for ln in next(s for s in sources if "tissuemc-bench" in s).splitlines()
        if "tissuemc-bench" in ln
    ]
    assert [ln.split("--rung ")[1].split()[0] for ln in benches] == ["6", "7", "6", "7"]
    assert ["--no-profiles" in ln for ln in benches] == [
        False,
        False,
        True,
        False,
    ]  # only the target
    profiles = next(s for s in sources if "--full" in s)
    assert profiles.count("--reuse-csv") == 2 and profiles.count("--no-profiles") == 1
    assert "not a result" in "".join(nb["cells"][0]["source"])


def test_the_r8_notebook_gates_its_long_steps_and_labels_the_diagnostic():
    nb = json.loads((ROOT / "notebooks" / "colab_gpu_r8.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    check = next(s for s in sources if "--photons 200000" in s)
    assert ".quick_ok" in check and "&&" in check and "--b-rung 8" in check
    long_keys = (
        "tissuemc-validate",
        "tissuemc-bench",
        "tissuemc-profile --rung",
        "--photons 2000000",
    )
    gated = [s for s in sources if any(k in s for k in long_keys)]
    assert len(gated) == 4
    for s in gated:
        commands = [ln for ln in s.splitlines() if ln.startswith("!")]
        assert commands and all(
            "[ -f results/.quick_ok ]" in ln and "SKIPPED" in ln for ln in commands
        )
    compare = next(s for s in sources if "--photons 2000000" in s)
    assert "--b-rung 7" in compare and "--b-rung 8" in compare
    benches = [
        ln
        for ln in next(s for s in sources if "tissuemc-bench" in s).splitlines()
        if "tissuemc-bench" in ln
    ]
    assert [ln.split("--rung ")[1].split()[0] for ln in benches] == ["7", "8", "6", "8"]
    assert ["--no-profiles" in ln for ln in benches] == [
        False,
        False,
        True,
        False,
    ]  # only the target
    profiles = next(s for s in sources if "--full" in s)
    assert profiles.count("--reuse-csv") == 2 and "--no-profiles" not in profiles
    assert "not a result" in "".join(nb["cells"][0]["source"])
    # A failed run explains itself: the build and the quick check are logged to files, and the
    # summary carries their tails, so pasting only the summary is enough to diagnose it.
    assert "results/quick.log" in check and "tail" in check
    build = next(s for s in sources if "cmake --build build" in s)
    assert "results/build.log" in build and "BUILD FAILED" in build
    summary = sources[-1]
    assert '"diagnostics"' in summary and "quick_check_passed" in summary
    assert "build_tail" in summary and "quick_check_tail" in summary
