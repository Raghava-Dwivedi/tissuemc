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
