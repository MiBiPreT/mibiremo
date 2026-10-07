"""Tests for the mibiremo.phreeqc module, with the PhreeqcRM library on a small batch problem."""

import os
from importlib.resources import files
import numpy as np
import pandas as pd
import pytest
import mibiremo

DATABASE = str(files("mibiremo") / "database" / "phreeqc.dat")
PQI = """
SOLUTION 1
    units mol/kgw
    pH 7
    C(4) 1e-3
SOLUTION 2
    units mol/kgw
    pH 7
EQUILIBRIUM_PHASES 1
    Calcite 0 10
SELECTED_OUTPUT 1
    -reset false
    -pH true
    -totals Ca C(4)
"""
# Solution, equilibrium phases, exchange, surface, gas phase, solid solutions, kinetics (-1: none) of the cells:
# 1 mmol/L dissolved carbon; pure water with calcite; pure water
IC = [[1, -1, -1, -1, -1, -1, -1], [2, 1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1]]


@pytest.fixture
def pqi_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # PhreeqcRM writes its output files in the working directory
    path = tmp_path / "calcite.pqi"
    path.write_text(PQI)
    return str(path)


@pytest.mark.parametrize("ic_type", [np.array, pd.DataFrame])
def test_calcite_equilibrium(pqi_file, ic_type):
    """Pure water in equilibrium with calcite has pH 9.9 and Ca = dissolved C; the cells keep the order of ic."""
    phr = mibiremo.PhreeqcRM()
    phr.create(nxyz=3, n_threads=-1)
    phr.initialize_phreeqc(DATABASE, units_solution=2)
    phr.run_initial_from_file(pqi_file, ic_type(IC))
    result = phr.get_selected_output_df()
    assert phr.n_threads == os.cpu_count()
    assert {"C", "Ca"} <= set(phr.components)
    assert {"Ca+2", "CO3-2"} <= set(phr.species)
    assert list(result.columns) == ["pH", "Ca(mol/kgw)", "C(4)(mol/kgw)"]
    assert result["pH"].to_numpy() == pytest.approx([7.0, 9.9, 7.0], abs=0.05)
    assert result["Ca(mol/kgw)"].to_numpy() == pytest.approx([0.0, 1.2e-4, 0.0], abs=1e-5)
    assert result["C(4)(mol/kgw)"].to_numpy() == pytest.approx([1e-3, 1.2e-4, 0.0], abs=1e-5)
    assert result["Ca(mol/kgw)"][1] == pytest.approx(result["C(4)(mol/kgw)"][1], rel=1e-4)  # CaCO3 dissolved


def test_errors(pqi_file):
    """PHREEQC errors raise exceptions and the initial conditions need one row per cell."""
    phr = mibiremo.PhreeqcRM()
    with pytest.raises(RuntimeError, match="create"):
        phr.initialize_phreeqc(DATABASE)
    phr.create(nxyz=2)
    with pytest.raises(RuntimeError, match="database"):
        phr.initialize_phreeqc("missing.dat")
    phr.initialize_phreeqc(DATABASE)
    with pytest.raises(RuntimeError, match="input file"):
        phr.run_initial_from_file("missing.pqi", np.array(IC[:2]))
    with pytest.raises(ValueError, match="shape"):
        phr.run_initial_from_file(pqi_file, np.array(IC))
