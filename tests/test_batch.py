"""Tests for the mibiremo.batch module."""

import math
import numpy as np
import pytest
from mibiremo.batch import BatchModel
from mibiremo.phreeqc import write_database

DAY = 86400.0
DEFINITIONS = """
SOLUTION_MASTER_SPECIES
Dk       Dk       0     Dk     1.0
SOLUTION_SPECIES
Dk = Dk
    log_k 0
PHASES
Dk_fixed
    Dk = Dk
    log_k -3
RATES
Dk
-start
10 SAVE CALC_VALUE("decay_rate") * TOT("Dk") * TOT("water") * TIME  # moles: decay rate x molality x mass of water
-end
"""


def test_first_order_decay(tmp_path):
    """A first-order decay gives C = C0 exp(-λ t) in a closed reactor; with the concentration held at C* by an
    equilibrium phase, the KINETICS reactant decreases linearly, m = m0 - λ C* t (zero order), in 1 L of water."""
    decay_rate = math.log(2) / (0.2 * DAY)  # λ [s-1]
    model = BatchModel(
        workspace=tmp_path,
        simulation_time=0.4 * DAY,
        time_step=0.05 * DAY,
        database=write_database(tmp_path / "decay.dat", DEFINITIONS, parameters={"decay_rate": decay_rate}),
        solutions={1: {"Dk": 1e-3}},  # mol/kgw
        equilibrium_phases={1: {"Dk_fixed": [0, 1]}},  # Dk at C* = 10^log_k = 1e-3 mol/kgw
        initial_equilibrium_phases=[-1, 1],
        kinetics={1: {"Dk": {"m0": 1.0, "formula": "Dk -1"}}},
        initial_kinetics=[1, 1],
    )
    model.run()
    dissolved = model.concentration("Dk").to_numpy() / 1000.0  # mol/L
    reactant = model.kinetic_reactant("Dk").to_numpy()  # mol per litre of water
    time = model.concentration("Dk").index.to_numpy()
    assert reactant.shape == (9, 2)
    assert dissolved[:, 0] == pytest.approx(dissolved[0, 0] * np.exp(-decay_rate * time), rel=1e-6)
    assert dissolved[:, 1] == pytest.approx(dissolved[0, 1], rel=1e-9)
    consumed = reactant[0, 1] - reactant[:, 1]  # from the stored T0 amount (1 s of reaction in the equilibration)
    assert consumed == pytest.approx(decay_rate * dissolved[:, 1] * (time - time[0]), rel=1e-6)
