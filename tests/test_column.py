"""Tests for the mibiremo.column module."""

import math
import numpy as np
import pytest
from mibiremo.column import ColumnModel
from .test_field import decay_database

DAY = 86400.0


def test_mibitrans_column(tmp_path):
    """A constant influent concentration gives the exact solution of mibitrans in its 1D limit (wide source, no
    transverse dispersion), for a conservative component and one with first-order decay; after a time t, the flushed
    pore volumes are V / V_p = v t / L."""
    mbt = pytest.importorskip("mibitrans")
    from mibitrans.analysis.differences import rmse

    length, diameter, porosity, dispersivity = 0.3, 0.05, 0.4, 0.005  # L [m], d [m], n [-], alpha_L [m]
    velocity, half_life = 1.0 / DAY, 0.2 * DAY  # v [m s-1], t_1/2 [s]
    model = ColumnModel(
        workspace=tmp_path,
        length=length,
        diameter=diameter,
        porosity=porosity,
        flow_rate=velocity * porosity * math.pi * diameter**2 / 4,  # Q = v n A
        longitudinal_dispersivity=dispersivity,
        n_cells=60,
        simulation_time=0.45 * DAY,
        time_step=0.005 * DAY,  # Courant number 1
        database=decay_database(tmp_path / "decay.dat"),
        solutions={1: {}, 2: {"Tr": 1e-3, "Dk": 1e-3}},
        influent_solution=2,
        kinetics={1: {"Dk": {"m0": 1.0, "parms": [math.log(2) / half_life], "formula": "Dk -1"}}},
        initial_kinetics=1,
    )
    model.run()
    assert model.effluent("Tr")["pore_volumes"].iloc[-1] == pytest.approx(velocity * 0.45 * DAY / length)

    influent = 1e-3 * 997.0  # mol kgw-1 x density of water [kg m-3]
    hydrology = mbt.HydrologicalParameters(velocity=1.0, porosity=porosity, alpha_x=dispersivity, alpha_y=1e-6)
    source = mbt.SourceParameters(np.array([0.05]), np.array([1.0]), depth=1.0)  # wide and deep for alpha_y, alpha_z
    grid = mbt.ModelParameters(model_length=length, model_width=0.1, model_time=0.45, dx=0.005, dy=0.05, dt=0.05)
    for component, decay_rate in [("Tr", 0.0), ("Dk", math.log(2) / 0.2)]:  # mibitrans: m, d
        plume = mbt.Mibitrans(hydrology, mbt.AttenuationParameters(decay_rate=decay_rate), source, grid)
        exact = plume.run().relative_cxyt[:, np.argmin(abs(plume.y)), 1:]  # centreline, x > 0
        numerical = np.array([model.concentration(component, t * DAY) for t in plume.t]) / influent
        assert plume.x[1:] == pytest.approx(model.x)
        assert rmse(exact, numerical, axis=(0, 1)) < 0.015  # decay: splitting of transport and reaction, ≈ λ Δt / 2
