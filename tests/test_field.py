"""Tests for the mibiremo.field module."""

import math
import shutil
from dataclasses import replace
import numpy as np
import pytest
from mibiremo.field import FieldModel
from mibiremo.field import intermittent
from mibiremo.field import structured_grid
from mibiremo.wells import Well
from mibiremo.wells import array_radial

# Example field locations (EPSG:27572): extraction well EXT_1, injection wells INJ_1-3 about 4 m around it
GEOMETRY = {"well_top": 96.3, "well_bottom": 90.5, "diameter": 0.1, "screen_top": 95.5, "screen_bottom": 90.5}
EXTRACTION = Well("EXT_1", x=222049.7016, y=2401808.1922, **GEOMETRY)
DESIGN = [EXTRACTION, *array_radial(replace(EXTRACTION, name="INJ"), n_wells=3, radius=4.0)]
REAL = [
    EXTRACTION,
    Well("INJ_1", x=222049.6162, y=2401812.2549, **GEOMETRY),  # 0.085 m west of EXT_1
    Well("INJ_2", x=222046.1198, y=2401806.2488, **GEOMETRY),
    Well("INJ_3", x=222053.1180, y=2401806.2389, **GEOMETRY),  # 0.010 m south of INJ_2
]
GRID = {"domain_size": (148.0, 116.0), "top": 95.5, "bottom": 90.5, "n_layers": 7, "grid_spacing": 4.0}
FINE = 0.5  # grid spacing at the wells [m]
HOUR, DAY = 3600.0, 86400.0
Q = 2.75e-5  # flow rate [m3 s-1]
MODEL = {  # 5 layers of 1 m
    "domain_size": (40.0, 40.0),
    "top": 95.5,
    "bottom": 90.5,
    "n_layers": 5,
    "hydraulic_conductivity": 5e-6,
    "porosity": 0.25,
    "reference_head": 93.7,
    "grid_spacing": 4.0,
    "grid_spacing_at_wells": FINE,
}
requires_mf6 = pytest.mark.skipif(shutil.which("mf6") is None, reason="MODFLOW 6 not found")


@pytest.mark.parametrize(
    ("wells", "tolerance"),
    [(DESIGN, 1e-6), (REAL, FINE / 4)],  # real wells nearly in line share a column (row)
)
def test_wells_at_cell_centres(wells, tolerance):
    """Wells are at cell centres of a domain centred on them, with layers of equal thickness."""
    grid = structured_grid(wells, grid_spacing_at_wells=FINE, **GRID)
    for w in wells:
        row, column = grid.intersect(w.x, w.y)
        assert grid.xcellcenters[row, column] == pytest.approx(w.x, abs=tolerance)
        assert grid.ycellcenters[row, column] == pytest.approx(w.y, abs=tolerance)
    x_min, x_max, y_min, y_max = grid.extent
    x, y = [w.x for w in wells], [w.y for w in wells]
    assert (x_max - x_min, y_max - y_min) == pytest.approx(GRID["domain_size"])
    assert ((x_min + x_max) / 2, (y_min + y_max) / 2) == pytest.approx(((min(x) + max(x)) / 2, (min(y) + max(y)) / 2))
    assert np.diff(grid.botm[:, 0, 0]) == pytest.approx([-5.0 / 7] * 6)  # layers of equal thickness


def test_cell_widths():
    """Cell widths are fine near the wells and grow by at most 1.5 up to the grid spacing."""
    margin = 5.0
    grid = structured_grid(DESIGN, grid_spacing_at_wells=FINE, refinement_margin=margin, **GRID)
    for widths, centres, wells_xy in [
        (grid.delr, grid.xcellcenters[0], [w.x for w in DESIGN]),
        (grid.delc, grid.ycellcenters[:, 0], [w.y for w in DESIGN]),
    ]:
        ratio = widths[1:] / widths[:-1]
        assert np.maximum(ratio, 1 / ratio).max() <= 1.5 + 1e-9
        assert widths.max() <= GRID["grid_spacing"] + 1e-9
        near_wells = (centres > min(wells_xy) - margin) & (centres < max(wells_xy) + margin)
        assert widths[near_wells] == pytest.approx(FINE, rel=0.05)


def test_stress_periods(tmp_path):
    """Stress periods start at every change time of the schedules and preserve the volume of each well."""
    model = FieldModel(
        **MODEL,
        workspace=tmp_path,
        wells=DESIGN,
        flow_rates={"EXT_1": -3 * Q, "INJ_1": intermittent(Q, 8 * HOUR, 16 * HOUR, end_time=2 * DAY)},
        injection_concentration=[(0.0, 1.0), (12 * HOUR, 0.0)],
        simulation_time=2 * DAY,
        time_step=[(0.0, HOUR), (DAY, 4 * HOUR)],
    )
    periods = model.stress_periods
    assert list(periods["start"]) == [0.0, 8 * HOUR, 12 * HOUR, DAY, DAY + 8 * HOUR]
    assert list(periods["n_time_steps"]) == [8, 4, 12, 2, 4]
    assert list(periods["injection_concentration"]) == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert (periods["INJ_1"] * periods["duration"]).sum() == pytest.approx(2 * 8 * HOUR * Q)
    assert (periods["EXT_1"] * periods["duration"]).sum() == pytest.approx(-3 * Q * 2 * DAY)


def test_well_rates(tmp_path):
    """In every stress period, Q of a well is split among the layers in proportion to the screened thickness."""
    partial = replace(EXTRACTION, screen_top=94.75, screen_bottom=92.0)  # 0.25, 1, 1, and 0.5 m in layers 1-4
    model = FieldModel(
        **MODEL,
        workspace=tmp_path,
        wells=[partial, *DESIGN[1:]],
        flow_rates={"EXT_1": -Q, "INJ_1": [(0.0, Q), (DAY, 0.0)]},
        simulation_time=2 * DAY,
        time_step=DAY,
    )
    model.build()
    wel = model.simulation.get_model("gwf").wel.stress_period_data
    for period, injection in enumerate([Q, 0.0]):
        records = wel.get_data(period)
        assert len(records) == 4 + 5  # all records in every stress period, with Q = 0 when off
        extraction = records[records["boundname"] == "EXT_1"]
        assert [layer for layer, _, _ in extraction["cellid"]] == [0, 1, 2, 3]
        assert extraction["q"] == pytest.approx(-Q * np.array([0.25, 1.0, 1.0, 0.5]) / 2.75)
        assert records[records["boundname"] == "INJ_1"]["q"].sum() == pytest.approx(injection)


@requires_mf6
def test_regional_flow(tmp_path):
    """Without pumping, the specific discharge is K i towards the regional flow azimuth (Darcy's law)."""
    gradient, azimuth = 0.008, math.radians(22.0)
    model = FieldModel(
        **MODEL,
        workspace=tmp_path,
        wells=DESIGN,
        flow_rates={"EXT_1": 0.0},
        regional_hydraulic_gradient=gradient,
        regional_flow_azimuth=22.0,
        simulation_time=1.0,
        time_step=1.0,
    )
    model.run()
    discharge = model.simulation.get_model("gwf").output.budget().get_data(text="DATA-SPDIS")[0]
    darcy = MODEL["hydraulic_conductivity"] * gradient
    assert discharge["qx"] == pytest.approx(darcy * math.sin(azimuth))
    assert discharge["qy"] == pytest.approx(darcy * math.cos(azimuth))
    assert discharge["qz"] == pytest.approx(0.0, abs=1e-15)


@requires_mf6
def test_tracer_mass(tmp_path):
    """The tracer mass in the aquifer, Σ n V C, equals the injected mass Q C_in t (no tracer reaches the boundary)."""
    model = FieldModel(
        **MODEL,
        workspace=tmp_path,
        wells=DESIGN,
        flow_rates={"INJ_1": Q},
        injection_concentration=[(0.0, 1.0), (12 * HOUR, 0.0)],  # pulse of 12 h, then clean water
        simulation_time=DAY,
        time_step=HOUR,
    )
    model.run()
    grid = model.grid
    volume = grid.delc[:, None] * grid.delr * grid.cell_thickness
    injected = Q * 1.0 * 12 * HOUR
    assert (MODEL["porosity"] * volume * model.concentration()).sum() == pytest.approx(injected, rel=1e-5)
    assert model.mass_balance().iloc[-1]["in_aquifer"] == pytest.approx(injected, rel=1e-4)


@requires_mf6
def test_extraction_concentration(tmp_path):
    """Without regional flow EXT_1 captures all the water of INJ_1: at steady state C_ext = Q_inj C_in / Q_ext."""
    model = FieldModel(
        **(MODEL | {"n_layers": 1}),
        workspace=tmp_path,
        wells=DESIGN,
        flow_rates={"EXT_1": -3 * Q, "INJ_1": Q},
        simulation_time=2000 * DAY,
        time_step=20 * DAY,
    )
    model.run()
    assert model.well_concentration("EXT_1")["concentration"].iloc[-1] == pytest.approx(1 / 3, rel=1e-3)
