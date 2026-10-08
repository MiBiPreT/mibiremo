"""Tests for the mibiremo.field module."""

import math
import shutil
from dataclasses import replace
from importlib.resources import files
import flopy
import geopandas
import numpy as np
import pytest
import shapely
from mibiremo.field import FieldModel
from mibiremo.field import intermittent_pumping
from mibiremo.field import structured_grid
from mibiremo.wells import Well
from mibiremo.wells import array_radial

# Example field locations (EPSG:27572): extraction well EXT_1, injection wells INJ_1-3 about 4 m around it
GEOMETRY = {"well_top": 96.3, "well_bottom": 90.5, "diameter": 0.1, "screen_top": 95.5, "screen_bottom": 90.5}
EXTRACTION = Well("EXT_1", x=222049.7016, y=2401808.1922, **GEOMETRY)
DESIGN = [EXTRACTION, *array_radial(replace(EXTRACTION, name="INJ"), n_wells=3, radius=4.0)]
GRID = {"domain_size": (148.0, 116.0), "top": 95.5, "layer_bottom": 90.5, "n_layers": 7, "grid_spacing": 4.0}
FINE = 0.5  # grid spacing at the wells [m]
HOUR, DAY = 3600.0, 86400.0
Q = 2.75e-5  # flow rate [m3 s-1]
MODEL = {  # 5 layers of 1 m
    "domain_size": (40.0, 40.0),
    "top": 95.5,
    "layer_bottom": 90.5,
    "n_layers": 5,
    "hydraulic_conductivity": 5e-6,
    "porosity": 0.25,
    "reference_head": 93.7,
    "grid_spacing": 4.0,
    "grid_spacing_at_wells": FINE,
}
requires_mf6 = pytest.mark.skipif(shutil.which("mf6") is None, reason="MODFLOW 6 not found")


def decay_database(path):
    """phreeqc.dat with the pseudo-components Tr (conservative) and Dk (first-order decay, PARM(1) = λ [s⁻¹])."""
    definitions = """
SOLUTION_MASTER_SPECIES
Tr       Tr       0     Tr     1.0
Dk       Dk       0     Dk     1.0
SOLUTION_SPECIES
Tr = Tr
    log_k 0
Dk = Dk
    log_k 0
RATES
Dk
-start
10 SAVE PARM(1) * TOT("Dk") * TIME
-end
"""
    text = (files("mibiremo") / "database" / "phreeqc.dat").read_text(encoding="latin-1")
    path.write_text(text.replace("\nEND\n", definitions + "END\n", 1), encoding="latin-1")
    return path


def dissolved_mass(model, time=None, component=None):
    """Mass in the aquifer, Σ n V C [concentration unit × m³]."""
    grid = model.grid
    volume = grid.delc[:, None] * grid.delr * grid.cell_thickness
    return (model.porosity * volume * model.concentration(time, component)).sum()


@pytest.mark.parametrize(
    ("wells", "tolerance", "centre"),
    [
        (DESIGN, 1e-6, None),
        ([EXTRACTION, replace(EXTRACTION, name="INJ_1", x=EXTRACTION.x + 0.6)], FINE / 4, None),  # cells share the gap
        (DESIGN, 1e-6, (EXTRACTION.x - 7.7, EXTRACTION.y - 7.2)),  # domain centred upgradient of the wells
    ],
)
def test_wells_at_cell_centres(wells, tolerance, centre):
    """Wells are at cell centres of a domain centred on them (or on a given centre), with layers of equal thickness."""
    grid = structured_grid(wells, grid_spacing_at_wells=FINE, domain_centre=centre, **GRID)
    for w in wells:
        row, column = grid.intersect(w.x, w.y)
        assert grid.xcellcenters[row, column] == pytest.approx(w.x, abs=tolerance)
        assert grid.ycellcenters[row, column] == pytest.approx(w.y, abs=tolerance)
    x_min, x_max, y_min, y_max = grid.extent
    x, y = [w.x for w in wells], [w.y for w in wells]
    centre = centre or ((min(x) + max(x)) / 2, (min(y) + max(y)) / 2)
    assert (x_max - x_min, y_max - y_min) == pytest.approx(GRID["domain_size"])
    assert ((x_min + x_max) / 2, (y_min + y_max) / 2) == pytest.approx(centre)
    assert np.diff(grid.botm[:, 0, 0]) == pytest.approx([-5.0 / 7] * 6)  # layers of equal thickness


def test_hydrostratigraphic_units():
    """Each unit is divided into layers of equal thickness, between a sloping top and the unit bottoms."""
    x0 = EXTRACTION.x
    grid = structured_grid(
        DESIGN,
        domain_size=(40.0, 40.0),
        top=lambda x, y: 95.5 + 0.01 * (x - x0),  # top of the aquifer rising towards the east
        layer_bottom=[93.5, lambda x, y: 90.5 - 0.01 * (x - x0)],  # bottom of each unit
        n_layers=[2, 3],
        grid_spacing=4.0,
        grid_spacing_at_wells=FINE,
    )
    upper = (grid.top - 93.5) / 2
    lower = (93.5 - (90.5 - 0.01 * (grid.xcellcenters - x0))) / 3
    assert grid.cell_thickness == pytest.approx(np.array([upper] * 2 + [lower] * 3))


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
        flow_rates={"EXT_1": -3 * Q, "INJ_1": intermittent_pumping(Q, 8 * HOUR, 16 * HOUR, end_time=2 * DAY)},
        tracer_concentration=[(0.0, 1.0), (12 * HOUR, 0.0)],
        simulation_time=2 * DAY,
        time_step=[(0.0, HOUR), (DAY, 4 * HOUR)],
    )
    periods = model.stress_periods
    assert list(periods["start"]) == [0.0, 8 * HOUR, 12 * HOUR, DAY, DAY + 8 * HOUR]
    assert list(periods["n_time_steps"]) == [8, 4, 12, 2, 4]
    assert list(periods["tracer_concentration"]) == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert (periods["INJ_1"] * periods["duration"]).sum() == pytest.approx(2 * 8 * HOUR * Q)
    assert (periods["EXT_1"] * periods["duration"]).sum() == pytest.approx(-3 * Q * 2 * DAY)


def test_well_rates(tmp_path):
    """In every stress period, Q of a well is split among the layers in proportion to the screened transmissivity."""
    partial = replace(EXTRACTION, screen_top=94.75, screen_bottom=92.0)  # 0.25, 1, 1, and 0.5 m in layers 1-4
    model = FieldModel(
        **(MODEL | {"layer_bottom": [93.5, 90.5], "n_layers": [2, 3], "hydraulic_conductivity": [1e-5, 1e-6]}),
        workspace=tmp_path,
        wells=[partial, *DESIGN[1:]],
        flow_rates={"EXT_1": -Q, "INJ_1": [(0.0, Q), (DAY, 0.0)]},
        simulation_time=2 * DAY,
        time_step=DAY,
    )
    model.build()
    wel = model.simulation.get_model("gwf").wel.stress_period_data
    transmissivity = np.array([0.25e-5, 1e-5, 1e-6, 0.5e-6])  # K b [m2 s-1] in layers 1-4
    for period, injection in enumerate([Q, 0.0]):
        records = wel.get_data(period)
        assert len(records) == 4 + 5  # all records in every stress period, with Q = 0 when off
        extraction = records[records["boundname"] == "EXT_1"]
        assert [layer for layer, _, _ in extraction["cellid"]] == [0, 1, 2, 3]
        assert extraction["q"] == pytest.approx(-Q * transmissivity / transmissivity.sum())
        assert records[records["boundname"] == "INJ_1"]["q"].sum() == pytest.approx(injection)


@requires_mf6
def test_regional_flow(tmp_path):
    """Without pumping, the hydraulic head is planar (also in the monitoring wells) and the specific discharge is K i
    towards the regional flow azimuth (Darcy's law)."""
    gradient, azimuth = 0.008, math.radians(22.0)
    model = FieldModel(
        **(MODEL | {"n_layers": 2}),
        workspace=tmp_path,
        wells=DESIGN,
        flow_rates={"EXT_1": 0.0},
        regional_hydraulic_gradient=gradient,
        regional_flow_azimuth=22.0,
        simulation_time=1.0,
        time_step=1.0,
    )
    model.run()
    planar = model.head_from_gradient(model.grid.xcellcenters, model.grid.ycellcenters)
    assert model.head() == pytest.approx(np.broadcast_to(planar, model.grid.shape))
    well = DESIGN[1]  # monitoring well, at a cell centre
    assert model.well_head(well.name)["head"].iloc[-1] == pytest.approx(model.head_from_gradient(well.x, well.y))
    discharge = model.simulation.get_model("gwf").output.budget().get_data(text="DATA-SPDIS")[0]
    darcy = MODEL["hydraulic_conductivity"] * gradient
    assert discharge["qx"] == pytest.approx(darcy * math.sin(azimuth))
    assert discharge["qy"] == pytest.approx(darcy * math.cos(azimuth))
    assert discharge["qz"] == pytest.approx(0.0, abs=1e-15)


@requires_mf6
def test_general_head_boundary(tmp_path):
    """Without regional flow, the water extracted by a well enters across the general-head boundary,
    Q = Σ C (h_b - h)."""
    conductance = 1e-4  # [m2 s-1]
    model = FieldModel(
        **(MODEL | {"n_layers": 1}),
        workspace=tmp_path,
        wells=DESIGN,
        flow_rates={"EXT_1": -Q},
        boundary_conductance=conductance,
        simulation_time=1.0,
        time_step=1.0,
    )
    model.run()
    flows = model.simulation.get_model("gwf").output.budget().get_data(text="GHB")[0]
    head = model.head().ravel()[flows["node"] - 1]
    assert flows["q"] == pytest.approx(conductance * (MODEL["reference_head"] - head))
    assert flows["q"].sum() == pytest.approx(Q, rel=1e-6)


@requires_mf6
def test_tracer_mass(tmp_path):
    """The tracer mass in the aquifer, Σ n V C, equals the injected mass Q C_in t (no tracer reaches the boundary)."""
    model = FieldModel(
        **(MODEL | {"n_layers": 2}),
        workspace=tmp_path,
        wells=DESIGN,
        flow_rates={"INJ_1": Q},
        tracer_concentration=[(0.0, 1.0), (12 * HOUR, 0.0)],  # pulse of 12 h, then clean water
        simulation_time=DAY,
        time_step=2 * HOUR,
    )
    model.run()
    injected = Q * 1.0 * 12 * HOUR
    assert dissolved_mass(model) == pytest.approx(injected, rel=1e-4)
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
        time_step=100 * DAY,  # implicit transport: steady state does not depend on the time step
    )
    model.run()
    assert model.well_concentration("EXT_1")["concentration"].iloc[-1] == pytest.approx(1 / 3, rel=1e-3)


def test_zone(tmp_path):
    """A zone holds the aquifer volume inside the polygon and between the elevations (all on cell faces)."""
    model = FieldModel(
        **MODEL, workspace=tmp_path, wells=DESIGN, flow_rates={"EXT_1": -Q}, simulation_time=DAY, time_step=DAY
    )
    x, y = EXTRACTION.x, EXTRACTION.y
    square = shapely.box(x - 1.25, y - 1.25, x + 1.25, y + 1.25)
    geopandas.GeoDataFrame(geometry=[square], crs="EPSG:27572").to_file(tmp_path / "a.shp")
    zone = model.zone(tmp_path / "a.shp", z_top=94.5, z_bottom=92.5)
    model.build()
    volume = model.grid.delc[:, None] * model.grid.delr * model.grid.cell_thickness
    assert volume[zone].sum() == pytest.approx(2.5 * 2.5 * 2.0)


@requires_mf6
def test_injected_solution(tmp_path):
    """Injected PHREEQC components move as the tracer, and the mass of a first-order decaying component follows the
    exact solution of dM/dt = Q C_in - λ M (relative to the conservative component)."""
    pulse, decay_rate = 12 * HOUR, math.log(2) / (2 * DAY)  # injection time [s], λ [s⁻¹] (half-life 2 d)
    settings = {
        **MODEL,
        "n_layers": 1,
        "domain_size": (20.0, 20.0),  # small grid: PHREEQC runs in every cell at every time step
        "grid_spacing": 2.0,
        "grid_spacing_at_wells": 1.0,
        "refinement_margin": 2.0,
        "wells": DESIGN,
        "flow_rates": {"EXT_1": -3 * Q, "INJ_1": Q},
        "boundary_conductance": 1e-4,  # inflow of the initial solution across a general-head boundary
        "simulation_time": DAY,
        "time_step": HOUR,
    }
    tracer = FieldModel(**settings, workspace=tmp_path / "tracer", tracer_concentration=[(0.0, 1.0), (pulse, 0.0)])
    tracer.run()
    model = FieldModel(
        **settings,
        workspace=tmp_path / "phreeqc",
        phreeqc_coupling=True,
        database=decay_database(tmp_path / "decay.dat"),
        solutions={1: {}, 2: {"Tr": 1e-3, "Dk": 1e-3}},
        injected_solution=[(0.0, 2), (pulse, 1)],
        kinetics={1: {"Dk": {"m0": 1.0, "parms": [decay_rate], "formula": "Dk -1"}}},
        initial_kinetics=1,
    )
    model.run()
    expected = 1e-3 * 997.0 * tracer.concentration()  # mol kgw⁻¹ × ρ_w [kg m⁻³]
    assert model.concentration(component="Tr") == pytest.approx(expected, rel=1e-3, abs=1e-9)  # mol m⁻³
    for time in [pulse, DAY]:
        injection = min(time, pulse)  # M = Q C_in (1 - exp(-λ t)) / λ while injecting, then decays as exp(-λ t)
        exact = (1 - math.exp(-decay_rate * injection)) / (decay_rate * injection)
        exact *= math.exp(-decay_rate * (time - injection))
        decayed = dissolved_mass(model, time, "Dk") / dissolved_mass(model, time, "Tr")
        assert decayed == pytest.approx(exact, rel=0.015)  # splitting of transport and reaction: error ≈ λ Δt / 2


@requires_mf6
def test_mibitrans_plume(tmp_path):
    """Plume of a constant-concentration source in uniform flow matches the exact solution of mibitrans."""
    mbt = pytest.importorskip("mibitrans")
    from mibitrans.analysis.differences import rmse

    conductivity, gradient, porosity = 2e-4, 0.01, 0.25  # K [m s-1], i [-], n [-]
    screen = {"well_top": 10.0, "well_bottom": 0.0, "diameter": 0.05, "screen_top": 10.0, "screen_bottom": 0.0}
    wells = [Well("SOURCE", x=0.0, y=0.0, **screen), Well("X_10", x=10.0, y=0.0, **screen)]
    model = FieldModel(
        wells=wells,
        flow_rates={"SOURCE": 0.0, "X_10": 0.0},  # no pumping: the wells place the uniform grid
        workspace=tmp_path,
        domain_size=(17.0, 11.0),
        top=10.0,
        layer_bottom=0.0,
        hydraulic_conductivity=conductivity,
        porosity=porosity,
        reference_head=10.0,
        regional_hydraulic_gradient=gradient,
        regional_flow_azimuth=90.0,
        advection_scheme="tvd",
        dispersion=True,
        longitudinal_dispersivity=1.0,
        transverse_dispersivity=0.1,
        grid_spacing=1.0,
        grid_spacing_at_wells=1.0,
        refinement_margin=0.0,
        simulation_time=8 * DAY,
        time_step=DAY,
    )
    source = list(zip(*np.nonzero(model.zone(shapely.box(-0.1, -2.1, 0.1, 2.1)))))  # 5 cells: half-width 2.5 m
    model.build()
    flopy.mf6.ModflowGwtcnc(model.simulation.get_model("gwt"), stress_period_data=[(cell, 1.0) for cell in source])
    model.run()

    plume = mbt.Mibitrans(
        mbt.HydrologicalParameters(
            h_conductivity=conductivity * DAY, h_gradient=gradient, porosity=porosity, alpha_x=1.0, alpha_y=0.1
        ),
        mbt.AttenuationParameters(decay_rate=0.0),
        mbt.SourceParameters(np.array([2.5]), np.array([1.0]), depth=10.0),
        mbt.ModelParameters(model_length=8.0, model_width=10.0, model_time=8.0, dx=1.0, dy=1.0, dt=4.0),
    )
    exact = plume.run().relative_cxyt[:, :, 1:]  # without x = 0, where mibitrans has C = 0 outside the source
    xc, yc = model.grid.xcellcenters[0], model.grid.ycellcenters[:, 0]
    rows, columns = np.ix_([np.argmin(abs(yc - y)) for y in plume.y], [np.argmin(abs(xc - x)) for x in plume.x[1:]])
    numerical = np.array([model.concentration(t)[0] for t in [4 * DAY, 8 * DAY]])[:, rows, columns]
    assert rmse(exact, numerical, concentration_cutoff=0.01) < 0.05
