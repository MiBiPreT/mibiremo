"""Creates a field installation model for simulating groundwater flow, transport and reactions.

Contents:
- `FieldModel`: wells in a homogeneous aquifer with constant gradient groundwater flow, tracer transport, and
  reactive transport (MODFLOW 6 coupled with PHREEQC by mf6rtm)
- `intermittent_pumping`: on/off schedule (e.g., of a flow rate)
- `structured_grid`: structured (DIS) grid centred on the wells
"""

import contextlib
import copy
import math
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import flopy
import numpy as np
import pandas as pd
from flopy.discretization import StructuredGrid
from mf6rtm.mup3d.base import ChemStress
from mf6rtm.mup3d.base import KineticPhases
from mf6rtm.mup3d.base import Mup3d
from mf6rtm.mup3d.base import Solutions

# Global constants
MAX_GROWTH_FACTOR = 1.5  # largest ratio between the widths of neighbouring cells outside the fine zone


@dataclass(kw_only=True)  # Allow keyword-only arguments, with any order of fields
class FieldModel:
    """MODFLOW 6 model of wells in a homogeneous confined aquifer with uniform regional groundwater flow and reactive transport.

    Time-dependent inputs are defined as [(t, value), ...]: t [s] from the simulation start,
    first t = 0, each value constant until the next t.

    Stress periods are then defined based on the times of the vector, flow is steady state
    within each stress period.

    By default, the model simulates the transport of a single tracer without reactions.

    If `phreeqc_coupling` is enabled, the transport of the components of PHREEQC solutions is simulated, with reactions
    computed by PHREEQC (MODFLOW 6 and PHREEQC coupled by mf6rtm).


    Args:
        workspace: Folder of the MODFLOW 6 files.
        wells: Wells, with unique names.
        flow_rates: Flow rate Q [m³ s⁻¹] by well name, constant or schedule; positive for injection, negative for
            extraction. Wells not listed are monitoring wells (Q = 0).
        domain_size: Size of the domain along x and y [m], centred on the pumped wells.
        top: Elevation of the top of the aquifer [m].
        bottom: Elevation of the bottom of the aquifer [m].
        n_layers: Number of layers; one is enough when all pumped wells are screened over the whole aquifer.
        hydraulic_conductivity: Horizontal hydraulic conductivity K [m s⁻¹].
        vertical_anisotropy: Ratio of vertical to horizontal hydraulic conductivity Kz/K [-].
        porosity: Effective porosity n [-].
        reference_head: Hydraulic head h at the domain centre without pumping [m].
        regional_hydraulic_gradient: Regional hydraulic gradient i [m/m].
        regional_flow_azimuth: Direction of the regional flow, clockwise from north [°].
        tracer_concentration: Tracer concentration of the injected water, constant or schedule, for all
            injection wells. The concentration is 0 in the aquifer at the start and in the inflow across the
            boundary. Default is 1.0.
        advection_scheme: "upstream", "central", or "tvd" (total variation diminishing).
        dispersion: If False, advection only.
        longitudinal_dispersivity: Longitudinal dispersivity alpha_L [m].
        transverse_dispersivity: Horizontal transverse dispersivity alpha_T [m].
        vertical_dispersivity: Vertical transverse dispersivity alpha_V [m].
        diffusion_coefficient: Effective molecular diffusion coefficient D* [m² s⁻¹].
        grid_spacing: Largest cell width Δx [m].
        grid_spacing_at_wells: Cell width Δx around the pumped wells [m].
        refinement_margin: Distance from the outermost pumped wells to the edge of the fine cells [m].
        simulation_time: Simulated time [s].
        time_step: Largest time step Δt [s], constant or schedule.
        phreeqc_coupling: If True, every component of the PHREEQC solutions is transported instead of the tracer
            (C in mol m⁻³), and PHREEQC computes the reactions in every cell at every time step.
        database: PHREEQC database file, needed with `phreeqc_coupling`; kinetic rates must be in its RATES block.
        solutions: PHREEQC solutions {number: {component: concentration [mol kgw⁻¹], "pH": pH, "pe": pe}},
            numbered 1, 2, ...; needed with `phreeqc_coupling`.
        initial_solution: Solution number in the aquifer at the start and in the inflow across the boundary, for
            all cells or as an array (layer, row, column).
        injected_solution: Solution number of the injected water, constant or schedule, for all injection wells.
        kinetics: PHREEQC KINETICS blocks {number: {reactant: {"m0": moles, "parms": [...], "formula": "..."}}},
            numbered 1, 2, ...
        initial_kinetics: KINETICS block number (-1 for none), for all cells or as an array (layer, row, column),
            e.g. from `zone`.
    """

    workspace: str | Path
    wells: list
    flow_rates: dict
    domain_size: tuple
    top: float
    bottom: float
    n_layers: int = 1
    hydraulic_conductivity: float
    vertical_anisotropy: float = 1.0
    porosity: float
    reference_head: float
    regional_hydraulic_gradient: float = 0.0
    regional_flow_azimuth: float = 0.0
    tracer_concentration: float | list = 1.0
    advection_scheme: str = "upstream"
    dispersion: bool = False
    longitudinal_dispersivity: float = 0.0
    transverse_dispersivity: float = 0.0
    vertical_dispersivity: float = 0.0
    diffusion_coefficient: float = 0.0
    grid_spacing: float
    grid_spacing_at_wells: float
    refinement_margin: float = 5.0
    simulation_time: float
    time_step: float | list
    phreeqc_coupling: bool = False
    database: str | Path | None = None
    solutions: dict | None = None
    initial_solution: int | np.ndarray = 1
    injected_solution: int | list = 1
    kinetics: dict | None = None
    initial_kinetics: int | np.ndarray = -1
    grid: StructuredGrid | None = field(default=None, init=False, repr=False)
    simulation: flopy.mf6.MFSimulation | None = field(default=None, init=False, repr=False)
    mup3d: Mup3d | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        names = [w.name for w in self.wells]
        if len(set(names)) < len(names):
            raise ValueError("Well names must be unique.")
        unknown = sorted(set(self.flow_rates) - set(names))
        if unknown:
            raise ValueError(f"flow_rates given for unknown wells: {', '.join(unknown)}.")
        if self.phreeqc_coupling:
            if self.database is None or self.solutions is None:
                raise ValueError("phreeqc_coupling needs a database and solutions.")
            for name, blocks in {"solutions": self.solutions, "kinetics": self.kinetics or {}}.items():
                if sorted(blocks) != list(range(1, len(blocks) + 1)):
                    raise ValueError(f"{name} must be numbered 1, 2, ...")

    @property
    def _pumped_wells(self):
        """Wells for which a flow rate is specified."""
        return [w for w in self.wells if w.name in self.flow_rates]

    @property
    def stress_periods(self):
        """Stress periods: start [s], duration [s], number of time steps, Q [m³ s⁻¹] of each pumped well, C_in, and
        the injected solution (with `phreeqc_coupling`)."""
        time_step = _schedule(self.time_step)
        schedules = {name: _schedule(q) for name, q in self.flow_rates.items()}
        schedules["tracer_concentration"] = _schedule(self.tracer_concentration)
        if self.phreeqc_coupling:
            schedules["injected_solution"] = _schedule(self.injected_solution)
        times = {t for schedule in [time_step, *schedules.values()] for t, _ in schedule}
        start = np.array(sorted(t for t in times if t < self.simulation_time))
        duration = np.diff([*start, self.simulation_time])
        n_time_steps = np.ceil(duration / _values_at(time_step, start) - 1e-9).astype(int)
        table = pd.DataFrame({"start": start, "duration": duration, "n_time_steps": n_time_steps})
        for name, schedule in schedules.items():
            table[name] = _values_at(schedule, start)
        if self.phreeqc_coupling:
            table["injected_solution"] = table["injected_solution"].astype(int)
        return table

    def head_from_gradient(self, x, y):
        """Hydraulic head of the regional flow without pumping, h = h_ref - i d.

        d is the distance from the domain centre (centre of the pumped wells) along the regional flow direction.

        Args:
            x: x coordinates [m].
            y: y coordinates [m].

        Returns:
            Hydraulic head h [m], with the shape of x and y.
        """
        x_wells = [w.x for w in self._pumped_wells]
        y_wells = [w.y for w in self._pumped_wells]
        azimuth = math.radians(self.regional_flow_azimuth)
        distance = (np.asarray(x) - (min(x_wells) + max(x_wells)) / 2) * math.sin(azimuth)
        distance += (np.asarray(y) - (min(y_wells) + max(y_wells)) / 2) * math.cos(azimuth)
        return self.reference_head - self.regional_hydraulic_gradient * distance

    def build(self):
        """Build the grid, the flopy simulation, and with `phreeqc_coupling` the mf6rtm model.

        MODFLOW 6 files are written by `run`; mf6rtm writes its PHREEQC files here.
        """
        pumped = self._pumped_wells
        grid = self._structured_grid()
        periods = self.stress_periods

        # Create the MF6 simulation object and the temporal discretization package (TDIS)
        simulation = flopy.mf6.MFSimulation(sim_name="field", sim_ws=str(self.workspace))
        flopy.mf6.ModflowTdis(
            simulation,
            time_units="seconds",
            nper=len(periods),
            perioddata=[(d, n, 1.0) for d, n in zip(periods["duration"], periods["n_time_steps"])],
        )

        # Initialise the Iterative Model Solution (IMS) package
        flopy.mf6.ModflowIms(
            simulation, filename="gwf.ims", complexity="simple", outer_dvclose=1e-6, inner_dvclose=1e-8
        )

        # Create the groundwater flow (GWF) object
        gwf = flopy.mf6.ModflowGwf(simulation, modelname="gwf", save_flows=True)
        dis = {
            "length_units": "meters",
            "nlay": grid.nlay,
            "nrow": grid.nrow,
            "ncol": grid.ncol,
            "delr": grid.delr,
            "delc": grid.delc,
            "top": grid.top,
            "botm": grid.botm,
            "xorigin": grid.xoffset,
            "yorigin": grid.yoffset,
        }
        flopy.mf6.ModflowGwfdis(gwf, **dis)

        # Create the node property flow (NPF) package for the groundwater flow model
        flopy.mf6.ModflowGwfnpf(
            gwf,
            icelltype=0,  # confined: constant saturated thickness
            k=self.hydraulic_conductivity,
            k33=self.hydraulic_conductivity * self.vertical_anisotropy,
            save_specific_discharge=True,
        )

        # Initial heads and lateral boundary heads from the regional flow
        head = self.head_from_gradient(grid.xcellcenters, grid.ycellcenters)

        # Create the initial conditions (IC) package for the groundwater flow model
        flopy.mf6.ModflowGwfic(gwf, strt=np.broadcast_to(head, grid.shape))

        # Create the constant head (CHD) package for the groundwater flow model; tracer-free inflow (C = 0)
        rows, columns = np.indices((grid.nrow, grid.ncol))
        boundary = (rows == 0) | (rows == grid.nrow - 1) | (columns == 0) | (columns == grid.ncol - 1)
        chd = [((k, i, j), head[i, j], 0.0) for k in range(grid.nlay) for i, j in zip(*np.nonzero(boundary))]
        flopy.mf6.ModflowGwfchd(gwf, pname="chd", auxiliary="concentration", stress_period_data={0: chd})

        # Create the well (WEL) package for the groundwater flow model, with C_in (used only where Q > 0)
        # All screened cells of all pumped wells in every stress period, in the same order (Q = 0 when off)
        cells = {w.name: _screened_cells(w, grid) for w in pumped}
        wel = {
            kper: [
                (cell, weight * period[name], period["tracer_concentration"], name)
                for name in cells
                for cell, weight in cells[name]
            ]
            for kper, period in enumerate(periods.to_dict("records"))
        }
        flopy.mf6.ModflowGwfwel(gwf, pname="wel", auxiliary="concentration", boundnames=True, stress_period_data=wel)

        # Create the output control (OC) package for the groundwater flow model
        flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord="gwf.hds",
            budget_filerecord="gwf.cbc",
            saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
            printrecord=[("BUDGET", "LAST")],
        )

        # Groundwater transport (GWT) model of the tracer, solved after the flow at every time step
        gwt = flopy.mf6.ModflowGwt(simulation, modelname="gwt")
        ims = flopy.mf6.ModflowIms(
            simulation,
            filename="gwt.ims",
            complexity="simple",
            linear_acceleration="bicgstab",
            outer_dvclose=1e-6,
            inner_dvclose=1e-8,
        )
        simulation.register_ims_package(ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, **dis)
        flopy.mf6.ModflowGwtic(gwt, strt=0.0)
        flopy.mf6.ModflowGwtmst(gwt, porosity=self.porosity)
        flopy.mf6.ModflowGwtadv(gwt, scheme=self.advection_scheme)
        if self.dispersion:
            flopy.mf6.ModflowGwtdsp(
                gwt,
                alh=self.longitudinal_dispersivity,
                ath1=self.transverse_dispersivity,
                ath2=self.vertical_dispersivity,  # vertical transverse spreading for horizontal flow
                diffc=self.diffusion_coefficient,
            )
        # Source and sink mixing (SSM): concentration of the water entering through WEL and CHD
        flopy.mf6.ModflowGwtssm(gwt, sources=[("wel", "AUX", "concentration"), ("chd", "AUX", "concentration")])
        flopy.mf6.ModflowGwtoc(
            gwt,
            concentration_filerecord="gwt.ucn",
            saverecord=[("CONCENTRATION", "ALL")],
            printrecord=[("BUDGET", "ALL")],
        )
        flopy.mf6.ModflowGwfgwt(simulation, exgtype="GWF6-GWT6", exgmnamea=gwf.name, exgmnameb=gwt.name)
        self.grid, self.simulation = grid, simulation
        self.mup3d = self._mup3d() if self.phreeqc_coupling else None

    def zone(self, polygon, z_top=None, z_bottom=None):
        """Cells whose centres lie inside a polygon and between two elevations (e.g., a contaminated zone).

        The model is not built, so the zone can set `initial_kinetics` or `initial_solution` before `run`.

        Args:
            polygon: Polygon in the coordinates of the wells: shapely geometry, GeoDataFrame, or file (e.g., shapefile).
            z_top: Upper elevation [m]; default: top of the aquifer.
            z_bottom: Lower elevation [m]; default: bottom of the aquifer.

        Returns:
            Boolean array (layer, row, column).
        """
        import geopandas
        import shapely

        grid = self._structured_grid() if self.grid is None else self.grid
        if isinstance(polygon, (str, Path)):
            polygon = geopandas.read_file(polygon)
        if isinstance(polygon, geopandas.GeoDataFrame):
            polygon = polygon.union_all()
        inside = shapely.contains_xy(polygon, grid.xcellcenters, grid.ycellcenters)
        z_top = self.top if z_top is None else z_top
        z_bottom = self.bottom if z_bottom is None else z_bottom
        return inside & (grid.zcellcenters <= z_top) & (grid.zcellcenters >= z_bottom)

    def _structured_grid(self):
        """Grid of the model, centred on the pumped wells."""
        return structured_grid(
            self._pumped_wells,
            self.domain_size,
            self.top,
            self.bottom,
            self.n_layers,
            self.grid_spacing,
            self.grid_spacing_at_wells,
            self.refinement_margin,
        )

    def _mup3d(self):
        """mf6rtm model of the simulation with the PHREEQC solutions, kinetics, and chemistry of the inflow."""
        # mf6rtm solutions: {component: [concentration in solution 1, 2, ...]}; PHREEQC defaults where not given
        names = sorted({name for solution in self.solutions.values() for name in solution})
        defaults = {"pH": 7.0, "pe": 4.0}
        numbers = sorted(self.solutions)
        solutions = Solutions(
            {name: [self.solutions[n].get(name, defaults.get(name, 0.0)) for n in numbers] for name in names}
        )
        solutions.set_ic(np.broadcast_to(self.initial_solution, self.grid.shape).astype(int))

        workspace = Path(self.workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        with open(workspace / "mf6rtm.log", "w") as log, contextlib.redirect_stdout(log):
            # from_mf6 creates a default working directory next to the workspace (with a warning): removed
            default = workspace.parent / "reactive"
            created = not default.exists()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mup3d = Mup3d.from_mf6(self.simulation, solutions)
            if created and default != workspace:
                default.rmdir()
            mup3d.set_wd(workspace)
            database = Path(self.database).resolve()
            if database.parent == workspace:  # set_database copies the file into the workspace
                mup3d.database = str(database)
            else:
                mup3d.set_database(str(database))
            if self.kinetics:
                phases = KineticPhases(copy.deepcopy(dict(sorted(self.kinetics.items()))))
                phases.set_ic(np.broadcast_to(self.initial_kinetics, self.grid.shape).astype(int))
                mup3d.set_phases(phases)
            mup3d.initialize()
            mup3d.set_diffusion_coeff(dict.fromkeys(mup3d.components, self.diffusion_coefficient))

            # Chemistry of the water entering through the wells (injected solution) and the boundary (initial solution)
            gwf = self.simulation.get_model("gwf")
            n_records = len(gwf.wel.stress_period_data.get_data(0))
            wel = ChemStress("wel")
            wel.set_spd({kper: [int(n)] * n_records for kper, n in enumerate(self.stress_periods["injected_solution"])})
            mup3d.set_chem_stress(wel)
            chd = ChemStress("chd")
            chd.set_spd([int(solutions.ic[tuple(cell)]) for cell in gwf.chd.stress_period_data.get_data(0)["cellid"]])
            mup3d.set_chem_stress(chd)
        return mup3d

    def run(self, n_threads=1):
        """Write the files and run MODFLOW 6, or mf6rtm with `phreeqc_coupling`; build first if needed.

        Args:
            n_threads: Number of threads for PHREEQC with `phreeqc_coupling`; -1 for all processors.
        """
        if self.simulation is None:
            self.build()
        if not self.phreeqc_coupling:
            self.simulation.write_simulation(silent=True)
            success, _ = self.simulation.run_simulation(silent=True)
            if not success:
                raise RuntimeError(f"MODFLOW 6 failed; see {Path(self.workspace) / 'mfsim.lst'}.")
            return
        workspace = Path(self.workspace).resolve()
        with open(workspace / "mf6rtm.log", "a") as log:
            if "gwt" in self.simulation.model_names:  # not written yet: mf6rtm replaces the tracer by the components
                with contextlib.redirect_stdout(log):
                    self.mup3d.write_simulation()
            log.flush()
            # Separate process, because an error of MODFLOW 6 stops the process running its library
            n_threads = os.cpu_count() if n_threads == -1 else n_threads
            solve = "import sys, mf6rtm; sys.exit(not mf6rtm.solve('.', nthread=int(sys.argv[1]), libname=sys.argv[2]))"
            command = [sys.executable, "-c", solve, str(n_threads), str(_libmf6())]
            result = subprocess.run(command, cwd=workspace, stdout=log, stderr=log)
        if result.returncode != 0:
            raise RuntimeError(f"mf6rtm failed; see {workspace / 'mf6rtm.log'} and {workspace / 'mfsim.lst'}.")

    def head(self, time=None):
        """Hydraulic head h [m] at the end of the time step containing time t (default: end of the simulation).

        Args:
            time: Time t [s] from the simulation start.

        Returns:
            Array (layer, row, column).
        """
        return _saved_array(self.simulation.get_model("gwf").output.head(), time)

    def concentration(self, time=None, component=None):
        """Concentration C at the end of the time step containing time t (default: end of the simulation).

        Args:
            time: Time t [s] from the simulation start.
            component: PHREEQC component with `phreeqc_coupling` (C in mol m⁻³); default: the tracer.

        Returns:
            Array (layer, row, column).
        """
        return _saved_array(self._transport_model(component).output.concentration(), time)

    def well_concentration(self, name, component=None):
        """Concentration of a well at the end of every time step, C_w = Σ w_k C_k with w_k = b_k / Σ b.

        b_k is the screened thickness in cell k, so C_w is the flow-weighted mean of a pumped well.

        Args:
            name: Well name.
            component: PHREEQC component with `phreeqc_coupling` (C in mol m⁻³); default: the tracer.

        Returns:
            DataFrame with time [s] and concentration.
        """
        well = {w.name: w for w in self.wells}[name]
        cells = _screened_cells(well, self.grid)
        series = self._transport_model(component).output.concentration().get_ts([cell for cell, _ in cells])
        return pd.DataFrame({"time": series[:, 0], "concentration": series[:, 1:] @ [w for _, w in cells]})

    def _transport_model(self, component):
        """Transport model of the tracer (component None) or of a PHREEQC component."""
        if (component is None) == self.phreeqc_coupling:
            raise ValueError("Give a PHREEQC component with phreeqc_coupling, and none without.")
        return self.simulation.get_model(component or "gwt")

    def mass_balance(self):
        """Cumulative tracer mass at the end of every time step, from the MODFLOW 6 budget.

        Returns:
            DataFrame with time [s] and the mass (concentration x m³) injected, extracted, entering and leaving
            across the lateral boundary, and dissolved in the aquifer.
        """
        _, budget = self._transport_model(None).output.list().get_dataframes(start_datetime=None)
        periods = self.stress_periods
        time_steps = (periods["duration"] / periods["n_time_steps"]).to_numpy()
        time = np.cumsum(np.repeat(time_steps, periods["n_time_steps"]))
        return pd.DataFrame(
            {
                "time": time,
                "injected": budget["WEL_IN"].to_numpy(),
                "extracted": budget["WEL_OUT"].to_numpy(),
                "boundary_inflow": budget["CHD_IN"].to_numpy(),
                "boundary_outflow": budget["CHD_OUT"].to_numpy(),
                "in_aquifer": (budget["STORAGE-AQUEOUS_OUT"] - budget["STORAGE-AQUEOUS_IN"]).to_numpy(),
            }
        )


def intermittent_pumping(value, on_duration, off_duration, end_time, start_time=0.0):
    """Schedule switching between value (on) and 0 (off), from start_time to end_time; 0 before and after.

    Args:
        value: Value during the on phases, e.g., Q [m³ s⁻¹].
        on_duration: Duration of each on phase [s].
        off_duration: Duration of each off phase [s].
        end_time: End of the last phase [s].
        start_time: Start of the first on phase [s].

    Returns:
        Schedule ``[(t, value), ...]``, t [s].
    """
    if on_duration <= 0 or off_duration <= 0:
        raise ValueError("on_duration and off_duration must be positive.")
    schedule = [(0.0, 0.0)] if start_time > 0 else []
    t = start_time
    while t < end_time:
        schedule += [(t, value), (min(t + on_duration, end_time), 0.0)]
        t += on_duration + off_duration
    return schedule


def structured_grid(
    wells, domain_size, top, bottom, n_layers, grid_spacing, grid_spacing_at_wells, refinement_margin=5.0
):
    """Structured grid centred on the wells and refined around them.

    Args:
        wells: Wells used to refine the grid (e.g., pumped wells).
        domain_size: Size of the domain along x and y [m].
        top: Elevation of the top of the aquifer [m].
        bottom: Elevation of the bottom of the aquifer [m].
        n_layers: Number of layers.
        grid_spacing: Largest cell width Δx [m].
        grid_spacing_at_wells: Cell width Δx around the wells [m].
        refinement_margin: Distance from the outermost wells to the edge of the fine cells [m].

    Returns:
        flopy StructuredGrid in the coordinates of the wells; row 1 is the northernmost.
    """
    if not wells:
        raise ValueError("At least one well is needed.")
    if grid_spacing_at_wells > grid_spacing:
        raise ValueError("grid_spacing_at_wells must not exceed grid_spacing.")
    x = [w.x for w in wells]
    y = [w.y for w in wells]
    xorigin = (min(x) + max(x) - domain_size[0]) / 2
    yorigin = (min(y) + max(y) - domain_size[1]) / 2
    spacing = {"coarse": grid_spacing, "fine": grid_spacing_at_wells, "margin": refinement_margin}
    delr = _cell_widths(x, xorigin, xorigin + domain_size[0], **spacing)
    delc = _cell_widths(y, yorigin, yorigin + domain_size[1], **spacing)[::-1]  # MODFLOW rows run north to south
    n_rows, n_columns = len(delc), len(delr)
    botm = np.linspace(top, bottom, n_layers + 1)[1:]
    return StructuredGrid(
        delc=delc,
        delr=delr,
        top=np.full((n_rows, n_columns), float(top)),
        botm=np.repeat(botm, n_rows * n_columns).reshape(n_layers, n_rows, n_columns),
        xoff=xorigin,
        yoff=yorigin,
    )


def _schedule(value):
    """Schedule ``[(t, value), ...]`` from a schedule or a constant."""
    if np.isscalar(value):
        return [(0.0, float(value))]
    times = [t for t, _ in value]
    if not times or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError(f"Schedule times must start at 0 and increase: {times}.")
    return [(float(t), float(v)) for t, v in value]


def _values_at(schedule, times):
    """Values of a schedule at the given times."""
    schedule_times, values = np.array(schedule).T
    return values[np.searchsorted(schedule_times, times, side="right") - 1]


def _libmf6():
    """Path of the MODFLOW 6 library, next to the mf6 program (both installed by get-modflow)."""
    name = {"win32": "libmf6.dll", "darwin": "libmf6.dylib"}.get(sys.platform, "libmf6.so")
    mf6 = shutil.which("mf6")
    if mf6 is None or not Path(mf6).with_name(name).exists():
        raise FileNotFoundError(f"{name} not found next to mf6; install both with get-modflow.")
    return Path(mf6).with_name(name)


def _saved_array(output, time):
    """Array of a flopy output file at the end of the time step containing time [s] (default: last saved)."""
    times = output.get_times()
    saved = times[-1] if time is None else times[min(np.searchsorted(times, time - 1e-3), len(times) - 1)]
    return output.get_data(totim=saved)


def _screened_cells(well, grid):
    """Cells (layer, row, column) crossed by a well screen, with weights proportional to the screened thickness."""
    row, column = grid.intersect(well.x, well.y, forgive=True)
    if np.isnan(row):
        raise ValueError(f"Well {well.name} lies outside the domain.")
    row, column = int(row), int(column)
    tops, bottoms = grid.top_botm[:-1, row, column], grid.top_botm[1:, row, column]
    thickness = np.minimum(tops, well.screen_top) - np.maximum(bottoms, well.screen_bottom)
    screened = np.flatnonzero(thickness > 0)
    if screened.size == 0:
        raise ValueError(f"Well {well.name}: the screen does not cross the aquifer.")
    return [((k, row, column), thickness[k] / thickness[screened].sum()) for k in screened]


def _cell_widths(coordinates, start, end, coarse, fine, margin):
    """Cell widths from start to end along one axis: fine cells centred on the wells, growing to coarse ones."""
    coordinates = np.sort(coordinates)
    # Wells closer than `fine` share a block of fine cells; a single well has one cell centred on it.
    clusters = np.split(coordinates, np.flatnonzero(np.diff(coordinates) >= fine) + 1)
    widths = []
    for previous, cluster in zip([None, *clusters], clusters):
        block = _equal_widths(cluster[-1] - cluster[0] + fine, fine)
        if previous is not None:
            gap = cluster[0] - previous[-1] - fine
            filler = _equal_widths(gap, fine)
            if not filler:  # a gap narrower than fine / 2 is shared by the cells on either side
                widths[-1] += gap / 2
                block[0] += gap / 2
            widths += filler
        widths += block
    n_margin = round(margin / fine)
    widths = [fine] * n_margin + widths + [fine] * n_margin

    # Outside the fine zone, the widths grow towards the domain boundaries.
    left = coordinates[0] - fine / 2 - n_margin * fine - start
    right = end - coordinates[-1] - fine / 2 - n_margin * fine
    if min(left, right) < 0:
        raise ValueError("The domain is too small for the wells and the refinement margin.")
    if left < fine / 2:  # too short for a cell of its own
        widths[0] += left
        left = 0.0
    if right < fine / 2:
        widths[-1] += right
        right = 0.0
    return np.array(_growing_widths(left, fine, coarse)[::-1] + widths + _growing_widths(right, fine, coarse))


def _equal_widths(length, spacing):
    """Equal widths, as close as possible to spacing, that fill length; none if length < spacing / 2."""
    n = round(length / spacing)
    return [length / n] * n if n else []


def _growing_widths(length, fine, coarse):
    """Widths growing from fine by MAX_GROWTH_FACTOR."""
    widths = []
    while sum(widths) < length:
        widths.append(min(fine * MAX_GROWTH_FACTOR ** (len(widths) + 1), coarse))
    return [w * length / sum(widths) for w in widths]
