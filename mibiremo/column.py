"""Creates a laboratory column model for simulating the reactive transport
    processes during the flushing of a solution through a column.

Contents:
- `ColumnModel`: advection and dispersion (semi-Lagrangian solver) coupled with PHREEQC reactions (PhreeqcRM)
"""

import math
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import numpy as np
import pandas as pd
from .field import _schedule
from .field import _stress_periods
from .phreeqc import PhreeqcRM
from .semilagsolver import SemiLagSolver


@dataclass(kw_only=True)  # Allow keyword-only arguments, with any order of fields
class ColumnModel:
    """Laboratory column model: 1D advection, dispersion, and PHREEQC reactions.

    The column is divided into cells of length Δx = L / n_cells. The concentration of each cell is computed at its
    downstream end, x = Δx, 2 Δx, ..., L, so the last cell gives the effluent. The influent enters at x = 0 with a fixed
    concentration; the outlet has a zero concentration gradient (Neumann boundary condition).
    At every time step, every component of the PHREEQC solutions is transported (`SemiLagSolver`),
    then PHREEQC computes the reactions in every cell (PhreeqcRM).

    Time-dependent inputs are defined as [(t, value), ...], as in `FieldModel`: t [s] from the start of the
    simulation, first t = 0, each value constant until the next t. Stress periods start at every change time.

    Args:
        workspace: Folder of the PHREEQC input file and of the PhreeqcRM output files.
        length: Length of the column L [m].
        diameter: Inner diameter of the column [m].
        porosity: Effective porosity n [-].
        flow_rate: Volumetric flow rate Q ≥ 0 [m³ s⁻¹], constant or schedule; 0 stops the flow.
        longitudinal_dispersivity: Longitudinal dispersivity alpha_L [m].
        diffusion_coefficient: Effective molecular diffusion coefficient D* [m² s⁻¹].
        n_cells: Number of cells.
        simulation_time: Simulated time [s].
        time_step: Largest time step Δt [s], constant or schedule.
        database: PHREEQC database file; kinetic rates must be in its RATES block.
        solutions: PHREEQC solutions {number: {component: concentration [mol kgw⁻¹], "pH": pH, "pe": pe}},
            numbered 1, 2, ...; pH 7 and pe 4 where not given.
        initial_solution: Solution number of the pore water at the start, for all cells or one per cell.
        influent_solution: Solution number of the influent, constant or schedule.
        kinetics: PHREEQC KINETICS blocks {number: {reactant: {"m0": moles per litre of water, "parms": [...],
            "formula": "..."}}}, numbered 1, 2, ...
        initial_kinetics: KINETICS block number (-1 for none), for all cells or one per cell.
    """

    workspace: str | Path
    length: float
    diameter: float
    porosity: float
    flow_rate: float | list
    longitudinal_dispersivity: float = 0.0
    diffusion_coefficient: float = 0.0
    n_cells: int
    simulation_time: float
    time_step: float | list
    database: str | Path
    solutions: dict
    initial_solution: int | np.ndarray = 1
    influent_solution: int | list = 1
    kinetics: dict | None = None
    initial_kinetics: int | np.ndarray = -1
    phreeqc: PhreeqcRM | None = field(default=None, init=False, repr=False)
    components: list | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        for name, blocks in {"solutions": self.solutions, "kinetics": self.kinetics or {}}.items():
            if sorted(blocks) != list(range(1, len(blocks) + 1)):
                raise ValueError(f"{name} must be numbered 1, 2, ...")
        if any(q < 0 for _, q in _schedule(self.flow_rate)):
            raise ValueError("The flow rate must not be negative.")

    @property
    def x(self) -> np.ndarray:
        """Distance from the inlet of the downstream end of each cell, where its concentration is computed [m]."""
        return np.arange(1, self.n_cells + 1) * self.length / self.n_cells

    @property
    def pore_volume(self) -> float:
        """Pore volume of the column V_p = n A L [m³]."""
        return self.porosity * math.pi * self.diameter**2 / 4 * self.length

    @property
    def stress_periods(self) -> pd.DataFrame:
        """Stress periods: start [s], duration [s], number of time steps, Q [m³ s⁻¹], and influent solution."""
        schedules = {"flow_rate": self.flow_rate, "influent_solution": self.influent_solution}
        table = _stress_periods(schedules, self.time_step, self.simulation_time)
        table["influent_solution"] = table["influent_solution"].astype(int)
        return table

    def run(self, n_threads: int = 1) -> None:
        """Write the PHREEQC input file and run the simulation.

        Args:
            n_threads: Number of threads for PHREEQC; -1 for all processors.
        """
        workspace = Path(self.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "column.pqi").write_text(self._phreeqc_input())

        # Concentrations in mol/L, other reactants in mol per litre of water; PhreeqcRM porosity 1: each cell holds
        # 1 L of water, and the porosity of the column acts on the transport only (as in mf6rtm)
        phreeqc = PhreeqcRM()
        phreeqc.create(nxyz=self.n_cells, n_threads=n_threads)
        phreeqc.initialize_phreeqc(
            str(self.database), units_solution=2, units=1, multicomponent=False, file_prefix=str(workspace / "phreeqc")
        )
        initial = np.full((self.n_cells, 7), -1)
        initial[:, 0] = np.broadcast_to(self.initial_solution, self.n_cells)
        initial[:, 6] = np.broadcast_to(self.initial_kinetics, self.n_cells)
        phreeqc.run_initial_from_file(str(workspace / "column.pqi"), initial)
        rm = phreeqc.rm
        n_components = len(phreeqc.components)

        periods = self.stress_periods
        influent = {n: np.asarray(rm.InitialPhreeqc2Concentrations([n])) for n in set(periods["influent_solution"])}
        area = math.pi * self.diameter**2 / 4
        concentration = np.reshape(rm.GetConcentrations(), (n_components, self.n_cells)).T
        time, volume = 0.0, 0.0
        times, volumes, saved = [time], [volume], [concentration]
        for period in periods.to_dict("records"):
            dt = period["duration"] / period["n_time_steps"]
            velocity = period["flow_rate"] / (area * self.porosity)  # average linear velocity v = Q / (n A)
            dispersion = self.longitudinal_dispersivity * velocity + self.diffusion_coefficient  # D = alpha_L v + D*
            for _ in range(period["n_time_steps"]):
                solver = SemiLagSolver(self.x, concentration, velocity, dispersion, dt)
                rm.SetConcentrations(solver.transport(influent[period["influent_solution"]]).T.ravel())
                time, volume = time + dt, volume + period["flow_rate"] * dt
                rm.SetTime(time)
                rm.SetTimeStep(dt)
                if rm.RunCells() < 0:
                    raise RuntimeError(f"PHREEQC failed; see {workspace / 'phreeqc.log.txt'}.")
                concentration = np.reshape(rm.GetConcentrations(), (n_components, self.n_cells)).T
                times.append(time)
                volumes.append(volume)
                saved.append(concentration)
        self.phreeqc, self.components = phreeqc, list(phreeqc.components)
        self._times, self._volumes = np.array(times), np.array(volumes)
        self._concentrations = 1000.0 * np.array(saved)  # mol/L -> mol m⁻³

    def concentration(self, component: str, time: float | None = None) -> np.ndarray:
        """Concentration profile C [mol m⁻³] at the end of the time step containing time t (default: end).

        Args:
            component: PHREEQC component.
            time: Time t [s] from the start of the simulation.

        Returns:
            Array with the concentration of each cell, at `x`.
        """
        step = len(self._times) - 1 if time is None else np.searchsorted(self._times, time - 1e-3)
        return self._concentrations[min(step, len(self._times) - 1), :, self.components.index(component)]

    def effluent(self, component: str) -> pd.DataFrame:
        """Effluent concentration at the end of every time step (breakthrough curve).

        Args:
            component: PHREEQC component.

        Returns:
            DataFrame with time [s], pore volumes flushed V/V_p [-], and concentration C [mol m⁻³].
        """
        return pd.DataFrame(
            {
                "time": self._times,
                "pore_volumes": self._volumes / self.pore_volume,
                "concentration": self._concentrations[:, -1, self.components.index(component)],
            }
        )

    def _phreeqc_input(self):
        """PHREEQC SOLUTION and KINETICS blocks of `solutions` and `kinetics`."""
        lines = []
        for number, solution in sorted(self.solutions.items()):
            lines += [f"SOLUTION {number}", "    units mol/kgw"]
            lines += [f"    {name} {value}" for name, value in ({"pH": 7.0, "pe": 4.0} | solution).items()]
        for number, block in sorted((self.kinetics or {}).items()):
            lines.append(f"KINETICS {number}")
            for reactant, settings in block.items():
                lines.append(f"    {reactant}")
                for key, value in settings.items():
                    value = " ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else value
                    lines.append(f"        -{key} {value}")
        return "\n".join([*lines, "END", ""])
