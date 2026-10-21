"""Creates a batch model for simulating the reactions in batches, e.g. soil mesocosms.

Contents:
- `BatchModel`: PHREEQC reactions (PhreeqcRM) in batch reactors, without transport
"""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import numpy as np
import pandas as pd
from .column import _check_numbering
from .column import _reactant_amounts
from .column import _reactants
from .column import _start_phreeqc
from .field import _stress_periods
from .phreeqc import PhreeqcRM


@dataclass(kw_only=True)  # Allow keyword-only arguments, with any order of fields
class BatchModel:
    """Batch model: PHREEQC reactions in batch reactors.

    Each reactor contains 1 L of water. One reactor is created for each `initial_kinetics` entry; the other initial
    conditions can be given for all reactors or one per reactor.

    Args:
        workspace: Folder of the PHREEQC input file and of the PhreeqcRM output files.
        simulation_time: Simulation duration [s].
        time_step: Largest time step Δt [s], constant or as a vector of tuples [(t, value), ...].
        database: PHREEQC database file; kinetic rates must be in its RATES block.
        solutions: PHREEQC solutions {number: {component: concentration [mol kgw⁻¹], "pH": pH, "pe": pe}},
            numbered 1, 2, ...; pH 7 and pe 4 where not given.
        initial_solution: Solution number of the water at the start, for all reactors or one per reactor.
        equilibrium_phases: PHREEQC EQUILIBRIUM_PHASES blocks {number: {phase: [saturation index, moles per litre of
            water]}}, numbered 1, 2, ...
        initial_equilibrium_phases: EQUILIBRIUM_PHASES block number (-1 for none), for all reactors or one per reactor.
        kinetics: PHREEQC KINETICS blocks {number: {reactant: {"m0": moles per litre of water, "parms": [...],
            "formula": "..."}}}, numbered 1, 2, ...
        initial_kinetics: KINETICS block number (-1 for none), one per reactor; a single number for one reactor.
    """

    workspace: str | Path
    simulation_time: float
    time_step: float | list
    database: str | Path
    solutions: dict
    initial_solution: int | np.ndarray = 1
    equilibrium_phases: dict | None = None
    initial_equilibrium_phases: int | np.ndarray = -1
    kinetics: dict | None = None
    initial_kinetics: int | list | np.ndarray = -1
    phreeqc: PhreeqcRM | None = field(default=None, init=False, repr=False)
    components: list | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        _check_numbering(self)

    @property
    def n_reactors(self) -> int:
        """Number of reactors, determined by the size of `initial_kinetics`."""
        return np.size(self.initial_kinetics)

    def run(self, n_threads: int = 1) -> None:
        """Write the PHREEQC input file and run the simulation.

        Args:
            n_threads: Number of threads for PHREEQC; -1 for all processors.
        """
        phreeqc = _start_phreeqc(self, "batch.pqi", self.n_reactors, n_threads)
        rm = phreeqc.rm
        n_components = len(phreeqc.components)
        reactants = _reactants(self.kinetics)
        time = 0.0
        times = [time]
        saved = [np.reshape(rm.GetConcentrations(), (n_components, self.n_reactors)).T]
        saved_amounts = [_reactant_amounts(phreeqc, reactants)]
        for period in _stress_periods({}, self.time_step, self.simulation_time).to_dict("records"):
            dt = period["duration"] / period["n_time_steps"]
            for _ in range(period["n_time_steps"]):
                time += dt
                rm.SetTime(time)
                rm.SetTimeStep(dt)
                if rm.RunCells() < 0:
                    raise RuntimeError(f"PHREEQC failed; see {Path(self.workspace) / 'phreeqc.log.txt'}.")
                times.append(time)
                saved.append(np.reshape(rm.GetConcentrations(), (n_components, self.n_reactors)).T)
                saved_amounts.append(_reactant_amounts(phreeqc, reactants))
        self.phreeqc, self.components = phreeqc, list(phreeqc.components)
        self._times = np.array(times)
        self._concentrations = 1000.0 * np.array(saved)  # mol/L -> mol m⁻³; (time steps, reactors, components)
        self._amounts = np.array(saved_amounts)  # (time steps, reactors, reactants)

    def concentration(self, component: str) -> pd.DataFrame:
        """Concentration C [mol m⁻³] of a component in the reactors at the end of every time step.

        Args:
            component: PHREEQC component.

        Returns:
            DataFrame with the time t [s] as index and one column per reactor.
        """
        return self._table(self._concentrations[:, :, self.components.index(component)])

    def kinetic_reactant(self, reactant: str) -> pd.DataFrame:
        """Amount of a KINETICS reactant, e.g. an immobile contaminant, in every reactor at the end of every time step.

        Args:
            reactant: KINETICS reactant.

        Returns:
            DataFrame with the time t [s] as index and one column per reactor; amounts in moles per litre of water, as
            `m0`.
        """
        return self._table(self._amounts[:, :, _reactants(self.kinetics).index(reactant)])

    def _table(self, values):
        """DataFrame of values (time steps, reactors)."""
        return pd.DataFrame(
            values, index=pd.Index(self._times, name="time"), columns=pd.RangeIndex(self.n_reactors, name="reactor")
        )
