# Batch model

`mibiremo.BatchModel` simulates laboratory batch experiments, e.g. soil mesocosms. PHREEQC reactions
(PhreeqcRM) are simulated in independent batch reactors, without transport. The inputs follow the conventions of the
[field](field_model.md) and [column](column_model.md) models, so the same PHREEQC solutions and kinetics can be used
in all three; the equilibrium phases are shared with the column model.

Example: [Aerobic biodegradation in mesocosms](notebooks/batch_first_order_decay.ipynb).

## Workflow

```python
import mibiremo as mb

day = 86400.0
database = mb.write_database(                      # phreeqc.dat + rate and its parameters (CALCULATE_VALUES)
    "output/mesocosms.dat", rates, parameters={"k": 0.02 / day, "K_o2": 0.1 / 32000}
)
model = mb.BatchModel(
    workspace="output/mesocosms",
    simulation_time=150 * day,
    time_step=day,                                 # constant or time-dependent
    database=database,
    solutions={1: pore_water},                     # {component: mol/kgw, "pH": ..., "pe": ...}
    equilibrium_phases={1: {"O2(g)": [-0.68, 10], "CO2(g)": [-3.4, 10]}},  # {phase: [SI, mol/L water]}
    initial_equilibrium_phases=[1, -1],            # for all reactors or one per reactor (-1 for none)
    kinetics={1: reaction},                        # PHREEQC KINETICS blocks
    initial_kinetics=[1, 1],                       # one per reactor: two reactors
)
model.run()
tph = model.kinetic_reactant("Hydrocarbon")        # DataFrame: index time [s], one column per reactor
dissolved = model.concentration("C")               # DataFrame: index time [s], one column per reactor
```

## Conventions

- **Reactors**: the number of reactors is determined by the size of `initial_kinetics` (`model.n_reactors`); 
  e.g., a single entry results in one reactor. The other initial conditions can be given for all reactors or one per reactor.
- **Units**: SI: s, mol m⁻³ (= mmol L⁻¹) for the concentrations. Each reactor contains 1 L of water: the
  PHREEQC solutions are in mol kgw⁻¹, and the KINETICS amounts `m0` and the EQUILIBRIUM_PHASES amounts in moles
  per litre of water. A PHREEQC rate is in moles per reactor: a rate written per kg of water (e.g. with `TOT`) is
  multiplied by `TOT("water")`.
- **Time steps**: `time_step` is the largest time step, constant or a list `[(t, value), ...]` starting at t = 0.
  The results are stored at the end of every time step.
- **Rate parameters**: `write_database` adds definitions (text or files) and parameters to the PHREEQC database 
  (`phreeqc.dat` by default) or to any database file. The parameters are written as CALCULATE_VALUES and read in the
  rates with `CALC_VALUE("name")`.

## Results

- `concentration(component)`: concentration of a component in the reactors at every time step.
- `kinetic_reactant(reactant)`: amount of a KINETICS reactant, e.g. an immobile contaminant, in the reactors at every
  time step, in moles per litre of water as `m0`.
