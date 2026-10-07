# Column model

`mibiremo.ColumnModel` simulates a laboratory column: a soil sample in a column flushed with a solution. It computes the
advection and dispersion of every component of the PHREEQC solutions with the semi-Lagrangian solver
(`SemiLagSolver`), and the reactions with PHREEQC through PhreeqcRM, sequentially at every time step
(sequential non-iterative approach). The inputs follow the conventions of the [field model](field_model.md), so the
same PHREEQC solutions and kinetics can be used in both.

Examples: [laboratory column test](notebooks/ex8-column_test.ipynb) and
[column validation against mibitrans](notebooks/ex9-column_validation-vs-mibitrans.ipynb).

## Workflow

```python
import math
import mibiremo as mb

minute, hour = 60.0, 3600.0
flow_rate = 1e-6 / minute  # 1 mL/min in m3/s

model = mb.ColumnModel(
    workspace="output/column",
    length=0.3,                                    # L [m]
    diameter=0.05,                                 # [m]
    porosity=0.38,
    flow_rate=[(0.0, flow_rate), (10 * hour, 0.0), (16 * hour, flow_rate)],  # Q [m3 s-1], stopped for 6 h
    longitudinal_dispersivity=0.005,               # alpha_L [m]
    n_cells=60,
    simulation_time=30 * hour,
    time_step=5 * minute,
    database="my_database.dat",                    # kinetic rates in its RATES block
    solutions={1: pore_water, 2: influent},        # {component: mol/kgw, "pH": ..., "pe": ...}
    initial_solution=1,
    influent_solution=2,                           # solution number, constant or schedule
    kinetics={1: reaction},                        # PHREEQC KINETICS blocks
    initial_kinetics=1,                            # KINETICS block number, for all cells or one per cell
)
model.run()
breakthrough = model.effluent("Tr")                # DataFrame: time [s], pore_volumes, concentration
profile = model.concentration("Tr", 2 * hour)      # array, one value per cell at model.x
```

## Conventions

- **Units**: SI throughout: m, s, m³ s⁻¹. Concentrations in mol m⁻³ (= mmol L⁻¹); the PHREEQC solutions are given in
  mol kgw⁻¹, and the KINETICS amounts `m0` in moles per litre of water, as in the field model.
- **Schedules**: the flow rate, the influent solution, and the time step are constants or lists `[(t, value), ...]`
  starting at t = 0, each value holding until the next t. Stress periods start at every change time
  (`model.stress_periods`). A flow rate of 0 stops the flow (stop-flow tests); diffusion continues.
- **Grid**: the column is divided into `n_cells` cells of length Δx = L / `n_cells`. The concentration of each cell
  is computed at its downstream end, `model.x` = Δx, 2 Δx, ..., L, so the last cell gives the effluent. A cell range,
  e.g. `model.x <= 0.1` for the first 10 cm, sets `initial_solution` or `initial_kinetics` per cell.

## Model

- One-dimensional flow with the average linear velocity v = Q / (n A), A = π d² / 4, and the longitudinal dispersion
  coefficient D = α_L v + D*.
- The influent enters at x = 0 with a fixed concentration (first-type boundary); the outlet has a zero concentration
  gradient.
- Transport: advection along the characteristics with monotone cubic interpolation, then dispersion with the Saul'yev
  alternating-direction scheme; both are stable for any time step, the accuracy needs Courant numbers v Δt / Δx of
  about 1 or less.
- Every component of the solutions is transported, including H, O, and the charge balance; immobile reactants must be
  KINETICS reactants.
- `model.phreeqc` is the `mibiremo.PhreeqcRM` object after `run`, and `model.components` the transported components.
  The workspace holds the PHREEQC input (`column.pqi`) and the PhreeqcRM output files.

## Results

- `effluent(component)`: concentration of the last cell at every time step, with the time and the pore volumes
  flushed V / V_p, where V_p = n A L (`model.pore_volume`).
- `concentration(component, time)`: concentration profile at the end of the time step containing `time`.

## Verification

The column model is verified against the exact analytical solution of
[mibitrans](https://github.com/MiBiPreT/mibitrans) (one-dimensional limit, constant influent concentration, no
transverse dispersion), for a conservative component and for one with first-order decay
([Example 9](notebooks/ex9-column_validation-vs-mibitrans.ipynb)).
