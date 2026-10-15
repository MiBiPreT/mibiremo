# Field model

`mibiremo.FieldModel` builds a MODFLOW 6 model of an in situ biological flushing installation: injection and
extraction wells in an aquifer with regional groundwater flow. It simulates groundwater flow, the transport of a
tracer, and reactive transport with PHREEQC through [mf6rtm](https://github.com/p-ortega/mf6rtm). The model is
built with [flopy](https://github.com/modflowpy/flopy), and the flopy simulation is always available as
`model.simulation`.

Examples: [tracer test](notebooks/field_tracer_test.ipynb),
[tracer test with three well layouts](notebooks/field_injection_extraction_distance.ipynb),
[tracer test in a layered aquifer](notebooks/field_layered_model.ipynb), and
[validation against mibitrans](notebooks/validation_field_vs_mibitrans.ipynb).

## Requirements

The MODFLOW 6 program and its shared library (used by mf6rtm), installed into the active Python environment:

```console
get-modflow :python --subset mf6,libmf6
```

## Workflow

```python
from dataclasses import replace
import mibiremo as mb

day, hour = 86400.0, 3600.0

# Wells: geometry only; layouts copy a well around its location, or read_wells reads a shapefile or CSV file
extraction = mb.Well("EXT_1", x=0.0, y=0.0, well_top=96.3, well_bottom=90.5, diameter=0.1,
                     screen_top=95.5, screen_bottom=90.5)
wells = [extraction, *mb.array_radial(replace(extraction, name="INJ"), n_wells=3, radius=4.0)]

pulsed = mb.intermittent_pumping(8.25e-5, on_duration=8 * hour, off_duration=16 * hour, end_time=30 * day)
model = mb.FieldModel(
    workspace="output/field",
    wells=wells,
    flow_rates={"EXT_1": -8.25e-5, "INJ_1": pulsed, "INJ_2": pulsed, "INJ_3": pulsed},  # Q [m3 s-1]
    domain_size=(100.0, 100.0),
    top=95.5,
    layer_bottom=90.5,
    hydraulic_conductivity=5e-6,
    porosity=0.25,
    reference_head=93.7,
    regional_hydraulic_gradient=0.008,
    regional_flow_azimuth=22.0,
    tracer_concentration=[(0.0, 1.0), (1 * day, 0.0)],
    grid_spacing=4.0,
    grid_spacing_at_wells=0.5,
    simulation_time=30 * day,
    time_step=[(0.0, 1 * hour), (1 * day, 4 * hour)],
)
model.run()
breakthrough = model.well_concentration("EXT_1")  # DataFrame: time [s], concentration
plume = model.concentration(time=2 * day)          # array (layer, row, column)
```

## Conventions

- **Units**: SI throughout: m, s, m³ s⁻¹, m s⁻¹. Times are counted from the start of the simulation.
- **Coordinates**: projected coordinates in metres, as in the well files; the grid is placed in the same coordinates.
  Azimuths are in degrees clockwise from north.
- **Elevations**: the top of the aquifer and the bottoms of the hydrostratigraphic units are constant or a function
  `z(x, y)` of the coordinates, e.g. `scipy.interpolate.NearestNDInterpolator(points, z)` for measured elevations.
- **Flow rates**: Q by well name, positive for injection and negative for extraction. Wells not listed in
  `flow_rates` are monitoring wells (Q = 0); their concentrations are available as for the pumped wells.
- **Schedules**: every time-dependent input is a list `[(t, value), ...]` starting at t = 0, each value holding until
  the next t; a number is a constant. Schedules are used for flow rates, the tracer concentration of the injected
  water, the time step, and the injected solution of PHREEQC coupling.
- **Stress periods** start at every change time of any schedule (`model.stress_periods`). Flow is steady within a
  stress period.
- **Tracer**: the aquifer and the inflow across the boundary are tracer-free. With the default tracer
  concentration of 1, the concentration is the fraction of injected water.

## Model

- Confined aquifer with horizontal hydraulic conductivity K, optional vertical anisotropy, and layers of equal
  thickness. One layer (the default) is enough when all pumped wells are screened over the whole aquifer.
- Layered aquifer: with `layer_bottom` as a list (one bottom per hydrostratigraphic unit, from top to bottom), each unit is
  divided into `n_layers` layers of equal thickness, and `n_layers`, `hydraulic_conductivity`, `vertical_anisotropy`,
  and `porosity` are one value for all units or a list with one value per unit:

    ```python
    model = mb.FieldModel(
        ...,
        top=96.0,
        layer_bottom=[93.5, 90.5, 87.0],                    # upper, middle, and lower unit [m]
        n_layers=[4, 3, 1],
        hydraulic_conductivity=[5e-5, 5e-6, 1e-7],    # K [m s-1]
        vertical_anisotropy=0.1,                      # Kz/K, all units
        porosity=0.25,
    )
    ```

- Regional groundwater flow with a uniform hydraulic gradient, imposed on the lateral boundary as fixed heads (CHD,
  the default) or as a general-head boundary (GHB) with `boundary_conductance` C [m² s⁻¹] in each boundary cell.
- The flow rate of a well is split among its screened cells in proportion to the screened transmissivity K b.
- Structured grid centred on the pumped wells (or on `domain_centre`), with fine cells around the pumped wells that grow
  to `grid_spacing`.
- Transport schemes: advection (upstream, central, or TVD scheme) and optional dispersion (longitudinal, transverse, and
  vertical dispersivities, molecular diffusion).
- By default a single tracer without reactions is simulated.
- If `phreeqc_coupling` is enabled, the transport of the components of PHREEQC solutions is simulated, with reactions computed by PHREEQC.

## Results

- `head(time)` and `concentration(time)`: arrays (layer, row, column) at the end of the time step containing `time`.
- `well_concentration(name)`: flow-weighted mean concentration of the screened cells at every time step.
- `well_head(name)`: head of a well at the end of every stress period, the mean head of the screened cells weighted by
  K b.
- `mass_balance()`: cumulative tracer mass injected, extracted, crossing the boundary, and in the aquifer.
- `mibiremo.plotting`: `map_view`, `grid`, `cross_section`, `wells`, and `flow_arrow`, e.g.
  `mb.plotting.map_view(model, model.concentration(2 * day), label="C")` or `mb.plotting.grid(model)`.

## Reactive transport: PHREEQC coupling

With `phreeqc_coupling=True`, MODFLOW 6 transports every component of the PHREEQC solutions instead of the tracer (one
transport model per component), and PHREEQC computes the reactions in every cell at every time step; `mf6rtm` couples
the two. `concentration` and `well_concentration` then take a `component` argument, with concentrations in mol m⁻³.

```python
model = mb.FieldModel(
    ...,                                                 # wells, aquifer, and time as above
    phreeqc_coupling=True,
    database="my_database.dat",                          # kinetic rates in its RATES block
    solutions={1: groundwater, 2: injected_water},       # {component: mol/kgw, "pH": ..., "pe": ...}
    initial_solution=1,                                  # in the aquifer and in the inflow across the boundary
    injected_solution=[(0.0, 2), (1 * day, 1)],          # solution number, for all injection wells
    kinetics={1: reaction},                              # PHREEQC KINETICS blocks
)
zone = model.zone("contaminated_area.shp", z_top=94.0, z_bottom=91.0)  # boolean array (layer, row, column)
model.initial_kinetics = np.where(zone, 1, -1)           # KINETICS block number per cell, -1 for none
model.run(n_threads=-1)
model.well_concentration("EXT_1", component="Tr")
```

- Solutions and KINETICS blocks are numbered 1, 2, ... as in PHREEQC.
- `zone` does not build the model, so its result can set `initial_kinetics` or `initial_solution` before `run`.
- `build` also creates the `mf6rtm` model, `model.mup3d`, e.g. to add boundary chemistry with an `mf6rtm` `ChemStress`
  before `run`.
- `mf6rtm` writes its output to the workspace; `mf6rtm.log` holds the messages of the run.
- Every component of the solutions is transported, including H, O, and the charge balance. Immobile reactants must be
  KINETICS reactants or phases, not solution components.

## Verification

The tracer and the reactive transport are verified against the exact analytical solution of
[mibitrans](https://github.com/MiBiPreT/mibitrans) for a plume from a constant-concentration source in uniform flow,
conservative and with first-order decay ([validation against mibitrans](notebooks/validation_field_vs_mibitrans.ipynb)).
