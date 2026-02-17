"""
BTEX dissolution modeling using MiBiReMo and PhreeqcRM.

Models the dissolution of benzene and ethylbenzene from NAPL phase into aqueous phase.

Author: Matteo Masi
Last revision: 10/06/2025
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import mibiremo
from pathlib import Path
from importlib.resources import files


# Simulation settings
database_path = str(files("mibiremo").joinpath("database/mibirem.dat"))  # Database path
n_cells = 1  # Number of model cells
n_threads = 4  # Threads for calculation (-1 for all CPUs)
pqi_file = str(Path(__file__).parent / "pqi/ex2_BTEX_dissolution.pqi")  # Phreeqc input file

# Unit settings
unit_solution = 2  # 1: mg/L; 2: mol/L; 3: kg/kgs
units = 1  # 0: mol/L cell; 1: mol/L water; 2: mol/L rock

# Physical properties
porosity = 1.0  # Porosity
saturation = 1.0  # Saturation

# Time settings
sim_duration = 7.0  # Simulation duration (days)
n_steps = 100  # Number of time steps


# Initialize PhreeqcRM
phr = mibiremo.PhreeqcRM()
phr.create(nxyz=n_cells, n_threads=n_threads)
phr.initialize_phreeqc(database_path, unit_solution, units, porosity, saturation)

# Prepare the initial conditions
# Use SOLUTION 1 and KINETICS 1, disable all model features (-1)
ic = [1, -1, -1, -1, -1, -1, 1]

# Repeat for all other cells (row-wise)
ic = np.tile(ic, (n_cells, 1))

phr.run_initial_from_file(pqi_file, ic)

# Get components and species
components = phr.components
species = phr.species
n_comps = len(components)
n_species = len(species)

# Get initial concentrations
cc = np.zeros(n_cells * n_comps, dtype=np.float64)
cs = np.zeros(n_cells * n_species, dtype=np.float64)
phr.rm_get_concentrations(cc)
phr.rm_get_species_concentrations(cs)

# Time step setup
dt = sim_duration / n_steps * 24 * 3600.0  # Convert days to seconds

# Initialize results storage
time_vector = np.zeros(n_steps)
component_headings = ["Benz", "Benznapl", "Ethyl", "Ethylnapl"]
concentration_results = np.zeros((n_steps, len(component_headings)))

# Create component mapping
component_map = np.zeros(len(component_headings), dtype=np.int32)
for i, comp in enumerate(component_headings):
    component_map[i] = np.where(components == comp)[0][0]

# Store initial concentrations
concentration_results[0, :] = cc[component_map]

# Main simulation loop
start_time = time.time()
for step in range(1, n_steps):
    current_time = step * dt
    time_vector[step] = current_time

    # Run simulation step
    phr.rm_set_time(current_time)
    phr.rm_set_time_step(dt)
    status = phr.rm_run_cells()

    # Store results
    status = phr.rm_get_concentrations(cc)
    concentration_results[step, :] = cc[component_map]

elapsed = time.time() - start_time
print(f"Simulation completed in {elapsed:.2f} seconds")

# Plot results
plt.figure(figsize=(10, 6))

# Plot NAPL concentrations (mol/L)
plt.plot(time_vector / 86400.0, concentration_results[:, [1, 3]])
plt.xlabel("Elapsed time (days)")
plt.ylabel("NAPL Concentration (mol/L)")
plt.legend([component_headings[1], component_headings[3]])
plt.title("BTEX dissolution")

# Create second y-axis for dissolved concentrations (mg/L)
ax2 = plt.twinx()
dissolved_mgl = concentration_results[:, [0, 2]].copy()
dissolved_mgl[:, 0] *= 78.11 * 1000  # Benzene: mol/L to mg/L
dissolved_mgl[:, 1] *= 106.17 * 1000  # Ethylbenzene: mol/L to mg/L
ax2.plot(time_vector / 86400.0, dissolved_mgl, "--")
ax2.set_ylabel("Dissolved concentration (mg/L)")
ax2.legend([component_headings[0], component_headings[2]])

plt.tight_layout()
plt.show()
