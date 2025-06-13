"""
MiBiReMo - Example - Calculate calcite titration curve.

This script models the reaction:
CaCO3(s) + 2HCl(aq) → CaCl2(aq) + CO2(g) + H2O(l)
assuming chemical reactions at equilibrium.

Author: Matteo Masi
Last revision: 10/06/2025
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import mibiremo


# Global settings
database_path = "../mibiremo/database/phreeqc.dat"  # .dat database path
n_cells = 1000  # Number of model cells
n_threads = 4  # Multithread calculation (-1 for all CPUs)
pqi_file = "pqi/ex1_Calcite_titration.pqi"  # Phreeqc input file
hcl_range = [0.0, 4.0]  # mol/L

# Unit settings
unit_solution = 2  # 1: mg/L; 2: mol/L; 3: kg/kgs
units = 1  # 0: mol/L cell; 1: mol/L water; 2: mol/L rock

# Physical properties
porosity = 1.0  # Porosity
saturation = 1.0  # Saturation


phr = mibiremo.PhreeqcRM()
phr.create(nxyz=n_cells, n_threads=n_threads)
phr.initialize_phreeqc(database_path, unit_solution, units, porosity, saturation)

# Prepare the initial conditions
# Use SOLUTION 1 and EQUILIBRIUM_PHASES 1, disable all other phases (-1)
ic = np.array([1, 1, -1, -1, -1, -1, -1]).astype(np.int32)

# Repeat for all other cells (row-wise)
ic = np.tile(ic, (n_cells, 1))

# Run initial conditions from file
phr.run_initial_from_file(pqi_file, ic)
components = phr.components
species = phr.species
n_comps = len(components)  # Number of components
n_species = len(species)  # Number of species

# Initialize concentration vectors
cc = np.zeros(n_cells * n_comps, dtype=np.float64)
cs = np.zeros(n_cells * n_species, dtype=np.float64)
phr.RM_GetConcentrations(cc)
phr.RM_GetSpeciesConcentrations(cs)

# Set HCl concentrations
hcl = np.linspace(hcl_range[0], hcl_range[1], n_cells)  # mol/L

# Find indices for Cl- and H+ species
indx_cl = np.where(species == "Cl-")[0][0]
indx_h = np.where(species == "H+")[0][0]

# Update species concentrations with HCl
cs_r = cs.reshape(n_species, n_cells).T
cs_r[:, indx_cl] += hcl  # Cl-
cs_r[:, indx_h] += hcl  # H+
cs1 = cs_r.T.reshape(n_cells * n_species)

# Run simulation with added HCl
phr.RM_SpeciesConcentrations2Module(cs1)
phr.RM_SetTime(1.0)
phr.RM_SetTimeStep(1.0)

start_time = time.time()
phr.RM_RunCells()
elapsed = time.time() - start_time
print(f"Simulation completed in {elapsed:.2f} seconds")

# Get results
results_df = phr.pdSelectedOutput()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(hcl, results_df["pH"])
plt.xlabel("mol H$^+$ added")
plt.ylabel("pH")
plt.title("Calcite Titration Curve")
plt.grid(True)
plt.show()
