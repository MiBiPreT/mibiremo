"""
BTEX dissolution modeling using MiBiReMo and PhreeqcRM.

Models the dissolution of benzene and ethylbenzene from NAPL phase into aqueous phase.

Author: Matteo Masi
Last revision: 09/06/2025
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import mibiremo


# Simulation settings
database_path = "../mibiremo/database/mibirem.dat"  # Database path
n_cells = 1  # Number of model cells
n_threads = 4  # Threads for calculation (-1 for all CPUs)
pqi_file = "pqi/ex2_BTEX_dissolution.pqi"  # Phreeqc input file

# Unit settings
unit_solution = 2  # 1: mg/L; 2: mol/L; 3: kg/kgs
units = 1  # 0: mol/L cell; 1: mol/L water; 2: mol/L rock

# Physical properties
porosity = 1.0  # Porosity
saturation = 1.0  # Saturation

# Time settings
sim_duration = 7.0  # Simulation duration (days)
n_steps = 100  # Number of time steps


def initialize_phreeqc():
    """Initialize and configure PhreeqcRM instance."""
    phr = mibiremo.PhreeqcRM()
    phr.create(nxyz=n_cells, n_threads=n_threads)

    # Load database
    status = phr.RM_LoadDatabase(database_path)
    if status != 0:
        raise RuntimeError("Failed to load Phreeqc database")

    # Set basic properties
    phr.RM_SetComponentH2O(0)  # Exclude H2O from component list
    phr.RM_SetRebalanceFraction(0.5)  # Thread load balancing

    # Configure units
    phr.RM_SetUnitsSolution(unit_solution)
    phr.RM_SetUnitsPPassemblage(units)
    phr.RM_SetUnitsExchange(units)
    phr.RM_SetUnitsSurface(units)
    phr.RM_SetUnitsGasPhase(units)
    phr.RM_SetUnitsSSassemblage(units)
    phr.RM_SetUnitsKinetics(units)

    # Set physical properties
    phr.RM_SetPorosity(porosity * np.ones(n_cells))
    phr.RM_SetSaturation(saturation * np.ones(n_cells))

    # Configure output
    phr.RM_SetFilePrefix("btex")
    phr.RM_OpenFiles()
    phr.RM_SetSpeciesSaveOn(1)

    return phr


def run_initial(phr):
    """Run initial setup and return component/species information."""
    status = phr.RM_RunFile(1, 1, 1, pqi_file)
    if status != 0:
        raise RuntimeError("Failed to run Phreeqc input file")

    # Log thread information
    th = phr.RM_GetThreadCount()
    phr.RM_OutputMessage(f"Number of threads: {th}\n")

    # Set initial conditions
    ic1 = -1 * np.ones(n_cells * 7, dtype=np.int32)
    for i in range(n_cells):
        ic1[i] = 1  # Solution 1
        ic1[i + n_cells] = -1  # No equilibrium phases
        ic1[i + 2*n_cells:i + 6*n_cells] = -1  # No other phases
        ic1[i + 6*n_cells] = 1  # Kinetics 1

    ic2 = -1 * np.ones(n_cells * 7, dtype=np.int32)
    f1 = np.ones(n_cells * 7, dtype=np.float64)

    status = phr.RM_InitialPhreeqc2Module(ic1, ic2, f1)

    # Get component and species information
    n_comps = phr.RM_FindComponents()
    n_species = phr.RM_GetSpeciesCount()

    components = np.zeros(n_comps, dtype="U20")
    for i in range(n_comps):
        phr.RM_GetComponent(i, components, 20)

    species = np.zeros(n_species, dtype="U20")
    for i in range(n_species):
        phr.RM_GetSpeciesName(i, species, 20)

    # Run initial step
    phr.RM_SetTime(0.0)
    phr.RM_SetTimeStep(0.1)  # 0.1 second
    status = phr.RM_RunCells()

    return components, species


def main():
    """Main simulation workflow for BTEX dissolution."""
    # Initialize and configure
    phr = initialize_phreeqc()
    components, species = run_initial(phr)

    # Get initial concentrations
    n_comps = len(components)
    n_species = len(species)

    cc = np.zeros(n_cells * n_comps, dtype=np.float64)
    cs = np.zeros(n_cells * n_species, dtype=np.float64)
    phr.RM_GetConcentrations(cc)
    phr.RM_GetSpeciesConcentrations(cs)

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
        phr.RM_SetTime(current_time)
        phr.RM_SetTimeStep(dt)
        status = phr.RM_RunCells()

        # Store results
        status = phr.RM_GetConcentrations(cc)
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
    dissolved_mgL = concentration_results[:, [0, 2]].copy()
    dissolved_mgL[:, 0] *= 78.11 * 1000  # Benzene: mol/L to mg/L
    dissolved_mgL[:, 1] *= 106.17 * 1000  # Ethylbenzene: mol/L to mg/L
    ax2.plot(time_vector / 86400.0, dissolved_mgL, "--")
    ax2.set_ylabel("Dissolved concentration (mg/L)")
    ax2.legend([component_headings[0], component_headings[2]])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
