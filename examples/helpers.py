
"""
Utility functions for initializing and configuring PhreeqcRM objects using the MiBiReMo package.
This module provides helper functions to run the examples.

Author: Matteo Masi
Last update: 10/06/2025

Dependencies:
    - mibiremo
    - numpy
"""

import mibiremo
import numpy as np


def initialize_phreeqc(database_path, n_cells = 1, n_threads = 1, unit_solution = 2, units = 1, porosity = 1.0, saturation = 1.0):
    """
    Initializes a PhreeqcRM object with specified parameters and loads the given database.

    Parameters:
        database_path (str): Path to the Phreeqc database file to be loaded.
        n_cells (int, optional): Number of cells to initialize in the model. Default is 1.
        n_threads (int, optional): Number of threads to use for parallel processing. Default is 1.
        unit_solution (int, optional): Units for solutions (e.g., 1 = mol/L, 2 = mmol/L). Default is 2.
        units (int, optional): Units for other phases (e.g., Exchange, Surface, etc.). Default is 1.
        porosity (float, optional): Porosity value to assign to all cells. Default is 1.0.
        saturation (float, optional): Saturation value to assign to all cells. Default is 1.0.

    Returns:
        PhreeqcRM: An initialized PhreeqcRM object with the specified configuration.

    Raises:
        RuntimeError: If the Phreeqc database fails to load.

    Notes:
        - Configures multicomponent diffusion settings.
        - Assumes `mibiremo` and `numpy` (as `np`) are available in the environment.
    """

    phr = mibiremo.PhreeqcRM()
    phr.create(nxyz=n_cells, n_threads=n_threads)

    # Load database
    status = phr.RM_LoadDatabase(database_path)
    if status != 0:
        raise RuntimeError("Failed to load Phreeqc database")

    # Set properties/parameters
    phr.RM_SetComponentH2O(0)  # Don't include H2O in component list
    phr.RM_SetRebalanceFraction(0.5)  # Rebalance thread load

    # Set units
    phr.RM_SetUnitsSolution(unit_solution)
    phr.RM_SetUnitsPPassemblage(units)
    phr.RM_SetUnitsExchange(units)
    phr.RM_SetUnitsSurface(units)
    phr.RM_SetUnitsGasPhase(units)
    phr.RM_SetUnitsSSassemblage(units)
    phr.RM_SetUnitsKinetics(units)

    # Set porosity and saturation
    phr.RM_SetPorosity(porosity * np.ones(n_cells))
    phr.RM_SetSaturation(saturation * np.ones(n_cells))

    # Create error log files
    phr.RM_SetFilePrefix("phr")
    phr.RM_OpenFiles()

    # Multicomponent diffusion settings
    phr.RM_SetSpeciesSaveOn(1)

    return phr


def run_initial(phr, pqi_file, n_cells):
    """
    Runs the initial setup for a PHREEQC module simulation and retrieves component and species information.

    Parameters
    ----------
    phr : object
        An instance of the PHREEQC RM (Reactive Transport Module) interface, providing methods to interact with the PHREEQC engine.
    pqi_file : str
        Path to the PHREEQC input file (.pqi) used for the initial setup.
    n_cells : int
        Number of cells in the simulation domain.

    Returns
    -------
    components : numpy.ndarray
        Array of component names (strings) present in the simulation after initialization.
    species : numpy.ndarray
        Array of species names (strings) present in the simulation after initialization.

    Raises
    ------
    RuntimeError
        If the PHREEQC input file fails to run.

    Notes
    -----
    This function:
      - Runs the initial PHREEQC input file.
      - Sets up initial conditions for all cells.
      - Transfers initial conditions to the PHREEQC module.
      - Retrieves the list of components and species.
      - Runs the initial equilibrium step.
    """

    # Run the initial setup file
    status = phr.RM_RunFile(1, 1, 1, pqi_file)
    if status != 0:
        raise RuntimeError("Failed to run Phreeqc input file")

    # Transfer initial conditions
    ic1 = -1 * np.ones(n_cells * 7, dtype=np.int32)
    for i in range(n_cells):
        ic1[i] = 1  # Solution 1
        ic1[i + n_cells] = 1  # Equilibrium phases 1
        # Other phases set to none (-1)
        ic1[i + 2*n_cells:i + 7*n_cells] = -1

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

    # Run initial equilibrium step
    phr.RM_SetTime(0.0)
    phr.RM_SetTimeStep(0.0)
    status = phr.RM_RunCells()

    return components, species
