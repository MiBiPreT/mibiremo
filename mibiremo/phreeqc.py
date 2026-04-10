"""Python interface to PhreeqcRM for biogeochemical reactive transport modeling.

PhreeqcRM is a reaction module developed by the U.S. Geological Survey (USGS)
for coupling biogeochemical calculations with transport models. It provides a
high-performance interface to PHREEQC modeling capabilities for reactive transport
simulations in environmental and hydrological applications.

PhreeqcRM enables:

- Multi-threaded biogeochemical calculations for large-scale transport models
- Equilibrium and kinetic biogeochemical reactions in porous media
- Parallel processing for computationally intensive reactive transport

This interface wraps the ``phreeqcrm`` pip package, exposing four high-level
convenience methods (``create``, ``initialize_phreeqc``,
``run_initial_from_file``, ``get_selected_output_df``) and a public
``self.rm`` attribute (the underlying ``phreeqcrm.PhreeqcRM`` instance) for
advanced users who need direct access to the full PhreeqcRM API.

PhreeqcRM documentation and source code can be found at:

- [PhreeqcRM Documentation](https://usgs-coupled.github.io/phreeqcrm/namespacephreeqcrm.html)
- [PhreeqcRM GitHub Repository](https://github.com/usgs-coupled/phreeqcrm)

Last revision: 10/04/2026
"""

import os
import numpy as np
import pandas as pd
import phreeqcrm


class PhreeqcRM:
    """Python interface to PhreeqcRM for geochemical reactive transport modeling.

    This class facilitates coupling between transport codes and geochemical
    reaction calculations by managing multiple reaction cells, each representing
    a grid cell in the transport model. The PhreeqcRM approach allows efficient
    parallel processing of geochemical calculations across large spatial domains.

    The class handles:
        - Creation and initialization of PhreeqcRM instances
        - Loading thermodynamic databases (PHREEQC format)
        - Setting up initial chemical conditions from input files
        - Running equilibrium and kinetic geochemical reactions
        - Transferring concentrations between transport and reaction modules
        - Managing porosity, saturation, temperature, and pressure fields
        - Retrieving calculated properties and concentrations

    Typical workflow:
        1. Create instance and call create() method to initialize with grid size
        2. Load thermodynamic database with initialize_phreeqc()
        3. Set initial conditions with run_initial_from_file()
        4. In transport time loop:
           - Transfer concentrations to reaction module with rm.SetConcentrations()
           - Advance time with rm.SetTime() and rm.SetTimeStep()
           - Run reactions with rm.RunCells()
           - Retrieve updated concentrations with rm.GetConcentrations()

    Attributes:
        nxyz (int): Number of grid cells in the reactive transport model.
        n_threads (int): Number of threads for parallel geochemical processing.
        rm (phreeqcrm.PhreeqcRM): The underlying PhreeqcRM instance. Use this
            attribute to access the full PhreeqcRM API directly (e.g.,
            ``rm.SetTime(t)``, ``rm.GetConcentrations()``).
        components (numpy.ndarray): Array of component names for transport.
        species (numpy.ndarray): Array of aqueous species names in the system.

    Examples:
        See page [Examples](examples.md) for usage examples.
    """

    def __init__(self):
        """Initialize PhreeqcRM instance.

        Creates a new PhreeqcRM object with default values. The instance must be
        created using the create() method before it can be used for calculations.
        """
        self._initialized = False
        self.nxyz = 1
        self.n_threads = 1
        self.rm = None
        self.components = None
        self.species = None

    def create(self, nxyz=1, n_threads=1) -> None:
        """Creates a PhreeqcRM reaction module instance.

        Initializes the PhreeqcRM library and creates a reaction module with
        the specified number of grid cells and threads.
        This method must be called before any other PhreeqcRM operations.

        Args:
            nxyz (int, optional): Number of grid cells in the model. Must be
                positive. Defaults to 1.
            n_threads (int, optional): Number of threads for parallel processing.
                Use -1 for automatic detection of CPU count. Defaults to 1.

        Raises:
            RuntimeError: If PhreeqcRM instance creation fails.

        Examples:
            >>> rm = PhreeqcRM()
            >>> rm.create(nxyz=100, n_threads=4)
        """
        if n_threads == -1:
            n_threads = os.cpu_count()

        self.n_threads = n_threads
        self.nxyz = nxyz
        try:
            self.rm = phreeqcrm.PhreeqcRM(nxyz, n_threads)
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to create PhreeqcRM instance: {e}")

    def initialize_phreeqc(
        self,
        database_path,
        units_solution=2,
        units=1,
        porosity=1.0,
        saturation=1.0,
        multicomponent=True,
    ) -> None:
        """Initialize PhreeqcRM with database and default parameters.

        Loads a thermodynamic database and sets up the PhreeqcRM instance with
        standard parameters for geochemical calculations. This is a convenience
        method that handles common initialization tasks.

        Args:
            database_path (str): Path to the PHREEQC database file (.dat format).
                Common databases include phreeqc.dat, Amm.dat, pitzer.dat.
            units_solution (int, optional): Units for solution concentrations.
                1 = mol/L, 2 = mmol/L, 3 = μmol/L. Defaults to 2.
            units (int, optional): Units for other phases (Exchange, Surface,
                Gas, Solid solutions, Kinetics). Defaults to 1.
            porosity (float, optional): Porosity value assigned to all cells.
                Must be between 0 and 1. Defaults to 1.0.
            saturation (float, optional): Saturation value assigned to all cells.
                Must be between 0 and 1. Defaults to 1.0.
            multicomponent (bool, optional): Enable multicomponent diffusion
                by saving species concentrations. Defaults to True.

        Raises:
            RuntimeError: If the PhreeqcRM instance is not initialized or if
                the database fails to load.

        Examples:
            >>> rm = PhreeqcRM()
            >>> rm.create(nxyz=100)
            >>> rm.initialize_phreeqc("phreeqc.dat", units_solution=1)
        """
        if not self._initialized:
            raise RuntimeError("PhreeqcRM instance not initialized. Call create() first.")

        status = self.rm.LoadDatabase(database_path)
        if status < 0:
            raise RuntimeError(f"Failed to load Phreeqc database (error code: {status})")

        self.rm.SetComponentH2O(False)
        self.rm.SetRebalanceFraction(0.5)
        self.rm.SetUnitsSolution(units_solution)
        self.rm.SetUnitsPPassemblage(units)
        self.rm.SetUnitsExchange(units)
        self.rm.SetUnitsSurface(units)
        self.rm.SetUnitsGasPhase(units)
        self.rm.SetUnitsSSassemblage(units)
        self.rm.SetUnitsKinetics(units)
        self.rm.SetPorosity(porosity * np.ones(self.nxyz))
        self.rm.SetSaturationUser(saturation * np.ones(self.nxyz))
        self.rm.SetFilePrefix("phr")
        self.rm.OpenFiles()
        if multicomponent:
            self.rm.SetSpeciesSaveOn(True)

    def run_initial_from_file(self, pqi_file, ic):
        """Set up initial conditions from PHREEQC input file and initial conditions array.

        Loads initial geochemical conditions by running a PHREEQC input file and
        mapping the defined solutions, phases, and other components to the grid cells.
        This method also retrieves component and species information for later use.

        Args:
            pqi_file (str): Path to the PHREEQC input file (.pqi format) containing
                definitions for solutions, equilibrium phases, exchange, surface,
                gas phases, solid solutions, and kinetic reactions.
            ic (numpy.ndarray): Initial conditions array with shape (nxyz, 7) where
                each row corresponds to a grid cell and columns represent:
                - Column 0: Solution ID
                - Column 1: Equilibrium phase ID
                - Column 2: Exchange ID
                - Column 3: Surface ID
                - Column 4: Gas phase ID
                - Column 5: Solid solution ID
                - Column 6: Kinetic reaction ID
                Use -1 for unused components.

        Raises:
            RuntimeError: If the PHREEQC input file fails to run.
            ValueError: If initial conditions array has incorrect shape or cannot
                be converted to integer array.

        Examples:
            >>> import numpy as np
            >>> ic = np.array([[1, -1, -1, -1, -1, -1, -1]])  # Only solution 1
            >>> rm.run_initial_from_file("initial.pqi", ic)
        """
        status = self.rm.RunFile(True, True, True, pqi_file)
        if status < 0:
            raise RuntimeError(f"Failed to run Phreeqc input file (error code: {status})")

        if ic.shape != (self.nxyz, 7):
            raise ValueError(f"Initial conditions array must have shape ({self.nxyz}, 7), got {ic.shape}")

        if not isinstance(ic, np.ndarray):
            try:
                ic = np.array(ic).astype(np.int32)
            except Exception as e:
                raise ValueError("Initial conditions must be convertible to a numpy array of integers") from e

        ic1 = ic.flatten("F").astype(np.int32)
        self.rm.InitialPhreeqc2Module(ic1)

        self.rm.FindComponents()
        self.components = np.array(list(self.rm.GetComponents()))
        self.species = np.array(list(self.rm.GetSpeciesNames()))

        self.rm.SetTime(0.0)
        self.rm.SetTimeStep(0.0)
        self.rm.RunCells()

    def get_selected_output_df(self) -> pd.DataFrame:
        """Retrieve selected output data as a pandas DataFrame.

        Extracts the current selected output data from PhreeqcRM and formats it
        as a pandas DataFrame with appropriate column headers. Selected output
        typically includes calculated properties like pH, pe, ionic strength,
        activities, saturation indices, and user-defined calculations.

        Returns:
            pandas.DataFrame: DataFrame containing selected output data with
                rows representing grid cells and columns representing the
                selected output variables defined in the PHREEQC input.

        Examples:
            >>> df = rm.get_selected_output_df()
            >>> print(df.columns)  # Show available output variables
            >>> print(df['pH'])     # Access pH values for all cells
        """
        headings = list(self.rm.GetSelectedOutputHeadings())
        ncolsel = len(headings)
        so = np.array(self.rm.GetSelectedOutput())
        return pd.DataFrame(so.reshape(ncolsel, self.nxyz).T, columns=headings)
