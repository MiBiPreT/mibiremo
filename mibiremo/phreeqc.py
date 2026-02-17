"""Python interface to PhreeqcRM for geochemical reactive transport modeling.

PhreeqcRM is a reaction module developed by the U.S. Geological Survey (USGS)
for coupling geochemical calculations with transport models. It provides a
high-performance interface to PHREEQC geochemical modeling capabilities for
reactive transport simulations in environmental and hydrological applications.

PhreeqcRM enables:

- Multi-threaded geochemical calculations for large-scale transport models
- Equilibrium and kinetic geochemical reactions in porous media
- Parallel processing for computationally intensive reactive transport

This interface provides a Python wrapper around the PhreeqcRM C++ library,
simplifying the process of integrating geochemical calculations with transport models.

All RM_* methods in this class correspond directly to PhreeqcRM C++ functions.
PhreeqcRM documentation and source code can be found at:

- [PhreeqcRM Documentation](https://usgs-coupled.github.io/phreeqcrm/namespacephreeqcrm.html)
- [PhreeqcRM GitHub Repository](https://github.com/usgs-coupled/phreeqcrm)

Last revision: 26/09/2025
"""

import ctypes
import os
import numpy as np
import pandas as pd
from .irmresult import irm_result


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
           - Transfer concentrations to reaction module with rm_set_concentrations()
           - Advance time with rm_set_time() and rm_set_time_step()
           - Run reactions with rm_run_cells()
           - Retrieve updated concentrations with rm_get_concentrations()

    Attributes:
        dllpath (str): Path to the PhreeqcRM dynamic library file.
        nxyz (int): Number of grid cells in the reactive transport model.
        n_threads (int): Number of threads for parallel geochemical processing.
        libc (ctypes.CDLL): Handle to the loaded PhreeqcRM dynamic library.
        id (int): Unique instance identifier returned by RM_Create.
        components (numpy.ndarray): Array of component names for transport.
        species (numpy.ndarray): Array of aqueous species names in the system.

    Note:
        This interface requires the PhreeqcRM dynamic library to be available
        in the lib/ subdirectory. The library handles the underlying PHREEQC
        calculations and memory management.

    Examples:
        See page [Examples](examples.md) for usage examples.
    """

    def __init__(self):
        """Initialize PhreeqcRM instance.

        Creates a new PhreeqcRM object with default values. The instance must be
        created using the create() method before it can be used for calculations.
        """
        self._initialized = False
        self.dllpath = None
        self.nxyz = 1
        self.n_threads = 1
        self.libc = None
        self.id = None
        self.components = None
        self.species = None

    def create(self, dllpath=None, nxyz=1, n_threads=1) -> None:
        """Creates a PhreeqcRM reaction module instance.

        Initializes the PhreeqcRM library, loads the dynamic library, and creates
        a reaction module with the specified number of grid cells and threads.
        This method must be called before any other PhreeqcRM operations.

        Args:
            dllpath (str, optional): Path to the PhreeqcRM library. If None,
                uses the default library path based on the operating system.
                Defaults to None.
            nxyz (int, optional): Number of grid cells in the model. Must be
                positive. Defaults to 1.
            n_threads (int, optional): Number of threads for parallel processing.
                Use -1 for automatic detection of CPU count. Defaults to 1.

        Raises:
            Exception: If the operating system is not supported (Windows/Linux only).
            RuntimeError: If PhreeqcRM instance creation fails.

        Examples:
            >>> rm = PhreeqcRM()
            >>> rm.create(nxyz=100, n_threads=4)
        """
        if dllpath is None:
            # If no path is provided, use the default path, based on operating system
            if os.name == "nt":
                dllpath = os.path.join(os.path.dirname(__file__), "lib", "PhreeqcRM.dll")
            elif os.name == "posix":
                dllpath = os.path.join(os.path.dirname(__file__), "lib", "PhreeqcRM.so")
            else:
                msg = "Operating system not supported"
                raise Exception(msg)
        self.dllpath = dllpath

        if n_threads == -1:
            n_threads = os.cpu_count()

        self.n_threads = n_threads
        self.nxyz = nxyz
        self.libc = ctypes.CDLL(dllpath)
        try:
            self.id = self.libc.RM_Create(nxyz, n_threads)
            self._initialized = True
        except Exception as e:
            msg = f"Failed to create PhreeqcRM instance: {e}"
            raise RuntimeError(msg)

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

        # Load database
        status = self.rm_load_database(database_path)
        if not status:
            raise RuntimeError(f"Failed to load Phreeqc database: {status}")

        # Set properties/parameters
        self.rm_set_component_h2o(0)  # Don't include H2O in component list
        self.rm_set_rebalance_fraction(0.5)  # Rebalance thread load

        # Set units
        self.rm_set_units_solution(units_solution)
        self.rm_set_units_p_passemblage(units)
        self.rm_set_units_exchange(units)
        self.rm_set_units_surface(units)
        self.rm_set_units_gas_phase(units)
        self.rm_set_units_ss_assemblage(units)
        self.rm_set_units_kinetics(units)

        # Set porosity and saturation
        self.rm_set_porosity(porosity * np.ones(self.nxyz))
        self.rm_set_saturation(saturation * np.ones(self.nxyz))

        # Create error log files
        self.rm_set_file_prefix("phr")
        self.rm_open_files()

        # Multicomponent diffusion settings
        if multicomponent:
            self.rm_set_species_save_on(1)

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

        # Run the initial setup file
        status = self.rm_run_file(1, 1, 1, pqi_file)
        if not status:
            raise RuntimeError(f"Failed to run Phreeqc input file: {status}")

        # Check size of initial conditions array
        if ic.shape != (self.nxyz, 7):
            raise ValueError(f"Initial conditions array must have shape ({self.nxyz}, 7), got {ic.shape}")

        # Be sure that ic is a numpy array
        if not isinstance(ic, np.ndarray):
            try:
                ic = np.array(ic).astype(np.int32)
            except Exception as e:
                raise ValueError("Initial conditions must be convertible to a numpy array of integers") from e

        # Reshape initial conditions to 1D array
        ic1 = np.reshape(ic.T, self.nxyz * 7).astype(np.int32)

        # ic2 contains numbers for a second entity that mixes with the first entity (here, it is not used)
        ic2 = -1 * np.ones(self.nxyz * 7, dtype=np.int32)

        # f1 contains the fractions of the first entity in each cell (here, it is set to 1.0)
        f1 = np.ones(self.nxyz * 7, dtype=np.float64)

        status = self.rm_initial_phreeqc2_module(ic1, ic2, f1)

        # Get component and species information
        n_comps = self.rm_find_components()
        n_species = self.rm_get_species_count()

        self.components = np.zeros(n_comps, dtype="U20")
        for i in range(n_comps):
            self.rm_get_component(i, self.components, 20)

        self.species = np.zeros(n_species, dtype="U20")
        for i in range(n_species):
            self.rm_get_species_name(i, self.species, 20)

        # Run initial equilibrium step
        self.rm_set_time(0.0)
        self.rm_set_time_step(0.0)
        status = self.rm_run_cells()

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
        # Get selected ouput headings
        ncolsel = self.rm_get_selected_output_column_count()
        selout_h = np.zeros(ncolsel, dtype="U100")
        for i in range(ncolsel):
            self.rm_get_selected_output_heading(i, selout_h, 100)
        so = np.zeros(ncolsel * self.nxyz).reshape(self.nxyz, ncolsel)
        self.rm_get_selected_output(so)
        return pd.DataFrame(so.reshape(ncolsel, self.nxyz).T, columns=selout_h)

    ### PhreeqcRM functions
    def rm_abort(self, result, err_str):
        """Abort the PhreeqcRM run.

        Args:
            result (int): Error code indicating reason for abort.
            err_str (str): Error message string.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.
        """
        return irm_result(self.libc.RM_Abort(self.id, result, err_str))

    def rm_close_files(self):
        """Close output files opened by RM_OpenFiles.

        Closes all output files that were opened by RM_OpenFiles, including
        error logs and debug files. This method should be called before
        destroying the PhreeqcRM instance to ensure proper file cleanup.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Examples:
            >>> rm.rm_open_files()  # Open files for logging
            >>> # ... perform calculations ...
            >>> result = rm.rm_close_files()  # Close files when done
            >>> if not result:
            >>>     print(f"Warning: {result}")
        """
        return irm_result(self.libc.RM_CloseFiles(self.id))

    def rm_concentrations2utility(self, c, n, tc, p_atm):
        """Transfer concentrations from a cell to the utility IPhreeqc instance.

        Transfers solution concentrations from a reaction cell to the utility
        IPhreeqc instance for further calculations or analysis. This method
        allows access to the full PHREEQC functionality for individual cells.

        Args:
            c (numpy.ndarray): Array of component concentrations to transfer.
            n (int): Cell number from which to transfer concentrations.
            tc (float): Temperature in Celsius for the utility calculation.
            p_atm (float): Pressure in atmospheres for the utility calculation.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.
        """
        return irm_result(self.libc.RM_Concentrations2Utility(self.id, c, n, tc, p_atm))

    def rm_create_mapping(self, grid2chem):
        """Create a mapping from grid cells to reaction cells.

        Establishes the relationship between transport grid cells and reaction
        cells, allowing for optimization when multiple grid cells share the
        same chemical composition. This can significantly reduce computational
        requirements for large models with repeating chemical conditions.

        Args:
            grid2chem (numpy.ndarray): Array mapping grid cells to reaction cells.
                Length should equal the number of grid cells in the transport model.
                Values are indices of reaction cells (0-based).

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            After calling this method, the number of reaction cells may be
            different from the number of grid cells, potentially reducing
            computational overhead.
        """
        return irm_result(self.libc.RM_CreateMapping(self.id, grid2chem.ctypes))

    def rm_decode_error(self, e):
        """Decode error code to human-readable message.

        Converts a numeric error code returned by PhreeqcRM functions into
        a descriptive error message string for debugging and logging purposes.

        Args:
            e (int): Error code to decode.

        Returns:
            str: Human-readable error message corresponding to the error code.

        Examples:
            >>> result = rm.rm_run_cells()
            >>> if not result:
            >>>     error_msg = rm.rm_decode_error(result.code)
            >>>     print(f"Error: {error_msg}")
        """
        return self.libc.RM_DecodeError(self.id, e)

    def rm_destroy(self):
        """Destroy a PhreeqcRM instance and free all associated memory.

        Deallocates all memory associated with the PhreeqcRM instance, including
        reaction cells, species data, and internal data structures. This method
        should be called when the PhreeqcRM instance is no longer needed to
        prevent memory leaks.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Warning:
            After calling this method, the PhreeqcRM instance should not be
            used for any further operations. All method calls will fail.

        Examples:
            >>> rm = PhreeqcRM()
            >>> rm.create(nxyz=100)
            >>> # ... use PhreeqcRM instance ...
            >>> rm.rm_destroy()  # Clean up when finished
        """
        return irm_result(self.libc.RM_Destroy(self.id))

    def rm_dump_module(self, dump_on, append):
        """Enable or disable dumping of reaction module data to file.
        Controls the output of detailed reaction module data to a dump file
        for debugging and analysis purposes. The dump file contains complete
        information about the state of all reaction cells.

        Args:
            dump_on (int): Enable (1) or disable (0) dump file creation.
            append (int): Append to existing file (1) or overwrite (0).

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            The dump file name is set by RM_SetDumpFileName(). If no name
            is set, a default name will be used.
        """
        return irm_result(self.libc.RM_DumpModule(self.id, dump_on, append))

    def rm_error_message(self, errstr):
        """Print an error message to the error output file.

        Writes an error message to the PhreeqcRM error log file. The message
        is formatted with timestamp and process information for debugging.

        Args:
            errstr (str): Error message string to write to the log.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Error logging must be enabled with RM_OpenFiles() for messages
            to be written to file.
        """
        return irm_result(self.libc.RM_ErrorMessage(self.id, errstr))

    def rm_find_components(self):
        """Find and count components for transport calculations.

        Analyzes all chemical definitions in the reaction module to determine
        the minimum set of components (elements plus charge) required for
        transport calculations. This method must be called after initial
        conditions are set but before starting transport calculations.

        The components identified include:
            - Chemical elements present in the system
            - Electric charge balance
            - Isotopes if defined in the database

        Returns:
            int: Number of components required for transport calculations.
                This defines the number of concentrations that must be
                transported for each grid cell.

        Note:
            This method should be called after RM_InitialPhreeqc2Module()
            and before beginning transport time stepping. The returned
            count determines array sizes for concentration transfers.

        Examples:
            >>> rm.run_initial_from_file("initial.pqi", ic_array)
            >>> ncomp = rm.rm_find_components()
            >>> print(f"Transport requires {ncomp} components")
        """
        return self.libc.RM_FindComponents(self.id)

    def rm_get_backward_mapping(self, n, list, size):
        """Get backward mapping from reaction cells to grid cells.

        Retrieves the list of grid cell numbers that map to a specific
        reaction cell. This is the inverse of the forward mapping and is
        useful for distributing reaction cell results back to grid cells.

        Args:
            n (int): Reaction cell number for which to get the mapping.
            list (numpy.ndarray): Array to receive grid cell numbers that
                map to the specified reaction cell.
            size (int): Size of the list array.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.
                The number of grid cells mapping to reaction cell n.
        """
        return irm_result(self.libc.RM_GetBackwardMapping(self.id, n, list, size))

    def rm_get_chemistry_cell_count(self):
        """Get the number of reaction cells in the module.

        Returns the number of reaction cells currently defined in the
        PhreeqcRM instance. This may be different from the number of
        grid cells if a mapping has been created to reduce computational
        requirements by grouping cells with identical chemistry.

        Returns:
            int: Number of reaction cells in the module.

        Note:
            Without cell mapping, this equals the number of grid cells.
            With mapping, this may be significantly smaller, improving
            computational efficiency.
        """
        return self.libc.RM_GetChemistryCellCount(self.id)

    def rm_get_component(self, num, chem_name, length):
        """Get the name of a component by index.

        Retrieves the name of a transport component identified by its index.
        Components are the chemical entities that must be transported in
        reactive transport simulations, typically elements plus charge.

        Args:
            num (int): Index of the component (0-based).
            chem_name (numpy.ndarray): String array to store component names.
                The name will be stored at index num.
            length (int): Maximum length of the component name string.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            This method is typically used in a loop to retrieve all component
            names after calling RM_FindComponents().
        """
        string = ctypes.create_string_buffer(length)
        status = self.libc.RM_GetComponent(self.id, num, string, length)
        chem_name[num] = string.value.decode()
        return irm_result(status)

    def rm_get_concentrations(self, c):
        """Retrieve component concentrations from reaction cells.

        Extracts current component concentrations from all reaction cells
        after geochemical calculations. These concentrations represent the
        dissolved components that must be transported in reactive transport
        simulations.

        Args:
            c (numpy.ndarray): Array to receive concentrations with shape
                (nxyz * ncomps). Will be filled with current concentrations
                in the units specified by RM_SetUnitsSolution().

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            The array is organized with all components for cell 0, followed
            by all components for cell 1, etc. Use this method after
            RM_RunCells() to get updated concentrations for transport.

        Examples:
            >>> ncomp = rm.rm_get_component_count()
            >>> conc = np.zeros(nxyz * ncomp)
            >>> result = rm.rm_get_concentrations(conc)
            >>> if result:
            >>>     # Reshape to (nxyz, ncomp) for easier handling
            >>>     conc_2d = conc.reshape(nxyz, ncomp)
        """
        return irm_result(self.libc.RM_GetConcentrations(self.id, c.ctypes))

    def rm_get_density(self, density):
        """Get solution density for all reaction cells.

        Retrieves the calculated solution density for each reaction cell
        based on the current chemical composition, temperature, and pressure.
        Densities are calculated using the thermodynamic database properties.

        Args:
            density (numpy.ndarray): Array to receive density values with
                length equal to the number of reaction cells. Units are kg/L.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Density calculations depend on the thermodynamic database and
            the specific solution composition. This method should be called
            after RM_RunCells() to get current density values.
        """
        return irm_result(self.libc.RM_GetDensity(self.id, density.ctypes))

    def rm_get_end_cell(self, ec):
        """Get the ending cell number for the current MPI process.

        In parallel (MPI) calculations, each process handles a subset of
        reaction cells. This method returns the index of the last cell
        handled by the current MPI process.

        Args:
            ec (ctypes pointer): Pointer to receive the ending cell index.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            For single-process calculations, this typically returns nxyz-1.
            For MPI calculations, the range [start_cell, end_cell] defines
            the cells handled by the current process.
        """
        return irm_result(self.libc.RM_GetEndCell(self.id, ec))

    def rm_get_equilibrium_phase_count(self):
        """Get the number of equilibrium phases defined in the system.

        Returns the count of mineral phases that can potentially precipitate
        or dissolve based on the thermodynamic database and initial conditions
        defined in the PhreeqcRM instance.

        Returns:
            int: Number of equilibrium phases defined in the system.

        Note:
            This count includes all phases that have been referenced in
            EQUILIBRIUM_PHASES blocks in the initial conditions, regardless
            of whether they are currently present in any cells.
        """
        return self.libc.RM_GetEquilibriumPhaseCount(self.id)

    def rm_get_equilibrium_phase_name(self, num, name, l1):
        """Get the name of an equilibrium phase by index.

        Retrieves the name of a mineral phase that can precipitate or dissolve
        in the geochemical system, identified by its index.

        Args:
            num (int): Index of the equilibrium phase (0-based).
            name (ctypes pointer): Buffer to receive the phase name.
            l1 (int): Maximum length of the name buffer.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.
        """
        return irm_result(self.libc.RM_GetEquilibriumPhaseName(self.id, num, name, l1))

    def rm_get_error_string(self, errstr, length):
        """Get the current error string from PhreeqcRM.

        Retrieves the most recent error message generated by PhreeqcRM
        operations. This provides detailed information about the last
        error that occurred during calculations.

        Args:
            errstr (ctypes pointer): Buffer to receive the error string.
            length (int): Maximum length of the error string buffer.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.
        """
        return irm_result(self.libc.RM_GetErrorString(self.id, errstr, length))

    def rm_get_error_string_length(self):
        """Get the length of the current error string.

        Returns the length of the error message string that can be retrieved
        with RM_GetErrorString(). Use this to allocate appropriate buffer
        size before calling RM_GetErrorString().

        Returns:
            int: Length of the current error string in characters.
        """
        return self.libc.RM_GetErrorStringLength(self.id)

    def rm_get_exchange_name(self, num, name, l1):
        return self.libc.RM_GetExchangeName(self.id, num, name, l1)

    def rm_get_exchange_species_count(self):
        return self.libc.RM_GetExchangeSpeciesCount(self.id)

    def rm_get_exchange_species_name(self, num, name, l1):
        return self.libc.RM_GetExchangeSpeciesName(self.id, num, name, l1)

    def rm_get_file_prefix(self, prefix, length):
        return self.libc.RM_GetFilePrefix(self.id, prefix.encode(), length)

    def rm_get_gas_components_count(self):
        return self.libc.RM_GetGasComponentsCount(self.id)

    def rm_get_gas_components_name(self, nun, name, l1):
        return self.libc.RM_GetGasComponentsName(self.id, nun, name, l1)

    def rm_get_gfw(self, gfw):
        """Get gram formula weights for transport components.

        Retrieves the gram formula weights (molecular weights) for all
        transport components. These weights are used to convert between
        molar and mass-based concentrations in transport calculations.

        Args:
            gfw (numpy.ndarray): Array to receive gram formula weights.
                Length should equal the number of components. Units are g/mol.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            The gram formula weights correspond to the components identified
            by RM_FindComponents() and can be retrieved by RM_GetComponent().
        """
        return irm_result(self.libc.RM_GetGfw(self.id, gfw.ctypes))

    def rm_get_grid_cell_count(self):
        """Get the number of grid cells in the model.

        Returns the total number of grid cells defined for the transport
        model, as specified when the PhreeqcRM instance was created.

        Returns:
            int: Number of grid cells in the transport model.

        Note:
            This is the nxyz parameter that was passed to RM_Create().
            It represents the total number of cells in the transport grid,
            which may be different from the number of reaction cells if
            cell mapping is used.
        """
        return self.libc.RM_GetGridCellCount(self.id)

    def rm_get_iphreeqc_id(self, i):
        return self.libc.RM_GetIPhreeqcId(self.id, i)

    def rm_get_kinetic_reactions_count(self):
        return self.libc.RM_GetKineticReactionsCount(self.id)

    def rm_get_kinetic_reactions_name(self, num, name, l1):
        return self.libc.RM_GetKineticReactionsName(self.id, num, name, l1)

    def rm_get_mpi_myself(self):
        return self.libc.RM_GetMpiMyself(self.id)

    def rm_get_mpi_tasks(self):
        return self.libc.RM_GetMpiTasks(self.id)

    def rm_get_nth_selected_output_user_number(self, n):
        return self.libc.RM_GetNthSelectedOutputUserNumber(self.id, n)

    def rm_get_saturation(self, sat_calc):
        """Get saturation values for all reaction cells.

        Retrieves the current saturation values for all reaction cells.
        Saturation represents the fraction of pore space occupied by water
        and affects the volume calculations in reactive transport.

        Args:
            sat_calc (numpy.ndarray): Array to receive saturation values.
                Length should equal the number of reaction cells.
                Values range from 0 to 1.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.
        """
        return irm_result(self.libc.RM_GetSaturation(self.id, sat_calc.ctypes))

    def rm_get_selected_output(self, so):
        """Retrieve selected output data from all reaction cells.

        Extracts the current selected output data, which includes calculated
        properties such as pH, pe, ionic strength, mineral saturation indices,
        species activities, and user-defined calculations specified in the
        PHREEQC input files.

        Args:
            so (numpy.ndarray): Array to receive selected output data with
                shape (nxyz, ncol) where ncol is the number of selected
                output columns defined in the PHREEQC input.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Use RM_GetSelectedOutputColumnCount() to determine the number
            of columns and RM_GetSelectedOutputHeading() to get column names.
            The get_selected_output_df() method provides a more convenient pandas
            DataFrame interface to this data.
        """
        return irm_result(self.libc.RM_GetSelectedOutput(self.id, so.ctypes))

    def rm_get_nth_selected_output_column_count(self):
        return self.libc.RM_GetNthSelectedOutputColumnCount(self.id)

    def rm_get_selected_output_count(self):
        return self.libc.RM_GetSelectedOutputCount(self.id)

    def rm_get_selected_output_heading(self, col, headings, length):
        string = ctypes.create_string_buffer(length)
        status = self.libc.RM_GetSelectedOutputHeading(self.id, col, string, length)
        headings[col] = string.value.decode()
        return irm_result(status)

    def rm_get_selected_output_column_count(self):
        """Get number of columns in selected output.

        Returns:
            int: Number of selected output columns.
        """
        return self.libc.RM_GetSelectedOutputColumnCount(self.id)

    def rm_get_selected_output_row_count(self):
        """Get number of rows in selected output.

        Returns:
            int: Number of selected output rows.
        """
        return self.libc.RM_GetSelectedOutputRowCount(self.id)

    def rm_get_si_count(self):
        return self.libc.RM_GetSICount(self.id)

    def rm_get_si_name(self, num, name, l1):
        return self.libc.RM_GetSIName(self.id, num, name, l1)

    def rm_get_solid_solution_components_count(self):
        return self.libc.RM_GetSolidSolutionComponentsCount(self.id)

    def rm_get_solid_solution_components_name(self, num, name, l1):
        return self.libc.RM_GetSolidSolutionComponentsName(self.id, num, name, l1)

    def rm_get_solid_solution_name(self, num, name, l1):
        return self.libc.RM_GetSolidSolutionName(self.id, num, name, l1)

    def rm_get_solution_volume(self, vol):
        return self.libc.RM_GetSolutionVolume(self.id, vol.ctypes)

    def rm_get_species_concentrations(self, species_conc):
        """Retrieve aqueous species concentrations from reaction cells.

        Extracts the concentrations of individual aqueous species from all
        reaction cells. This provides more detailed chemical information than
        component concentrations, including the speciation of dissolved elements.

        Args:
            species_conc (numpy.ndarray): Array to receive species concentrations
                with shape (nxyz * nspecies). Species concentrations are in
                the same units as solution concentrations (mol/L, mmol/L, or μmol/L).

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Species saving must be enabled with RM_SetSpeciesSaveOn(1) before
            this method can be used. Use RM_GetSpeciesCount() to determine the
            number of species and RM_GetSpeciesName() to get species names.

        Examples:
            >>> rm.rm_set_species_save_on(1)  # Enable species saving
            >>> rm.rm_run_cells()  # Run reactions
            >>> nspecies = rm.rm_get_species_count()
            >>> species_c = np.zeros(nxyz * nspecies)
            >>> rm.rm_get_species_concentrations(species_c)
        """
        return irm_result(self.libc.RM_GetSpeciesConcentrations(self.id, species_conc.ctypes))

    def rm_get_species_count(self):
        """Get the number of aqueous species in the geochemical system.

        Returns the total number of dissolved aqueous species that can exist
        in the current geochemical system based on the loaded thermodynamic
        database and the chemical components present in the system.

        Returns:
            int: Number of aqueous species defined in the system. This includes
                primary species (elements and basis species) and secondary species
                (complexes) formed from the primary species.

        Note:
            The species count is determined after loading the database and
            running initial equilibrium calculations. Species names can be
            retrieved using RM_GetSpeciesName() with indices from 0 to count-1.

        Examples:
            >>> nspecies = rm.rm_get_species_count()
            >>> print(f"System contains {nspecies} aqueous species")
            >>> # Get all species names
            >>> species_names = np.zeros(nspecies, dtype='U20')
            >>> for i in range(nspecies):
            >>>     rm.rm_get_species_name(i, species_names, 20)
        """
        return self.libc.RM_GetSpeciesCount(self.id)

    def rm_get_species_d25(self, diffc):
        """Get diffusion coefficients at 25°C for all aqueous species.

        Retrieves the reference diffusion coefficients (at 25°C in water)
        for all aqueous species in the system. These values are used in
        multicomponent diffusion calculations in reactive transport models.

        Args:
            diffc (numpy.ndarray): Array to receive diffusion coefficients
                with length equal to the number of species. Units are m²/s.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Diffusion coefficients are taken from the thermodynamic database.
            For transport at different temperatures, these values should be
            corrected using appropriate temperature relationships.
        """
        return irm_result(self.libc.RM_GetSpeciesD25(self.id, diffc.ctypes))

    def rm_get_species_log10_gammas(self, species_log10gammas):
        """Get log10 activity coefficients for all aqueous species.

        Retrieves the base-10 logarithm of activity coefficients for all
        aqueous species in each reaction cell. Activity coefficients account
        for non-ideal solution behavior and are used to calculate activities
        from concentrations.

        Args:
            species_log10gammas (numpy.ndarray): Array to receive log10 activity
                coefficients with shape (nxyz * nspecies). Values are dimensionless.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Activity coefficients depend on ionic strength, temperature, and
            the activity model used in the thermodynamic database (e.g.,
            Debye-Hückel, Pitzer, SIT). Species saving must be enabled.
        """
        return irm_result(self.libc.RM_GetSpeciesLog10Gammas(self.id, species_log10gammas.ctypes))

    def rm_get_species_name(self, num, chem_name, length):
        """Get the name of an aqueous species by index.

        Retrieves the name of an aqueous species identified by its index.
        Species names follow PHREEQC conventions and include charge states
        for ionic species.

        Args:
            num (int): Index of the species (0-based).
            chem_name (numpy.ndarray): String array to store species names.
                The name will be stored at index num.
            length (int): Maximum length of the species name string.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Examples:
            >>> nspecies = rm.rm_get_species_count()
            >>> species_names = np.zeros(nspecies, dtype='U20')
            >>> for i in range(nspecies):
            >>>     rm.rm_get_species_name(i, species_names, 20)
            >>> print(species_names)  # ['H2O', 'H+', 'OH-', 'Ca+2', ...]
        """
        string = ctypes.create_string_buffer(length)
        status = self.libc.RM_GetSpeciesName(self.id, num, string, length)
        chem_name[num] = string.value.decode()
        return irm_result(status)

    def rm_get_species_save_on(self):
        """Check if species concentration saving is enabled.

        Returns the current setting for species concentration saving. When
        enabled, PhreeqcRM calculates and stores individual species
        concentrations that can be retrieved with RM_GetSpeciesConcentrations().

        Returns:
            int: 1 if species saving is enabled, 0 if disabled.

        Note:
            Species saving increases memory usage and computation time but
            provides detailed speciation information useful for analysis
            and multicomponent diffusion calculations.
        """
        return self.libc.RM_GetSpeciesSaveOn(self.id)

    def rm_get_species_z(self, z):
        """Get charge values for all aqueous species.

        Retrieves the electric charge (valence) for each aqueous species
        in the system. Charge values are essential for electrostatic
        calculations and charge balance constraints.

        Args:
            z (numpy.ndarray): Array to receive charge values with length
                equal to the number of species. Values are dimensionless
                (e.g., +2 for Ca+2, -1 for Cl-, 0 for neutral species).

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Charge values are defined in the thermodynamic database and
            are used in activity coefficient calculations and electroneutrality
            constraints.
        """
        return irm_result(self.libc.RM_GetSpeciesZ(self.id, z.ctypes))

    def rm_get_start_cell(self, sc):
        return self.libc.RM_GetStartCell(self.id, sc)

    def rm_get_surface_name(self, num, name, l1):
        return self.libc.RM_GetSurfaceName(self.id, num, name, l1)

    def rm_get_surface_type(self, num, name, l1):
        return self.libc.RM_GetSurfaceType(self.id, num, name, l1)

    def rm_get_thread_count(self):
        return self.libc.RM_GetThreadCount(self.id)

    def rm_get_time(self):
        """Get the current simulation time.

        Returns the current time in the reactive transport simulation.
        This time is used for kinetic rate calculations and time-dependent
        boundary conditions.

        Returns:
            float: Current simulation time in user-defined units (typically
                seconds, days, or years depending on the model setup).

        Note:
            The simulation time is set by RM_SetTime() and is used internally
            by PhreeqcRM for kinetic calculations. The time units should be
            consistent with kinetic rate constants in the database.

        Examples:
            >>> current_time = rm.rm_get_time()
            >>> print(f"Current simulation time: {current_time} days")
        """
        self.libc.RM_GetTime.restype = ctypes.c_double
        return self.libc.RM_GetTime(self.id)

    def rm_get_time_conversion(self):
        self.libc.RM_GetTimeConversion.restype = ctypes.c_double
        return self.libc.RM_GetTimeConversion(self.id)

    def rm_get_time_step(self):
        """Get the current time step duration.

        Returns the duration of the current time step used for kinetic
        calculations and time-dependent processes in the geochemical system.

        Returns:
            float: Current time step duration in user-defined units (typically
                seconds, days, or years consistent with the simulation time units).

        Note:
            The time step is set by RM_SetTimeStep() and affects the integration
            of kinetic rate equations. Smaller time steps provide more accurate
            solutions but require more computational time.

        Examples:
            >>> dt = rm.rm_get_time_step()
            >>> print(f"Current time step: {dt} days")
        """
        self.libc.RM_GetTimeStep.restype = ctypes.c_double
        return self.libc.RM_GetTimeStep(self.id)

    def rm_initial_phreeqc2_module(self, ic1, ic2, f1):
        """Transfer initial conditions from InitialPhreeqc to reaction module.

        Args:
            ic1 (numpy.ndarray): Initial condition indices for primary entities.
            ic2 (numpy.ndarray): Initial condition indices for secondary entities.
            f1 (numpy.ndarray): Mixing fractions for primary entities.

        Returns:
            int: irm_result status code (0 for success).
        """
        return irm_result(self.libc.RM_InitialPhreeqc2Module(self.id, ic1.ctypes, ic2.ctypes, f1.ctypes))

    def rm_initial_phreeqc2_concentrations(self, c, n_boundary, boundary_solution1, boundary_solution2, fraction1):
        return self.libc.RM_InitialPhreeqc2Concentrations(
            self.id, c.ctypes, n_boundary, boundary_solution1.ctypes, boundary_solution2.ctypes, fraction1.ctypes
        )

    def rm_initial_phreeqc2_species_concentrations(
        self, species_c, n_boundary, boundary_solution1, boundary_solution2, fraction1
    ):
        return self.libc.RM_InitialPhreeqc2SpeciesConcentrations(
            self.id,
            species_c.ctypes,
            n_boundary.ctypes,
            boundary_solution1.ctypes,
            boundary_solution2.ctypes,
            fraction1.ctypes,
        )

    def rm_initial_phreeqc_cell2_module(self, n, module_numbers, dim_module_numbers):
        return self.libc.RM_InitialPhreeqcCell2Module(self.id, n, module_numbers, dim_module_numbers)

    def rm_load_database(self, db_name):
        """Load a thermodynamic database for geochemical calculations.

        Loads a PHREEQC-format thermodynamic database containing thermodynamic
        data for aqueous species, minerals, gases, and other phases. The database
        defines the chemical system and enables geochemical calculations.

        Args:
            db_name (str): Path to the database file. Common databases include:
                - "phreeqc.dat": Standard PHREEQC database (25°C)
                - "Amm.dat": Extended database with ammonia species
                - "pitzer.dat": Pitzer interaction parameter database
                - "sit.dat": Specific ion interaction theory database

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Raises:
            RuntimeError: If the database file cannot be found or loaded.

        Note:
            This method must be called before setting up initial conditions
            or running calculations. The database determines which species,
            minerals, and reactions are available for calculations.

        Examples:
            >>> rm = PhreeqcRM()
            >>> rm.create(nxyz=100)
            >>> result = rm.rm_load_database("phreeqc.dat")
            >>> if not result:
            >>>     raise RuntimeError(f"Failed to load database: {result}")
        """
        return irm_result(self.libc.RM_LoadDatabase(self.id, db_name.encode()))

    def rm_log_message(self, str):
        return self.libc.RM_LogMessage(self.id, str.encode())

    def rm_mpi_worker(self):
        return self.libc.RM_MpiWorker(self.id)

    def rm_mpi_worker_break(self):
        return self.libc.RM_MpiWorkerBreak(self.id)

    def rm_open_files(self):
        """Open output files for logging, debugging, and error reporting.

        Creates and opens output files for PhreeqcRM logging, error messages,
        and debugging information. File names are based on the prefix set
        by RM_SetFilePrefix().

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Files created:
            - {prefix}.log: General log messages and information
            - {prefix}.err: Error messages and warnings
            - {prefix}.out: PHREEQC output from calculations

        Note:
            Call RM_SetFilePrefix() before this method to set the file name
            prefix. Use RM_CloseFiles() to properly close files when finished.

        Examples:
            >>> rm.rm_set_file_prefix("simulation")
            >>> rm.rm_open_files()  # Creates simulation.log, simulation.err, etc.
            >>> # ... run calculations ...
            >>> rm.rm_close_files()  # Clean up
        """
        return irm_result(self.libc.RM_OpenFiles(self.id))

    def rm_output_message(self, str):
        return self.libc.RM_OutputMessage(self.id, str.encode())

    def rm_run_cells(self):
        """Run geochemical reactions for all reaction cells.

        Performs equilibrium speciation and kinetic reactions for the current
        time step in all reaction cells. This is the core computational method
        that updates chemical compositions based on thermodynamic equilibrium
        and reaction kinetics.

        The method performs:
            - Aqueous speciation calculations
            - Mineral precipitation/dissolution equilibrium
            - Ion exchange equilibrium
            - Surface complexation equilibrium
            - Gas phase equilibrium
            - Kinetic reaction integration over the time step

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Before calling this method, ensure that:
            - Concentrations are set with RM_SetConcentrations()
            - Current time is set with RM_SetTime()
            - Time step is set with RM_SetTimeStep()
            - Temperature and pressure are set if needed

        Examples:
            >>> rm.rm_set_concentrations(concentrations)
            >>> rm.rm_set_time(current_time)
            >>> rm.rm_set_time_step(dt)
            >>> result = rm.rm_run_cells()
            >>> if result:
            >>>     rm.rm_get_concentrations(updated_concentrations)
            >>> else:
            >>>     print(f"Reaction failed: {result}")
        """
        return irm_result(self.libc.RM_RunCells(self.id))

    def rm_run_file(self, workers, initial_phreeqc, utility, chem_name):
        """Run a PHREEQC input file in specified PhreeqcRM instances.

        Executes a PHREEQC input file (.pqi format) in one or more PhreeqcRM
        instances. This is used to define initial conditions, equilibrium phases,
        exchange assemblages, surface complexation sites, gas phases, solid
        solutions, and kinetic reactions.

        Args:
            workers (int): Run in worker instances (1) or not (0). Worker instances
                handle the main geochemical calculations for reaction cells.
            initial_phreeqc (int): Run in initial PhreeqcRM instance (1) or not (0).
                Used for defining initial chemical conditions and templates.
            utility (int): Run in utility instance (1) or not (0). Utility instance
                provides access to full PHREEQC functionality for special calculations.
            chem_name (str): Path to the PHREEQC input file containing chemical
                definitions and calculations.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            PHREEQC input files define the chemical system using standard PHREEQC
            syntax. Common blocks include SOLUTION, EQUILIBRIUM_PHASES, EXCHANGE,
            SURFACE, GAS_PHASE, SOLID_SOLUTIONS, KINETICS, and SELECTED_OUTPUT.

        Examples:
            >>> # Run initial conditions file in initial PhreeqcRM instance
            >>> result = rm.rm_run_file(0, 1, 0, "initial_conditions.pqi")
            >>> if not result:
            >>>     print(f"Error running initial conditions file: {result}")
        """
        return irm_result(self.libc.RM_RunFile(self.id, workers, initial_phreeqc, utility, chem_name.encode()))

    def rm_run_string(self, workers, initial_phreeqc, utility, input_string):
        """Run PHREEQC input from a string in specified instances.

        Executes PHREEQC input commands provided as a string in one or more
        PhreeqcRM instances. This allows dynamic generation of PHREEQC input
        without creating temporary files.

        Args:
            workers (int): Run in worker instances (1) or not (0).
            initial_phreeqc (int): Run in initial PhreeqcRM instance (1) or not (0).
            utility (int): Run in utility instance (1) or not (0).
            input_string (str): PHREEQC input commands as a string, using
                standard PHREEQC syntax with newline separators.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Examples:
            >>> phreeqc_input = '''
            >>> SOLUTION 1
            >>>     pH 7.0
            >>>     Ca 1.0
            >>>     Cl 2.0
            >>> END
            >>> '''
            >>> rm.rm_run_string(0, 1, 0, phreeqc_input)
        """
        return irm_result(self.libc.RM_RunString(self.id, workers, initial_phreeqc, utility, input_string.encode()))

    def rm_screen_message(self, str):
        return self.libc.RM_ScreenMessage(self.id, str.encode())

    def rm_set_component_h2o(self, tf):
        """Set whether to include H2O as a transport component.

        Controls whether water (H2O) is included in the list of components
        that must be transported in reactive transport simulations. This
        setting affects the component count and transport requirements.

        Args:
            tf (int): Include H2O as a component:
                1 = Include H2O as a transport component
                0 = Exclude H2O from transport components (default)

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Typically, H2O is not transported as a separate component because
            water content is determined by porosity, saturation, and density.
            Including H2O increases the number of transport equations but may
            be necessary for some specialized applications.

        Examples:
            >>> rm.rm_set_component_h2o(0)  # Standard: don't transport H2O
            >>> ncomp = rm.rm_find_components()  # Get component count
        """
        return irm_result(self.libc.RM_SetComponentH2O(self.id, tf))

    def rm_set_concentrations(self, c):
        """Set component concentrations for all reaction cells.

        Transfers concentration data from the transport model to the reaction
        module. This method is typically called at each transport time step
        to provide updated concentrations for geochemical calculations.

        Args:
            c (numpy.ndarray): Concentration array with shape (nxyz * ncomps)
                containing concentrations for all cells and components.
                Concentrations must be in the units specified by RM_SetUnitsSolution().
                Array is organized with all components for cell 0, followed by
                all components for cell 1, etc.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            This method sets the starting concentrations for the next call to
            RM_RunCells(). The concentrations are used as initial conditions
            for equilibrium and kinetic calculations.

        Examples:
            >>> # Transport model updates concentrations
            >>> new_conc = transport_step(old_conc, velocity, dt)
            >>> # Transfer to reaction module
            >>> result = rm.rm_set_concentrations(new_conc.flatten())
            >>> if result:
            >>>     rm.rm_run_cells()  # Run geochemical reactions
        """
        return irm_result(self.libc.RM_SetConcentrations(self.id, c.ctypes))

    def rm_set_current_selected_output_user_number(self, n_user):
        return self.libc.RM_SetCurrentSelectedOutputUserNumber(self.id, n_user)

    def rm_set_density(self, density):
        return self.libc.RM_SetDensity(self.id, density.ctypes)

    def rm_set_dump_file_name(self, dump_name):
        return self.libc.RM_SetDumpFileName(self.id, dump_name)

    def rm_set_error_handler_mode(self, mode):
        return self.libc.RM_SetErrorHandlerMode(self.id, mode)

    def rm_set_file_prefix(self, prefix):
        """Set prefix for output files.

        Args:
            prefix (str): Prefix string for output file names.

        Returns:
            int: irm_result status code (0 for success).
        """
        return irm_result(self.libc.RM_SetFilePrefix(self.id, prefix.encode()))

    def rm_set_mpi_worker_callback_cookie(self, cookie):
        return self.libc.RM_SetMpiWorkerCallbackCookie(self.id, cookie)

    def rm_set_partition_uz_solids(self, tf):
        return self.libc.RM_SetPartitionUZSolids(self.id, tf)

    def rm_set_porosity(self, por):
        """Set porosity values for all grid cells.

        Defines the porosity (void fraction) for each grid cell, which
        represents the fraction of bulk volume occupied by pore space.
        Porosity affects volume calculations for concentration conversions
        and reaction extent calculations.

        Args:
            por (numpy.ndarray): Array of porosity values for each cell.
                Length should equal the number of grid cells (nxyz).
                Values must be between 0 and 1, where:
                - 0 = no pore space (solid rock)
                - 1 = completely porous (pure fluid)
                - Typical values: 0.1-0.4 for sedimentary rocks

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Porosity is used with saturation to calculate the water volume
            in each cell: water_volume = porosity × saturation × bulk_volume.
            This affects concentration calculations and reaction stoichiometry.

        Examples:
            >>> porosity = np.full(nxyz, 0.25)  # 25% porosity for all cells
            >>> rm.rm_set_porosity(porosity)
        """
        return irm_result(self.libc.RM_SetPorosity(self.id, por.ctypes))

    def rm_set_pressure(self, p):
        return self.libc.RM_SetPressure(self.id, p.ctypes)

    def rm_set_print_chemistry_mask(self, cell_mask):
        return self.libc.RM_SetPrintChemistryMask(self.id, cell_mask.ctypes)

    def rm_set_print_chemistry_on(self, workers, initial_phreeqc, utility):
        return self.libc.RM_SetPrintChemistryOn(self.id, workers, initial_phreeqc, utility)

    def rm_set_rebalance_by_cell(self, method):
        return self.libc.RM_SetRebalanceByCell(self.id, method)

    def rm_set_rebalance_fraction(self, f):
        """Set load balancing algorithm fraction.

        Args:
            f (float): Fraction for load balancing (typically 0.5).

        Returns:
            int: irm_result status code (0 for success).
        """
        return irm_result(self.libc.RM_SetRebalanceFraction(self.id, ctypes.c_double(f)))

    def rm_set_representative_volume(self, rv):
        return self.libc.RM_SetRepresentativeVolume(self.id, rv.ctypes)

    def rm_set_saturation(self, sat):
        """Set water saturation values for all grid cells.

        Defines the water saturation for each grid cell, representing the
        fraction of pore space occupied by water. Saturation affects volume
        calculations and is particularly important in unsaturated zone
        modeling where gas phase may occupy part of the pore space.

        Args:
            sat (numpy.ndarray): Array of saturation values for each cell.
                Length should equal the number of grid cells (nxyz).
                Values must be between 0 and 1, where:
                - 0 = dry (no water in pores)
                - 1 = fully saturated (all pores filled with water)
                - Typical values: 0.1-1.0 depending on vadose zone conditions

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            Water saturation is used with porosity to calculate the actual
            water volume: water_volume = porosity × saturation × bulk_volume.
            This directly affects solution concentrations and reaction rates.

        Examples:
            >>> # Fully saturated conditions
            >>> saturation = np.ones(nxyz)
            >>> rm.rm_set_saturation(saturation)
            >>>
            >>> # Partially saturated (vadose zone)
            >>> sat_profile = np.linspace(0.3, 1.0, nxyz)  # Increasing with depth
            >>> rm.rm_set_saturation(sat_profile)
        """
        return irm_result(self.libc.RM_SetSaturation(self.id, sat.ctypes))

    def rm_set_screen_on(self, tf):
        return self.libc.RM_SetScreenOn(self.id, tf)

    def rm_set_selected_output_on(self, selected_output):
        return self.libc.RM_SetSelectedOutputOn(self.id, selected_output)

    def rm_set_species_save_on(self, save_on):
        """Enable or disable saving of aqueous species concentrations.

        Controls whether PhreeqcRM calculates and stores individual aqueous
        species concentrations that can be retrieved with RM_GetSpeciesConcentrations().
        This provides detailed speciation information but increases memory usage
        and computation time.

        Args:
            save_on (int): Species saving option:
                1 = Enable species concentration saving
                0 = Disable species saving (default, saves memory and time)

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            When enabled, PhreeqcRM stores concentrations for all aqueous species
            after each call to RM_RunCells(). This is required for:
            - Multicomponent diffusion calculations
            - Detailed speciation analysis
            - Species-specific output and post-processing

        Examples:
            >>> rm.rm_set_species_save_on(1)  # Enable species saving
            >>> rm.rm_run_cells()  # Calculate with species saving
            >>> species_conc = np.zeros(nxyz * nspecies)
            >>> rm.rm_get_species_concentrations(species_conc)
        """
        return irm_result(self.libc.RM_SetSpeciesSaveOn(self.id, save_on))

    def rm_set_temperature(self, t):
        return self.libc.RM_SetTemperature(self.id, t.ctypes)

    def rm_set_time(self, time):
        """Set the current simulation time for geochemical calculations.

        Updates the simulation time used by PhreeqcRM for kinetic rate
        calculations and time-dependent processes. The time should be
        consistent with the time stepping in the transport model.

        Args:
            time (float): Current simulation time in user-defined units.
                Units should be consistent with kinetic rate constants
                in the thermodynamic database (typically seconds, days, or years).

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            This method should be called before each call to RM_RunCells()
            to ensure kinetic calculations use the correct time. The time
            is used to integrate kinetic rate equations over the time step.

        Examples:
            >>> for step in range(num_steps):
            >>>     current_time = step * dt
            >>>     rm.rm_set_time(current_time)
            >>>     rm.rm_set_time_step(dt)
            >>>     rm.rm_run_cells()
        """
        return irm_result(self.libc.RM_SetTime(self.id, ctypes.c_double(time)))

    def rm_set_time_conversion(self, conv_factor):
        return self.libc.RM_SetTimeConversion(self.id, ctypes.c_double(conv_factor))

    def rm_set_time_step(self, time_step):
        """Set the time step duration for kinetic calculations.

        Specifies the time interval over which kinetic reactions will be
        integrated during the next call to RM_RunCells(). The time step
        affects the accuracy of kinetic calculations.

        Args:
            time_step (float): Time step duration in user-defined units.
                Must be positive and consistent with simulation time units.
                Smaller time steps provide better accuracy for fast kinetic
                reactions but increase computational cost.

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            The time step is used for integrating kinetic rate equations.
            For stiff kinetic systems, smaller time steps may be required
            for numerical stability and accuracy.

        Examples:
            >>> dt = 0.1  # 0.1 day time step
            >>> rm.rm_set_time_step(dt)
            >>> rm.rm_run_cells()  # Integrate kinetics over dt
        """
        return irm_result(self.libc.RM_SetTimeStep(self.id, ctypes.c_double(time_step)))

    def rm_set_units_exchange(self, option):
        """Set units for exchange reactions.

        Args:
            option (int): Units option for exchange calculations.

        Returns:
            int: irm_result status code (0 for success).
        """
        return irm_result(self.libc.RM_SetUnitsExchange(self.id, option))

    def rm_set_units_gas_phase(self, option) -> None:
        self.libc.RM_SetUnitsGasPhase(self.id, option)

    def rm_set_units_kinetics(self, option) -> None:
        self.libc.RM_SetUnitsKinetics(self.id, option)

    def rm_set_units_p_passemblage(self, option):
        return self.libc.RM_SetUnitsPPassemblage(self.id, option)

    def rm_set_units_solution(self, option):
        """Set concentration units for aqueous solutions.

        Specifies the units for solution concentrations used in all
        concentration transfers between the transport model and PhreeqcRM.
        This affects how concentration data is interpreted and returned.

        Args:
            option (int): Units option for solution concentrations:
                1 = mol/L (molar)
                2 = mmol/L (millimolar) - commonly used
                3 = μmol/L (micromolar) - for trace species

        Returns:
            IRMStatus: Status object with code, name, and message. Use bool(result) to check success.

        Note:
            This setting affects:
            - RM_SetConcentrations() input interpretation
            - RM_GetConcentrations() output units
            - RM_GetSpeciesConcentrations() output units
            - Initial condition concentration scaling

        Examples:
            >>> rm.rm_set_units_solution(2)  # Use mmol/L
            >>> # Now all concentrations are in millimolar units
            >>> conc_mmol = np.array([1.0, 0.5, 2.0])  # 1, 0.5, 2 mmol/L
            >>> rm.rm_set_concentrations(conc_mmol)
        """
        return irm_result(self.libc.RM_SetUnitsSolution(self.id, option))

    def rm_set_units_ss_assemblage(self, option):
        return self.libc.RM_SetUnitsSSassemblage(self.id, option)

    def rm_set_units_surface(self, option):
        """Set units for surface complexation reactions.

        Args:
            option (int): Units option for surface calculations.

        Returns:
            int: irm_result status code (0 for success).
        """
        return irm_result(self.libc.RM_SetUnitsSurface(self.id, option))

    def rm_species_concentrations2_module(self, species_conc):
        return self.libc.RM_SpeciesConcentrations2Module(self.id, species_conc.ctypes)

    def rm_use_solution_density_volume(self, tf):
        return self.libc.RM_UseSolutionDensityVolume(self.id, tf)

    def rm_warning_message(self, warn_str):
        return self.libc.RM_WarningMessage(self.id, warn_str)

    def rm_get_component_count(self):
        """Get the number of transport components in the system.

        Returns the number of chemical components (elements plus charge balance)
        that must be transported in reactive transport simulations. This count
        is determined after calling RM_FindComponents() and defines the size
        of concentration arrays.

        Returns:
            int: Number of transport components, which includes:
                - Chemical elements present in the system (e.g., Ca, Cl, C)
                - Electric charge balance component
                - Isotopes if defined in the database
                - Excludes H2O unless RM_SetComponentH2O(1) was called

        Note:
            This method should be called after RM_FindComponents() to get the
            correct component count. The returned value determines array sizes
            for RM_SetConcentrations() and RM_GetConcentrations().

        Examples:
            >>> rm.run_initial_from_file("initial.pqi", ic_array)
            >>> ncomp = rm.rm_get_component_count()
            >>> conc_array = np.zeros(nxyz * ncomp)
            >>> rm.rm_get_concentrations(conc_array)
        """
        return self.libc.RM_GetComponentCount(self.id)
