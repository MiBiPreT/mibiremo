"""Tests for the mibiremo.phreeqcrm module.
The C library calls are mocked to allow testing without the actual library.
The objective is to test the Python wrapper logic.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import mibiremo


class TestPhreeqcRMInitialization:
    """Test PhreeqcRM initialization and basic setup."""

    def test_phreeqcrm_init(self):
        """Test PhreeqcRM instance initialization."""
        phr = mibiremo.PhreeqcRM()
        assert phr is not None
        assert not phr._initialized
        assert phr.dllpath is None
        assert phr.nxyz == 1
        assert phr.n_threads == 1
        assert phr.libc is None
        assert phr.id is None
        assert phr.components is None
        assert phr.species is None

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_create_default_params(self, mock_cdll):
        """Test create method with default parameters, mocking the C library."""
        mock_lib = MagicMock()  # Create a mock to simulate the C library
        mock_lib.RM_Create.return_value = 1  # Simulate successful instance creation
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        assert phr._initialized
        assert phr.nxyz == 1
        assert phr.n_threads == 1
        assert phr.id == 1
        mock_lib.RM_Create.assert_called_once_with(1, 1)

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_create_custom_params(self, mock_cdll):
        """Test create method with custom parameters."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 2
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=100, n_threads=4)

        assert phr._initialized
        assert phr.nxyz == 100
        assert phr.n_threads == 4
        assert phr.id == 2
        mock_lib.RM_Create.assert_called_once_with(100, 4)

    @patch("mibiremo.phreeqc.os.cpu_count")
    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_create_auto_threads(self, mock_cdll, mock_cpu_count):
        """Test automatic detection of CPU threads."""
        mock_cpu_count.return_value = 8
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 3
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=50, n_threads=-1)

        assert phr.n_threads == 8
        mock_lib.RM_Create.assert_called_once_with(50, 8)

    @patch("mibiremo.phreeqc.os.name", "nt")
    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_create_windows_dll_path(self, mock_cdll):
        """Test DLL path generation on Windows."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 4
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        assert phr.dllpath.endswith("PhreeqcRM.dll")
        assert "lib" in phr.dllpath

    @patch("mibiremo.phreeqc.os.name", "posix")
    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_create_linux_dll_path(self, mock_cdll):
        """Test DLL path generation on Linux."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 5
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        assert phr.dllpath.endswith("PhreeqcRM.so")
        assert "lib" in phr.dllpath

    @patch("mibiremo.phreeqc.os.name", "unknown")
    def test_create_unsupported_os(self):
        """Test exception for unsupported operating system."""
        phr = mibiremo.PhreeqcRM()
        with pytest.raises(Exception, match="Operating system not supported"):
            phr.create()

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_create_failure(self, mock_cdll):
        """Test failure in PhreeqcRM instance creation."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.side_effect = Exception("Creation failed")
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        with pytest.raises(RuntimeError, match="Failed to create PhreeqcRM instance"):
            phr.create()


class TestPhreeqcRMInitializePhreeqc:
    """Test PhreeqcRM initialization with database and parameters."""

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_initialize_phreeqc_not_initialized(self, mock_cdll):
        """Test initialize_phreeqc before create() is called."""
        phr = mibiremo.PhreeqcRM()
        with pytest.raises(RuntimeError, match="PhreeqcRM instance not initialized"):
            phr.initialize_phreeqc("test.dat")

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_initialize_phreeqc_success(self, mock_cdll):
        """Test successful initialize_phreeqc call."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_LoadDatabase.return_value = 0  # Success
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=10)
        phr.initialize_phreeqc("phreeqc.dat")

        # Verify key method calls
        mock_lib.RM_LoadDatabase.assert_called_once_with(1, b"phreeqc.dat")
        mock_lib.RM_SetComponentH2O.assert_called_once_with(1, 0)
        mock_lib.RM_SetRebalanceFraction.assert_called_once()
        mock_lib.RM_SetUnitsSolution.assert_called_once_with(1, 2)
        mock_lib.RM_SetFilePrefix.assert_called_once_with(1, b"phr")
        mock_lib.RM_OpenFiles.assert_called_once_with(1)

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_initialize_phreeqc_database_failure(self, mock_cdll):
        """Test initialize_phreeqc with database loading failure."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_LoadDatabase.return_value = -1  # Failure
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=10)

        with pytest.raises(RuntimeError, match="Failed to load Phreeqc database"):
            phr.initialize_phreeqc("invalid.dat")

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_initialize_phreeqc_custom_params(self, mock_cdll):
        """Test initialize_phreeqc with custom parameters."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_LoadDatabase.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=5)
        phr.initialize_phreeqc(
            "custom.dat", units_solution=1, units=2, porosity=0.3, saturation=0.8, multicomponent=False
        )

        mock_lib.RM_SetUnitsSolution.assert_called_with(1, 1)
        mock_lib.RM_SetUnitsPPassemblage.assert_called_with(1, 2)
        mock_lib.RM_SetSpeciesSaveOn.assert_not_called()  # multicomponent=False


class TestPhreeqcRMRunInitialFromFile:
    """Test running initial conditions from file."""

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_run_initial_from_file_success(self, mock_cdll):
        """Test successful run_initial_from_file."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_RunFile.return_value = 0  # Success
        mock_lib.RM_InitialPhreeqc2Module.return_value = 0
        mock_lib.RM_FindComponents.return_value = 3
        mock_lib.RM_GetSpeciesCount.return_value = 10
        mock_lib.RM_SetTime.return_value = 0
        mock_lib.RM_SetTimeStep.return_value = 0
        mock_lib.RM_RunCells.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=2)

        ic = np.array([[1, -1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1]])

        phr.run_initial_from_file("test.pqi", ic)

        mock_lib.RM_RunFile.assert_called_once_with(1, 1, 1, 1, b"test.pqi")
        assert phr.components is not None
        assert phr.species is not None

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_run_initial_from_file_failure(self, mock_cdll):
        """Test run_initial_from_file with file failure."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_RunFile.return_value = -1  # Failure
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)

        ic = np.array([[1, -1, -1, -1, -1, -1, -1]])

        with pytest.raises(RuntimeError, match="Failed to run Phreeqc input file"):
            phr.run_initial_from_file("invalid.pqi", ic)

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_run_initial_from_file_wrong_ic_shape(self, mock_cdll):
        """Test run_initial_from_file with incorrect IC array shape."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_RunFile.return_value = 0  # Success for file run
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)  # Need to create first

        # Wrong shape - should be (nxyz, 7)
        ic = np.array([[1, -1, -1, -1]])

        with pytest.raises(ValueError, match="Initial conditions array must have shape"):
            phr.run_initial_from_file("test.pqi", ic)

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_run_initial_from_file_invalid_ic_type(self, mock_cdll):
        """Test run_initial_from_file with non-convertible IC data."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_RunFile.return_value = 0  # Success for file run
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)

        # Create a numpy array with string data that can't be converted to int
        ic = np.array([["invalid", "data", "here", "x", "y", "z", "w"]])

        with pytest.raises(ValueError, match="invalid literal for int"):
            phr.run_initial_from_file("test.pqi", ic)


class TestPhreeqcRMSelectedOutput:
    """Test selected output functionality."""

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_get_selected_output_df_basic(self, mock_cdll):
        """Test get_selected_output_df method basic functionality."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_GetSelectedOutputColumnCount.return_value = 2
        mock_lib.RM_GetSelectedOutputHeading.return_value = 0
        mock_lib.RM_GetSelectedOutput.return_value = 0
        mock_cdll.return_value = mock_lib

        # Mock the string buffer creation for headings
        def mock_get_heading(id, col, buffer, length):
            buffer.value = f"col_{col}".encode()
            return 0

        mock_lib.RM_GetSelectedOutputHeading.side_effect = mock_get_heading

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=2)

        with patch("mibiremo.phreeqc.ctypes.create_string_buffer") as mock_buffer:
            mock_str_buffer = MagicMock()
            mock_str_buffer.value.decode.return_value = "test_heading"
            mock_buffer.return_value = mock_str_buffer

            df = phr.get_selected_output_df()

            assert isinstance(df, pd.DataFrame)
            # Verify the method was called
            mock_lib.RM_GetSelectedOutputColumnCount.assert_called_once()
            mock_lib.RM_GetSelectedOutput.assert_called_once()

    def test_selected_output_column_count(self):
        """Test RM_GetSelectedOutputColumnCount method directly."""
        with patch("mibiremo.phreeqc.ctypes.CDLL") as mock_cdll:
            mock_lib = MagicMock()
            mock_lib.RM_Create.return_value = 1
            mock_lib.RM_GetSelectedOutputColumnCount.return_value = 5
            mock_cdll.return_value = mock_lib

            phr = mibiremo.PhreeqcRM()
            phr.create()

            result = phr.rm_get_selected_output_column_count()
            assert result == 5


class TestPhreeqcRMGetterMethods:
    """Test various getter methods."""

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_basic_getters(self, mock_cdll):
        """Test basic getter methods that return simple values."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_GetGridCellCount.return_value = 100
        mock_lib.RM_GetChemistryCellCount.return_value = 100
        mock_lib.RM_GetComponentCount.return_value = 5
        mock_lib.RM_GetSpeciesCount.return_value = 20
        mock_lib.RM_GetEquilibriumPhaseCount.return_value = 3
        mock_lib.RM_GetThreadCount.return_value = 4
        mock_lib.RM_GetSpeciesSaveOn.return_value = 1
        mock_lib.RM_GetSelectedOutputColumnCount.return_value = 10
        mock_lib.RM_GetSelectedOutputRowCount.return_value = 100
        mock_lib.RM_GetTime.return_value = 1000.0
        mock_lib.RM_GetTimeStep.return_value = 1.0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=100)

        assert phr.rm_get_grid_cell_count() == 100
        assert phr.rm_get_chemistry_cell_count() == 100
        assert phr.rm_get_component_count() == 5
        assert phr.rm_get_species_count() == 20
        assert phr.rm_get_equilibrium_phase_count() == 3
        assert phr.rm_get_thread_count() == 4
        assert phr.rm_get_species_save_on() == 1
        assert phr.rm_get_selected_output_column_count() == 10
        assert phr.rm_get_selected_output_row_count() == 100
        assert phr.rm_get_time() == 1000.0
        assert phr.rm_get_time_step() == 1.0

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_array_getters(self, mock_cdll):
        """Test getter methods that fill arrays."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_GetConcentrations.return_value = 0
        mock_lib.RM_GetSpeciesConcentrations.return_value = 0
        mock_lib.RM_GetDensity.return_value = 0
        mock_lib.RM_GetSaturation.return_value = 0
        mock_lib.RM_GetGfw.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=10)

        # Test concentration getter
        conc = np.zeros(50)  # 10 cells * 5 components
        result = phr.rm_get_concentrations(conc)
        assert result.code == 0

        # Test species concentration getter
        species_conc = np.zeros(200)  # 10 cells * 20 species
        result = phr.rm_get_species_concentrations(species_conc)
        assert result.code == 0

        # Test density getter
        density = np.zeros(10)
        result = phr.rm_get_density(density)
        assert result.code == 0

        # Test saturation getter
        saturation = np.zeros(10)
        result = phr.rm_get_saturation(saturation)
        assert result.code == 0

        # Test gram formula weights
        gfw = np.zeros(5)
        result = phr.rm_get_gfw(gfw)
        assert result.code == 0


class TestPhreeqcRMSetterMethods:
    """Test various setter methods."""

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_basic_setters(self, mock_cdll):
        """Test basic setter methods."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_SetComponentH2O.return_value = 0
        mock_lib.RM_SetSpeciesSaveOn.return_value = 0
        mock_lib.RM_SetUnitsSolution.return_value = 0
        mock_lib.RM_SetUnitsExchange.return_value = 0
        mock_lib.RM_SetUnitsSurface.return_value = 0
        mock_lib.RM_SetTime.return_value = 0
        mock_lib.RM_SetTimeStep.return_value = 0
        mock_lib.RM_SetRebalanceFraction.return_value = 0
        mock_lib.RM_SetFilePrefix.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        # Test various setter methods
        result = phr.rm_set_component_h2o(1)
        assert result.code == 0

        result = phr.rm_set_species_save_on(1)
        assert result.code == 0

        result = phr.rm_set_units_solution(2)
        assert result.code == 0

        result = phr.rm_set_units_exchange(1)
        assert result.code == 0

        result = phr.rm_set_units_surface(1)
        assert result.code == 0

        result = phr.rm_set_time(100.0)
        assert result.code == 0

        result = phr.rm_set_time_step(1.0)
        assert result.code == 0

        result = phr.rm_set_rebalance_fraction(0.5)
        assert result.code == 0

        result = phr.rm_set_file_prefix("test")
        assert result.code == 0

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_array_setters(self, mock_cdll):
        """Test setter methods that take arrays."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_SetConcentrations.return_value = 0
        mock_lib.RM_SetPorosity.return_value = 0
        mock_lib.RM_SetSaturation.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=5)

        # Test concentration setter
        conc = np.ones(25)  # 5 cells * 5 components
        result = phr.rm_set_concentrations(conc)
        assert result.code == 0

        # Test porosity setter
        porosity = np.full(5, 0.3)
        result = phr.rm_set_porosity(porosity)
        assert result.code == 0

        # Test saturation setter
        saturation = np.full(5, 0.8)
        result = phr.rm_set_saturation(saturation)
        assert result.code == 0


class TestPhreeqcRMRunMethods:
    """Test methods that run PHREEQC calculations."""

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_run_cells(self, mock_cdll):
        """Test RM_RunCells method."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_RunCells.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        result = phr.rm_run_cells()
        assert result.code == 0
        assert result
        mock_lib.RM_RunCells.assert_called_once_with(1)

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_run_file(self, mock_cdll):
        """Test RM_RunFile method."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_RunFile.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        result = phr.rm_run_file(1, 1, 0, "test.pqi")
        assert result.code == 0
        mock_lib.RM_RunFile.assert_called_once_with(1, 1, 1, 0, b"test.pqi")

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_run_string(self, mock_cdll):
        """Test RM_RunString method."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_RunString.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        input_string = "SOLUTION 1\npH 7\nEND"
        result = phr.rm_run_string(0, 1, 0, input_string)
        assert result.code == 0
        mock_lib.RM_RunString.assert_called_once_with(1, 0, 1, 0, input_string.encode())

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_load_database(self, mock_cdll):
        """Test RM_LoadDatabase method."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_LoadDatabase.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        result = phr.rm_load_database("phreeqc.dat")
        assert result.code == 0
        mock_lib.RM_LoadDatabase.assert_called_once_with(1, b"phreeqc.dat")

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_run_initial_ic_conversion_error(self, mock_cdll):
        """Test run_initial_from_file with IC conversion that fails."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_RunFile.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)

        # Create a numpy array with non-convertible data
        ic_array = np.array([["text", "data", "here", "x", "y", "z", "w"]], dtype=str)

        with pytest.raises(ValueError, match="invalid literal for int"):
            phr.run_initial_from_file("test.pqi", ic_array)


class TestPhreeqcRMFileOperations:
    """Test file operation methods."""

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_file_operations(self, mock_cdll):
        """Test file open/close operations."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_OpenFiles.return_value = 0
        mock_lib.RM_CloseFiles.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        # Test opening files
        result = phr.rm_open_files()
        assert result.code == 0
        mock_lib.RM_OpenFiles.assert_called_once_with(1)

        # Test closing files
        result = phr.rm_close_files()
        assert result.code == 0
        mock_lib.RM_CloseFiles.assert_called_once_with(1)


class TestPhreeqcRMCleanup:
    """Test cleanup and destruction methods."""

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_destroy(self, mock_cdll):
        """Test RM_Destroy method."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_Destroy.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        result = phr.rm_destroy()
        assert result.code == 0
        mock_lib.RM_Destroy.assert_called_once_with(1)


class TestPhreeqcRMAdditionalMethods:
    """Test additional methods."""

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_error_and_message_methods(self, mock_cdll):
        """Test error handling and message methods."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_Abort.return_value = 0
        mock_lib.RM_ErrorMessage.return_value = 0
        mock_lib.RM_DecodeError.return_value = "Test error message"
        mock_lib.RM_LogMessage.return_value = 0
        mock_lib.RM_ScreenMessage.return_value = 0
        mock_lib.RM_OutputMessage.return_value = 0
        mock_lib.RM_WarningMessage.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        # Test error methods
        result = phr.rm_abort(1, "Test abort")
        assert result.code == 0

        result = phr.rm_error_message("Test error")
        assert result.code == 0

        error_msg = phr.rm_decode_error(1)
        assert error_msg == "Test error message"

        # Test message methods (these return raw integers)
        result = phr.rm_log_message("Test log message")
        assert result == 0

        result = phr.rm_screen_message("Test screen message")
        assert result == 0

        result = phr.rm_output_message("Test output message")
        assert result == 0

        result = phr.rm_warning_message("Test warning")
        assert result == 0

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_utility_and_mapping_methods(self, mock_cdll):
        """Test utility and mapping methods."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_Concentrations2Utility.return_value = 0
        mock_lib.RM_CreateMapping.return_value = 0
        mock_lib.RM_GetBackwardMapping.return_value = 0
        mock_lib.RM_InitialPhreeqc2Concentrations.return_value = 0
        mock_lib.RM_InitialPhreeqc2SpeciesConcentrations.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=5)

        # Test utility methods
        conc = np.array([1.0, 2.0, 3.0])
        result = phr.rm_concentrations2utility(conc, 0, 25.0, 1.0)
        assert result.code == 0

        # Test mapping methods
        grid2chem = np.array([0, 1, 2, 3, 4])
        result = phr.rm_create_mapping(grid2chem)
        assert result.code == 0

        mapping_list = np.zeros(10, dtype=int)
        result = phr.rm_get_backward_mapping(0, mapping_list, 10)
        assert result.code == 0

        # Test initial phreeqc methods (these return raw integers)
        conc_array = np.zeros(15)  # 5 cells * 3 components
        boundary_sol1 = np.array([1, 2, 3])
        boundary_sol2 = np.array([4, 5, 6])
        fraction1 = np.array([0.5, 0.6, 0.7])

        result = phr.rm_initial_phreeqc2_concentrations(conc_array, 3, boundary_sol1, boundary_sol2, fraction1)
        assert result == 0  # Raw integer return

        species_conc = np.zeros(50)  # 5 cells * 10 species
        n_boundary = np.array([3])
        result = phr.rm_initial_phreeqc2_species_concentrations(
            species_conc, n_boundary, boundary_sol1, boundary_sol2, fraction1
        )
        assert result == 0  # Raw integer return

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_species_and_diffusion_methods(self, mock_cdll):
        """Test species information and diffusion methods."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_GetSpeciesD25.return_value = 0
        mock_lib.RM_GetSpeciesLog10Gammas.return_value = 0
        mock_lib.RM_GetSpeciesZ.return_value = 0
        mock_lib.RM_SpeciesConcentrations2Module.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=3)

        # Test species diffusion coefficients
        species_d25 = np.zeros(10)  # 10 species
        result = phr.rm_get_species_d25(species_d25)
        assert result.code == 0

        # Test species activity coefficients
        species_gammas = np.zeros(30)  # 3 cells * 10 species
        result = phr.rm_get_species_log10_gammas(species_gammas)
        assert result.code == 0

        # Test species charges
        species_z = np.zeros(10)
        result = phr.rm_get_species_z(species_z)
        assert result.code == 0

        # Test species concentrations to module transfer
        species_conc = np.ones(30)  # 3 cells * 10 species
        result = phr.rm_species_concentrations2_module(species_conc)
        assert result == 0  # This returns raw integer

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_additional_setter_methods(self, mock_cdll):
        """Test additional setter methods."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_SetDensity.return_value = 0
        mock_lib.RM_SetPressure.return_value = 0
        mock_lib.RM_SetTemperature.return_value = 0
        mock_lib.RM_SetTimeConversion.return_value = 0
        mock_lib.RM_UseSolutionDensityVolume.return_value = 0
        mock_lib.RM_SetCurrentSelectedOutputUserNumber.return_value = 0
        mock_lib.RM_SetScreenOn.return_value = 0
        mock_lib.RM_SetSelectedOutputOn.return_value = 0
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=5)

        # Test array setters (these return raw integers, not irm_result)
        density = np.full(5, 1.0)
        result = phr.rm_set_density(density)
        assert result == 0  # Direct return value check

        pressure = np.full(5, 1.0)
        result = phr.rm_set_pressure(pressure)
        assert result == 0

        temperature = np.full(5, 25.0)
        result = phr.rm_set_temperature(temperature)
        assert result == 0

        # Test scalar setters (these return raw integers, not irm_result)
        result = phr.rm_set_time_conversion(86400.0)  # seconds to days
        assert result == 0

        result = phr.rm_use_solution_density_volume(1)
        assert result == 0

        result = phr.rm_set_current_selected_output_user_number(1)
        assert result == 0

        result = phr.rm_set_screen_on(1)
        assert result == 0

        result = phr.rm_set_selected_output_on(1)
        assert result == 0

    @patch("mibiremo.phreeqc.ctypes.CDLL")
    def test_get_methods_simple(self, mock_cdll):
        """Test simple getter methods that return integers."""
        mock_lib = MagicMock()
        mock_lib.RM_Create.return_value = 1
        mock_lib.RM_GetExchangeSpeciesCount.return_value = 5
        mock_lib.RM_GetGasComponentsCount.return_value = 3
        mock_lib.RM_GetKineticReactionsCount.return_value = 2
        mock_lib.RM_GetMpiMyself.return_value = 0
        mock_lib.RM_GetMpiTasks.return_value = 1
        mock_lib.RM_GetNthSelectedOutputUserNumber.return_value = 1
        mock_lib.RM_GetSelectedOutputCount.return_value = 1
        mock_lib.RM_GetSICount.return_value = 4
        mock_lib.RM_GetSolidSolutionComponentsCount.return_value = 2
        mock_lib.RM_GetTimeConversion.return_value = 86400.0
        mock_lib.RM_GetErrorStringLength.return_value = 50
        mock_cdll.return_value = mock_lib

        phr = mibiremo.PhreeqcRM()
        phr.create()

        # Test getter methods
        assert phr.rm_get_exchange_species_count() == 5
        assert phr.rm_get_gas_components_count() == 3
        assert phr.rm_get_kinetic_reactions_count() == 2
        assert phr.rm_get_mpi_myself() == 0
        assert phr.rm_get_mpi_tasks() == 1
        assert phr.rm_get_nth_selected_output_user_number(0) == 1
        assert phr.rm_get_selected_output_count() == 1
        assert phr.rm_get_si_count() == 4
        assert phr.rm_get_solid_solution_components_count() == 2
        assert phr.rm_get_time_conversion() == 86400.0
        assert phr.rm_get_error_string_length() == 50
