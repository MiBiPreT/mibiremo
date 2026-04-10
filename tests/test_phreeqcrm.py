"""Tests for the mibiremo.phreeqcrm module.
The phreeqcrm.PhreeqcRM class is mocked to allow testing without the actual library.
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
        assert phr.nxyz == 1
        assert phr.n_threads == 1
        assert phr.rm is None
        assert phr.components is None
        assert phr.species is None

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_create_default_params(self, mock_phreeqcrm_cls):
        """Test create method with default parameters."""
        mock_rm = MagicMock()
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create()

        assert phr._initialized
        assert phr.nxyz == 1
        assert phr.n_threads == 1
        assert phr.rm is mock_rm
        mock_phreeqcrm_cls.assert_called_once_with(1, 1)

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_create_custom_params(self, mock_phreeqcrm_cls):
        """Test create method with custom parameters."""
        mock_rm = MagicMock()
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=100, n_threads=4)

        assert phr._initialized
        assert phr.nxyz == 100
        assert phr.n_threads == 4
        assert phr.rm is mock_rm
        mock_phreeqcrm_cls.assert_called_once_with(100, 4)

    @patch("mibiremo.phreeqc.os.cpu_count")
    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_create_auto_threads(self, mock_phreeqcrm_cls, mock_cpu_count):
        """Test automatic detection of CPU threads."""
        mock_cpu_count.return_value = 8
        mock_rm = MagicMock()
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=50, n_threads=-1)

        assert phr.n_threads == 8
        mock_phreeqcrm_cls.assert_called_once_with(50, 8)

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_create_failure(self, mock_phreeqcrm_cls):
        """Test failure in PhreeqcRM instance creation."""
        mock_phreeqcrm_cls.side_effect = Exception("Creation failed")

        phr = mibiremo.PhreeqcRM()
        with pytest.raises(RuntimeError, match="Failed to create PhreeqcRM instance"):
            phr.create()


class TestPhreeqcRMInitializePhreeqc:
    """Test PhreeqcRM initialization with database and parameters."""

    def test_initialize_phreeqc_not_initialized(self):
        """Test initialize_phreeqc before create() is called."""
        phr = mibiremo.PhreeqcRM()
        with pytest.raises(RuntimeError, match="PhreeqcRM instance not initialized"):
            phr.initialize_phreeqc("test.dat")

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_initialize_phreeqc_success(self, mock_phreeqcrm_cls):
        """Test successful initialize_phreeqc call."""
        mock_rm = MagicMock()
        mock_rm.LoadDatabase.return_value = 0
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=10)
        phr.initialize_phreeqc("phreeqc.dat")

        mock_rm.LoadDatabase.assert_called_once_with("phreeqc.dat")
        mock_rm.SetComponentH2O.assert_called_once_with(False)
        mock_rm.SetRebalanceFraction.assert_called_once_with(0.5)
        mock_rm.SetUnitsSolution.assert_called_once_with(2)
        mock_rm.SetFilePrefix.assert_called_once_with("phr")
        mock_rm.OpenFiles.assert_called_once()
        mock_rm.SetSpeciesSaveOn.assert_called_once_with(True)

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_initialize_phreeqc_database_failure(self, mock_phreeqcrm_cls):
        """Test initialize_phreeqc with database loading failure."""
        mock_rm = MagicMock()
        mock_rm.LoadDatabase.return_value = -1
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=10)

        with pytest.raises(RuntimeError, match="Failed to load Phreeqc database"):
            phr.initialize_phreeqc("invalid.dat")

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_initialize_phreeqc_custom_params(self, mock_phreeqcrm_cls):
        """Test initialize_phreeqc with custom parameters."""
        mock_rm = MagicMock()
        mock_rm.LoadDatabase.return_value = 0
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=5)
        phr.initialize_phreeqc(
            "custom.dat", units_solution=1, units=2, porosity=0.3, saturation=0.8, multicomponent=False
        )

        mock_rm.SetUnitsSolution.assert_called_with(1)
        mock_rm.SetUnitsPPassemblage.assert_called_with(2)
        mock_rm.SetSpeciesSaveOn.assert_not_called()


class TestPhreeqcRMRunInitialFromFile:
    """Test running initial conditions from file."""

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_run_initial_from_file_success(self, mock_phreeqcrm_cls):
        """Test successful run_initial_from_file."""
        mock_rm = MagicMock()
        mock_rm.RunFile.return_value = 0
        mock_rm.GetComponents.return_value = ["H", "O", "Charge"]
        mock_rm.GetSpeciesNames.return_value = ["H+", "OH-", "H2O"]
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=2)

        ic = np.array([[1, -1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1]])
        phr.run_initial_from_file("test.pqi", ic)

        mock_rm.RunFile.assert_called_once_with(True, True, True, "test.pqi")
        mock_rm.InitialPhreeqc2Module.assert_called_once()
        mock_rm.FindComponents.assert_called_once()
        assert phr.components is not None
        assert phr.species is not None
        np.testing.assert_array_equal(phr.components, ["H", "O", "Charge"])
        np.testing.assert_array_equal(phr.species, ["H+", "OH-", "H2O"])

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_run_initial_from_file_failure(self, mock_phreeqcrm_cls):
        """Test run_initial_from_file with file failure."""
        mock_rm = MagicMock()
        mock_rm.RunFile.return_value = -1
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)

        ic = np.array([[1, -1, -1, -1, -1, -1, -1]])

        with pytest.raises(RuntimeError, match="Failed to run Phreeqc input file"):
            phr.run_initial_from_file("invalid.pqi", ic)

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_run_initial_from_file_wrong_ic_shape(self, mock_phreeqcrm_cls):
        """Test run_initial_from_file with incorrect IC array shape."""
        mock_rm = MagicMock()
        mock_rm.RunFile.return_value = 0
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)

        ic = np.array([[1, -1, -1, -1]])

        with pytest.raises(ValueError, match="Initial conditions array must have shape"):
            phr.run_initial_from_file("test.pqi", ic)

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_run_initial_from_file_invalid_ic_type(self, mock_phreeqcrm_cls):
        """Test run_initial_from_file with non-convertible IC data."""
        mock_rm = MagicMock()
        mock_rm.RunFile.return_value = 0
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)

        ic = np.array([["invalid", "data", "here", "x", "y", "z", "w"]])

        with pytest.raises(ValueError, match="invalid literal for int"):
            phr.run_initial_from_file("test.pqi", ic)

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_run_initial_from_file_non_ndarray_success(self, mock_phreeqcrm_cls):
        """Test run_initial_from_file with non-ndarray IC that converts to int."""
        mock_rm = MagicMock()
        mock_rm.RunFile.return_value = 0
        mock_rm.GetComponents.return_value = ["H"]
        mock_rm.GetSpeciesNames.return_value = ["H+"]
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)

        ic = pd.DataFrame([[1, -1, -1, -1, -1, -1, -1]])
        phr.run_initial_from_file("test.pqi", ic)

        mock_rm.InitialPhreeqc2Module.assert_called_once()

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_run_initial_from_file_non_ndarray_failure(self, mock_phreeqcrm_cls):
        """Test run_initial_from_file with non-ndarray IC that cannot convert."""
        mock_rm = MagicMock()
        mock_rm.RunFile.return_value = 0
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)

        ic = pd.DataFrame([["a", "b", "c", "d", "e", "f", "g"]])

        with pytest.raises(ValueError, match="Initial conditions must be convertible to a numpy array"):
            phr.run_initial_from_file("test.pqi", ic)

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_run_initial_from_file_ic_flattened_fortran_order(self, mock_phreeqcrm_cls):
        """Test that IC array is flattened in Fortran order."""
        mock_rm = MagicMock()
        mock_rm.RunFile.return_value = 0
        mock_rm.GetComponents.return_value = ["H"]
        mock_rm.GetSpeciesNames.return_value = ["H+"]
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=2)

        ic = np.array([[1, -1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1]])
        phr.run_initial_from_file("test.pqi", ic)

        call_args = mock_rm.InitialPhreeqc2Module.call_args[0][0]
        expected = ic.flatten("F").astype(np.int32)
        np.testing.assert_array_equal(call_args, expected)


class TestPhreeqcRMSelectedOutput:
    """Test selected output functionality."""

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_get_selected_output_df_basic(self, mock_phreeqcrm_cls):
        """Test get_selected_output_df method basic functionality."""
        mock_rm = MagicMock()
        mock_rm.GetSelectedOutputHeadings.return_value = ["pH", "pe"]
        mock_rm.GetSelectedOutput.return_value = [7.0, 4.0, 8.0, 3.0]
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=2)

        df = phr.get_selected_output_df()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["pH", "pe"]
        assert df.shape == (2, 2)
        mock_rm.GetSelectedOutputHeadings.assert_called_once()
        mock_rm.GetSelectedOutput.assert_called_once()

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_get_selected_output_df_values(self, mock_phreeqcrm_cls):
        """Test get_selected_output_df returns correct values."""
        mock_rm = MagicMock()
        mock_rm.GetSelectedOutputHeadings.return_value = ["pH", "pe"]
        # Data layout: ncolsel values per column, nxyz rows
        # Column-major: [pH_cell0, pH_cell1, pe_cell0, pe_cell1]
        mock_rm.GetSelectedOutput.return_value = [7.0, 8.0, 4.0, 3.0]
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=2)

        df = phr.get_selected_output_df()

        assert df["pH"].iloc[0] == 7.0
        assert df["pH"].iloc[1] == 8.0
        assert df["pe"].iloc[0] == 4.0
        assert df["pe"].iloc[1] == 3.0


class TestPhreeqcRMPublicAttribute:
    """Test that self.rm is accessible as a public attribute."""

    @patch("mibiremo.phreeqc.phreeqcrm.PhreeqcRM")
    def test_rm_attribute_accessible(self, mock_phreeqcrm_cls):
        """Test that self.rm is the underlying PhreeqcRM instance."""
        mock_rm = MagicMock()
        mock_phreeqcrm_cls.return_value = mock_rm

        phr = mibiremo.PhreeqcRM()
        phr.create(nxyz=1)

        assert phr.rm is mock_rm
        # Users can call methods directly on phr.rm
        phr.rm.SetTime(100.0)
        mock_rm.SetTime.assert_called_once_with(100.0)
