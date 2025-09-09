"""Tests for the mibiremo.irmresult module."""

import pytest
from mibiremo.irmresult import IRMStatus


class TestIRMStatus:
    """Test the IRMStatus named tuple and its methods."""

    def test_irmstatus_creation(self):
        """Test IRMStatus creation with all attributes."""
        status = IRMStatus(0, "IRM_OK", "Success")
        assert status.code == 0
        assert status.name == "IRM_OK"
        assert status.message == "Success"

    def test_bool_success(self):
        """Test __bool__ method for success case."""
        status = IRMStatus(0, "IRM_OK", "Success")
        assert bool(status) is True
        assert status  # Direct boolean evaluation

    def test_bool_failure(self):
        """Test __bool__ method for failure cases."""
        status = IRMStatus(-1, "IRM_OUTOFMEMORY", "Failure, Out of memory")
        assert bool(status) is False
        assert not status  # Direct boolean evaluation

        # Test with various failure codes
        for code in [-1, -2, -3, -4, -5, -6, -7, 1, 42]:
            status = IRMStatus(code, "ERROR", "Some error")
            assert bool(status) is False

    def test_int_conversion(self):
        """Test __int__ method for backwards compatibility."""
        status = IRMStatus(0, "IRM_OK", "Success")
        assert int(status) == 0

        status = IRMStatus(-5, "IRM_INVALIDCOL", "Failure, Invalid column")
        assert int(status) == -5

        status = IRMStatus(42, "CUSTOM", "Custom error")
        assert int(status) == 42

    def test_str_representation(self):
        """Test __str__ method for formatted output."""
        status = IRMStatus(0, "IRM_OK", "Success")
        assert str(status) == "IRM_OK: Success"

        status = IRMStatus(-1, "IRM_OUTOFMEMORY", "Failure, Out of memory")
        assert str(status) == "IRM_OUTOFMEMORY: Failure, Out of memory"

        status = IRMStatus(99, "UNKNOWN", "Unknown error")
        assert str(status) == "UNKNOWN: Unknown error"

    def test_is_success_property(self):
        """Test is_success property."""
        status = IRMStatus(0, "IRM_OK", "Success")
        assert status.is_success is True

        status = IRMStatus(-1, "IRM_FAIL", "Failure")
        assert status.is_success is False

        status = IRMStatus(1, "ERROR", "Some error")
        assert status.is_success is False

    def test_raise_for_status_success(self):
        """Test raise_for_status method with success status."""
        status = IRMStatus(0, "IRM_OK", "Success")
        # Should not raise any exception
        status.raise_for_status()
        status.raise_for_status("Operation context")

    def test_raise_for_status_failure(self):
        """Test raise_for_status method with failure status."""
        status = IRMStatus(-1, "IRM_OUTOFMEMORY", "Failure, Out of memory")

        # Test without context
        with pytest.raises(RuntimeError, match="IRM_OUTOFMEMORY: Failure, Out of memory"):
            status.raise_for_status()

    def test_raise_for_status_failure_with_context(self):
        """Test raise_for_status method with failure status and context."""
        status = IRMStatus(-3, "IRM_INVALIDARG", "Failure, Invalid argument")

        # Test with context
        with pytest.raises(RuntimeError, match="Loading database: IRM_INVALIDARG: Failure, Invalid argument"):
            status.raise_for_status("Loading database")

        # Test with empty context string
        with pytest.raises(RuntimeError, match="IRM_INVALIDARG: Failure, Invalid argument"):
            status.raise_for_status("")

    def test_named_tuple_behavior(self):
        """Test that IRMStatus behaves like a named tuple."""
        status = IRMStatus(0, "IRM_OK", "Success")

        # Test tuple unpacking
        code, name, message = status
        assert code == 0
        assert name == "IRM_OK"
        assert message == "Success"

        # Test indexing
        assert status[0] == 0
        assert status[1] == "IRM_OK"
        assert status[2] == "Success"

        # Test immutability
        with pytest.raises(AttributeError):
            status.code = 1
