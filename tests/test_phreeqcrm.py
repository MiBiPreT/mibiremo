"""Tests for the mibiremo.phreeqcrm module."""

import pytest
import mibiremo


def test_phreeqcrm():
    """Example using assert."""
    assert mibiremo.PhreeqcRM()


def test_hello_with_error():
    """Example of testing for raised errors."""
    phr = mibiremo.PhreeqcRM()
    phr.create(nxyz=10)
    assert phr.nxyz == 10


@pytest.fixture
def some_nr_cells():
    """Example fixture."""
    return 1000


def test_hello_with_fixture(some_nr_cells: str):
    """Example using a fixture."""
    phr = mibiremo.PhreeqcRM()
    phr.create(nxyz=some_nr_cells)
    assert phr.nxyz == 1000
