"""Tests for the mibiremo.semilagsolver module, against analytical solutions of the transport equation."""

import numpy as np
import pytest
from scipy.special import erfc
from mibiremo.semilagsolver import SemiLagSolver


def gaussian(x, centre, width):
    return np.exp(-((x - centre) ** 2) / (2 * width**2))


def test_advection_translates_profile():
    """Pure advection shifts a smooth profile by v t at Courant number 2.5, conserving mass without new extrema."""
    x = np.linspace(0.0, 20.0, 201)  # [m], Δx = 0.1 m
    v, dt, n_steps = 0.5, 0.5, 20  # v [m s⁻¹], Δt [s]: v Δt / Δx = 2.5, shift v t = 5 m
    solver = SemiLagSolver(x, gaussian(x, 4.0, 0.8), v, 0.0, dt)
    for _ in range(n_steps):
        solver.transport(0.0)
    exact = gaussian(x, 4.0 + v * dt * n_steps, 0.8)
    assert solver.C == pytest.approx(exact, abs=0.02)
    assert solver.C.min() >= 0.0
    assert solver.C.max() <= 1.0
    assert np.trapezoid(solver.C, x) == pytest.approx(np.trapezoid(exact, x), rel=1e-4)


def test_dispersion_variance():
    """Pure dispersion conserves mass and the variance of a Gaussian profile grows as 2 D t."""
    x = np.linspace(0.0, 20.0, 201)  # [m]
    d, dt, n_steps, width = 0.05, 0.1, 100, 0.8  # D [m² s⁻¹], Δt [s], initial standard deviation [m]
    initial = gaussian(x, 10.0, width)
    solver = SemiLagSolver(x, initial, 0.0, d, dt)
    for _ in range(n_steps):
        solver.transport(0.0)
    mass = np.trapezoid(solver.C, x)
    variance = np.trapezoid(solver.C * (x - 10.0) ** 2, x) / mass
    assert mass == pytest.approx(np.trapezoid(initial, x), rel=1e-5)
    assert variance == pytest.approx(width**2 + 2 * d * dt * n_steps, rel=1e-3)


def test_ogata_banks():
    """A constant inlet concentration gives the Ogata-Banks solution (cell Péclet number 1, Courant number 0.5)."""
    v, d, dx, dt, t = 0.5, 0.05, 0.1, 0.1, 10.0  # v [m s⁻¹], D [m² s⁻¹], Δx [m], Δt [s], t [s]
    x = np.arange(0.0, 12.0 + dx / 2, dx)
    solver = SemiLagSolver(x, np.zeros_like(x), v, d, dt)
    for _ in range(round(t / dt)):
        solver.transport(1.0)
    spread = 2 * np.sqrt(d * t)
    exact = 0.5 * (erfc((x - v * t) / spread) + np.exp(v * x / d) * erfc((x + v * t) / spread))
    assert solver.C == pytest.approx(exact, abs=0.03)  # numerical dispersion of the scheme


@pytest.mark.parametrize(
    ("x", "c_init"),
    [(np.linspace(0.0, 5.0, 51), np.ones(50)), (np.array([0.0]), np.array([1.0]))],
)
def test_invalid_grid(x, c_init):
    """The grid needs at least 2 points, with one concentration at each."""
    with pytest.raises(ValueError):
        SemiLagSolver(x, c_init, 0.5, 0.05, 0.01)
