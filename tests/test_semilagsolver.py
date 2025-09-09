"""Tests for the mibiremo.semilagsolver module."""

import pytest
import numpy as np
from mibiremo.semilagsolver import SemiLagSolver


class TestSemiLagSolverInitialization:
    """Test SemiLagSolver initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic initialization with valid parameters."""
        x = np.linspace(0, 5, 51)
        C_init = np.exp(-(x**2))  # Gaussian initial condition
        v = 0.5
        D = 0.05
        dt = 0.01

        solver = SemiLagSolver(x, C_init, v, D, dt)

        assert np.array_equal(solver.x, x)
        assert np.array_equal(solver.C, C_init)
        assert solver.v == v
        assert solver.D == D
        assert solver.dt == dt
        assert solver.dx == pytest.approx(0.1, rel=1e-10)

    def test_initialization_with_different_grid_sizes(self):
        """Test initialization with different grid sizes."""
        # Small grid
        x_small = np.linspace(0, 1, 11)
        C_small = np.ones_like(x_small)
        solver_small = SemiLagSolver(x_small, C_small, 0.1, 0.01, 0.001)
        assert solver_small.dx == pytest.approx(0.1, rel=1e-10)

        # Large grid
        x_large = np.linspace(0, 100, 1001)
        C_large = np.zeros_like(x_large)
        solver_large = SemiLagSolver(x_large, C_large, 1.0, 0.1, 0.01)
        assert solver_large.dx == pytest.approx(0.1, rel=1e-10)

    def test_initialization_zero_diffusion(self):
        """Test initialization with zero diffusion coefficient."""
        x = np.linspace(0, 5, 51)
        C_init = np.ones_like(x)
        solver = SemiLagSolver(x, C_init, 0.5, 0.0, 0.01)
        assert solver.D == 0.0

    def test_initialization_negative_velocity(self):
        """Test initialization with negative velocity (reverse flow)."""
        x = np.linspace(0, 5, 51)
        C_init = np.ones_like(x)
        solver = SemiLagSolver(x, C_init, -0.5, 0.05, 0.01)
        assert solver.v == -0.5

    def test_mismatched_array_lengths(self):
        """Test that ValueError is raised when x and C_init have different lengths."""
        x = np.linspace(0, 5, 51)
        C_init = np.ones(50)  # Different length

        with pytest.raises(ValueError, match="Length of x \\(51\\) must match length of C_init \\(50\\)"):
            SemiLagSolver(x, C_init, 0.5, 0.05, 0.01)

    def test_single_point_grid(self):
        """Test that ValueError is raised with single point grid."""
        x = np.array([0.0])
        C_init = np.array([1.0])

        with pytest.raises(ValueError, match="Grid must have at least 2 points, got 1"):
            SemiLagSolver(x, C_init, 0.5, 0.05, 0.01)


class TestCubicSplineAdvection:
    """Test the cubic spline advection method."""

    def test_advection_no_movement(self):
        """Test advection with zero velocity (minimal movement expected)."""
        x = np.linspace(0, 5, 51)
        C_init = np.exp(-((x - 2.5) ** 2))  # Gaussian centered at x=2.5
        solver = SemiLagSolver(x, C_init, 0.0, 0.0, 0.01)

        C_original = solver.C.copy()
        # Use boundary condition equal to first point to minimize changes
        solver.cubic_spline_advection(C_original[0])

        # With zero velocity, changes should be minimal (interpolation effects only)
        # The first point might change due to boundary condition logic
        np.testing.assert_array_almost_equal(solver.C[1:], C_original[1:], decimal=5)

    def test_advection_positive_velocity(self):
        """Test advection with positive velocity (left to right movement)."""
        x = np.linspace(0, 10, 101)
        # Create a sharp peak at x=2
        C_init = np.exp(-10 * (x - 2) ** 2)
        solver = SemiLagSolver(x, C_init, 1.0, 0.0, 0.1)  # v*dt = 0.1

        C_before = solver.C.copy()
        solver.cubic_spline_advection(0.0)

        # Peak should have moved to the right
        peak_before = np.argmax(C_before)
        peak_after = np.argmax(solver.C)
        assert peak_after >= peak_before

    def test_advection_negative_velocity(self):
        """Test advection with negative velocity (right to left movement)."""
        x = np.linspace(0, 10, 101)
        # Create a sharp peak at x=8
        C_init = np.exp(-10 * (x - 8) ** 2)
        solver = SemiLagSolver(x, C_init, -1.0, 0.0, 0.1)  # v*dt = -0.1

        C_before = solver.C.copy()
        solver.cubic_spline_advection(0.0)

        # Peak should have moved to the left
        peak_before = np.argmax(C_before)
        peak_after = np.argmax(solver.C)
        assert peak_after <= peak_before

    def test_advection_boundary_condition(self):
        """Test advection with inlet boundary condition."""
        x = np.linspace(0, 5, 51)
        C_init = np.zeros_like(x)
        solver = SemiLagSolver(x, C_init, 1.0, 0.0, 0.5)  # Large time step

        C_bound = 2.0
        solver.cubic_spline_advection(C_bound)

        # Left boundary should have the prescribed concentration
        assert solver.C[0] == pytest.approx(C_bound)
        # Some concentration should have entered the domain
        assert np.any(solver.C > 0)

    def test_advection_high_courant_number(self):
        """Test advection with high Courant number (v*dt/dx > 1)."""
        x = np.linspace(0, 5, 51)
        C_init = np.exp(-((x - 1) ** 2))
        # High Courant number: v*dt/dx = 2*0.5/0.1 = 10
        solver = SemiLagSolver(x, C_init, 2.0, 0.0, 0.5)

        solver.cubic_spline_advection(0.0)

        # Method should be stable even with high Courant number
        assert np.all(np.isfinite(solver.C))
        assert not np.any(np.isnan(solver.C))

    def test_advection_conservation_no_boundary_input(self):
        """Test mass conservation in advection with no boundary input."""
        x = np.linspace(0, 10, 101)
        C_init = np.exp(-((x - 5) ** 2))
        solver = SemiLagSolver(x, C_init, 0.5, 0.0, 0.1)

        mass_before = np.trapezoid(solver.C, x)
        solver.cubic_spline_advection(0.0)  # No inlet concentration
        mass_after = np.trapezoid(solver.C, x)

        # Mass should be approximately conserved (some loss at boundaries is expected)
        assert mass_after <= mass_before


class TestSaulyevSolver:
    """Test the Saul'yev diffusion solver method."""

    def test_diffusion_no_diffusion_coefficient(self):
        """Test diffusion with zero diffusion coefficient."""
        x = np.linspace(0, 5, 51)
        C_init = np.exp(-((x - 2.5) ** 2))
        solver = SemiLagSolver(x, C_init, 0.0, 0.0, 0.01)

        C_original = solver.C.copy()
        solver.saulyev_solver_alt(1.0)

        # With zero diffusion, concentration should remain unchanged
        np.testing.assert_array_almost_equal(solver.C, C_original)

    def test_diffusion_smoothing_effect(self):
        """Test that diffusion smooths concentration profiles."""
        x = np.linspace(0, 10, 101)
        # Create a sharp step function
        C_init = np.where(x < 5, 1.0, 0.0)
        solver = SemiLagSolver(x, C_init, 0.0, 0.1, 0.01)

        # Calculate initial gradient magnitude
        grad_initial = np.max(np.abs(np.diff(solver.C)))

        solver.saulyev_solver_alt(1.0)

        # Calculate final gradient magnitude
        grad_final = np.max(np.abs(np.diff(solver.C)))

        # Diffusion should reduce the maximum gradient
        assert grad_final < grad_initial

    def test_diffusion_boundary_conditions(self):
        """Test diffusion boundary conditions."""
        x = np.linspace(0, 5, 51)
        C_init = np.zeros_like(x)
        solver = SemiLagSolver(x, C_init, 0.0, 0.1, 0.01)

        C_bound = 2.0
        solver.saulyev_solver_alt(C_bound)

        # Left boundary should have concentration influenced by boundary condition
        assert solver.C[0] > 0  # Should be positive due to boundary influence

    def test_diffusion_stability(self):
        """Test diffusion solver stability with large time steps."""
        x = np.linspace(0, 5, 51)
        C_init = np.random.random(len(x))
        # Large diffusion number: D*dt/dx² >> 1
        solver = SemiLagSolver(x, C_init, 0.0, 1.0, 1.0)

        solver.saulyev_solver_alt(0.0)

        # Solution should remain finite and stable
        assert np.all(np.isfinite(solver.C))
        assert not np.any(np.isnan(solver.C))

    def test_diffusion_maximum_principle(self):
        """Test that diffusion preserves maximum principle."""
        x = np.linspace(0, 5, 51)
        C_init = np.exp(-((x - 2.5) ** 2))
        max_initial = np.max(C_init)
        min_initial = np.min(C_init)

        solver = SemiLagSolver(x, C_init, 0.0, 0.1, 0.01)
        solver.saulyev_solver_alt(0.0)  # No boundary input

        # No new extrema should be created (within numerical tolerance)
        assert np.max(solver.C) <= max_initial + 1e-10
        assert np.min(solver.C) >= min_initial - 1e-10


class TestTransportMethod:
    """Test the complete transport method (advection + diffusion)."""

    def test_transport_complete_step(self):
        """Test complete transport step with both advection and diffusion."""
        x = np.linspace(0, 10, 101)
        C_init = np.exp(-((x - 2) ** 2))
        solver = SemiLagSolver(x, C_init, 0.5, 0.05, 0.1)

        C_before = solver.C.copy()
        C_result = solver.transport(1.0)

        # Result should be different from initial condition
        assert not np.allclose(C_result, C_before)
        # Result should be the same as solver.C
        np.testing.assert_array_equal(C_result, solver.C)
        # All values should be finite
        assert np.all(np.isfinite(C_result))

    def test_transport_multiple_steps(self):
        """Test multiple transport steps."""
        x = np.linspace(0, 10, 101)
        C_init = np.exp(-((x - 5) ** 2))
        solver = SemiLagSolver(x, C_init, 0.2, 0.02, 0.05)

        # Perform multiple transport steps
        concentrations = [solver.C.copy()]
        for _ in range(10):
            solver.transport(0.0)
            concentrations.append(solver.C.copy())

        # Each step should produce different results
        for i in range(1, len(concentrations)):
            assert not np.allclose(concentrations[i], concentrations[i - 1])

    def test_transport_pure_advection(self):
        """Test transport with pure advection (D=0)."""
        x = np.linspace(0, 10, 101)
        C_init = np.exp(-10 * (x - 2) ** 2)
        solver = SemiLagSolver(x, C_init, 1.0, 0.0, 0.1)

        peak_before = np.argmax(solver.C)
        solver.transport(0.0)
        peak_after = np.argmax(solver.C)

        # Peak should move to the right for positive velocity
        assert peak_after >= peak_before

    def test_transport_pure_diffusion(self):
        """Test transport with pure diffusion (v=0)."""
        x = np.linspace(0, 10, 101)
        # Sharp initial condition
        C_init = np.where(np.abs(x - 5) < 0.1, 1.0, 0.0)
        solver = SemiLagSolver(x, C_init, 0.0, 0.1, 0.01)

        width_before = np.sum(solver.C > 0.1)
        solver.transport(0.0)
        width_after = np.sum(solver.C > 0.1)

        # Diffusion should spread the concentration
        assert width_after >= width_before

    def test_transport_with_injection(self):
        """Test transport with continuous injection at boundary."""
        x = np.linspace(0, 5, 51)
        C_init = np.zeros_like(x)
        solver = SemiLagSolver(x, C_init, 0.5, 0.01, 0.01)

        C_bound = 1.0
        # Multiple steps with injection
        for _ in range(50):
            solver.transport(C_bound)

        # Concentration should have spread into the domain
        # Boundary condition is approximate due to numerical scheme
        assert solver.C[0] == pytest.approx(C_bound, rel=0.01)  # Allow 1% relative error
        assert np.any(solver.C[1:] > 0)

    def test_transport_return_value(self):
        """Test that transport method returns updated concentration."""
        x = np.linspace(0, 5, 51)
        C_init = np.ones_like(x)
        solver = SemiLagSolver(x, C_init, 0.1, 0.01, 0.01)

        C_result = solver.transport(0.5)

        # Returned array should be the same object as solver.C
        assert C_result is solver.C
        # Should be a numpy array
        assert isinstance(C_result, np.ndarray)
