"""Semi-Lagrangian solver for the 1D advection-dispersion equation on a uniform grid.

Author: Matteo Masi
Last revision: 07/10/2026

"""

import numpy as np
from numba import njit
from scipy.interpolate import PchipInterpolator


@njit(cache=True)
def _saulyev(c, diffusion_number, c_bound):
    """JIT-compiled Saul'yev alternating-direction sweeps for c (points, components)."""
    n = c.shape[0]
    a = diffusion_number
    left_right = c.copy()
    right_left = c.copy()
    for i in range(n):
        upstream = c_bound if i == 0 else left_right[i - 1]
        downstream = c[i + 1] if i < n - 1 else c[i]  # zero gradient at the outlet
        left_right[i] = (a * upstream + (1.0 - a) * c[i] + a * downstream) / (1.0 + a)
    for i in range(n - 1, -1, -1):
        downstream = left_right[n - 1] if i == n - 1 else right_left[i + 1]
        upstream = c_bound if i == 0 else c[i - 1]
        right_left[i] = (a * downstream + (1.0 - a) * c[i] + a * upstream) / (1.0 + a)
    return (left_right + right_left) / 2


class SemiLagSolver:
    """Semi-Lagrangian solver for 1D advection-dispersion transport equations.

    This class implements a semi-Lagrangian numerical scheme for solving the
    one-dimensional advection-dispersion equation on uniform grids. The solver
    uses operator splitting to handle advection and dispersion separately,
    providing accurate and stable solutions for transport problems. Several
    components can be transported at once (one column of the concentration
    array per component).

    The numerical approach consists of two sequential steps:
        1. **Advection**: Solved using the Method of Characteristics (MOC) with
           cubic spline interpolation (PCHIP - Piecewise Cubic Hermite Interpolating
           Polynomial) to maintain monotonicity and prevent oscillations.
        2. **Dispersion**: Solved using the Saul'yev alternating direction method,
           which provides unconditional stability for the dispersion equation.

    Boundary Conditions:
        - **Inlet**: Dirichlet-type condition with prescribed concentration value,
          located one grid spacing upstream of the first grid point (x[0] - Δx),
          for both the advection and the dispersion steps.
        - **Outlet (right boundary)**: Neumann-type condition (zero gradient) allowing
          natural outflow of transported species.

    Mathematical Formulation:
        The solver addresses the 1D advection-dispersion equation:

        ∂C/∂t + v∂C/∂x = D∂²C/∂x²

        where:
        - C(x,t): Concentration field
        - v: Average linear (pore-water) velocity (constant, v ≥ 0)
        - D: Longitudinal hydrodynamic dispersion coefficient (constant)

    Numerical Stability:
        - The cubic spline advection step is stable for any Courant number
        - The Saul'yev dispersion solver is unconditionally stable
        - Combined scheme maintains stability and accuracy for typical transport problems

    Applications:
        - Reactive transport modeling in porous media
        - Contaminant transport in groundwater systems
        - Chemical species transport in environmental flows
        - Coupling with geochemical reaction modules (e.g., PhreeqcRM)

    Attributes:
        x (numpy.ndarray): Spatial coordinate array (uniform spacing required).
        C (numpy.ndarray): Current concentration field at grid points, shape (points,)
            or (points, components).
        v (float): Average linear velocity in consistent units with spatial coordinates.
        d (float): Dispersion coefficient in consistent units (L²/T).
        dt (float): Time step for numerical integration in consistent time units.
        dx (float): Spatial grid spacing (automatically calculated from x).

    Note:
        The spatial grid must be uniformly spaced for the numerical scheme to
        work correctly. Non-uniform grids are not supported in this implementation.
    """

    def __init__(self, x: np.ndarray, c_init: np.ndarray, v: float, d: float, dt: float):
        """Initialize the Semi-Lagrangian solver with transport parameters.

        Sets up the numerical solver with spatial discretization, initial conditions,
        and transport parameters. Validates input consistency and calculates derived
        parameters needed for the numerical scheme.

        Args:
            x (numpy.ndarray): Spatial coordinate array defining the 1D computational
                domain. Must be uniformly spaced with at least 2 points. Units should
                be consistent with velocity and dispersion coefficient.
            c_init (numpy.ndarray): Initial concentration field at each grid point,
                shape (points,) for one component or (points, components) for several.
                Its length must match the spatial coordinate array. Units are user-defined
                but should be consistent throughout the simulation.
            v (float): Average linear velocity (non-negative, flow from left to right).
                Units must be consistent with spatial coordinates and time step
                (e.g., if x is in meters and dt in days, v should be in m/day).
            d (float): Dispersion coefficient (must be non-negative).
                Units must be L²/T where L and T are consistent with spatial
                coordinates and time step (e.g., m²/day).
            dt (float): Time step for numerical integration (must be positive).
                Units should be consistent with velocity and dispersion coefficient.

        Raises:
            ValueError: If the grid has fewer than 2 points or if the concentration
                array length doesn't match spatial coordinates.
            ValueError: If transport parameters are not physically reasonable
                (negative velocity or dispersion, zero or negative time step).

        Examples:
            >>> x = np.linspace(0, 5, 51)      # 5 m domain, 0.1 m spacing
            >>> C0 = np.exp(-x**2)             # Gaussian initial condition
            >>> solver = SemiLagSolver(x, C0, v=0.5, d=0.05, dt=0.01)
        """
        if len(x) < 2 or len(x) != len(c_init):
            raise ValueError(f"Give at least 2 grid points and one concentration at each ({len(x)}, {len(c_init)}).")
        if v < 0 or d < 0 or dt <= 0:
            raise ValueError(
                "The velocity and the dispersion coefficient must not be negative, the time step positive."
            )
        self.x = np.asarray(x, dtype=float)
        self.C = np.asarray(c_init, dtype=float)
        self.v = v
        self.d = d
        self.dt = dt
        self.dx = self.x[1] - self.x[0]

    def cubic_spline_advection(self, c_bound: float | np.ndarray) -> None:
        """Solve the advection step using cubic spline interpolation.

        Implements the Method of Characteristics (MOC) for the advection equation
        ∂C/∂t + v∂C/∂x = 0 using backward tracking of characteristic lines.
        Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) to maintain
        monotonicity and prevent numerical oscillations.

        The method works by:
            1. Computing departure points: xi = x - v*dt (backward tracking)
            2. Interpolating concentrations at departure points using cubic splines,
               with the inlet concentration as first data point at x[0] - Δx
            3. Applying the inlet concentration to departure points upstream of the inlet

        Args:
            c_bound (float or numpy.ndarray): Inlet concentration value applied at the
                inlet (x[0] - Δx) for any characteristic lines that originated from outside
                the computational domain, one value or one per component. Units should match
                the concentration field.

        Note:
            This method modifies self.C in-place. The cubic spline interpolation
            preserves monotonicity, making it suitable for concentration fields
            where spurious oscillations must be avoided.

        Numerical Properties:
            - Unconditionally stable (no CFL restriction)
            - Maintains monotonicity (no new extrema created)
            - Handles arbitrary Courant numbers (v*dt/dx)
            - Exact for linear concentration profiles
        """
        x = np.concatenate([[self.x[0] - self.dx], self.x])
        c = np.concatenate([np.broadcast_to(c_bound, (1, *self.C.shape[1:])), self.C])
        departure = np.maximum(self.x - self.v * self.dt, x[0])
        self.C = PchipInterpolator(x, c, axis=0)(departure)

    def saulyev_solver_alt(self, c_bound: float | np.ndarray) -> None:
        """Solve the dispersion step using the Saul'yev alternating direction method.

        Implements the Saul'yev scheme for the dispersion equation ∂C/∂t = D∂²C/∂x²
        using alternating direction sweeps to achieve unconditional stability.
        The method performs two passes:
            1. Left-to-right sweep using forward differences
            2. Right-to-left sweep using backward differences
            3. Final solution is the average of both sweeps

        Args:
            c_bound (float or numpy.ndarray): Inlet concentration value applied at the
                inlet (x[0] - Δx) during the dispersion solve, one value or one per
                component. This maintains consistency with the advection boundary condition.

        Algorithm Details:
            - **Left-to-Right Pass**: For each cell i, uses implicit treatment of
              left neighbor and explicit treatment of right neighbor
            - **Right-to-Left Pass**: For each cell i, uses implicit treatment of
              right neighbor and explicit treatment of left neighbor
            - **Averaging**: Combines both solutions to achieve second-order accuracy

        Boundary Conditions:
            - **Inlet (x[0] - Δx)**: Dirichlet condition with prescribed c_bound, in
              both sweeps
            - **Right boundary**: Zero gradient (Neumann) condition implemented
              by using the same concentration as the last interior point

        Numerical Properties:
            - Unconditionally stable for any time step size
            - Second-order accurate in space and time
            - Preserves the maximum principle (no spurious extrema) for diffusion
              numbers D*dt/dx² ≤ 1
            - Handles arbitrary diffusion numbers (D*dt/dx²)

        Note:
            This method modifies self.C in-place. The alternating direction
            approach eliminates the restrictive stability constraint of explicit
            methods while maintaining computational efficiency.
        """
        c = self.C.reshape(len(self.x), -1)
        bound = np.broadcast_to(np.asarray(c_bound, dtype=float), c.shape[1:]).copy()
        self.C = _saulyev(c, self.d * self.dt / self.dx**2, bound).reshape(self.C.shape)

    def transport(self, c_bound: float | np.ndarray) -> np.ndarray:
        """Perform one complete transport time step with coupled advection-dispersion.

        Executes the full semi-Lagrangian algorithm by sequentially applying
        the advection and dispersion operators using operator splitting. This
        approach decouples the hyperbolic (advection) and parabolic (dispersion)
        aspects of the transport equation for enhanced numerical stability.

        The operator splitting sequence:
            1. **Advection Step** using cubic spline MOC
            2. **Dispersion Step** using Saul'yev method

        Args:
            c_bound (float or numpy.ndarray): Inlet boundary concentration applied at
                x[0] - Δx for both advection and dispersion steps, one value or one per
                component. This represents the concentration of material entering the
                domain (e.g., injection well concentration, column influent, upstream
                boundary condition, etc.).

        Returns:
            numpy.ndarray: Updated concentration field after the complete transport
                step. The array has the same shape as the initial concentration
                and represents C(x, t+dt).

        Note:
            This method updates the internal concentration field (self.C) and
            returns the updated values. For reactive transport coupling, call
            this method to advance transport, then apply geochemical reactions
            to the returned concentration field.
        """
        # Step 1: Solve advection equation using cubic spline MOC
        self.cubic_spline_advection(c_bound)

        # Step 2: Solve diffusion equation using Saul'yev alternating direction method
        self.saulyev_solver_alt(c_bound)

        return self.C
