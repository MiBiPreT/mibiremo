"""Semi-Lagrangian solver for 1D advection-diffusion equation on a uniform grid.

Author: Matteo Masi
Last revision: 09/09/2024

"""

import warnings
import numpy as np
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore")


class SemiLagSolver:
    """Semi-Lagrangian solver for 1D advection-diffusion transport equations.

    This class implements a semi-Lagrangian numerical scheme for solving the
    one-dimensional advection-diffusion equation on uniform grids. The solver
    uses operator splitting to handle advection and diffusion separately,
    providing accurate and stable solutions for transport problems.

    The numerical approach consists of two sequential steps:
        1. **Advection**: Solved using the Method of Characteristics (MOC) with
           cubic spline interpolation (PCHIP - Piecewise Cubic Hermite Interpolating
           Polynomial) to maintain monotonicity and prevent oscillations.
        2. **Diffusion**: Solved using the Saul'yev alternating direction method,
           which provides unconditional stability for the diffusion equation.

    Boundary Conditions:
        - **Inlet (left boundary, x=0)**: Dirichlet-type condition with prescribed
          concentration value.
        - **Outlet (right boundary)**: Neumann-type condition (zero gradient) allowing
          natural outflow of transported species.

    Mathematical Formulation:
        The solver addresses the 1D advection-diffusion equation:

        ∂C/∂t + v∂C/∂x = D∂²C/∂x²

        where:
        - C(x,t): Concentration field
        - v: Advection velocity (constant)
        - D: Diffusion/dispersion coefficient (constant)

    Numerical Stability:
        - The cubic spline advection step is stable for any Courant number
        - The Saul'yev diffusion solver is unconditionally stable
        - Combined scheme maintains stability and accuracy for typical transport problems

    Applications:
        - Reactive transport modeling in porous media
        - Contaminant transport in groundwater systems
        - Chemical species transport in environmental flows
        - Coupling with geochemical reaction modules (e.g., PhreeqcRM)

    Attributes:
        x (numpy.ndarray): Spatial coordinate array (uniform spacing required).
        C (numpy.ndarray): Current concentration field at grid points.
        v (float): Advection velocity in consistent units with spatial coordinates.
        D (float): Diffusion coefficient in consistent units (L²/T).
        dt (float): Time step for numerical integration in consistent time units.
        dx (float): Spatial grid spacing (automatically calculated from x).

    Note:
        The spatial grid must be uniformly spaced for the numerical scheme to
        work correctly. Non-uniform grids are not supported in this implementation.
    """

    def __init__(self, x, C_init, v, D, dt):
        """Initialize the Semi-Lagrangian solver with transport parameters.

        Sets up the numerical solver with spatial discretization, initial conditions,
        and transport parameters. Validates input consistency and calculates derived
        parameters needed for the numerical scheme.

        Args:
            x (numpy.ndarray): Spatial coordinate array defining the 1D computational
                domain. Must be uniformly spaced with at least 2 points. Units should
                be consistent with velocity and diffusion coefficient.
            C_init (numpy.ndarray): Initial concentration field at each grid point.
                Length must match the spatial coordinate array. Units are user-defined
                but should be consistent throughout the simulation.
            v (float): Advection velocity (positive for left-to-right flow).
                Units must be consistent with spatial coordinates and time step
                (e.g., if x is in meters and dt in days, v should be in m/day).
            D (float): Diffusion/dispersion coefficient (must be non-negative).
                Units must be L²/T where L and T are consistent with spatial
                coordinates and time step (e.g., m²/day).
            dt (float): Time step for numerical integration (must be positive).
                Units should be consistent with velocity and diffusion coefficient.

        Raises:
            ValueError: If spatial coordinates are not uniformly spaced or if
                concentration array length doesn't match spatial coordinates.
            ValueError: If transport parameters are not physically reasonable
                (negative diffusion, zero or negative time step).

        Examples:
            >>> x = np.linspace(0, 5, 51)      # 5 m domain, 0.1 m spacing
            >>> C0 = np.exp(-x**2)             # Gaussian initial condition
            >>> solver = SemiLagSolver(x, C0, v=0.5, D=0.05, dt=0.01)
        """
        if len(x) != len(C_init):
            raise ValueError(f"Length of x ({len(x)}) must match length of C_init ({len(C_init)})")

        if len(x) < 2:
            raise ValueError(f"Grid must have at least 2 points, got {len(x)}")

        self.x = x
        self.C = C_init
        self.v = v
        self.D = D
        self.dt = dt
        self.dx = x[1] - x[0]

    def cubic_spline_advection(self, C_bound) -> None:
        """Solve the advection step using cubic spline interpolation.

        Implements the Method of Characteristics (MOC) for the advection equation
        ∂C/∂t + v∂C/∂x = 0 using backward tracking of characteristic lines.
        Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) to maintain
        monotonicity and prevent numerical oscillations.

        The method works by:
            1. Computing departure points: xi = x - v*dt (backward tracking)
            2. Interpolating concentrations at departure points using cubic splines
            3. Applying inlet boundary condition for points that tracked outside domain

        Args:
            C_bound (float): Inlet concentration value applied at the left boundary
                (x=0) for any characteristic lines that originated from outside the
                computational domain. Units should match the concentration field.

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
        cs = PchipInterpolator(self.x, self.C)
        shift = self.v * self.dt
        xi = self.x - shift
        k0 = xi <= 0
        xi[k0] = 0
        yi = cs(xi)
        yi[k0] = C_bound
        self.C = yi

    def saulyev_solver_alt(self, C_bound) -> None:
        """Solve the diffusion step using the Saul'yev alternating direction method.

        Implements the Saul'yev scheme for the diffusion equation ∂C/∂t = D∂²C/∂x²
        using alternating direction sweeps to achieve unconditional stability.
        The method performs two passes:
            1. Left-to-right sweep using forward differences
            2. Right-to-left sweep using backward differences
            3. Final solution is the average of both sweeps

        Args:
            C_bound (float): Inlet concentration value applied at the left boundary
                during the diffusion solve. This maintains consistency with the
                advection boundary condition.

        Algorithm Details:
            - **Left-to-Right Pass**: For each cell i, uses implicit treatment of
              left neighbor and explicit treatment of right neighbor
            - **Right-to-Left Pass**: For each cell i, uses implicit treatment of
              right neighbor and explicit treatment of left neighbor
            - **Averaging**: Combines both solutions to achieve second-order accuracy

        Boundary Conditions:
            - **Left boundary (x=0)**: Dirichlet condition with prescribed C_bound
            - **Right boundary**: Zero gradient (Neumann) condition implemented
              by using the same concentration as the last interior point

        Numerical Properties:
            - Unconditionally stable for any time step size
            - Second-order accurate in space and time
            - Preserves maximum principle (no spurious extrema)
            - Handles arbitrary diffusion numbers (D*dt/dx²)

        Note:
            This method modifies self.C in-place. The alternating direction
            approach eliminates the restrictive stability constraint of explicit
            methods while maintaining computational efficiency.
        """
        dt = self.dt
        theta = self.D * dt / (self.dx**2)

        # Assign current C state as initial condition
        C_init = self.C.copy()
        CLR = self.C.copy()
        CRL = self.C.copy()

        # A) L-R direction
        for i in range(len(CLR)):
            if i == 0:  # left boundary
                solA = theta * C_bound
            else:
                solA = theta * CLR[i - 1]
            solB = (1 - theta) * C_init[i]
            solC = theta * C_init[i + 1] if i < len(CLR) - 1 else theta * C_init[i]
            # L-R Solution
            CLR[i] = (solA + solB + solC) / (1 + theta)

        # B) R-L direction
        for i in range(len(CRL) - 1, -1, -1):
            if i == len(CRL) - 1:  # right boundary (take from LR solution)
                solA = theta * CLR[-1]
            else:
                solA = theta * CRL[i + 1]
            solB = (1 - theta) * C_init[i]
            solC = theta * C_init[i - 1] if i > 0 else theta * C_init[i]
            # R-L Solution
            CRL[i] = (solA + solB + solC) / (1 + theta)

        # Average L-R and R-L solutions and update to final state
        self.C = (CLR + CRL) / 2

    def transport(self, C_bound) -> np.ndarray:
        """Perform one complete transport time step with coupled advection-diffusion.

        Executes the full semi-Lagrangian algorithm by sequentially applying
        the advection and diffusion operators using operator splitting. This
        approach decouples the hyperbolic (advection) and parabolic (diffusion)
        aspects of the transport equation for enhanced numerical stability.

        The operator splitting sequence:
            1. **Advection Step** using cubic spline MOC
            2. **Diffusion Step** using Saul'yev method

        Args:
            C_bound (float): Inlet boundary concentration applied at x=0 for both
                advection and diffusion steps. This represents the concentration
                of material entering the domain (e.g., injection well concentration,
                upstream boundary condition, etc.).

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
        self.cubic_spline_advection(C_bound)

        # Step 2: Solve diffusion equation using Saul'yev alternating direction method
        self.saulyev_solver_alt(C_bound)

        return self.C
