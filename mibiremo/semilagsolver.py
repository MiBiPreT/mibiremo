"""
    Semi-lagrangian solver for 1D advection on a uniform grid

    1) Advection solved with semi-lagrangian method with cubic spline interpolation
    2) Diffusion solved with Saul'yev finite difference scheme (alternating directions)
    
    Author: Matteo Masi
    Last revision: 03/07/2024

"""



import numpy as np
from scipy.interpolate import PchipInterpolator
import warnings

warnings.filterwarnings("ignore")

class SemiLagSolver:
    """
    Implements semi-lagrangian integration schemes with Dirichlet-type boundary 
    condition at the inlet of the domain (left boundary x = 0) 
    and Neumann-type boundary condition at the outlet (right boundary).

    Usage:
        obj = SemiLagSolver(x, C, v, dt)
        
    Parameters:
        x (np.ndarray): Spatial coordinates (must be equally spaced)
        C (np.ndarray): Initial concentration
        v (float): Velocity
        D (float): Diffusion (dispersion) coefficient
        dt (float): Time-step

    """

    def __init__(self, x, C_init, v, D, dt):
        """
        Initializes the LagSolver object with spatial coordinates, 
        initial concentration, velocity, time-step, and solver type.
        """
        self.x = x
        self.C = C_init
        self.v = v
        self.D = D
        self.dt = dt
        self.dx = x[1] - x[0]


    def cubic_spline_advection(self, C_bound):
        """
        Propagates the current variable using a cubic spline interpolation.
        """
        cs = PchipInterpolator(self.x, self.C)
        shift = self.v * self.dt
        xi = self.x - shift
        k0 = xi <= 0
        xi[k0] = 0
        yi = cs(xi)
        yi[k0] = C_bound
        self.C = yi

    def saulyev_solver_alt(self, C_bound):
        """
        Saul'yev solver (integration in alternating directions)
        """
        dt = self.dt
        theta = self.D * dt / (self.dx ** 2)
        
        # Assign current C state as initial condition
        C_init = self.C.copy()
        CLR = self.C.copy()
        CRL = self.C.copy()
        
        # A) L-R direction
        for i in range(len(CLR)):
            if i == 0:  # left boundary
                solA = theta * C_bound
            else:
                solA = theta * CLR[i-1]
            solB = (1 - theta) * C_init[i]
            if i < len(CLR) - 1:
                solC = theta * C_init[i+1]
            else:
                solC = theta * C_init[i]
            # L-R Solution
            CLR[i] = (solA + solB + solC) / (1 + theta)

        # B) R-L direction
        for i in range(len(CRL) - 1, -1, -1):
            if i == len(CRL) - 1:  # right boundary (take from LR solution)
                solA = theta * CLR[-1]
            else:
                solA = theta * CRL[i+1]
            solB = (1 - theta) * C_init[i]
            if i > 0:
                solC = theta * C_init[i-1]
            else:
                solC = theta * C_init[i]
            # R-L Solution
            CRL[i] = (solA + solB + solC) / (1 + theta)
        
        # Average L-R and R-L solutions
        C = (CLR + CRL) / 2
        
        # Update initial condition for next step k
        C_init = C.copy()
        
        # Update to the final state
        self.C = C


    def transport(self, C_bound):
        """
        Couple advection and diffusion
        """
        # Advection
        self.cubic_spline_advection(C_bound)

        # Diffusion
        self.saulyev_solver_alt(C_bound)

        return self.C
