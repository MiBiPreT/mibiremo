"""pytest configuration for the mibiremo test suite.

Set NUMBA_DISABLE_JIT=1 at module level to disable JIT compilation during testing,
and MPLBACKEND=Agg to plot without a display
"""

import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLBACKEND"] = "Agg"
