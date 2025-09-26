"""PhreeqcRM result code mapping and error handling utilities.

This module provides utilities for interpreting and handling result codes returned
by PhreeqcRM operations. PhreeqcRM functions return integer status codes that
indicate success, failure, or specific error conditions. This module maps these
numeric codes to human-readable error messages for improved debugging and logging.

The result codes follow the PhreeqcRM C++ library conventions and correspond
to the IrmResult enumeration defined in IrmResult.h of the original PhreeqcRM
source code.

Classes:
    IRMStatus: Named tuple containing status code, name, and message with convenience methods.

Functions:
    IRM_RESULT: Maps integer error codes to IRMStatus objects with enhanced functionality.

References:

- [PhreeqcRM IrmResult.h File Reference](https://usgs-coupled.github.io/phreeqcrm/IrmResult_8h.html)

"""

from typing import NamedTuple


class IRMStatus(NamedTuple):
    """PhreeqcRM status result with enhanced functionality.

    This named tuple extends the basic error code mapping with convenience methods
    for better error handling and user experience.

    Attributes:
        code (int): The raw integer error code from PhreeqcRM.
        name (str): Symbolic name of the error code (e.g., "IRM_OK", "IRM_FAIL").
        message (str): Human-readable description of the error.
    """

    code: int
    name: str
    message: str

    def __bool__(self) -> bool:
        """Return True if the operation was successful (code == 0).

        Examples:
            >>> result = rm.RM_RunCells()
            >>> if result:
            >>>     print("Success!")
            >>> else:
            >>>     print(f"Error: {result}")
        """
        return self.code == 0

    def __int__(self) -> int:
        """Return the raw integer code for backwards compatibility.

        Examples:
            >>> result = rm.RM_RunCells()
            >>> if int(result) == 0:  # Still works
            >>>     print("Success!")
        """
        return self.code

    def __str__(self) -> str:
        """Return a formatted string representation of the status.

        Returns:
            str: Formatted string in the form "ERROR_NAME: Error description"
        """
        return f"{self.name}: {self.message}"

    def raise_for_status(self, context: str = "") -> None:
        """Raise an exception if the operation failed.

        Args:
            context (str, optional): Additional context for the error message.

        Raises:
            RuntimeError: If the status code indicates failure (non-zero).

        Examples:
            >>> result = rm.RM_LoadDatabase("invalid.dat")
            >>> result.raise_for_status("Loading database")
            RuntimeError: Loading database: IRM_FAIL: Failure, Unspecified
        """
        if not self:
            prefix = f"{context}: " if context else ""
            raise RuntimeError(f"{prefix}{self}")


def IRM_RESULT(code: int) -> IRMStatus:
    """Map PhreeqcRM integer error codes to enhanced status objects.

    Args:
        code (int): Integer error code returned by PhreeqcRM functions.
            Return codes are listed below:
            - 0: Success (IRM_OK)
            - -1: Out of memory error
            - -2: Invalid variable type
            - -3: Invalid argument
            - -4: Invalid row index
            - -5: Invalid column index
            - -6: Invalid PhreeqcRM instance ID
            - -7: Unspecified failure

    Returns:
        IRMStatus: A named tuple containing:
            - code (int): The raw integer error code
            - name (str): Symbolic error code name (e.g., "IRM_OK", "IRM_FAIL")
            - message (str): Human-readable error description

    Examples:
        >>> result = IRM_RESULT(0)
        >>> print(result)  # "IRM_OK: Success"
        >>> if result:  # True for success
        >>>     print("Operation successful")
        >>>
        >>> error = IRM_RESULT(-1)
        >>> print(int(error))  # -1 (backwards compatibility)
        >>> error.raise_for_status("Memory allocation")  # Raises RuntimeError
    """
    mapping = {
        0: ("IRM_OK", "Success"),
        -1: ("IRM_OUTOFMEMORY", "Failure, Out of memory"),
        -2: ("IRM_BADVARTYPE", "Failure, Invalid VAR type"),
        -3: ("IRM_INVALIDARG", "Failure, Invalid argument"),
        -4: ("IRM_INVALIDROW", "Failure, Invalid row"),
        -5: ("IRM_INVALIDCOL", "Failure, Invalid column"),
        -6: ("IRM_BADINSTANCE", "Failure, Invalid rm instance id"),
        -7: ("IRM_FAIL", "Failure, Unspecified"),
    }

    if code in mapping:
        name, message = mapping[code]
        return IRMStatus(code, name, message)
    return IRMStatus(code, "IRM_UNKNOWN", f"Unknown error code: {code}")
