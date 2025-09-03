"""PhreeqcRM result code mapping and error handling utilities.

This module provides utilities for interpreting and handling result codes returned
by PhreeqcRM operations. PhreeqcRM functions return integer status codes that
indicate success, failure, or specific error conditions. This module maps these
numeric codes to human-readable error messages for improved debugging and logging.

The result codes follow the PhreeqcRM C++ library conventions and correspond
to the IrmResult enumeration defined in IrmResult.h of the original PhreeqcRM
source code.

Functions:
    IRM_RESULT: Maps integer error codes to descriptive error messages.

References:
    PhreeqcRM IrmResult.h File Reference:
    https://usgs-coupled.github.io/phreeqcrm/IrmResult_8h.html

"""


def IRM_RESULT(code):
    """Map PhreeqcRM integer error codes to descriptive error messages.

    Args:
        code (int): Integer error code returned by PhreeqcRM functions.
            Common codes include:
            - 0: Success (IRM_OK)
            - -1: Out of memory error
            - -2: Invalid variable type
            - -3: Invalid argument
            - -4: Invalid row index
            - -5: Invalid column index
            - -6: Invalid PhreeqcRM instance ID
            - -7: Unspecified failure

    Returns:
        tuple: A 2-element tuple containing:
            - str: Symbolic error code name (e.g., "IRM_OK", "IRM_FAIL")
            - str: Human-readable error description
    """
    map = {
        0: ("IRM_OK", "Success"),
        -1: ("IRM_OUTOFMEMORY", "Failure, Out of memory"),
        -2: ("IRM_BADVARTYPE", "Failure, Invalid VAR type"),
        -3: ("IRM_INVALIDARG", "Failure, Invalid argument"),
        -4: ("IRM_INVALIDROW", "Failure, Invalid row"),
        -5: ("IRM_INVALIDCOL", "Failure, Invalid column"),
        -6: ("IRM_BADINSTANCE", "Failure, Invalid rm instance id"),
        -7: ("IRM_FAIL", "Failure, Unspecified"),
    }

    if code in map:
        return map[code]
    return ("UNSPECIFIED", "Invalid error code")
