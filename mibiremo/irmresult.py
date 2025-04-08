def IRM_RESULT(code):
    """Map error codes (integer) to error messages
    Returns the error as a tuple (ERR_CODE, DESCRIPTION)
    See: IrmResult.h File Reference.
    """
    map = {
        0:("IRM_OK","Success"),
        -1:("IRM_OUTOFMEMORY","Failure, Out of memory"),
        -2:("IRM_BADVARTYPE","Failure, Invalid VAR type"),
        -3:("IRM_INVALIDARG","Failure, Invalid argument"),
        -4:("IRM_INVALIDROW","Failure, Invalid row"),
        -5:("IRM_INVALIDCOL","Failure, Invalid column"),
        -6:("IRM_BADINSTANCE","Failure, Invalid rm instance id"),
        -7:("IRM_FAIL","Failure, Unspecified"),
    }

    if code in map:
        return map[code]
    return ("UNSPECIFIED", "Invalid error code")
