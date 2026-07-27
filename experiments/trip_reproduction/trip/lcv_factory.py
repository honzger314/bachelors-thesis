def get_lcv_function(method):

    if method == "original":
        from trip.lcv import compute_lcv

    elif method == "modified":
        from trip.lcv_modified import compute_lcv

    else:
        raise ValueError(
            f"Unknown LCV method: {method}"
        )

    return compute_lcv