from turn_clearance import evaluate_turn_clearance


def authorize_turn(
    direction,
    obstacle_distance,
    *,
    shape_verified=False,
    swept_area_observed=False,
    distance_reference_verified=False,
    evidence_fresh=False,
):
    """Preliminary evidence gate; not a physical safety certificate.

    All evidence flags must be explicitly established by a trusted
    upstream verification process. A single forward ToF hit alone
    must never satisfy these conditions.
    """

    flags = (
        shape_verified,
        swept_area_observed,
        distance_reference_verified,
        evidence_fresh,
    )

    if not all(type(flag) is bool for flag in flags):
        return False

    if not all(flags):
        return False

    try:
        return evaluate_turn_clearance(direction, obstacle_distance)
    except (TypeError, ValueError, OverflowError):
        return False
