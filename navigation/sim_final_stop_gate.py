"""Step 1252: simulation-only final velocity stop gate."""

import math

ZERO_COMMAND = (0.0, 0.0, 0.0)


def apply_sim_final_stop_gate(command, emergency_stop=False):
    """Reject unsafe command values; prioritize an emergency stop.

    Does not authorize navigation, validate sensor health,
    or certify path clearance.
    """
    if emergency_stop:
        return ZERO_COMMAND

    if command is None:
        return ZERO_COMMAND

    try:
        values = tuple(float(v) for v in command)
    except (TypeError, ValueError, OverflowError):
        return ZERO_COMMAND

    if len(values) != 3:
        return ZERO_COMMAND

    if not all(math.isfinite(v) for v in values):
        return ZERO_COMMAND

    return values
