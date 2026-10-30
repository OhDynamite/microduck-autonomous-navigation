"""Simulation-only final command arbitration for one first turn.

Does not clear an emergency latch or start a controller.
It must be used as the LAST velocity gate before set_vel_cmd.
"""
import math
from sim_final_stop_gate import (
    apply_sim_final_stop_gate,
)

ZERO = (0.0, 0.0, 0.0)


def arbitrate_first_turn(
    *,
    normal_command,
    emergency_latched,
    emergency_reason,
    trial_enabled,
    gate,
    gate_proposal,
    explicit_trial_permission,
):
    # This branch preserves the original final stop semantics.
    if emergency_latched is not True:
        return apply_sim_final_stop_gate(
            normal_command,
            emergency_stop=bool(emergency_latched),
        )

    # No trial, no exception.
    if (
        trial_enabled is not True
        or explicit_trial_permission is not True
        or emergency_reason != "TOF_NEAR"
        or gate is None
        or getattr(gate, "state", None) != "ACTIVE"
    ):
        return ZERO

    # Never accept normal navigation commands during a stop.
    # Only the independently checked first-turn proposal is considered.
    try:
        if gate_proposal is None:
            return ZERO
        cmd = tuple(float(v) for v in gate_proposal)
    except (ValueError, TypeError, OverflowError):
        return ZERO

    if len(cmd) != 3 or not all(math.isfinite(v) for v in cmd):
        return ZERO

    vx, vy, wz = cmd
    if vx != 0.0 or vy != 0.0:
        return ZERO
    if abs(wz) > 0.50:
        return ZERO

    # Do not implicitly unlock the emergency stop.
    return (0.0, 0.0, wz)
