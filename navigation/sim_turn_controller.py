"""MuJoCo-only PPO V3 turn state controller.

This module manages test state and timing only.
It does not authorize movement, control motors, or bypass
the existing Phase 4 safety gate.
"""

import math
from enum import Enum


class SimTurnState(Enum):
    WAITING = "WAITING"
    TURNING = "TURNING"
    SETTLING = "SETTLING"
    READY_TO_RESUME = "READY_TO_RESUME"
    ABORTED = "ABORTED"


def angle_error(target, current):
    return math.atan2(
        math.sin(target - current),
        math.cos(target - current),
    )


class SimTurnController:

    def __init__(
        self,
        *,
        turn_speed=0.50,
        turn_tolerance_deg=12.0,
        settling_duration=0.50,
        max_turn_duration=4.0,
    ):
        self.state = SimTurnState.WAITING
        self.turn_speed = turn_speed
        self.tolerance = math.radians(turn_tolerance_deg)
        self.settling_duration = settling_duration
        self.max_turn_duration = max_turn_duration

        self.target_yaw = None
        self.start_time = None
        self.settling_start = None
        self.abort_reason = None

    def start(self, *, current_time, target_yaw):
        if self.state is not SimTurnState.WAITING:
            return False

        if not (
            math.isfinite(current_time)
            and math.isfinite(target_yaw)
        ):
            return False

        self.target_yaw = target_yaw
        self.start_time = current_time
        self.state = SimTurnState.TURNING
        return True

    def update(self, *, current_time, current_yaw, contact=False, fallen=False):
        if contact or fallen:
            self.abort("CONTACT" if contact else "FALL")
            return 0.0

        if self.state is SimTurnState.TURNING:
            elapsed = current_time - self.start_time

            if elapsed >= self.max_turn_duration:
                self.abort("TURN_TIMEOUT")
                return 0.0

            error = angle_error(self.target_yaw, current_yaw)

            if abs(error) <= self.tolerance:
                self.state = SimTurnState.SETTLING
                self.settling_start = current_time
                return 0.0

            return math.copysign(self.turn_speed, error)

        if self.state is SimTurnState.SETTLING:
            if current_time - self.settling_start >= self.settling_duration:
                self.state = SimTurnState.READY_TO_RESUME
            return 0.0

        return 0.0

    def abort(self, reason):
        self.state = SimTurnState.ABORTED
        self.abort_reason = reason
