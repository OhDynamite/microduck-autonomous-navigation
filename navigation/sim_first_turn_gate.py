"""Simulation-only, one-shot first-turn command gate.

This module cannot release an emergency stop, drive forward,
start PPO inference, or issue motor commands.

Its output is only a proposed yaw command. Integration into
MuJoCo's final command arbitration requires separate validation.
"""

import math

ZERO = (0.0, 0.0, 0.0)


class SimFirstTurnGate:
    def __init__(
        self,
        *,
        max_duration_s=4.0,
        max_yaw_speed=0.50,
        min_tof_distance_m=0.30,
    ):
        self.state = "IDLE"
        self.reason = None
        self.started_at = None
        self.target_yaw = None
        self.max_duration_s = float(max_duration_s)
        self.max_yaw_speed = float(max_yaw_speed)
        self.min_tof_distance_m = float(min_tof_distance_m)

        if not (
            0 < self.max_duration_s <= 4.0
            and 0 < self.max_yaw_speed <= 0.50
            and 0 < self.min_tof_distance_m <= 0.60
        ):
            raise ValueError("Invalid safety limits")

    def block(self, reason):
        self.state = "BLOCKED"
        self.reason = str(reason)
        return ZERO

    def arm(
        self,
        *,
        emergency_latched,
        emergency_reason,
        task,
        now,
        current_yaw,
    ):
        if self.state != "IDLE":
            return False

        if emergency_latched is not True or emergency_reason != "TOF_NEAR":
            self.block("NO_VERIFIED_STOP")
            return False

        if not isinstance(task, dict):
            self.block("INVALID_TASK")
            return False

        if (
            task.get("segment") != 1
            or task.get("task_type") != "TURN_THEN_WALK"
            or task.get("movement_authorized") is not False
            or task.get("execution_state") != "NOT_STARTED"
        ):
            self.block("UNEXPECTED_TASK")
            return False

        try:
            now = float(now)
            current_yaw = float(current_yaw)
            target = float(task["target_yaw_rad"])
            delta = math.atan2(
                math.sin(target - current_yaw),
                math.cos(target - current_yaw),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            self.block("INVALID_HEADING")
            return False

        if not all(math.isfinite(v) for v in (now, current_yaw, target)):
            self.block("NONFINITE_HEADING")
            return False

        if not (
            math.radians(5) <= abs(delta) <= math.radians(90)
        ):
            self.block("TURN_ANGLE_OUT_OF_RANGE")
            return False

        self.started_at = now
        self.target_yaw = target
        self.state = "ARMED"
        return True

    def command(
        self,
        *,
        now,
        proposed_yaw_rate,
        emergency_latched,
        emergency_reason,
        sensor_healthy,
        tof_distance_m,
        pose_healthy,
        contact,
        clearance_ok,
        turn_complete=False,
        turn_aborted=False,
    ):
        if self.state not in ("ARMED", "ACTIVE"):
            return ZERO

        if emergency_latched is not True or emergency_reason != "TOF_NEAR":
            return self.block("STOP_LATCH_CHANGED")

        if turn_aborted is True:
            return self.block("TURN_ABORTED")

        if turn_complete is True:
            self.state = "COMPLETE"
            return ZERO

        if (
            sensor_healthy is not True
            or pose_healthy is not True
            or contact is not False
            or clearance_ok is not True
        ):
            return self.block("SAFETY_CHECK_FAILED")

        try:
            values = (
                float(now),
                float(proposed_yaw_rate),
                float(tof_distance_m),
            )
        except (TypeError, ValueError, OverflowError):
            return self.block("INVALID_COMMAND_INPUT")

        if not all(math.isfinite(v) for v in values):
            return self.block("NONFINITE_INPUT")

        now, wz, distance = values

        if now < self.started_at:
            return self.block("TIME_REVERSED")

        if now - self.started_at >= self.max_duration_s:
            return self.block("TURN_TIMEOUT")

        if distance < self.min_tof_distance_m:
            return self.block("TOF_TOO_CLOSE")

        if abs(wz) > self.max_yaw_speed + 1e-9:
            return self.block("EXCESSIVE_YAW_RATE")

        self.state = "ACTIVE"
        return (0.0, 0.0, wz)

    def snapshot(self):
        return {
            "state": self.state,
            "reason": self.reason,
            "target_yaw": self.target_yaw,
            "movement_authorized": False,
            "emergency_stop_release_authorized": False,
            "walking_authorized": False,
            "simulation_only": True,
        }
