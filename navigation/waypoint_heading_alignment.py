"""Simulation-only WP1 small-angle heading alignment.

Produces a bounded yaw-only PROPOSAL.
Does not issue motor commands or unlock emergency stops.
"""

import math

ZERO = (0.0, 0.0, 0.0)


def signed_angle_error(target, current):
    return math.atan2(
        math.sin(target - current),
        math.cos(target - current),
    )


class WP1HeadingAligner:
    def __init__(
        self,
        *,
        max_speed=0.18,
        tolerance_deg=5.0,
        settle_s=0.30,
        timeout_s=2.0,
    ):
        self.max_speed = max_speed
        self.tolerance = math.radians(tolerance_deg)
        self.settle_s = settle_s
        self.timeout_s = timeout_s
        self.state = "IDLE"
        self.reason = None
        self.start_time = None
        self.target = None
        self.within_since = None

    def block(self, reason):
        self.state = "BLOCKED"
        self.reason = str(reason)
        return ZERO

    def start(
        self, *,
        now, current_yaw, target_yaw,
        emergency_latched, emergency_reason,
        route_clear,
    ):
        if self.state != "IDLE":
            return False

        values = (now, current_yaw, target_yaw)
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in values
        ):
            self.block("INVALID_START")
            return False

        error = abs(signed_angle_error(target_yaw, current_yaw))

        if not (
            emergency_latched is True
            and emergency_reason == "TOF_NEAR"
            and route_clear is True
            and math.radians(8) < error <= math.radians(25)
        ):
            self.block("START_NOT_AUTHORIZED")
            return False

        self.start_time = float(now)
        self.target = float(target_yaw)
        self.state = "ALIGNING"
        return True

    def command(
        self, *,
        now, current_yaw,
        emergency_latched, emergency_reason,
        sensor_healthy, distance_m,
        pose_healthy, contact, clearance_ok,
    ):
        if self.state != "ALIGNING":
            return ZERO

        values = (now, current_yaw, distance_m)
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in values
        ):
            return self.block("INVALID_SENSOR_OR_POSE")

        if (
            emergency_latched is not True
            or emergency_reason != "TOF_NEAR"
        ):
            return self.block("STOP_LATCH_CHANGED")

        if (
            sensor_healthy is not True
            or pose_healthy is not True
            or contact is not False
            or clearance_ok is not True
            or distance_m < 0.30
        ):
            return self.block("SAFETY_FAILURE")

        elapsed = now - self.start_time

        if elapsed < 0 or elapsed >= self.timeout_s:
            return self.block("TIMEOUT")

        error = signed_angle_error(self.target, current_yaw)

        if abs(error) <= self.tolerance:
            if self.within_since is None:
                self.within_since = now

            if now - self.within_since >= self.settle_s:
                self.state = "COMPLETE"
                return ZERO

            return ZERO

        self.within_since = None

        # Small-angle proportional correction.
        wz = max(
            -self.max_speed,
            min(self.max_speed, 1.0 * error),
        )

        return (0.0, 0.0, float(wz))

    def snapshot(self):
        return {
            "state": self.state,
            "reason": self.reason,
            "movement_authorized": False,
            "stop_latch_release_authorized": False,
            "simulation_only": True,
        }
