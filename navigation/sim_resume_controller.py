"""Simulation-only navigation resume state controller.

This controller does not authorize real robot movement.
"""

import math
from enum import Enum


class SimResumeState(Enum):
    WAITING = "WAITING"
    VALIDATING = "VALIDATING"
    WALKING = "WALKING"
    STOPPED = "STOPPED"
    GOAL_REACHED = "GOAL_REACHED"


class SimResumeController:

    def __init__(self, heading_tolerance_deg=8.0):
        self.state = SimResumeState.WAITING
        self.heading_tolerance_deg = float(heading_tolerance_deg)
        self.stop_reason = None

    def begin_validation(self):
        if self.state is not SimResumeState.WAITING:
            return False
        self.state = SimResumeState.VALIDATING
        return True

    def validate(
        self,
        *,
        pose,
        heading_error_deg,
        path_verified,
        contact=False,
        fallen=False,
    ):
        if self.state is not SimResumeState.VALIDATING:
            return False

        if contact or fallen:
            self.stop("CONTACT" if contact else "FALL")
            return False

        if not all(math.isfinite(float(v)) for v in pose):
            self.stop("INVALID_POSE")
            return False

        if not math.isfinite(float(heading_error_deg)):
            self.stop("INVALID_HEADING")
            return False

        if not path_verified:
            self.stop("PATH_NOT_VERIFIED")
            return False

        if abs(heading_error_deg) > self.heading_tolerance_deg:
            self.stop("HEADING_NOT_ALIGNED")
            return False

        self.state = SimResumeState.WALKING
        return True

    def update(self, *, contact=False, fallen=False, path_verified=True):
        if self.state is not SimResumeState.WALKING:
            return False

        if contact or fallen or not path_verified:
            reason = (
                "CONTACT" if contact
                else "FALL" if fallen
                else "PATH_LOST"
            )
            self.stop(reason)
            return False

        return True

    def goal_reached(self):
        if self.state is not SimResumeState.WALKING:
            return False
        self.state = SimResumeState.GOAL_REACHED
        return True

    def stop(self, reason):
        self.state = SimResumeState.STOPPED
        self.stop_reason = str(reason)
