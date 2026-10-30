from enum import Enum


class TurnState(Enum):
    STOPPED = "STOPPED"
    TURN_PENDING = "TURN_PENDING"
    TURNING = "TURNING"
    TURN_COMPLETE = "TURN_COMPLETE"
    TURN_ABORTED = "TURN_ABORTED"


class StoppedTurnController:
    """Manage Phase 5 turning states without issuing motor commands."""

    def __init__(self):
        self.state = TurnState.STOPPED
        self.direction = None
        self.target_deg = None
        self.abort_reason = None

    def request_turn(self, direction, target_deg):
        if self.state != TurnState.STOPPED:
            return False
        if direction not in ("LEFT", "RIGHT"):
            return False
        if not (0.0 < target_deg <= 90.0):
            return False

        self.direction = direction
        self.target_deg = float(target_deg)
        self.abort_reason = None
        self.state = TurnState.TURN_PENDING
        return True

    def authorize_turn(self, path_ready, clearance_verified):
        if self.state != TurnState.TURN_PENDING:
            return False
        if not path_ready or not clearance_verified:
            return False

        self.state = TurnState.TURNING
        return True

    def complete_turn(self):
        if self.state != TurnState.TURNING:
            return False

        self.state = TurnState.TURN_COMPLETE
        return True

    def abort_turn(self, reason):
        self.abort_reason = str(reason)
        self.state = TurnState.TURN_ABORTED

    def reset_to_stopped(self):
        self.state = TurnState.STOPPED
        self.direction = None
        self.target_deg = None
        self.abort_reason = None

    def turning_allowed(self):
        return self.state == TurnState.TURNING
