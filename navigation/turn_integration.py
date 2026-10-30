"""Fail-closed Phase 4/5 turning integration.

This adapter is state management only. It never issues velocity
commands and does not unlock Phase 4 navigation.

Safety evidence must come from a trusted sensor-verification
process. Simulator ground truth is not accepted as sensor evidence.
"""

from navigation_state import NavigationState
from stopped_turn_controller import (
    StoppedTurnController,
    TurnState,
)
from turn_safety_gate import authorize_turn as safety_gate


class Phase45TurnIntegration:
    """Coordinate a stopped navigation state and pending turn."""

    def __init__(self, navigation_controller):
        self.navigation = navigation_controller
        self.turn = StoppedTurnController()
        self.path_ready = False

    def request(self, direction, target_deg, *, path_ready=False):
        """Register a turn request without authorizing movement."""
        if self.navigation.state is not NavigationState.STOPPED:
            return False

        if type(path_ready) is not bool:
            return False

        if self.turn.state is not TurnState.STOPPED:
            return False

        if not self.turn.request_turn(direction, target_deg):
            return False

        self.path_ready = path_ready
        return True

    def evaluate(
        self,
        obstacle_distance,
        *,
        shape_verified=False,
        swept_area_observed=False,
        distance_reference_verified=False,
        evidence_fresh=False,
    ):
        """Evaluate clearance; never bypass the navigation stop."""
        if self.navigation.state is not NavigationState.STOPPED:
            return False

        if self.turn.state is not TurnState.TURN_PENDING:
            return False

        if not self.path_ready:
            return False

        clearance_ok = safety_gate(
            self.turn.direction,
            obstacle_distance,
            shape_verified=shape_verified,
            swept_area_observed=swept_area_observed,
            distance_reference_verified=distance_reference_verified,
            evidence_fresh=evidence_fresh,
        )

        if not clearance_ok:
            return False

        # This adapter deliberately does not transition to TURNING.
        # Motor-command integration requires a separately verified
        # authority boundary and continuous safety monitoring.
        return True

    def movement_authorized(self):
        """Physical turning is not enabled by this adapter."""
        return False

    def reset(self):
        self.path_ready = False
        self.turn.reset_to_stopped()
