from enum import Enum


class NavigationState(Enum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    REPLANNING = "REPLANNING"
    BLOCKED = "BLOCKED"


class ReplanningController:
    """Manage safe navigation transitions."""

    def __init__(self):
        self.state = NavigationState.RUNNING
        self.replan_count = 0

    def obstacle_detected(self):
        if self.state == NavigationState.RUNNING:
            self.state = NavigationState.STOPPED
            return True
        return False

    def start_replanning(self):
        if self.state != NavigationState.STOPPED:
            return False

        self.state = NavigationState.REPLANNING
        self.replan_count += 1
        return True

    def finish_replanning(self, path_valid):
        if self.state != NavigationState.REPLANNING:
            return False

        if path_valid:
            self.state = NavigationState.RUNNING
        else:
            self.state = NavigationState.BLOCKED

        return True

    def velocity_allowed(self):
        return self.state == NavigationState.RUNNING


if __name__ == "__main__":
    print("===== Step 120: Navigation State Test =====")

    controller = ReplanningController()

    assert controller.velocity_allowed()
    print("[OK] Initial RUNNING state")

    assert controller.obstacle_detected()
    assert controller.state == NavigationState.STOPPED
    assert not controller.velocity_allowed()
    print("[OK] Obstacle detection requests stop")

    assert controller.start_replanning()
    assert controller.state == NavigationState.REPLANNING
    assert not controller.velocity_allowed()
    print("[OK] Replanning state blocks movement")

    assert controller.finish_replanning(path_valid=True)
    assert controller.velocity_allowed()
    assert controller.replan_count == 1
    print("[OK] Valid path permits resuming")

    assert controller.obstacle_detected()
    assert controller.start_replanning()
    assert controller.finish_replanning(path_valid=False)

    assert controller.state == NavigationState.BLOCKED
    assert not controller.velocity_allowed()
    assert controller.replan_count == 2
    print("[OK] Failed replanning keeps robot blocked")

    print()
    print("SUCCESS: Navigation state transitions verified")
    print("===== Step 120 Completed =====")
