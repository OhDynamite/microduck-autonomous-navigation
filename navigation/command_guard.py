from navigation_state import (
    NavigationState,
    ReplanningController,
)


ZERO_CMD = (0.0, 0.0, 0.0)


def guarded_command(controller, requested_command):
    """Apply navigation safety state to velocity command."""

    if not controller.velocity_allowed():
        return ZERO_CMD

    command = tuple(float(v) for v in requested_command)

    if len(command) != 3:
        raise ValueError("Velocity command must have 3 elements")

    return command


if __name__ == "__main__":
    print("===== Step 122: Safety Command Guard Test =====")

    controller = ReplanningController()

    walk_command = (0.30, 0.0, 0.0)
    turn_command = (0.0, 0.0, 0.50)

    # Test 1: Normal navigation
    assert guarded_command(controller, walk_command) == walk_command
    print("[OK] RUNNING allows navigation command")

    # Test 2: Obstacle detected during WARMUP
    assert controller.obstacle_detected()
    assert controller.state == NavigationState.STOPPED

    assert guarded_command(controller, walk_command) == ZERO_CMD
    print("[OK] STOPPED overrides WARMUP forward command")

    # Test 3: Replanning
    assert controller.start_replanning()

    assert guarded_command(controller, turn_command) == ZERO_CMD
    print("[OK] REPLANNING overrides turning command")

    # Test 4: Valid path permits resuming
    assert controller.finish_replanning(path_valid=True)

    assert guarded_command(controller, turn_command) == turn_command
    print("[OK] Valid replanning permits turning")

    # Test 5: Failed replanning blocks movement
    assert controller.obstacle_detected()
    assert controller.start_replanning()
    assert controller.finish_replanning(path_valid=False)

    assert guarded_command(controller, walk_command) == ZERO_CMD
    assert guarded_command(controller, turn_command) == ZERO_CMD
    print("[OK] BLOCKED prevents forward and turning commands")

    print()
    print("SUCCESS: Safety command guard verified")
    print("===== Step 122 Completed =====")
