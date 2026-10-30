"""Route execution sequencing: NO motor-control capability.

This module reports desired control phases. It never sends velocity,
starts PPO controllers, or authorizes release of an emergency stop.
"""


class ShadowRouteSequencer:
    def __init__(self, tasks):
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("Nonempty task list required")

        self.tasks = []
        for index, task in enumerate(tasks, 1):
            if not isinstance(task, dict):
                raise ValueError("Task must be a dict")
            if task.get("segment") != index:
                raise ValueError("Task ordering invalid")
            if task.get("task_type") != "TURN_THEN_WALK":
                raise ValueError("Unsupported task type")
            if task.get("movement_authorized") is not False:
                raise ValueError("Unsafe task authorization")
            if task.get("execution_state") != "NOT_STARTED":
                raise ValueError("Unexpected execution state")

            target = task.get("target_xy")
            if not isinstance(target, list) or len(target) != 2:
                raise ValueError("Invalid target")
            import math
            if not all(math.isfinite(float(v)) for v in target):
                raise ValueError("Nonfinite target")

            yaw = float(task.get("target_yaw_rad", float("nan")))
            length = float(task.get("length_m", float("nan")))
            if not math.isfinite(yaw) or not math.isfinite(length):
                raise ValueError("Nonfinite task geometry")
            if length <= 0:
                raise ValueError("Invalid segment length")

            self.tasks.append(dict(task))

        self.index = 0
        self.state = "STOPPED"
        self.block_reason = None
        self.events = []

    def start_shadow(self):
        if self.state != "STOPPED":
            return self.block("INVALID_START")
        self.state = "TURN_PENDING"
        self.events.append("TURN_PENDING:1")
        return True

    def turn_verified(self, completed):
        if self.state != "TURN_PENDING":
            return self.block("INVALID_TURN_EVENT")
        if completed is not True:
            return self.block("TURN_NOT_VERIFIED")
        self.state = "WALK_PENDING"
        self.events.append(f"WALK_PENDING:{self.index + 1}")
        return True

    def waypoint_verified(self, reached):
        if self.state != "WALK_PENDING":
            return self.block("INVALID_WAYPOINT_EVENT")
        if reached is not True:
            return self.block("WAYPOINT_NOT_VERIFIED")
        self.events.append(f"WAYPOINT_VERIFIED:{self.index + 1}")
        self.index += 1
        if self.index >= len(self.tasks):
            self.state = "ROUTE_COMPLETE_SHADOW"
        else:
            self.state = "TURN_PENDING"
            self.events.append(f"TURN_PENDING:{self.index + 1}")
        return True

    def block(self, reason):
        self.state = "BLOCKED"
        self.block_reason = str(reason)
        self.events.append("BLOCKED:" + self.block_reason)
        return False

    def snapshot(self):
        return {
            "state": self.state,
            "current_segment": (
                self.index + 1
                if self.index < len(self.tasks) else None
            ),
            "task_count": len(self.tasks),
            "block_reason": self.block_reason,
            "events": list(self.events),
            "movement_authorized": False,
            "emergency_stop_release_authorized": False,
        }
