import math


class SafetyMap:
    """Create provisional exclusion zones around ToF hit points."""

    def __init__(self, safety_radius=0.20):
        if safety_radius <= 0:
            raise ValueError("safety_radius must be positive")

        self.safety_radius = float(safety_radius)
        self.zones = []

    def add_detection(self, point):
        x, y = map(float, point)
        zone = (x, y, self.safety_radius)
        self.zones.append(zone)
        return zone

    def is_blocked(self, x, y):
        return any(
            math.hypot(x - cx, y - cy) <= radius
            for cx, cy, radius in self.zones
        )

    def segment_blocked(self, start, end):
        """Check whether a line segment intersects any exclusion zone."""
        ax, ay = map(float, start)
        bx, by = map(float, end)

        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy

        for cx, cy, radius in self.zones:
            if length_sq == 0:
                nearest_x, nearest_y = ax, ay
            else:
                t = (
                    (cx - ax) * dx + (cy - ay) * dy
                ) / length_sq
                t = max(0.0, min(1.0, t))
                nearest_x = ax + t * dx
                nearest_y = ay + t * dy

            if math.hypot(
                cx - nearest_x,
                cy - nearest_y
            ) <= radius:
                return True

        return False


if __name__ == "__main__":
    print("===== Step 115: Safety Zone Test =====")

    safety_map = SafetyMap(safety_radius=0.20)

    zone = safety_map.add_detection((0.90, 0.00))

    print(f"[MAP] Safety zone: {zone}")
    assert len(safety_map.zones) == 1
    print("[OK] Safety zone registered")

    assert safety_map.is_blocked(0.90, 0.00)
    assert safety_map.is_blocked(0.80, 0.00)
    assert not safety_map.is_blocked(0.50, 0.00)
    print("[OK] Point collision checks verified")

    assert safety_map.segment_blocked(
        (0.0, 0.0), (1.0, 0.0)
    )
    print("[OK] Crossing path correctly blocked")

    assert not safety_map.segment_blocked(
        (0.0, 0.5), (1.0, 0.5)
    )
    print("[OK] Clear path correctly accepted")

    assert safety_map.segment_blocked(
        (0.75, -0.3), (0.75, 0.3)
    )
    print("[OK] Side crossing correctly blocked")

    print()
    print("SUCCESS: Safety zone generation verified")
    print("===== Step 115 Completed =====")
