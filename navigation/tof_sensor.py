import mujoco
import numpy as np


class SimulatedToFSensor:
    """MuJoCo ray-based, single-direction distance sensor."""

    def __init__(self, max_distance=2.0):
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        self.max_distance = float(max_distance)

    def measure(self, model, data, origin, direction, bodyexclude=-1):
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)

        if origin.shape != (3,) or direction.shape != (3,):
            raise ValueError("origin and direction must be 3D vectors")

        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError("direction must be nonzero")

        direction = direction / norm
        geomid = np.array([-1], dtype=np.int32)

        distance = mujoco.mj_ray(
            model,
            data,
            origin,
            direction,
            None,
            1,
            bodyexclude,
            geomid,
        )

        if distance < 0 or distance > self.max_distance:
            return None

        return float(distance)


if __name__ == "__main__":
    print("===== Step 106: Simulated ToF Sensor Test =====")

    xml = """
    <mujoco model="tof_test">
      <worldbody>
        <geom name="obstacle_1"
              type="box"
              pos="1.0 0.0 0.2"
              size="0.1 0.1 0.2"/>
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    origin = [0.2, 0.0, 0.2]
    forward = [1.0, 0.0, 0.0]

    sensor = SimulatedToFSensor(max_distance=2.0)

    distance = sensor.measure(model, data, origin, forward)
    print(f"[TOF] Forward distance: {distance:.3f} m")
    assert abs(distance - 0.7) < 1e-6
    print("[OK] Obstacle distance verified")

    side = sensor.measure(model, data, origin, [0.0, 1.0, 0.0])
    assert side is None
    print("[OK] No obstacle in side direction")

    short_sensor = SimulatedToFSensor(max_distance=0.5)
    short_result = short_sensor.measure(model, data, origin, forward)
    assert short_result is None
    print("[OK] Maximum detection range verified")

    print()
    print("SUCCESS: Simulated ToF sensor verified")
    print("===== Step 106 Completed =====")
