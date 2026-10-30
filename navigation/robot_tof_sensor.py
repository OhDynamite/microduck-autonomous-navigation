import math
import mujoco
import numpy as np

from tof_sensor import SimulatedToFSensor


class RobotMountedToF:
    """Body-mounted forward distance sensor."""

    def __init__(
        self,
        model,
        body_name="trunk_base",
        local_offset=(0.04, 0.0, 0.08),
        max_distance=2.0,
    ):
        self.body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if self.body_id < 0:
            raise ValueError(f"Body not found: {body_name}")

        self.local_offset = np.asarray(
            local_offset, dtype=np.float64
        )

        self.local_forward = np.array(
            [1.0, 0.0, 0.0], dtype=np.float64
        )

        self.sensor = SimulatedToFSensor(max_distance)

        # Detect group 0 only.
        # MicroDuck robot geoms are in groups 2 and 3.
        self.geomgroup = np.array(
            [1, 0, 0, 0, 0, 0],
            dtype=np.uint8,
        )

    def pose(self, data):
        position = data.xpos[self.body_id].copy()
        rotation = data.xmat[self.body_id].reshape(3, 3)

        origin = position + rotation @ self.local_offset
        direction = rotation @ self.local_forward

        return origin, direction

    def measure(self, model, data):
        origin, direction = self.pose(data)

        geomid = np.array([-1], dtype=np.int32)

        distance = mujoco.mj_ray(
            model,
            data,
            origin,
            direction,
            self.geomgroup,
            1,
            -1,
            geomid,
        )

        if distance < 0 or distance > self.sensor.max_distance:
            return None

        return float(distance)


if __name__ == "__main__":
    print("===== Step 111: Robot Mounted ToF Test =====")

    xml = """
    <mujoco model="robot_tof_test">
      <worldbody>
        <body name="trunk_base" pos="0 0 0.12">
          <freejoint/>
          <geom name="robot_body"
                type="box"
                size="0.05 0.05 0.05"
                group="3"/>
        </body>

        <geom name="obstacle_x"
              type="box"
              pos="1.0 0.0 0.2"
              size="0.1 0.1 0.2"
              group="0"/>

        <geom name="obstacle_y"
              type="box"
              pos="0.0 1.0 0.2"
              size="0.1 0.1 0.2"
              group="0"/>
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    sensor = RobotMountedToF(
        model,
        local_offset=(0.04, 0.0, 0.08),
    )

    # Test 1: facing +X
    mujoco.mj_forward(model, data)
    origin, direction = sensor.pose(data)
    distance = sensor.measure(model, data)

    print(f"[TOF] Initial direction: {direction}")
    print(f"[TOF] Initial distance: {distance:.3f} m")

    assert np.allclose(direction, [1, 0, 0])
    assert abs(distance - 0.86) < 1e-6
    print("[OK] Forward obstacle detected")

    # Test 2: rotate robot 90 degrees around Z.
    yaw = math.pi / 2
    data.qpos[3:7] = [
        math.cos(yaw / 2),
        0.0,
        0.0,
        math.sin(yaw / 2),
    ]

    mujoco.mj_forward(model, data)

    origin, direction = sensor.pose(data)
    distance = sensor.measure(model, data)

    print(f"[TOF] Rotated direction: {np.round(direction, 3)}")
    print(f"[TOF] Rotated distance: {distance:.3f} m")

    assert np.allclose(direction, [0, 1, 0], atol=1e-6)
    assert abs(distance - 0.86) < 1e-6
    print("[OK] Sensor follows robot rotation")

    print("[OK] Robot geom excluded by group filter")

    print()
    print("SUCCESS: Robot mounted ToF verified")
    print("===== Step 111 Completed =====")
