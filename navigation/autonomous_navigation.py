#!/usr/bin/env python3
"""Simple script to run ONNX policy inference in MuJoCo with rendering."""

import argparse
import csv
import json
import math
import os
import pickle
import queue
import select
import sys
import termios
import threading
import time
import tty
import numpy as np
import mujoco
import mujoco.viewer
from scenario_config import (
    load_scenario,
    validate_scene,
    planning_obstacles_from_config,
)
from contextlib import nullcontext
import onnxruntime as ort
from path_planner import Obstacle, plan_path
from path_planner_turn_aware import plan_path_turn_aware
from path_optimizer_turn_aware import optimize_path
from path_optimizer_warmup_aware import optimize_warmup_waypoint
import path_optimizer_warmup_aware as warmup_optimizer
from path_planner_safety_fixed import segment_clear as fixed_segment_clear
from navigation_state import ReplanningController
from turn_integration import Phase45TurnIntegration
from sim_turn_controller import SimTurnController, SimTurnState, angle_error
from sim_turn_clearance import evaluate_sim_turn_clearance
from command_guard import guarded_command
from robot_tof_sensor import RobotMountedToF
from obstacle_map import ObstacleMap
from safety_map import SafetyMap
from dynamic_replanning import convert_obstacles, make_path

# Phase 3: unify collision safety checks.
import path_planner as base_planner
import path_optimizer_turn_aware as turn_optimizer

base_planner.segment_clear = fixed_segment_clear
turn_optimizer.segment_clear = fixed_segment_clear
warmup_optimizer.segment_clear = fixed_segment_clear

MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene_ramps.xml"
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene_floor_objects.xml"
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene_robot_walk.xml"
MICRODUCK_ROLLERS_XML = "src/mjlab_microduck/robot/microduck/scene_rollers.xml"
MICRODUCK_BALL_XML = "src/mjlab_microduck/robot/microduck/scene_ball.xml"

# BAM M6 defaults — MUST mirror `_BAM_ACTUATOR_KWARGS` in
# src/mjlab_microduck/robot/microduck_constants.py (the actuator every policy is
# trained against in warp). Not imported from there: that module drags in
# mjlab/torch/warp (~16 s import) for a CPU rehearsal script. Locked by
# tests/test_infer_policy_bam.py.
BAM_MOTOR_NAME = "xl330"
BAM_MODEL = "m6"
BAM_KP_FW = 200.0                 # microduck's preserved firmware stiffness
BAM_VIN_RANGE = (6.5, 8.2)        # per-env battery voltage DR in training
BAM_VIN_DROP_GAIN_RANGE = (0.0, 0.2)  # load-dependent sag V_drop = gain * sum|tau|
BAM_VIN_MIN = 6.0                 # floor on effective voltage after sag
BAM_MAX_CURRENT = None            # training runs WITHOUT the firmware current limiter
# Stiff joint-friction constraint, copied from bam.mjlab.BamActuator
# (stiff_frictionloss=True in training): warp has no noslip solver, so BAM
# stiffens frictionloss so a statically-held joint does not creep. Mirrored
# here so CPU and warp apply the same friction budget the same way.
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def load_bam_model(kp_fw: float, vin: float, max_current):
    """Build the BAM M6 model + XL330 voltage-controlled actuator."""
    from bam.model import load_model
    bam_model = load_model(motor_name=BAM_MOTOR_NAME, model=BAM_MODEL)
    bam_model.actuator.kp = kp_fw
    bam_model.actuator.vin = vin
    bam_model.actuator.max_current = max_current if (max_current and max_current > 0) else None
    return bam_model


def load_mujoco_with_bam(xml_path: str, bam_model, timestep: float, vin_drop_gain, vin_min):
    """Load the scene and hand every non-passive actuator to bam.mujoco.MujocoController.

    Mirrors bam.mjlab.BamActuator.edit_spec (what warp does at training time):
    position actuators -> torque motors with the voltage-bounded forcerange,
    joint damping/frictionloss zeroed (BAM rewrites them every step), stiff
    friction constraint. Armature is set on the dofs by MujocoController.
    Returns (model, data, bam_ctrl, actuator_names).
    """
    from bam.mujoco import MujocoController

    kt = bam_model.kt.value
    R = bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R

    spec = mujoco.MjSpec.from_file(xml_path)
    names = []
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))  # MjsJoint expects a (3,1) array
                joint.frictionloss = 0.0
                joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                break

    model = spec.compile()
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    bam_ctrl = MujocoController(bam_model, names, model, data,
                                vin_drop_gain=vin_drop_gain, vin_min=vin_min)
    print(f"BAM {BAM_MODEL} actuators on {len(names)} joints: kt={kt:.4f} R={R:.4f} "
          f"vin={bam_model.actuator.vin:.2f}V kp_fw={bam_model.actuator.kp:.0f} "
          f"vin_drop_gain={vin_drop_gain} vin_min={vin_min} "
          f"max_current={bam_model.actuator.max_current} forcerange=+/-{force_limit:.3f}Nm "
          f"armature={bam_model.actuator.get_extra_inertia():.2e}")
    return model, data, bam_ctrl, names


# Body pose command constants (must match training constants)
BODY_CMD_MAX_Z = 0.03              # ±30 mm
BODY_CMD_MAX_XY = 0.02             # ±20 mm
BODY_CMD_MAX_ANGLE = math.radians(30)  # ±30°

# Ball placement for kick behaviors (must match microduck_ball_kick_env_cfg's
# reset_ball_in_front_of_foot params: ball center in the robot's yaw frame).
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035

# Default pose used by the policy (legs flexed, standing position)
# This is the reference pose that:
# - Actions are offsets from (motor_target = DEFAULT_POSE + action * scale)
# - Joint observations are relative to (obs_joint_pos = current_pos - DEFAULT_POSE)
# STAND2 pose (matches HOME_FRAME in microduck_constants.py): trunk shifted
# ~5mm forward so the CoM sits over the ankle axis. Leg pitch chain leaned
# forward vs the old pose: hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°.
DEFAULT_POSE = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
    0.4530,   # left_ankle
    0.3491,   # neck_pitch
    0.3491,   # head_pitch
    0.0,      # head_yaw
    0.0,      # head_roll
    0.0,      # right_hip_yaw
    0.0873,   # right_hip_roll
    0.4579,   # right_hip_pitch
    0.0049,   # right_knee
    -0.4530,  # right_ankle
], dtype=np.float32)


class TerminalInput:
    """Single-keypress reader on stdin (cbreak mode, background thread).

    Replaces the MuJoCo viewer key_callback: keypresses in the viewer window
    also fire the viewer's built-in visualization shortcuts (frames, labels,
    rendering toggles…), so commands are read from the terminal instead.
    Arrow keys arrive as ESC [ A/B/C/D escape sequences and are translated to
    symbolic names ("up"/"down"/"left"/"right"); letters are lowercased.
    cbreak (not raw) mode keeps ISIG enabled, so Ctrl+C still works.
    """

    _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def __init__(self):
        self._queue = queue.Queue()
        self.enabled = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self.enabled else -1
        self._old_attrs = None
        self._stop = threading.Event()

    def __enter__(self):
        if not self.enabled:
            print("WARNING: stdin is not a TTY — keyboard control disabled")
            return self
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def _read1(self, timeout):
        """Read one byte from stdin, or None on timeout. os.read (unbuffered):
        buffered sys.stdin.read would swallow escape-sequence bytes past what
        select reported ready."""
        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return None
        data = os.read(self._fd, 1)
        return data.decode(errors="ignore") if data else None

    def _reader(self):
        while not self._stop.is_set():
            ch = self._read1(0.1)
            if not ch:
                continue
            if ch == "\x1b":  # possible arrow-key escape sequence
                if self._read1(0.05) == "[":
                    final = self._read1(0.05)
                    name = self._ARROWS.get(final) if final else None
                    if name:
                        self._queue.put(name)
                continue  # bare ESC / unknown sequence: ignore
            self._queue.put(ch.lower() if ch.isalpha() else ch)

    def get_keys(self):
        """Drain and return all pending keys (symbolic names / characters)."""
        keys = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                return keys



def load_ppo_v3_turn_policy(model_path):
    """Load protected PPO V3 turn policy without activating movement."""
    from pathlib import Path
    import onnxruntime as ort

    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(f"PPO V3 model not found: {path}")

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print("[PPO_V3] Model loaded")
    print(f"[PPO_V3] Input: {[(x.name, x.shape) for x in inputs]}")
    print(f"[PPO_V3] Output: {[(x.name, x.shape) for x in outputs]}")

    return session


class PolicyInference:
    def __init__(self, model, data, walking_onnx_path=None, action_scale=1.0, bam_ctrl=None,
                 delay_min_lag=0, delay_max_lag=0,
                 standing_onnx_path=None, switch_threshold=0.05,
                 use_projected_gravity=False, ground_pick_onnx_path=None, ground_pick_period=4.0,
                 sit_onnx_path=None, new_cmd_obs=False, slope_onnx_path=None,
                 sitstand_onnx_path=None,
                 kick_left_onnx_path=None, kick_right_onnx_path=None,
                 roulade_onnx_path=None,
                 kick_duration=3.0, roulade_duration=2.0):
        self.bam_ctrl = bam_ctrl  # bam.mujoco.MujocoController (None = legacy position actuators)
        self.model = model
        self.data = data
        self.action_scale = action_scale
        self.use_projected_gravity = use_projected_gravity
        self.delay_min_lag = delay_min_lag
        self.delay_max_lag = delay_max_lag
        self.switch_threshold = switch_threshold
        # When True: emit the unified 13D command vector and treat head_offset /
        # body_cmd as policy COMMANDS (no add to ctrl, no joint_pos correction).
        # When False: legacy behaviour (3D command, head_offset added to ctrl[5:9]).
        self.new_cmd_obs = new_cmd_obs

        # Load walking policy
        self.walking_session = None
        self.default_gait_period_from_onnx = None
        if walking_onnx_path:
            print(f"Loading walking policy from: {walking_onnx_path}")
            self.walking_session = ort.InferenceSession(walking_onnx_path)
            w_input_shape = self.walking_session.get_inputs()[0].shape
            w_output_shape = self.walking_session.get_outputs()[0].shape
            print(f"Walking policy input: {self.walking_session.get_inputs()[0].name}, shape: {w_input_shape}")
            print(f"Walking policy output: {self.walking_session.get_outputs()[0].name}, shape: {w_output_shape}")

            # Try to read gait period from ONNX metadata
            try:
                model_metadata = self.walking_session.get_modelmeta()
                if hasattr(model_metadata, 'custom_metadata_map') and 'gait_period' in model_metadata.custom_metadata_map:
                    self.default_gait_period_from_onnx = float(model_metadata.custom_metadata_map['gait_period'])
                    print(f"Found gait period in ONNX metadata: {self.default_gait_period_from_onnx:.4f}s")
            except Exception as e:
                print(f"Could not read gait period from ONNX metadata: {e}")

        # Load standing policy
        self.standing_session = None
        if standing_onnx_path:
            print(f"\nLoading standing policy from: {standing_onnx_path}")
            self.standing_session = ort.InferenceSession(standing_onnx_path)
            s_input_shape = self.standing_session.get_inputs()[0].shape
            s_output_shape = self.standing_session.get_outputs()[0].shape
            print(f"Standing policy input: {self.standing_session.get_inputs()[0].name}, shape: {s_input_shape}")
            print(f"Standing policy output: {self.standing_session.get_outputs()[0].name}, shape: {s_output_shape}")
            if self.walking_session:
                print(f"Policy switching threshold: {switch_threshold} (vel command magnitude)")

        # Load ground pick policy
        self.ground_pick_session = None
        self.ground_pick_mode = False
        self.ground_pick_phase = 0.0
        self.ground_pick_period = ground_pick_period
        if ground_pick_onnx_path:
            print(f"\nLoading ground pick policy from: {ground_pick_onnx_path}")
            self.ground_pick_session = ort.InferenceSession(ground_pick_onnx_path)
            gp_input_shape = self.ground_pick_session.get_inputs()[0].shape
            print(f"Ground pick policy input shape: {gp_input_shape}")

        # Load sit policy. Two flavours share the Y key and self.sit_session:
        #  - --sit (is_sitstand=False): the OLD one-way sit policy. Sits
        #    unconditionally on a zero twist command; standing back up is done
        #    by switching back to the standing/walking session.
        #  - --sitstand (is_sitstand=True): the commanded sit↔stand policy.
        #    twist[0] is a posture flag (0=stand, 1=sit); the SAME policy sits,
        #    holds, and stands back up — Y just flips the flag.
        self.sit_session = None
        self.sit_mode = False
        self.is_sitstand = False
        if sit_onnx_path and sitstand_onnx_path:
            raise ValueError("Provide only one of --sit / --sitstand")
        if sit_onnx_path:
            print(f"\nLoading sit policy from: {sit_onnx_path}")
            self.sit_session = ort.InferenceSession(sit_onnx_path)
            sit_input_shape = self.sit_session.get_inputs()[0].shape
            print(f"Sit policy input shape: {sit_input_shape}")
        elif sitstand_onnx_path:
            if not self.new_cmd_obs:
                raise ValueError(
                    "--sitstand policies use the unified 13D command obs (61D); run with --new-cmd-obs"
                )
            print(f"\nLoading sitstand policy from: {sitstand_onnx_path}")
            self.sit_session = ort.InferenceSession(sitstand_onnx_path)
            self.is_sitstand = True
            ss_input_shape = self.sit_session.get_inputs()[0].shape
            print(f"Sitstand policy input shape: {ss_input_shape}")

        # Load slope policy (passive descent, runs with zero twist command)
        self.slope_session = None
        self.slope_mode = False
        if slope_onnx_path:
            print(f"\nLoading slope policy from: {slope_onnx_path}")
            self.slope_session = ort.InferenceSession(slope_onnx_path)
            sl_input_shape = self.slope_session.get_inputs()[0].shape
            print(f"Slope policy input shape: {sl_input_shape}")

        # Episodic behavior policies (kick left/right, roulade). All three use
        # the unified 61D obs layout with an ALL-ZERO 13D command (twist forced
        # ~0 in training, head/body slots zero-padded), so triggering one is a
        # plain session swap; after `duration` seconds control hands back to
        # walking/standing (the behavior policies end standing on their own).
        self.behavior_sessions = {}
        self.behavior_durations = {}
        self.behavior_mode = None       # name of the running behavior, or None
        self.behavior_time_left = 0.0
        for name, path, duration in (
            ("kick_left", kick_left_onnx_path, kick_duration),
            ("kick_right", kick_right_onnx_path, kick_duration),
            ("roulade", roulade_onnx_path, roulade_duration),
        ):
            if not path:
                continue
            if not self.new_cmd_obs:
                raise ValueError(
                    f"--{name.replace('_', '-')} policies use the unified 13D "
                    "command obs (61D); run with --new-cmd-obs"
                )
            print(f"\nLoading {name} policy from: {path}")
            self.behavior_sessions[name] = ort.InferenceSession(path)
            self.behavior_durations[name] = duration
            print(f"{name} policy input shape: {self.behavior_sessions[name].get_inputs()[0].shape}"
                  f"  (auto-return after {duration:.1f}s)")

        # Validate at least one policy loaded. A sitstand policy can run alone
        # (it holds the stand at flag=0), unlike the old one-way sit policy.
        if not self.walking_session and not self.standing_session and not self.is_sitstand:
            raise ValueError("At least one of --walking, --standing or --sitstand must be provided")

        # Determine initial active session and policy
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        elif self.standing_session:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        else:
            # sitstand-only: start standing (posture flag 0).
            self.current_policy = "sit"
            self.ort_session = self.sit_session

        # Get input/output names from active session
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        # Get sensor IDs and body IDs
        self.imu_ang_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.trunk_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

        # Trunk freejoint qpos address (needed to place the ball in the robot's
        # yaw frame) and optional ball freejoint (present in scene_ball.xml).
        _trunk_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self._trunk_qpos_adr = int(model.jnt_qposadr[_trunk_jid])
        _ball_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        if _ball_jid >= 0:
            self.ball_qpos_adr = int(model.jnt_qposadr[_ball_jid])
            self.ball_qvel_adr = int(model.jnt_dofadr[_ball_jid])
        else:
            self.ball_qpos_adr = None
            self.ball_qvel_adr = None

        print(f"Sensors found:")
        print(f"  imu_ang_vel: id={self.imu_ang_vel_id}")
        print(f"Body IDs:")
        print(f"  trunk_base: id={self.trunk_base_id}")

        # Joint information
        self.n_joints = model.nu

        # For robots with passive/interspersed joints (e.g. roller skates), the actuated
        # joints are not contiguous in qpos/qvel. Compute the correct indices from the
        # actuator transmission joint IDs so extraction works for any joint ordering.
        self.joint_qpos_indices = [
            int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
        ]
        self.joint_qvel_indices = [
            int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
        ]

        # Default pose for the policy (flexed legs)
        self.default_pose = DEFAULT_POSE[:self.n_joints]
        print(f"Number of actuators: {self.n_joints}")
        print(f"Default pose: {self.default_pose}")
        print(f"Action scale: {self.action_scale}")

        # Last action (for observation history)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

        # Velocity command [lin_vel_x, lin_vel_y, ang_vel_z] — controls walking / policy switching
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        # Key-press step sizes and limits (overridden per mode in main())
        self.vel_step_x = 0.05
        self.vel_step_y = 0.05
        self.vel_step_ang = 0.3
        self.vel_max_x = 0.3
        self.vel_min_x = -0.3
        self.vel_max_y = 0.3
        self.vel_min_y = -0.3
        self.vel_max_ang = 1.5
        # Body pose command. In new_cmd_obs mode this is 6D
        #   [x, y, z, roll, pitch, yaw] (m, m, m, rad, rad, rad)
        # In legacy mode only [z, pitch, roll] (first 3 indices reused as
        # [z, pitch, roll] to keep the legacy normalization path working).
        self.body_cmd = np.zeros(6 if self.new_cmd_obs else 3, dtype=np.float32)
        # Obs command vector (3D in legacy mode, 13D when new_cmd_obs=True).
        self.command = np.zeros(13 if self.new_cmd_obs else 3, dtype=np.float32)

        # Body pose mode (like head mode but for standing body pose control)
        self.body_pose_mode = False
        self.body_cmd_step_xy = 0.005             # 5 mm per keypress (4 to max)
        self.body_cmd_step_z = 0.01               # 10 mm per keypress (3 to max)
        self.body_cmd_step_angle = math.radians(10) # 10° per keypress (3 to max)

        # Head control mode. In legacy mode head_offset is added on top of
        # ctrl[5:9]; in new_cmd_obs mode it's a *command* fed to the policy.
        # Final per-joint training caps: neck/head_pitch ±1.1, head_yaw ±1.4,
        # head_roll ±0.31. Slider max = widest joint cap; head_roll naturally
        # gets clipped by the policy since it was never trained beyond 0.31.
        self.head_mode = False
        self.head_offset = np.zeros(4, dtype=np.float32)
        if self.new_cmd_obs:
            self.head_max = 1.4
            self.head_step = 0.1
        else:
            self.head_max = 2.5
            self.head_step = 0.83

        # Action delay buffer
        self.use_delay = self.delay_max_lag > 0
        if self.use_delay:
            buffer_size = self.delay_max_lag + 1
            self.action_buffer = [np.zeros(self.n_joints, dtype=np.float32) for _ in range(buffer_size)]
            self.buffer_index = 0
            self.current_lag = np.random.randint(self.delay_min_lag, self.delay_max_lag + 1)
            print(f"\nActuator delay enabled:")
            print(f"  Min lag: {self.delay_min_lag} timesteps")
            print(f"  Max lag: {self.delay_max_lag} timesteps")
            print(f"  Sampled lag: {self.current_lag} timesteps")
            print(f"  Buffer size: {buffer_size}")
        else:
            self.action_buffer = None
            self.current_lag = 0

    def _update_command(self):
        """Update self.command (fed into obs) based on current policy and commands.

        Legacy mode (new_cmd_obs=False): self.command is 3D.
        New mode (new_cmd_obs=True): self.command is 13D:
            [vx, vy, vtheta,                                  ← twist
             neck_pitch, head_pitch, head_yaw, head_roll,     ← head_pose deltas
             body_x, body_y, body_z, body_roll, body_pitch, body_yaw]  ← body_pose
        We keep the existing keyboard mappings: head_offset (4D) drives the head
        slots; body_cmd[0..2] currently mean (Δz, Δpitch, Δroll) and are routed
        into body_pose slots [z, pitch, roll]; x/y/yaw stay 0 (not exposed on
        keyboard yet). ground_pick still owns slots [0..2] for phase encoding.
        """
        if self.new_cmd_obs:
            if self.behavior_mode is not None:
                # Kick/roulade were trained with an all-zero 13D command
                # (twist ~0, head/body slots zero-padded) — feeding stale
                # head/body commands would be out-of-distribution.
                self.command = np.zeros(13, dtype=np.float32)
                return
            cmd = np.zeros(13, dtype=np.float32)
            # twist slot (or phase encoding for ground_pick — overwritten there)
            if self.current_policy in ("walking", "ppo_v3_turn"):
                cmd[0:3] = self.vel_cmd
            elif self.current_policy == "sit" and self.is_sitstand:
                # Sitstand posture flag: 1 = sit, 0 = stand. NOT zeros — the
                # all-zero twist is the STAND command for this policy, which is
                # why feeding it the old sit-policy zero command did nothing.
                cmd[0] = 1.0 if self.sit_mode else 0.0
            # else standing/old-sit/ground_pick: leave twist 0 (ground_pick
            # writes its phase encoding later)
            cmd[3:7]  = self.head_offset
            cmd[7:13] = self.body_cmd  # [x, y, z, roll, pitch, yaw]
            self.command = cmd
            return

        # Legacy 3D command
        if self.current_policy == "walking":
            self.command = self.vel_cmd.copy()
        elif self.current_policy == "sit":
            # Sit was trained with a near-zero twist command.
            self.command = np.zeros(3, dtype=np.float32)
        elif self.current_policy == "standing":
            # Normalize body pose cmd to match training's body_pose_cmd_obs
            self.command = np.array([
                self.body_cmd[0] / BODY_CMD_MAX_Z,
                self.body_cmd[1] / BODY_CMD_MAX_ANGLE,
                self.body_cmd[2] / BODY_CMD_MAX_ANGLE,
            ], dtype=np.float32)
        elif self.current_policy == "slope":
            # Passive descent: zero command (like standing coast)
            self.command = np.zeros(3, dtype=np.float32)
        # ground_pick: command is set directly by update_ground_pick_phase


    def configure_ppo_v3_turn(self, model_path):
        """Load PPO V3 without activating turning."""
        import hashlib

        expected_hash = (
            "70afbeacdf6d520c6c862fbeb2851ceba01e3e661b30304d8d6b7123a3aa6ac1"
        )

        from pathlib import Path
        path = Path(model_path)

        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("PPO V3 model hash mismatch")

        self.ppo_v3_turn_session = load_ppo_v3_turn_policy(path)
        self.ppo_v3_turn_active = False

        print("[PPO_V3] Turn policy configured (inactive)")

    def activate_ppo_v3_turn(self):
        """Switch inference session only; caller must enforce safety."""
        if not getattr(self, "ppo_v3_turn_session", None):
            raise RuntimeError("PPO V3 turn policy not configured")

        if getattr(self, "ppo_v3_turn_active", False):
            return

        self.ppo_v3_turn_active = True
        self.ort_session = self.ppo_v3_turn_session
        self.current_policy = "ppo_v3_turn"

        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        print("[PPO_V3] Turn policy activated")

    def step1309_direct_ppo_to_walking(self):
        """One-shot transition; caller must already authorize walking.

        Does not send a nonzero motor command.
        The existing final arbiter issues the first walking command.
        """
        if (
            not getattr(self, "ppo_v3_turn_active", False)
            or self.current_policy != "ppo_v3_turn"
            or self.walking_session is None
            or not np.all(np.isfinite(self.vel_cmd))
            or float(np.linalg.norm(self.vel_cmd)) > 1e-6
        ):
            raise RuntimeError(
                "STEP1309_DIRECT_TRANSITION_PRECONDITION"
            )

        self.ppo_v3_turn_active = False
        self.ort_session = self.walking_session
        self.current_policy = "walking"
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        self._update_command()

        print(
            "[SIM_DIRECT_POLICY_SWITCH] "
            "from=ppo_v3_turn to=walking "
            "intermediate=NONE cmd=ZERO",
            flush=True,
        )

    def deactivate_ppo_v3_turn(self):
        """Return to existing walking/standing policy selection."""
        if not getattr(self, "ppo_v3_turn_active", False):
            return

        self.ppo_v3_turn_active = False

        magnitude = float(np.linalg.norm(self.vel_cmd))

        if self.standing_session and magnitude <= self.switch_threshold:
            self.ort_session = self.standing_session
            self.current_policy = "standing"
        else:
            self.ort_session = self.walking_session
            self.current_policy = "walking"

        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        print(f"[PPO_V3] Returned to {self.current_policy} policy")

    def _update_policy_session(self):
        """Switch between walking and standing sessions based on vel_cmd magnitude."""

        if getattr(self, "ppo_v3_turn_active", False):
            return

        if not (self.walking_session and self.standing_session):
            return  # Only one policy loaded, no switching
        if self.ground_pick_mode:
            return  # Don't switch during ground pick
        if self.sit_mode:
            return  # Don't switch while sitting
        if self.slope_mode:
            return  # Don't switch during slope mode
        if self.behavior_mode is not None:
            return  # Don't switch during a kick/roulade

        magnitude = float(np.linalg.norm(self.vel_cmd))
        new_policy = "standing" if magnitude <= self.switch_threshold else "walking"
        if new_policy != self.current_policy:
            self.current_policy = new_policy
            self.ort_session = self.standing_session if new_policy == "standing" else self.walking_session
            print(f"Switched to {self.current_policy} policy (vel magnitude: {magnitude:.3f})")
            self._update_command()

    def set_vel_cmd(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        """Set velocity command (used for walking / policy switching)."""
        self.vel_cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
        self._update_policy_session()
        self._update_command()
        print(f"Vel cmd: [{lin_vel_x:.2f}, {lin_vel_y:.2f}, {ang_vel_z:.2f}] [{self.current_policy}]")

    def toggle_body_pose_mode(self):
        """Toggle body pose control mode on/off."""
        self.body_pose_mode = not self.body_pose_mode
        if self.body_pose_mode:
            print("Body pose mode: ON")
            print(f"  UP/DOWN: Δz ±{self.body_cmd_step_z*1000:.0f}mm  (max ±{BODY_CMD_MAX_Z*1000:.0f}mm)")
            print(f"  LEFT/RIGHT: Δpitch ±{math.degrees(self.body_cmd_step_angle):.0f}°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
            print(f"  A/E: Δroll ±{math.degrees(self.body_cmd_step_angle):.0f}°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
            if self.new_cmd_obs:
                print(f"  Z/S: Δyaw ±{math.degrees(self.body_cmd_step_angle):.0f}°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
            print(f"  SPACE: reset body pose to zero")
            self._print_body_cmd()
        else:
            print("Body pose mode: OFF")

    def toggle_slope_mode(self):
        """Toggle slope policy mode on/off (passive descent, zero twist command)."""
        if self.slope_session is None:
            print("Slope unavailable: no --slope policy loaded")
            return
        if self.behavior_mode is not None:
            print(f"Cannot toggle slope mode during {self.behavior_mode}")
            return
        self.slope_mode = not self.slope_mode
        if self.slope_mode:
            self.ort_session = self.slope_session
            self.current_policy = "slope"
            self.set_vel_cmd(0.0, 0.0, 0.0)  # passive descent: zero command
            print("Slope mode: ON (passive descent)")
        else:
            self.vel_cmd = np.zeros(3, dtype=np.float32)
            if self.walking_session:
                self.current_policy = "walking"
                self.ort_session = self.walking_session
            else:
                self.current_policy = "standing"
                self.ort_session = self.standing_session
            self._update_command()
            print("Slope mode: OFF")

    def _print_body_cmd(self):
        if self.new_cmd_obs:
            x, y, z, roll, pitch, yaw = self.body_cmd
            print(
                f"Body cmd: x={x*1000:5.1f}mm  y={y*1000:5.1f}mm  z={z*1000:5.1f}mm  "
                f"roll={math.degrees(roll):5.1f}°  pitch={math.degrees(pitch):5.1f}°  "
                f"yaw={math.degrees(yaw):5.1f}°"
            )
        else:
            print(
                f"Body cmd: z={self.body_cmd[0]*1000:.1f}mm  "
                f"pitch={math.degrees(self.body_cmd[1]):.1f}°  "
                f"roll={math.degrees(self.body_cmd[2]):.1f}°"
            )

    # --- body command bumpers (index differs between legacy 3D and new 6D) ---
    def _body_idx(self, axis: str) -> int:
        """Map an axis name to the body_cmd index, depending on the active mode."""
        if self.new_cmd_obs:
            return {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}[axis]
        return {"z": 0, "pitch": 1, "roll": 2}[axis]

    def bump_body(self, axis: str, delta: float):
        idx = self._body_idx(axis)
        cap = BODY_CMD_MAX_Z if axis == "z" else BODY_CMD_MAX_XY if axis in ("x", "y") else BODY_CMD_MAX_ANGLE
        self.body_cmd[idx] = float(np.clip(self.body_cmd[idx] + delta, -cap, cap))
        self._update_command()
        self._print_body_cmd()

    def quat_rotate_inverse(self, quat, vec):
        """Rotate a vector by the inverse of a quaternion [w, x, y, z]."""
        w = quat[0]
        xyz = quat[1:4]
        t = np.cross(xyz, vec) * 2
        return vec - w * t + np.cross(xyz, t)

    def get_raw_accelerometer(self):
        """Get raw accelerometer reading from MuJoCo sensor."""
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if sensor_id < 0:
            raise ValueError("Sensor 'imu_accel' not found in model")

        sensor_adr = self.model.sensor_adr[sensor_id]
        accel_raw = self.data.sensordata[sensor_adr:sensor_adr+3].copy().astype(np.float32)
        accel_negated = -accel_raw
        mag = np.linalg.norm(accel_negated)
        if mag > 0.1:
            return accel_negated / mag
        else:
            quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
            world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            return self.quat_rotate_inverse(quat, world_gravity)

    def get_projected_gravity(self):
        """Get projected gravity in body frame."""
        quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return self.quat_rotate_inverse(quat, world_gravity)

    def get_base_ang_vel(self):
        """Get base angular velocity from IMU gyro sensor."""
        sensor_adr = self.model.sensor_adr[self.imu_ang_vel_id]
        return self.data.sensordata[sensor_adr:sensor_adr + 3].copy().astype(np.float32)

    def get_joint_pos_relative(self):
        """Get joint positions relative to default pose."""
        current_pos = self.data.qpos[self.joint_qpos_indices].copy().astype(np.float32)
        return current_pos - self.default_pose

    def get_joint_vel(self):
        """Get joint velocities."""
        return self.data.qvel[self.joint_qvel_indices].copy().astype(np.float32)

    def get_observations(self):
        """Collect observations matching policy input.

        Order for velocity/standing task:
        1. base_ang_vel (3D)
        2. raw_accelerometer OR projected_gravity (3D)
        3. joint_pos (14D) - relative to default
        4. joint_vel (14D)
        5. actions (14D) - last action
        6. command (3D) - vel cmd (walking) or normalized body pose cmd (standing)
        Total: 51D
        """
        obs = []

        obs.append(self.get_base_ang_vel())

        if self.use_projected_gravity:
            obs.append(self.get_projected_gravity())
        else:
            obs.append(self.get_raw_accelerometer())

        obs.append(self.get_joint_pos_relative())
        obs.append(self.get_joint_vel())
        obs.append(self.last_action)
        obs.append(self.command)

        return np.concatenate(obs).astype(np.float32)

    def trigger_ground_pick(self):
        """Start one ground pick cycle. Automatically returns to walking when done."""
        if self.ground_pick_session is None:
            print("Ground pick unavailable: no --ground-pick policy loaded")
            return
        if self.ground_pick_mode:
            print("Ground pick already in progress")
            return
        if self.sit_mode:
            print("Cannot ground pick while sitting (press Y to stand up first)")
            return
        if self.behavior_mode is not None:
            print(f"Cannot ground pick during {self.behavior_mode}")
            return
        self.ground_pick_mode = True
        self.ground_pick_phase = 0.0
        self.ort_session = self.ground_pick_session
        self.current_policy = "ground_pick"
        print(f"Ground pick: started (period={self.ground_pick_period:.1f}s)")

    def _end_ground_pick(self):
        """Switch back after a ground pick cycle completes."""
        self.ground_pick_mode = False
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        else:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        self._update_command()
        print(f"Ground pick: done → back to {self.current_policy}")

    def update_ground_pick_phase(self, dt: float):
        """Advance the ground pick phase; auto-exit when one full cycle completes."""
        if not self.ground_pick_mode:
            return
        new_phase = self.ground_pick_phase + dt / self.ground_pick_period
        if new_phase >= 0.7:
            self._end_ground_pick()
            return
        self.ground_pick_phase = new_phase
        # ground_pick policies use the first 3 slots (twist) as phase encoding.
        # Higher slots (head/body) stay at whatever _update_command set them to.
        self.command[0] = np.cos(2 * np.pi * self.ground_pick_phase)
        self.command[1] = np.sin(2 * np.pi * self.ground_pick_phase)
        self.command[2] = 0.0

    def trigger_behavior(self, name):
        """Start an episodic behavior (kick_left / kick_right / roulade).

        The behavior policies were trained to run from a standing start with an
        all-zero command and end standing, so triggering is a session swap; a
        timer hands control back to walking/standing afterwards.
        """
        session = self.behavior_sessions.get(name)
        if session is None:
            print(f"{name} unavailable: no --{name.replace('_', '-')} policy loaded")
            return
        if self.behavior_mode is not None:
            print(f"Cannot start {name}: {self.behavior_mode} already in progress")
            return
        if self.ground_pick_mode:
            print(f"Cannot start {name} during ground pick")
            return
        if self.sit_mode:
            print(f"Cannot start {name} while sitting (press Y to stand up first)")
            return
        if self.slope_mode:
            print(f"Cannot start {name} during slope mode")
            return
        if name in ("kick_left", "kick_right"):
            self._place_ball(name)
        self.behavior_mode = name
        self.behavior_time_left = self.behavior_durations[name]
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.current_policy = name
        self.ort_session = session
        self._update_command()
        print(f"{name}: started (auto-return in {self.behavior_time_left:.1f}s)")

    def _place_ball(self, behavior):
        """Teleport the ball in front of the kicking foot, matching training's
        reset_ball_in_front_of_foot (offset in the robot's yaw frame)."""
        if self.ball_qpos_adr is None or self.ball_qvel_adr is None:
            print("No ball in scene (kick will swing at air)")
            return
        adr = self._trunk_qpos_adr
        x, y = float(self.data.qpos[adr]), float(self.data.qpos[adr + 1])
        qw, qx, qy, qz = self.data.qpos[adr + 3:adr + 7]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        off_y = -BALL_OFFSET_ABS_Y if behavior == "kick_right" else BALL_OFFSET_ABS_Y
        bx = x + math.cos(yaw) * BALL_OFFSET_X - math.sin(yaw) * off_y
        by = y + math.sin(yaw) * BALL_OFFSET_X + math.cos(yaw) * off_y
        self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr + 7] = [bx, by, BALL_RADIUS, 1, 0, 0, 0]
        self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr + 6] = 0.0
        foot = behavior.split("_")[1]
        print(f"Ball placed at ({bx:.3f}, {by:.3f}) in front of the {foot} foot")

    def update_behavior(self, dt: float):
        """Advance the behavior timer; hand back to walking/standing when done."""
        if self.behavior_mode is None:
            return
        self.behavior_time_left -= dt
        if self.behavior_time_left <= 0.0:
            self._end_behavior()

    def _end_behavior(self):
        name = self.behavior_mode
        self.behavior_mode = None
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        elif self.standing_session:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        else:
            # sitstand-only setup: the sitstand policy holds the stand (flag 0).
            self.current_policy = "sit"
            self.ort_session = self.sit_session
        self._update_command()
        print(f"{name}: done → back to {self.current_policy}")

    def toggle_sit(self):
        """Toggle sitting on/off (Y key).

        Old one-way sit policy (--sit): Y off switches back to the standing/
        walking session, which does the standing back up.
        Sitstand policy (--sitstand): Y just flips the posture flag — the SAME
        policy sits, holds the sit, and stands back up gently (trained response
        to a flag flip is a ~2 s glide). The session stays active after
        standing (it holds the stand); a velocity command switches back to
        walking/standing as usual.
        """
        if self.sit_session is None:
            print("Sit unavailable: no --sit/--sitstand policy loaded")
            return
        if self.ground_pick_mode:
            print("Cannot sit during ground pick")
            return
        if self.behavior_mode is not None:
            print(f"Cannot sit during {self.behavior_mode}")
            return
        self.sit_mode = not self.sit_mode
        if self.sit_mode:
            self.vel_cmd = np.zeros(3, dtype=np.float32)
            self.current_policy = "sit"
            self.ort_session = self.sit_session
            print("Sit: ON" + (" (sitstand flag=1; Y again to stand up)" if self.is_sitstand else ""))
        elif self.is_sitstand:
            # Stay on the sitstand session — it stands up itself (flag → 0).
            # Do NOT swap to the standing policy here: it would take over
            # mid-rise from a seated state it wasn't trained on.
            print("Sit: OFF → sitstand policy standing up (flag=0)")
        else:
            if self.standing_session:
                self.current_policy = "standing"
            else:
                self.current_policy = "walking"
            self.ort_session = self.standing_session if self.current_policy == "standing" else self.walking_session
            print(f"Sit: OFF → back to {self.current_policy}")
        self._update_command()

    def toggle_head_mode(self):
        """Toggle head control mode on/off."""
        self.head_mode = not self.head_mode
        if self.head_mode:
            print("Head mode: ON")
            print(f"  Z/S: neck_pitch  |  UP/DOWN: head_pitch  |  LEFT/RIGHT: head_yaw  |  A/E: head_roll  |  SPACE: reset  (max ±{self.head_max:.2f} rad)")
        else:
            print("Head mode: OFF")

    def infer(self):
        """Run policy inference and return action."""
        obs = self.get_observations()
        obs_batch = obs.reshape(1, -1)
        action = self.ort_session.run([self.output_name], {self.input_name: obs_batch})[0]
        action = action.squeeze(0).astype(np.float32)
        self.last_action = action.copy()
        return action

    def apply_action(self, action):
        """Apply action to MuJoCo controls with optional delay."""
        if self.use_delay:
            self.action_buffer[self.buffer_index] = action.copy()
            delayed_index = (self.buffer_index - self.current_lag) % len(self.action_buffer)
            delayed_action = self.action_buffer[delayed_index]
            self.buffer_index = (self.buffer_index + 1) % len(self.action_buffer)
            target_positions = self.default_pose + delayed_action * self.action_scale
        else:
            target_positions = self.default_pose + action * self.action_scale

        # Legacy mode: head_offset is an external perturbation added on top of
        # the policy output. New mode: head_offset is a COMMAND fed into the
        # policy's obs, so the policy itself produces the offset head pose.
        if not self.new_cmd_obs:
            target_positions = target_positions.copy()
            target_positions[5:9] += self.head_offset
        self.set_position_targets(target_positions)

    def set_position_targets(self, target_positions):
        """Send joint position targets to the actuators.

        BAM: the firmware position loop lives in the controller (ctrl is the
        motor TORQUE it writes on update()). Legacy: MuJoCo position actuators.
        """
        if self.bam_ctrl is not None:
            self.bam_ctrl.q_target[:] = target_positions
        else:
            self.data.ctrl[:] = target_positions


# ---------------------------------------------------------------------------
# Odometry anchor points (--odom-anchor-points, --odom-compare)
#
# The robot's contact odometry (microduck/odometry/src/lib.rs) tracks the trunk
# from the lowest of a set of candidate contact points on the soles, expressed
# in each foot's site frame. scripts/odom_anchor_points.py samples those sets
# on the sole mesh and writes them to scripts/odom_anchor_sets.json:
#   v15      the legacy ±27.0 x ±20.6 mm bbox on the site plane (production)
#   alpha4   the flat patch's four corners, on the mesh
#   alpha16  a 4x4 grid over the whole footprint, on the mesh
# --odom-anchor-points draws them; --odom-compare runs a Python replica of the
# estimator per set on the simulated legs + a perfect IMU and scores each
# against MuJoCo's ground-truth trunk pose.
# ---------------------------------------------------------------------------

ODOM_ANCHOR_SETS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odom_anchor_sets.json")
ODOM_SET_RGBA = {
    "v15": (0.1, 0.5, 1.0, 1.0),       # blue
    "alpha4": (0.1, 0.9, 0.3, 1.0),    # green
    "alpha16": (1.0, 0.3, 1.0, 1.0),   # magenta
}
_ODOM_FALLBACK_RGBA = (1.0, 0.8, 0.1, 1.0)


def load_odom_anchor_sets(names="all"):
    """{name: [left (N,3), right (N,3)]} from the generated JSON, in the order
    the file lists them; `names` is 'all' or a comma-separated subset."""
    with open(ODOM_ANCHOR_SETS_JSON) as f:
        sets = json.load(f)["sets"]
    if names != "all":
        wanted = names.split(",")
        missing = [n for n in wanted if n not in sets]
        if missing:
            raise SystemExit(f"unknown odometry anchor set(s) {missing}; have {list(sets)}")
        sets = {n: sets[n] for n in wanted}
    return {n: [np.asarray(v["left"], dtype=np.float64), np.asarray(v["right"], dtype=np.float64)]
            for n, v in sets.items()}


def _quat2mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


class FootFrames:
    """Foot-site poses relative to the trunk, and the trunk's world pose — what
    the robot's odometry gets from FK and the IMU, here from the simulator."""

    def __init__(self, model):
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.sites = []
        for side in ("left", "right"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
            if sid < 0:
                raise RuntimeError(f"scene has no '{side}_foot' site; the odometry needs it")
            self.sites.append(sid)

    def trunk_world(self, data):
        return data.xpos[self.trunk].copy(), data.xmat[self.trunk].reshape(3, 3).copy()

    def feet_in_trunk(self, data):
        p_t, R_t = self.trunk_world(data)
        out = []
        for sid in self.sites:
            out.append((R_t.T @ (data.site_xpos[sid] - p_t), R_t.T @ data.site_xmat[sid].reshape(3, 3)))
        return out

    def feet_world(self, data):
        return [(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()) for sid in self.sites]


class PyOdometry:
    """Line-for-line replica of microduck/odometry/src/lib.rs `Odometry::update`
    for one anchor set: anchor the lowest sole point to the ground, orient the
    trunk by the IMU, move the anchor when another point drops below it and
    holds the claim for SWITCH_CONFIRM_TICKS."""

    SWITCH_MARGIN = -0.010
    SWITCH_CONFIRM_TICKS = 2

    def __init__(self, name, corners):
        self.name = name
        self.corners = corners            # [left (N,3), right (N,3)] in the site frames
        self.anchor_foot = 0
        self.anchor_local = np.zeros(3)
        self.anchor_xy = np.zeros(2)
        self.position = np.zeros(3)
        self.pending = None               # (foot, local, world_xy)
        self.pending_ticks = 0
        self.needs_init = True

    def update(self, feet, R):
        """feet: [(pos, rot)] of the two foot sites in the trunk frame; R: the
        IMU's trunk-in-world rotation (3x3)."""
        if self.needs_init:
            self.anchor_xy = (R @ feet[self.anchor_foot][0])[:2].copy()
            self.needs_init = False
        self._reproject(R, feet)
        cand = self._lowest_corner(R, feet)
        if cand is None:
            self.pending = None
            self.pending_ticks = 0
        else:
            foot, local, world_xy = cand
            if self.pending is not None and self.pending[0] == foot:
                self.pending_ticks += 1
            else:
                self.pending = cand
                self.pending_ticks = 1
            if self.pending_ticks >= self.SWITCH_CONFIRM_TICKS:
                foot, local, world_xy = self.pending
                self.pending = None
                self.anchor_foot = foot
                self.anchor_local = local
                self.anchor_xy = world_xy
                self._reproject(R, feet)
                self.pending_ticks = 0

    def _reproject(self, R, feet):
        pos, rot = feet[self.anchor_foot]
        contact = R @ (rot @ self.anchor_local + pos)
        self.position = np.array([self.anchor_xy[0] - contact[0], self.anchor_xy[1] - contact[1], -contact[2]])

    def _lowest_corner(self, R, feet):
        lowest = -self.SWITCH_MARGIN
        best = None
        for foot, (pos, rot) in enumerate(feet):
            world = self.position + (self.corners[foot] @ rot.T + pos) @ R.T
            i = int(np.argmin(world[:, 2]))
            if world[i, 2] < lowest:
                lowest = world[i, 2]
                best = (foot, self.corners[foot][i].copy(), world[i, :2].copy())
        return best

    def anchor_world(self, feet, R):
        pos, rot = feet[self.anchor_foot]
        return self.position + R @ (rot @ self.anchor_local + pos)


class OdomCompare:
    """Run one PyOdometry per anchor set at the control rate and score each
    against the simulator's trunk pose. The IMU is perfect (true trunk
    rotation), so the sets differ only by their contact geometry."""

    def __init__(self, model, sets):
        self.frames = FootFrames(model)
        self.odoms = [PyOdometry(n, c) for n, c in sets.items()]
        self.start = None
        self.path_len = 0.0
        self.last_gt = None
        self.ticks = 0
        self.max_err = {o.name: 0.0 for o in self.odoms}
        print("\nOdometry comparison: " + ", ".join(f"{o.name} ({len(o.corners[0])} pts/foot)" for o in self.odoms))

    def step(self, data):
        p_t, R_t = self.frames.trunk_world(data)
        feet = self.frames.feet_in_trunk(data)
        if self.start is None:
            self.start = p_t.copy()
            self.last_gt = p_t.copy()
        self.path_len += float(np.linalg.norm((p_t - self.last_gt)[:2]))
        self.last_gt = p_t.copy()
        self.ticks += 1
        for o in self.odoms:
            o.update(feet, R_t)
            self.max_err[o.name] = max(self.max_err[o.name], self._err_xy(o, p_t))

    def _err_xy(self, o, p_t):
        return float(np.linalg.norm(o.position[:2] - (p_t - self.start)[:2]))

    def _line(self, data, final=False):
        p_t, _ = self.frames.trunk_world(data)
        parts = []
        for o in self.odoms:
            exy = self._err_xy(o, p_t) * 1000
            ez = (o.position[2] - p_t[2]) * 1000
            foot = "L" if o.anchor_foot == 0 else "R"
            parts.append(f"{o.name}: xy {exy:5.1f} mm  z {ez:+5.1f} mm  [{foot}]")
        head = f"[odom {self.ticks / 50:.0f}s  path {self.path_len:.2f} m]"
        return head + "  " + "  |  ".join(parts)

    def print_status(self, data):
        print(self._line(data))

    def print_summary(self, data):
        p_t, _ = self.frames.trunk_world(data)
        print("\nOdometry comparison summary (perfect IMU; errors vs MuJoCo trunk pose)")
        print(f"  {self.ticks / 50:.1f} s, ground-truth XY path {self.path_len:.3f} m")
        print(f"  {'set':>8}  {'pts':>3}  {'final xy err':>12}  {'max xy err':>10}  {'xy err / path':>13}  {'final z err':>11}")
        for o in self.odoms:
            exy = self._err_xy(o, p_t)
            ez = o.position[2] - p_t[2]
            rel = f"{100 * exy / self.path_len:.1f} %" if self.path_len > 0.05 else "   n/a"
            print(f"  {o.name:>8}  {len(o.corners[0]):>3}  {exy * 1000:9.1f} mm  {self.max_err[o.name] * 1000:7.1f} mm  {rel:>13}  {ez * 1000:+8.1f} mm")

    def anchors_world(self, data):
        """[(name, world point)] — where each replica believes it stands, drawn
        on the real foot so a wrong anchor shows as a floating sphere."""
        feet = self.frames.feet_world(data)
        out = []
        for o in self.odoms:
            pos, rot = feet[o.anchor_foot]
            out.append((o.name, rot @ o.anchor_local + pos))
        return out


class OdomAnchorOverlay:
    """Draw anchor sets into viewer.user_scn every frame: each set's points on
    both feet in the set's colour (4-point sets also as a rectangle), the
    lowest point of each set in red, the foot-site origin in white, and, with
    --odom-compare, each replica's current anchor as a big sphere."""

    ANCHOR_RGBA = (1.0, 0.1, 0.1, 1.0)
    SITE_RGBA = (1.0, 1.0, 1.0, 1.0)

    def __init__(self, model, sets):
        self.frames = FootFrames(model)
        self.sets = sets
        print("\nOdometry anchor points drawn: " + ", ".join(
            f"{n} ({len(c[0])} pts/foot, {self._colour_name(n)})" for n, c in sets.items())
            + "; red = lowest point of a set, white = foot site")

    @staticmethod
    def _colour_name(name):
        return {"v15": "blue", "alpha4": "green", "alpha16": "magenta"}.get(name, "yellow")

    def draw(self, data, scn, compare=None):
        scn.ngeom = 0
        feet = self.frames.feet_world(data)
        for pos, _rot in feet:
            self._sphere(scn, pos, 0.002, self.SITE_RGBA)
        for name, corners in self.sets.items():
            rgba = ODOM_SET_RGBA.get(name, _ODOM_FALLBACK_RGBA)
            world = [corners[f] @ rot.T + pos for f, (pos, rot) in enumerate(feet)]
            lowest = min(((f, i) for f in range(2) for i in range(len(world[f]))), key=lambda fi: world[fi[0]][fi[1], 2])
            for f in range(2):
                if len(world[f]) == 4:
                    self._polygon(scn, world[f], 0.0006, rgba)
                for i, p in enumerate(world[f]):
                    if (f, i) == lowest:
                        self._sphere(scn, p, 0.0035, self.ANCHOR_RGBA)
                    else:
                        self._sphere(scn, p, 0.002, rgba)
        if compare is not None:
            for name, p in compare.anchors_world(data):
                self._sphere(scn, p, 0.006, ODOM_SET_RGBA.get(name, _ODOM_FALLBACK_RGBA))

    @staticmethod
    def _sphere(scn, pos, radius, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                            np.asarray(pos, dtype=np.float64), np.eye(3).flatten(),
                            np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

    @staticmethod
    def _polygon(scn, pts, width, rgba):
        n = len(pts)
        for i in range(n):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                                 np.asarray(pts[i], dtype=np.float64),
                                 np.asarray(pts[(i + 1) % n], dtype=np.float64))
            scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description="Run ONNX policy in MuJoCo")
    parser.add_argument("--roller", action="store_true", help="Use roller skate robot XML (robot_walk_rollers.xml)")
    parser.add_argument("--scene", type=str, default=None, help="Path to a scene XML, overriding the default pick (e.g. src/mjlab_microduck/robot/microduck/scene_allcollisions.xml)")
    parser.add_argument("--walking", type=str, default=None, help="Path to walking policy ONNX file")
    parser.add_argument("--standing", "-s", type=str, default=None, help="Path to standing policy ONNX file")
    parser.add_argument("--ground-pick", type=str, default=None, help="Path to ground pick policy ONNX file (press G to activate)")
    parser.add_argument("--sit", type=str, default=None, help="Path to OLD one-way sitting policy ONNX file (press Y to sit, Y again switches back to standing/walking policy)")
    parser.add_argument("--sitstand", type=str, default=None, help="Path to sitstand policy ONNX (commanded sit<->stand; press Y to sit, Y again the SAME policy stands back up). Requires --new-cmd-obs. Can run standalone.")
    parser.add_argument("--slope", type=str, default=None, help="Path to slope policy ONNX file (press Y to toggle)")
    parser.add_argument("--kick-left", type=str, default=None, help="Path to LEFT-foot ball kick policy ONNX (press K to trigger). Requires --new-cmd-obs. Loads a scene with a ball.")
    parser.add_argument("--kick-right", type=str, default=None, help="Path to RIGHT-foot ball kick policy ONNX (press L to trigger). Requires --new-cmd-obs. Loads a scene with a ball.")
    parser.add_argument("--roulade", type=str, default=None, help="Path to roulade (forward roll) policy ONNX (press R to trigger). Requires --new-cmd-obs.")
    parser.add_argument("--kick-duration", type=float, default=3.0, help="Seconds a kick policy stays active before handing back to standing/walking (default: 3.0)")
    parser.add_argument("--roulade-duration", type=float, default=2.0, help="Seconds the roulade policy stays active before handing back to standing/walking (default: 2.0, ~the roll itself; the standing/walking policy takes over for the settle)")
    parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Initial linear velocity X command (m/s)")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Initial linear velocity Y command (m/s)")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Initial angular velocity Z command (rad/s)")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Action scale (default: 1.0)")
    parser.add_argument("--raw-accelerometer", action="store_true", help="Use raw accelerometer instead of projected gravity")
    parser.add_argument("--delay", type=int, nargs='*', default=None, help="Enable actuator delay: --delay MIN MAX or --delay LAG")
    parser.add_argument("--debug", action="store_true", help="Print observations and actions")
    parser.add_argument("--save-csv", type=str, default=None, help="Save observations and actions to CSV file")
    parser.add_argument("--record", type=str, default=None, help="Enable recording mode: save observations to pickle file on Ctrl+C")
    parser.add_argument("--switch-threshold", type=float, default=0.05, help="Vel command magnitude threshold for walking/standing switch (default: 0.05)")
    parser.add_argument("--ground-pick-period", type=float, default=4.0, help="Ground pick phase period in seconds (default: 4.0)")
    parser.add_argument("--new-cmd-obs", action="store_true",
                        help="Use the unified 13D command obs layout (twist+head_pose+body_pose). "
                             "Required for policies trained with the new pose-command-tracking setup. "
                             "Old policies (51D obs, head_offset added to ctrl) need this flag OFF.")
    parser.add_argument("--no-bam", action="store_true",
                        help="Use the XML MuJoCo position actuators instead of the BAM M6 "
                             "voltage/friction model the policies are trained against.")
    parser.add_argument("--vin", type=float, default=7.4,
                        help="BAM battery voltage [V]. Training samples per-env in "
                             f"{BAM_VIN_RANGE}; 7.4 = nominal 2S LiPo.")
    parser.add_argument("--vin-drop-gain", type=float, default=0.1,
                        help="BAM load-dependent voltage sag gain [V/Nm], V = vin - gain*sum|tau|. "
                             f"Training samples per-env in {BAM_VIN_DROP_GAIN_RANGE}. 0 disables.")
    parser.add_argument("--kp-fw", type=float, default=BAM_KP_FW,
                        help="BAM firmware P-gain (training uses %(default)s).")
    parser.add_argument("--current-limit", type=float, default=0.0,
                        help="XL330 firmware current limit [A]. With BAM this is the duty-cycle "
                             "limiter of the voltage model (as bam models it); with --no-bam the "
                             "actuator force is clipped to +/- current_limit * kt. Training runs "
                             "WITHOUT a current limit, so the default is off (<=0).")
    parser.add_argument("--foot-friction", type=float, default=None,
                        help="Override the foot sliding friction (mu) to emulate the real grippy "
                             "PU sole. Training used mu~1.0 (range 0.7-1.3); real PU is likely "
                             "~1.5-2.5. e.g. --foot-friction 2.0")
    parser.add_argument("--foot-solref", type=float, default=None,
                        help="Soften foot contact: solref time constant (s) for the foot geoms "
                             "(default sim ~0.02 = stiff/rigid). Larger = softer, to emulate the "
                             "compliant PU sole. e.g. --foot-solref 0.04")
    parser.add_argument("--odom-anchor-points", nargs="?", const="all", default=None, metavar="SETS",
                        help="Draw the odometry's candidate contact points on both feet, from "
                             "scripts/odom_anchor_sets.json: 'all' (default when given bare) or a "
                             "comma-separated subset of v15,alpha4,alpha16. Red = lowest point of a set.")
    parser.add_argument("--odom-compare", action="store_true",
                        help="Run a replica of the robot's contact odometry per anchor set on the "
                             "simulated legs (perfect IMU) and print each set's error against the "
                             "true trunk pose every second and at exit. Drives the robot around to "
                             "compare drift; combine with --odom-anchor-points to see each anchor.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run MuJoCo without opening the Viewer",
    )
    parser.add_argument(
        "--sim-ppo-v3-turn",
        action="store_true",
        help="Enable MuJoCo-only PPO V3 turning experiment (headless only)",
    )
    parser.add_argument(
        "--sim-turn-speed",
        type=float,
        default=0.50,
        help="MuJoCo-only turn speed, 0 < speed <= 0.50 rad/s",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Scenario JSON configuration (optional)",
    )
    args = parser.parse_args()

    if not args.walking and not args.standing and not args.sitstand:
        parser.error("At least one of --walking, --standing or --sitstand must be provided")
    if args.sitstand and not args.new_cmd_obs:
        parser.error("--sitstand policies use the unified 13D command obs (61D); add --new-cmd-obs")
    if (args.kick_left or args.kick_right or args.roulade) and not args.new_cmd_obs:
        parser.error("--kick-left/--kick-right/--roulade policies use the unified 13D command obs (61D); add --new-cmd-obs")
    if (args.kick_left or args.kick_right or args.roulade) and args.roller:
        parser.error("kick/roulade policies are trained on the walking robot, not the roller model")

    # Parse delay arguments
    delay_min_lag = 0
    delay_max_lag = 0
    if args.delay is not None:
        if len(args.delay) == 0:
            delay_min_lag = 1
            delay_max_lag = 2
        elif len(args.delay) == 1:
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[0]
        elif len(args.delay) == 2:
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[1]
        else:
            print("Error: --delay accepts 0, 1, or 2 arguments")
            return

    # Load MuJoCo model. Kick policies get a scene with a ball to kick.
    # --scene overrides everything (any scene whose robot has the standard
    # 14-servo layout works, e.g. scene_allcollisions.xml).
    if args.scene:
        xml_path = args.scene
    elif args.roller:
        xml_path = MICRODUCK_ROLLERS_XML
    elif args.kick_left or args.kick_right:
        xml_path = MICRODUCK_BALL_XML
    else:
        xml_path = MICRODUCK_XML
    print(f"Loading MuJoCo model from: {xml_path}")
    bam_ctrl = None
    if not args.no_bam:
        # Same actuator the policies are trained against in warp (BAM M6 XL330,
        # voltage control + load-dependent friction budget), driven on CPU by
        # bam.mujoco.MujocoController. Voltage DR collapses to fixed --vin /
        # --vin-drop-gain (training samples them per env).
        bam_model = load_bam_model(args.kp_fw, args.vin, args.current_limit)
        vin_drop_gain = args.vin_drop_gain if args.vin_drop_gain > 0 else None
        model, data, bam_ctrl, _bam_names = load_mujoco_with_bam(
            xml_path, bam_model, 0.005, vin_drop_gain, BAM_VIN_MIN)
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)
        print("Legacy MuJoCo position actuators (--no-bam): NOT the actuator the policy was trained with")

    # (--no-bam only) XL330 firmware current limit. The motors saturate current
    # at ~1.75 A; since torque = kt * current, this caps the actuator force at
    # +/- kt * I_max. With BAM the limiter is modelled inside the voltage
    # controller instead (see load_bam_model). kt comes from the bam package.
    if args.no_bam and args.current_limit and args.current_limit > 0:
        from bam.model import load_model
        kt = load_model(motor_name="xl330", model="m6").kt.value
        torque_limit = kt * args.current_limit
        model.actuator_forcerange[:, 0] = -torque_limit
        model.actuator_forcerange[:, 1] = torque_limit
        model.actuator_forcelimited[:] = 1
        print(f"Current limit: {args.current_limit:.2f} A -> torque limit "
              f"+/-{torque_limit:.4f} Nm (kt={kt:.4f})")

    # Foot contact override — emulate the real grippy + soft PU sole to check
    # whether it reproduces the on-robot forward-fall-at-speed. Training used
    # rigid feet at mu~1.0; the real sole is grippier (higher mu) and compliant
    # (softer solref). Applied to the foot collision geoms only.
    if args.foot_friction is not None or args.foot_solref is not None:
        import re as _re
        n_feet = 0
        for g in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if gname and _re.match(r"^(left|right)_foot_collision$", gname):
                if args.foot_friction is not None:
                    model.geom_friction[g, 0] = args.foot_friction  # tangential mu
                if args.foot_solref is not None:
                    model.geom_solref[g, 0] = args.foot_solref       # softer contact
                    model.geom_solref[g, 1] = 1.0
                n_feet += 1
        print(f"Foot override on {n_feet} geoms: "
              f"mu={args.foot_friction if args.foot_friction is not None else 'default'}, "
              f"solref={args.foot_solref if args.foot_solref is not None else 'default'}")

    # Initialize policy
    policy = PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=args.walking,
        action_scale=args.action_scale,
        delay_min_lag=delay_min_lag,
        delay_max_lag=delay_max_lag,
        standing_onnx_path=args.standing,
        switch_threshold=args.switch_threshold,
        use_projected_gravity=not args.raw_accelerometer,
        ground_pick_onnx_path=args.ground_pick,
        ground_pick_period=args.ground_pick_period,
        sit_onnx_path=args.sit,
        new_cmd_obs=args.new_cmd_obs,
        slope_onnx_path=args.slope,
        sitstand_onnx_path=args.sitstand,
        kick_left_onnx_path=args.kick_left,
        kick_right_onnx_path=args.kick_right,
        roulade_onnx_path=args.roulade,
        kick_duration=args.kick_duration,
        roulade_duration=args.roulade_duration,
    )

    # PPO V3: load protected turn policy without activating movement.
    from pathlib import Path
    ppo_v3_model_path = (
        Path(__file__).resolve().parent
        / "phase5_turn_ppo/protected/v3_model99/model_99.onnx"
    )
    policy.configure_ppo_v3_turn(ppo_v3_model_path)
    policy.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)

    # Set realistic wheel bearing friction for roller inference (must be done
    # programmatically — non-zero frictionloss in the XML breaks training)
    if args.roller:
        import re
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and re.match(r"^passive_.*", name):
                dof_adr = model.jnt_dofadr[j]
                model.dof_frictionloss[dof_adr] = 0.003

    # Per-mode velocity command limits matching training ranges
    if args.roller:
        policy.vel_step_x = 0.05      # lin_vel_x step (range -0.5..0.6)
        policy.vel_step_y = 0.0       # no lateral command for rollers
        policy.vel_step_ang = 0.1     # heading error step (range ±1.0 rad)
        policy.vel_max_x = 0.6
        policy.vel_min_x = -0.5       # negative = brake
        policy.vel_max_y = 0.0
        policy.vel_min_y = 0.0
        policy.vel_max_ang = 1.0      # ±1.0 rad heading error
    else:
        policy.vel_max_x = 0.3
        policy.vel_min_x = -0.3
        policy.vel_max_y = 0.2
        policy.vel_min_y = -0.2
        policy.vel_max_ang = 1.5

    # Set initial position to default pose
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.1385 if args.roller else 0.125  # rollers add 13.5mm height
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)   # clears voltage-drop state, q_target = current qpos
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    # Verify observation size
    test_obs = policy.get_observations()
    cmd_dim = 13 if policy.new_cmd_obs else 3
    expected_obs_size = 3 + 3 + policy.n_joints + policy.n_joints + policy.n_joints + cmd_dim
    breakdown = (
        f"3(ang_vel) + 3(proj_grav) + {policy.n_joints}(joint_pos) + "
        f"{policy.n_joints}(joint_vel) + {policy.n_joints}(last_action) + {cmd_dim}(command)"
    )

    if test_obs.size != expected_obs_size:
        print(f"\nWARNING: Observation size mismatch!")
        print(f"  Expected: {expected_obs_size}")
        print(f"  Got: {test_obs.size}")
        print(f"  Breakdown: {breakdown}")
        print()

    print("\n" + "="*80)
    print("MicroDuck Policy Inference")
    print("="*80)
    print(f"Control frequency: 50 Hz (decimation: 4)")
    print(f"Simulation timestep: {model.opt.timestep}s")
    print(f"Observation size: {test_obs.size} (expected: {expected_obs_size})")
    if policy.walking_session:
        print(f"Walking policy: loaded")
    if policy.standing_session:
        print(f"Standing policy: loaded  (body pose: z=±{BODY_CMD_MAX_Z*1000:.0f}mm, pitch/roll=±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    if policy.walking_session and policy.standing_session:
        print(f"  Switch threshold: {policy.switch_threshold} (vel cmd magnitude)")
    if policy.ground_pick_session:
        print(f"Ground pick policy: loaded  (press G)")
    if policy.sit_session:
        kind = "Sitstand" if policy.is_sitstand else "Sit"
        print(f"{kind} policy: loaded  (press Y to toggle)")
    if policy.slope_session:
        print(f"Slope policy: loaded  (press Y to toggle, passive descent)")
    _behavior_keys = {"kick_left": "K", "kick_right": "L", "roulade": "R"}
    for _name in policy.behavior_sessions:
        print(f"{_name} policy: loaded  (press {_behavior_keys[_name]}, "
              f"auto-return after {policy.behavior_durations[_name]:.1f}s)")
    print(f"Active policy: {policy.current_policy}")
    print("Close viewer window to exit")
    print()

    decimation = 4
    control_step_count = 0
    control_dt = decimation * model.opt.timestep

    # Rolling buffer of trunk world-frame xy velocity over the last 1 s, used
    # to print a running average so we can compare commanded vs achieved speed.
    from collections import deque
    _vel_window_steps = max(1, int(round(1.0 / control_dt)))   # ≈ 50 @ 50 Hz
    vel_history = deque(maxlen=_vel_window_steps)

    csv_data = [] if args.save_csv else None
    recorded_observations = [] if args.record else None
    policy_enabled = not args.record
    policy_enable_time = None
    original_kp = None
    if args.record:
        original_kp = model.actuator_gainprm[:, 0].copy()

    # Standby (--record) gains: the legacy path sets the position-actuator kp
    # to 2.0 (XML kp 0.55 ~ kp_fw 200). Under BAM apply the same ratio to the
    # firmware gain so both paths hold with the same relative stiffness.
    _XML_KP_NOMINAL = 0.55
    _STANDBY_KP = 2.0

    def set_standby_gains(on: bool):
        if bam_ctrl is not None:
            bam_ctrl.model.actuator.kp = args.kp_fw * (_STANDBY_KP / _XML_KP_NOMINAL if on else 1.0)
            print(f"  BAM kp_fw set to {bam_ctrl.model.actuator.kp:.0f}")
            return
        for i in range(model.nu):
            kp = _STANDBY_KP if on else original_kp[i]
            model.actuator_gainprm[i, 0] = kp
            model.actuator_biasprm[i, 1] = -kp

    # Cache the trunk freejoint qvel address so the push handler can write to
    # the trunk's world-frame linear velocity directly (qvel[0..3]).
    _freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    _trunk_qvel_adr = int(model.jnt_dofadr[_freejoint_id])
    PUSH_MAX = 1.0   # matches the final velstand push_magnitude curriculum cap

    def random_push():
        """Set the trunk's world-frame xy velocity to a random vector of
        magnitude PUSH_MAX, simulating the push_by_setting_velocity training
        event. Doesn't accumulate — overwrites current linear velocity."""
        import random
        angle = random.uniform(0, 2 * np.pi)
        vx = PUSH_MAX * np.cos(angle)
        vy = PUSH_MAX * np.sin(angle)
        data.qvel[_trunk_qvel_adr + 0] = vx
        data.qvel[_trunk_qvel_adr + 1] = vy
        print(f"PUSH applied: v=[{vx:.2f}, {vy:.2f}, 0] m/s (angle={np.degrees(angle):.0f}°)")

    # Keys come from the TERMINAL (raw stdin, see TerminalInput) — not from the
    # MuJoCo viewer window, whose keypresses also fire built-in visualization
    # shortcuts. `key` is a symbolic name: "up"/"down"/"left"/"right", " ", or
    # a lowercase letter.
    quit_requested = False

    def handle_key(key):
        nonlocal policy_enabled, quit_requested
        try:
            if key == "up":
                if policy.head_mode:
                    policy.head_offset[1] = np.clip(policy.head_offset[1] + policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("z", policy.body_cmd_step_z)
                else:
                    policy.set_vel_cmd(policy.vel_max_x, policy.vel_cmd[1], policy.vel_cmd[2])
            elif key == "down":
                if policy.head_mode:
                    policy.head_offset[1] = np.clip(policy.head_offset[1] - policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("z", -policy.body_cmd_step_z)
                else:
                    policy.set_vel_cmd(policy.vel_min_x, policy.vel_cmd[1], policy.vel_cmd[2])
            elif key == "right":
                if policy.head_mode:
                    policy.head_offset[2] = np.clip(policy.head_offset[2] - policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("pitch", -policy.body_cmd_step_angle)
                elif args.roller:
                    new_ang = np.clip(policy.vel_cmd[2] - policy.vel_step_ang, -policy.vel_max_ang, policy.vel_max_ang)
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], new_ang)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_min_y, policy.vel_cmd[2])
            elif key == "left":
                if policy.head_mode:
                    policy.head_offset[2] = np.clip(policy.head_offset[2] + policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("pitch", policy.body_cmd_step_angle)
                elif args.roller:
                    new_ang = np.clip(policy.vel_cmd[2] + policy.vel_step_ang, -policy.vel_max_ang, policy.vel_max_ang)
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], new_ang)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_max_y, policy.vel_cmd[2])
            elif key == " ":
                if policy.head_mode:
                    policy.head_offset[:] = 0.0
                    policy._update_command()
                    print("Head offset reset to zero")
                elif policy.body_pose_mode:
                    policy.body_cmd[:] = 0.0
                    policy._update_command()
                    print("Body pose cmd reset to zero")
                else:
                    policy.set_vel_cmd(0.0, 0.0, 0.0)
            elif key == "t":
                # Toggle policy inference on/off. When OFF the controller stops
                # querying the ONNX policy and the motors hold the last applied
                # target (no fresh ctrl writes).
                policy_enabled = not policy_enabled
                print(f"Policy inference: {'ON' if policy_enabled else 'OFF (paused)'}")
            elif key == "g":
                policy.trigger_ground_pick()
            elif key == "k":
                policy.trigger_behavior("kick_left")
            elif key == "l":
                policy.trigger_behavior("kick_right")
            elif key == "r":
                policy.trigger_behavior("roulade")
            elif key == "q":
                quit_requested = True
                print("Quit requested")
            elif key == "y":
                # Y toggles whichever aux policy is loaded (--sit or --slope).
                if policy.sit_session is not None:
                    policy.toggle_sit()
                else:
                    policy.toggle_slope_mode()
            elif key == "h":
                policy.toggle_head_mode()
            elif key == "b":
                policy.toggle_body_pose_mode()
            elif key == "p":
                random_push()
            elif key == "a":
                if policy.head_mode:
                    policy.head_offset[3] = np.clip(policy.head_offset[3] + policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("roll", policy.body_cmd_step_angle)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], policy.vel_max_ang)
            elif key == "e":
                if policy.head_mode:
                    policy.head_offset[3] = np.clip(policy.head_offset[3] - policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("roll", -policy.body_cmd_step_angle)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], -policy.vel_max_ang)
            elif key == "z":
                if policy.head_mode:
                    policy.head_offset[0] = np.clip(policy.head_offset[0] + policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode and policy.new_cmd_obs:
                    policy.bump_body("yaw", policy.body_cmd_step_angle)
            elif key == "s":
                if policy.head_mode:
                    policy.head_offset[0] = np.clip(policy.head_offset[0] - policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode and policy.new_cmd_obs:
                    policy.bump_body("yaw", -policy.body_cmd_step_angle)
        except Exception as e:
            print(f"Key press error: {e}")

    print("\nKeyboard controls (type in THIS terminal — the viewer window no longer captures keys):")
    print("  [ Velocity mode (default) ]")
    print("  UP arrow:         increase lin_vel_x (push/accelerate)")
    print("  DOWN arrow:       decrease lin_vel_x (0=coast, negative=brake)")
    if args.roller:
        print("  LEFT/RIGHT arrow: turn left/right (ang_vel_z heading error)")
        print("  A / E:            turn left/right (ang_vel_z, incremental)")
    else:
        print("  LEFT/RIGHT arrow: strafe left/right (lin_vel_y)")
        print("  A / E:            turn left/right (ang_vel_z)")
    print("  SPACE:            coast (zero all commands)")
    print("  T:                toggle policy inference on/off (paused = motors hold last target)")
    print("  G:                trigger ground pick (requires --ground-pick)")
    print("  Y:                toggle sit (with --sit/--sitstand) or slope mode (with --slope)")
    print("  K:                kick with LEFT foot (requires --kick-left)")
    print("  L:                kick with RIGHT foot (requires --kick-right)")
    print("  R:                roulade / forward roll (requires --roulade)")
    print(f"  P:                random push (trunk vel = {PUSH_MAX:.1f} m/s in random direction)")
    print("  Q:                quit")
    print("  [ Body pose mode — press B to toggle ]")
    print(f"  UP/DOWN arrow:    Δz ±10mm  (max ±{BODY_CMD_MAX_Z*1000:.0f}mm)")
    print(f"  LEFT/RIGHT arrow: Δpitch ±10°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    print(f"  A / E:            Δroll ±10°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    if args.new_cmd_obs:
        print(f"  Z / S:            Δyaw ±10°  (new_cmd_obs only, max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    print("  SPACE:            reset body pose to zero")
    print("  [ Head mode — press H to toggle ]")
    print("  Z / S:            neck_pitch ±step")
    print("  UP/DOWN arrow:    head_pitch ±step")
    print("  LEFT/RIGHT arrow: head_yaw ±step")
    print("  A / E:            head_roll ±step")
    print("  SPACE:            reset head offset to zero")

    viewer_context = (
        nullcontext(None)
        if args.headless
        else mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
    )

    with TerminalInput() as term, viewer_context as viewer:
        odom_overlay = OdomAnchorOverlay(model, load_odom_anchor_sets(args.odom_anchor_points))\
            if args.odom_anchor_points else None
        odom_compare = OdomCompare(model, load_odom_anchor_sets()) if args.odom_compare else None
        if odom_overlay is not None and viewer is not None:
            odom_overlay.draw(data, viewer.user_scn, odom_compare)
        if viewer is not None:
            viewer.sync()
        start_time = time.time()

        # Automatic navigation state (MuJoCo simulation time)
        auto_nav_start_time = data.time
        last_auto_phase = -1
        yaw_correction_mode = 0
        yaw_correction_count = 0
        last_yaw_cmd = None

        # Phase 4 safety state controller
        replanning_controller = ReplanningController()
        phase45_adapter = Phase45TurnIntegration(replanning_controller)

        # Simulation-only PPO V3 turn state controller.
        # Initialization does not authorize movement.
        if args.sim_ppo_v3_turn and not (0.0 < args.sim_turn_speed <= 0.50):
            raise ValueError("Invalid --sim-turn-speed: expected > 0 and <= 0.50")
        sim_turn_controller = SimTurnController(
            turn_speed=args.sim_turn_speed if args.sim_ppo_v3_turn else 0.50,
        )
        print(f"[PHASE45_SIM] Configured turn speed: {sim_turn_controller.turn_speed:.2f} rad/s")
        sim_turn_policy_active = False
        sim_turn_start_pose = None
        sim_turn_trajectory = []
        sim_turn_start_yaw = None
        sim_turn_last_state = SimTurnState.WAITING
        sim_ema_rate = None
        sim_ema_prev_time = None
        sim_turn_finish_logged = False
        sim_turn_test_enabled = bool(args.sim_ppo_v3_turn)
        from sim_resume_controller import SimResumeController, SimResumeState
        sim_precision_resume = SimResumeController(heading_tolerance_deg=8.0)
        sim_precision_walk_started = False
        sim_precision_walk_finished = False
        sim_precision_walk_status = None
        sim_precision_walk_start = None
        sim_precision_walk_waypoint_index = None
        sim_precision_walk_start_time = None
        sim_staging_state = 'WAITING'
        sim_staging_target = None
        sim_staging_start = None
        sim_staging_start_time = None
        sim_staging_status = None
        sim_goal_walk_state = "WAITING"
        sim_goal_walk_start = None
        sim_goal_walk_target = None
        sim_goal_walk_start_time = None
        sim_goal_walk_status = None
        sim_second_turn_state = "WAITING"
        sim_staging_stop_time = None
        sim_staging_stop_pose = None

        if sim_turn_test_enabled and not args.headless:
            raise ValueError(
                "--sim-ppo-v3-turn requires --headless"
            )

        print("[PHASE45_SIM] Turn controller initialized")
        print("[PHASE45_SIM] Autonomous turn execution disabled")



        phase4_sensor = None
        # Step 1252: Simulation-only emergency stop latch.
        # Sensor-triggered activation is NOT integrated yet.
        from sim_final_stop_gate import apply_sim_final_stop_gate
        sim_emergency_stop_requested = False
        step1272_shadow_attempted = False
        # Step 1288: MuJoCo-only first-turn experiment.
        # The emergency-stop latch is never cleared.
        from sim_first_turn_gate import SimFirstTurnGate
        from sim_turn_final_arbiter import arbitrate_first_turn
        step1288_gate = SimFirstTurnGate(
            max_duration_s=4.0,
            max_yaw_speed=0.50,
            min_tof_distance_m=0.30,
        )
        step1288_trial_started = False
        step1288_trial_finished = False
        step1288_trial_start_time = None
        step1288_trial_target_yaw = None
        step1288_trial_start_xy = None
        step1288_trial_start_yaw = None
        step1288_trial_reason = None
        from waypoint_heading_alignment import WP1HeadingAligner
        step1300_aligner = WP1HeadingAligner()
        step1300_started = False
        step1300_start_xy = None
        step1300_start_yaw = None
        step1300_target = (0.20, 1.00)
        step1300_next = (0.95, 1.00)
        step1300_last_cmd = (0.0, 0.0, 0.0)
        step1305_hold_until = None
        step1305_hold_start_xy = None
        step1305_hold_start_yaw = None
        step1305_walk_revalidated = False
        step1305_hold_samples = []
        step1301_started = False
        step1301_finished = False
        step1301_reason = None
        step1301_start_time = None
        step1301_start_xy = None
        step1301_direction = None
        step1301_travel_m = 0.0
        step1301_cross_track_m = 0.0
        step1301_last_motor_command = (0.0, 0.0, 0.0)
        # Step 1337-B: separate VIA trial, WP1 state preserved.
        step1337b_state = None
        step1337b_hold_time = None
        step1337b_origin = None
        step1337b_walk_time = None
        step1337b_reason = None
        step1337b_last_sample = -1.0
        step1340s_max_drift = 0.0
        step1340s_max_error = 0.0
        step1341_walk_origin = None
        # Step 1349-R: simulation-only VIA right-turn trial.
        step1349_mode = "WAITING"
        step1349_controller = SimTurnController(
            turn_speed=0.50,
            turn_tolerance_deg=8.0,
            settling_duration=0.50,
            max_turn_duration=4.0,
        )
        step1349_origin = None
        step1349_target_yaw = None
        step1349_start_time = None
        step1349_reason = None
        step1349_max_drift = 0.0
        step1349_last_sample = -1.0
        step1356_state = "WAITING"
        step1356_start_time = None
        step1356_origin = None
        step1356_reason = None
        step1356_last_sample = -1.0
        step1356_command = (0.0, 0.0, 0.0)
        step1356_max_drift = 0.0
        step1376_heading_excursion_since = None
        step1376_heading_excursion_count = 0
        step1376_max_heading_error = 0.0
        step1349_command = (0.0, 0.0, 0.0)

        # Step 1258: Diagnostic stop-hold state.
        sim_stop_hold_start_time = None
        sim_stop_hold_start_xy = None
        sim_stop_hold_last_sample = -1
        sim_stop_hold_max_displacement = 0.0
        sim_stop_hold_min_point_box_distance = float("inf")
        # Step 1244: Persist detections across replanning events.
        obstacle_map = ObstacleMap()
        safety_map = SafetyMap(safety_radius=0.20)
        phase4_stop_threshold = 0.60
        phase4_stop_time = None
        phase4_stop_origin = None
        phase4_replan_ready = False
        sim_resume_obstacles = None
        sim_resume_path_checked = False
        sim_resume_path_safe = False



        # Yaw smoothing history
        from collections import deque
        yaw_history = deque()
        yaw_smoothing_window = 0.10

        # Step 9-7-10: Obstacle contact tracking
        # Step 1237: Discover every obstacle geometry.
        obstacle_geom_ids = {}
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if geom_name and geom_name.startswith("obstacle_"):
                obstacle_geom_ids[geom_id] = geom_name

        if not obstacle_geom_ids:
            raise RuntimeError("No obstacle geoms found")

        # Step 1240: Identify robot geometry by body hierarchy.
        trunk_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
        )
        if trunk_body_id < 0:
            raise RuntimeError("trunk_base body not found")

        robot_geom_ids = set()
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            while body_id > 0:
                if body_id == trunk_body_id:
                    robot_geom_ids.add(geom_id)
                    break
                body_id = int(model.body_parentid[body_id])

        if not robot_geom_ids:
            raise RuntimeError("No robot geoms found")

        if robot_geom_ids.intersection(obstacle_geom_ids):
            raise RuntimeError("Robot and obstacle geom IDs overlap")

        print(
            f"[CONTACT] Robot geoms: {len(robot_geom_ids)} "
            f"obstacle geoms: {len(obstacle_geom_ids)}"
        )
        obstacle_contact_events = 0
        obstacle_contact_steps = 0
        obstacle_contact_previous = False
        obstacle_first_contact_time = None

        print(f"[CONTACT] Monitored obstacles: {obstacle_geom_ids}")

        # Goal navigation settings
        nav_goal_x = 1.0
        nav_goal_y = 0.5
        nav_goal_tolerance = 0.03 if sim_turn_test_enabled else 0.10
        nav_timeout = 80.0
        nav_mode = "WALK"
        last_nav_command = None

        # Phase 1: A* automatic path planning
        # Obstacle information is known in advance.
        # Planning logic is independent of MuJoCo.
        if args.config:
            config = load_scenario(args.config)

            validate_scene(model, config)

            # Phase 4: initialize ToF only after scenario is loaded.
            if not config.get(
                "phase4", {}
            ).get("initial_obstacles_known", True):
                phase4_sensor = RobotMountedToF(
                    model,
                    body_name="trunk_base",
                    max_distance=2.0,
                )
                print("[PHASE4] ToF safety detection enabled")
                print("[PHASE4] Stop threshold: 0.60 m")

            start = (
                float(config["start"]["x"]),
                float(config["start"]["y"]),
            )

            if start != (0.0, 0.0):
                raise ValueError(
                    "Non-zero START is not yet supported"
                )

            nav_goal_x = float(config["goal"]["x"])
            nav_goal_y = float(config["goal"]["y"])

            nav_goal_tolerance = float(
                config["navigation"]["goal_tolerance"]
            )
            nav_timeout = float(
                config["navigation"]["timeout"]
            )

            # Phase 4: Physical obstacles remain in MuJoCo,
            # but unknown obstacles are hidden from initial A*.
            initial_obstacles_known = config.get(
                "phase4", {}
            ).get("initial_obstacles_known", True)

            if initial_obstacles_known:
                planning_obstacles = (
                    planning_obstacles_from_config(config)
                )
            else:
                planning_obstacles = []
                print("[PHASE4] Initial obstacle knowledge disabled")
                print("[PHASE4] Initial A* known obstacles: 0")

            resolution = float(
                config["planner"]["resolution"]
            )
            margin = float(
                config["planner"]["margin"]
            )

            print(
                f"[SCENARIO] Loaded: "
                f"{config['scenario_id']}"
            )
            print("[SCENARIO] MuJoCo geometry validated")

        else:
            start = (0.0, 0.0)
            planning_obstacles = [
                Obstacle(0.50, 0.25, 0.20, 0.20)
            ]
            resolution = 0.05
            margin = 0.15

        # Phase 3: Scenario C uses predicted WARMUP position.
        planning_start = start
        planning_yaw = 0.0

        if args.config is not None and config["scenario_id"] == "scenario_c":
            from path_optimizer_warmup_aware import predict_warmup_state

            planning_start, planning_yaw = predict_warmup_state(
                start,
                initial_yaw=0.0,
                warmup_time=2.02,
                warmup_speed=0.10,
            )

            if not fixed_segment_clear(
                start, planning_start, planning_obstacles, margin
            ):
                raise RuntimeError("Unsafe START to WARMUP segment")

            print("[WARMUP_MULTI] Predicted position:", planning_start)

        # Experimental: turn-aware A* followed by waypoint optimization.
        grid_path = plan_path_turn_aware(
            start=planning_start,
            goal=(nav_goal_x, nav_goal_y),
            obstacles=planning_obstacles,
            initial_yaw=planning_yaw,
            resolution=resolution,
            margin=margin,
        )

        planned_path = optimize_path(
            grid_path,
            planning_obstacles,
            margin=margin,
            initial_yaw=planning_yaw,
        )

        print("[TURN_AWARE] Optimized navigation path")
        print(f"[TURN_AWARE] Grid points: {len(grid_path)}")
        print(f"[TURN_AWARE] Optimized points: {len(planned_path)}")

        # Experiment: automatically generate warmup-aware path.
        if args.config is not None and config["scenario_id"] in ("baseline", "scenario_a"):
            warmup_result = optimize_warmup_waypoint(
                start=start,
                goal=(nav_goal_x, nav_goal_y),
                obstacles=planning_obstacles,
                margin=margin,
                resolution=resolution,
                initial_yaw=0.0,
                warmup_time=2.02,
                warmup_speed=0.10,
                min_segment_length=0.20,
                search_padding=0.50,
            )

            planned_path = warmup_result["path"]

            print("[WARMUP_AUTO] Automatic path generated")
            print(
                f"[WARMUP_AUTO] Candidates: "
                f"{warmup_result['candidate_count']}"
            )
            print(
                f"[WARMUP_AUTO] Predicted cost: "
                f"{warmup_result['predicted_cost']:.2f}s"
            )
            print(
                f"[WARMUP_AUTO] WP1={planned_path[1]}"
            )

        if args.config is not None and config["scenario_id"] == "scenario_c":
            for a, b in zip(planned_path, planned_path[1:]):
                if not fixed_segment_clear(
                    a, b, planning_obstacles, margin
                ):
                    raise RuntimeError("Unsafe Phase 3 planned segment")

            print("[WARMUP_MULTI] All planned segments safe")

        # Exclude START; navigation targets are waypoints and GOAL.
        nav_waypoints = planned_path[1:]

        if not nav_waypoints:
            raise RuntimeError("A* generated no navigation targets")

        nav_waypoint_index = 0

        print("[NAV] A* automatic waypoints configured")
        print(f"[NAV] Generated {len(nav_waypoints)} targets")

        for i, (wx, wy) in enumerate(nav_waypoints):
            print(f"[NAV] WP{i + 1}: X={wx:.3f} Y={wy:.3f}")

        auto_phase_start_time = data.time
        turn_start_yaw = None
        last_pose_log_time = data.time

        if args.record:
            policy_enable_time = start_time + 1.0
            print("Recording mode: policy will be enabled after 1 second standby")
            set_standby_gains(True)
            print("  Standby mode: kp set to 2.0 (XML units)")

        try:
            prev_step_time = time.time()

            while (viewer is None or viewer.is_running()) and not quit_requested:
                step_start = time.time()

                for key in term.get_keys():
                    handle_key(key)

                # === AUTO NAVIGATION START ===

                sim_elapsed = data.time - auto_nav_start_time

                # Current position
                nav_x = float(data.qpos[qpos_adr])
                nav_y = float(data.qpos[qpos_adr + 1])

                # Current yaw
                quat_nav = data.qpos[qpos_adr + 3:qpos_adr + 7]
                qw, qx, qy, qz = quat_nav

                current_yaw = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz)
                )

                # Smooth yaw using circular mean
                yaw_history.append((data.time, current_yaw))

                while (
                    yaw_history
                    and data.time - yaw_history[0][0]
                    > yaw_smoothing_window
                ):
                    yaw_history.popleft()

                avg_sin = sum(math.sin(v) for _, v in yaw_history)
                avg_cos = sum(math.cos(v) for _, v in yaw_history)
                smooth_yaw = math.atan2(avg_sin, avg_cos)

                # Step 9-7: Active waypoint navigation
                nav_goal_x, nav_goal_y = nav_waypoints[nav_waypoint_index]

                dx = nav_goal_x - nav_x
                dy = nav_goal_y - nav_y
                distance = math.hypot(dx, dy)

                # Waypoint reached / final goal reached / timeout
                if nav_mode not in ("DONE", "TIMEOUT"):
                    if distance <= nav_goal_tolerance:
                        if nav_waypoint_index < len(nav_waypoints) - 1:
                            print(
                                f"[NAV] WAYPOINT REACHED! "
                                f"WP{nav_waypoint_index + 1} "
                                f"t={sim_elapsed:.2f}s "
                                f"distance={distance:.3f}m"
                            )

                            nav_waypoint_index += 1
                            nav_goal_x, nav_goal_y = nav_waypoints[nav_waypoint_index]
                            nav_mode = "TURN"

                            print(
                                f"[NAV] NEXT TARGET: "
                                f"WP{nav_waypoint_index + 1} "
                                f"({nav_goal_x:.2f}, {nav_goal_y:.2f})"
                            )

                            # Recalculate for the new target immediately
                            dx = nav_goal_x - nav_x
                            dy = nav_goal_y - nav_y
                            distance = math.hypot(dx, dy)

                        else:
                            nav_mode = "DONE"
                            print(
                                f"[NAV] GOAL REACHED! "
                                f"t={sim_elapsed:.2f}s "
                                f"distance={distance:.3f}m"
                            )

                    elif sim_elapsed >= nav_timeout:
                        nav_mode = "TIMEOUT"
                        print(
                            f"[NAV] TIMEOUT "
                            f"WP{nav_waypoint_index + 1} "
                            f"distance={distance:.3f}m"
                        )

                # Direction and heading error for active target
                target_yaw = math.atan2(dy, dx)

                heading_error = math.atan2(
                    math.sin(target_yaw - smooth_yaw),
                    math.cos(target_yaw - smooth_yaw)
                )

                error_deg = math.degrees(heading_error)

                # Navigation state machine
                if nav_mode == "WALK":
                    if sim_elapsed >= 2.0:
                        nav_mode = "TURN"
                        print("[NAV] WALK -> TURN after 2s warmup")

                if nav_mode == "TURN":
                    if abs(error_deg) <= 8.0:
                        nav_mode = "DRIVE"
                        print(
                            f"[NAV] TURN -> DRIVE "
                            f"error={error_deg:.1f}deg"
                        )

                elif nav_mode == "DRIVE":
                    if abs(error_deg) > 35.0:
                        nav_mode = "TURN"
                        print(
                            f"[NAV] DRIVE -> TURN "
                            f"error={error_deg:.1f}deg"
                        )

                # Decide velocity command
                if nav_mode == "WALK":
                    nav_command = (0.30, 0.0, 0.0)

                elif nav_mode == "TURN":
                    turn_direction = 1.0 if error_deg > 0 else -1.0
                    nav_command = (0.0, 0.0, 0.50 * turn_direction)

                elif nav_mode == "DRIVE":
                    if error_deg > 5.0:
                        turn_cmd = 0.40
                    elif error_deg < -5.0:
                        turn_cmd = -0.40
                    else:
                        turn_cmd = 0.0

                    nav_command = (0.30, 0.0, turn_cmd)

                else:
                    nav_command = (0.0, 0.0, 0.0)

                # Update only when command changes
                # Phase 4: Detect unknown obstacles before
                # sending navigation commands to the policy.
                if (
                    phase4_sensor is not None
                    and replanning_controller.velocity_allowed()
                ):
                    tof_distance = phase4_sensor.measure(model, data)

                    if (
                        tof_distance is None
                        or tof_distance <= phase4_stop_threshold
                    ):
                        if tof_distance is None:
                            print("[PHASE4] TOF NO RETURN: safety stop")
                        else:
                            print(
                                f"[PHASE4] OBSTACLE DETECTED "
                                f"distance={tof_distance:.3f}m"
                            )

                        replanning_controller.obstacle_detected()
                        phase4_stop_time = data.time
                        phase4_stop_origin = (
                            float(data.qpos[0]),
                            float(data.qpos[1]),
                        )
                        print("[PHASE4] STATE: RUNNING -> STOPPED")

                        # Step 142: Generate and register a new route.
                        # Keep STOPPED until resume control is verified.
                        if tof_distance is not None:
                            try:
                                body_id = mujoco.mj_name2id(
                                    model,
                                    mujoco.mjtObj.mjOBJ_BODY,
                                    "trunk_base",
                                )

                                rotation = data.xmat[body_id].reshape(3, 3)
                                sensor_origin = (
                                    data.xpos[body_id]
                                    + rotation @ np.array(
                                        [0.04, 0.0, 0.08]
                                    )
                                )
                                sensor_direction = (
                                    rotation @ np.array([1.0, 0.0, 0.0])
                                )

                                # Reuse the persistent obstacle map.
                                hit_point = obstacle_map.register(
                                    sensor_origin,
                                    sensor_direction,
                                    tof_distance,
                                )

                                # Preserve prior safety exclusion zones.
                                safety_map.add_detection(hit_point)

                                estimated_obstacles = convert_obstacles(
                                    safety_map
                                )
                                sim_resume_obstacles = estimated_obstacles

                                # Use final GOAL, not the active waypoint.
                                final_goal = (
                                    float(config["goal"]["x"]),
                                    float(config["goal"]["y"]),
                                )

                                new_path = make_path(
                                    (nav_x, nav_y),
                                    final_goal,
                                    current_yaw,
                                    estimated_obstacles,
                                )

                                if len(new_path) < 2:
                                    raise RuntimeError(
                                        "Replanned path has no targets"
                                    )

                                # Replace all previous navigation targets.
                                nav_waypoints = new_path[1:]
                                nav_waypoint_index = 0
                                nav_goal_x, nav_goal_y = nav_waypoints[0]
                                nav_mode = "TURN"

                                phase4_replan_ready = True
                                # Phase 4/5: request only; never issue movement.
                                phase45_dx = nav_goal_x - nav_x
                                phase45_dy = nav_goal_y - nav_y
                                phase45_target_yaw = math.atan2(
                                    phase45_dy, phase45_dx
                                )
                                phase45_yaw_error = math.atan2(
                                    math.sin(phase45_target_yaw - current_yaw),
                                    math.cos(phase45_target_yaw - current_yaw),
                                )
                                phase45_direction = (
                                    "LEFT" if phase45_yaw_error >= 0
                                    else "RIGHT"
                                )
                                phase45_target_deg = min(
                                    abs(math.degrees(phase45_yaw_error)),
                                    90.0,
                                )

                                if phase45_target_deg > 0.0:
                                    phase45_requested = phase45_adapter.request(
                                        phase45_direction,
                                        phase45_target_deg,
                                        path_ready=phase4_replan_ready,
                                    )
                                    print(
                                        f"[PHASE45] TURN REQUEST "
                                        f"direction={phase45_direction} "
                                        f"target={phase45_target_deg:.2f}deg "
                                        f"accepted={phase45_requested}"
                                    )
                                    print(
                                        f"[PHASE45] TURN_PENDING "
                                        f"state={phase45_adapter.turn.state.value}"
                                    )

                                    # A forward-only ToF cannot verify the
                                    # full swept area or obstacle shape.
                                    # Evidence flags remain False.
                                    phase45_clearance = phase45_adapter.evaluate(
                                        None,
                                        shape_verified=False,
                                        swept_area_observed=False,
                                        distance_reference_verified=False,
                                        evidence_fresh=False,
                                    )

                                    print(
                                        f"[PHASE45] SAFETY GATE: "
                                        f"{'PASSED' if phase45_clearance else 'BLOCKED'}"
                                    )
                                    print(
                                        f"[PHASE45] MOVEMENT AUTHORIZED: "
                                        f"{phase45_adapter.movement_authorized()}"
                                    )


                                # Simulation-only diagnostic; no movement authorization.
                                sim_turn_safe, sim_turn_clearance, sim_turn_reason = (
                                    evaluate_sim_turn_clearance(
                                        model,
                                        data,
                                        clearance_margin=0.10,
                                        robot_radius=0.10,
                                    )
                                )
                                print(
                                    f"[PHASE45_SIM] TURN CLEARANCE "
                                    f"safe={sim_turn_safe} "
                                    f"clearance={sim_turn_clearance} "
                                    f"reason={sim_turn_reason}"
                                )
                                # MuJoCo-only turn start request; no motor movement here.
                                if sim_turn_test_enabled:
                                    sim_turn_start_allowed = (
                                        phase4_replan_ready
                                        and phase45_requested
                                        and sim_turn_safe
                                        and sim_turn_controller.state is SimTurnState.WAITING
                                    )
                                    if sim_turn_start_allowed:
                                        sim_turn_started = sim_turn_controller.start(
                                            current_time=float(data.time),
                                            target_yaw=phase45_target_yaw,
                                        )
                                        print(
                                            f"[PHASE45_SIM] TURN START READY "
                                            f"started={sim_turn_started} "
                                            f"state={sim_turn_controller.state.value}"
                                        )
                                    else:
                                        print('[PHASE45_SIM] TURN START BLOCKED')


                                print(
                                    f"[PHASE4] REPLAN READY "
                                    f"points={len(new_path)}"
                                )
                                print(
                                    f"[PHASE4] New targets: "
                                    f"{len(nav_waypoints)}"
                                )
                                print(
                                    f"[PHASE4] First target: "
                                    f"({nav_goal_x:.3f}, "
                                    f"{nav_goal_y:.3f})"
                                )
                                print(
                                    "[PHASE4] STATE: STOPPED "
                                    "(resume disabled)"
                                )

                            except Exception as exc:
                                import traceback
                                print('[PHASE45_DIAG] TRACEBACK START')
                                traceback.print_exc()
                                print('[PHASE45_DIAG] TRACEBACK END')
                                phase4_replan_ready = False
                                print(
                                    f"[PHASE4] REPLAN FAILED: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                                print(
                                    "[PHASE4] STATE: STOPPED "
                                    "(no safe route confirmed)"
                                )

                # MuJoCo-only PPO V3 turn experiment.
                # The production navigation gate remains unchanged.
                sim_turn_command = None

                if sim_turn_test_enabled and sim_turn_controller.state in (
                    SimTurnState.TURNING,
                    SimTurnState.SETTLING,
                ):
                    # Abort on obstacle contact, abnormal height or invalid pose.
                    sim_fallen = (
                        not math.isfinite(float(data.qpos[2]))
                        or float(data.qpos[2]) < 0.09
                    )

                    sim_contact = bool(obstacle_contact_previous)

                    # Recheck the simulation clearance during rotation.
                    sim_dynamic_safe, _, _ = evaluate_sim_turn_clearance(
                        model,
                        data,
                        clearance_margin=0.10,
                        robot_radius=0.10,
                    )

                    if not sim_dynamic_safe:
                        sim_turn_controller.abort("CLEARANCE_LOST")

                    if sim_turn_controller.state is not SimTurnState.ABORTED:
                        sim_wz = sim_turn_controller.update(
                            current_time=float(data.time),
                            current_yaw=float(current_yaw),
                            contact=sim_contact,
                            fallen=sim_fallen,
                        )
                    else:
                        sim_wz = 0.0

                    if sim_turn_controller.state is SimTurnState.TURNING:
                        if not sim_turn_policy_active:
                            sim_ema_rate = None
                            sim_ema_prev_time = None
                            policy.activate_ppo_v3_turn()
                            sim_turn_policy_active = True
                            print("[PHASE45_SIM] PPO V3 TURN ACTIVE")
                            sim_turn_start_pose = (
                                float(data.qpos[qpos_adr]),
                                float(data.qpos[qpos_adr + 1]),
                            )
                            sim_turn_start_yaw = float(current_yaw)
                            sim_turn_trajectory = [
                                (
                                    float(data.time),
                                    sim_turn_start_pose[0],
                                    sim_turn_start_pose[1],
                                )
                            ]
                            print(
                                f"[SIM_TURN_DRIFT] START "
                                f"pos={sim_turn_start_pose} "
                                f"yaw={math.degrees(sim_turn_start_yaw):.2f}deg"
                            )

                        sim_turn_command = (0.0, 0.0, float(sim_wz))

                    elif sim_turn_controller.state is SimTurnState.SETTLING:
                        # Keep PPO active during settling and apply EMA-IMU PD.
                        sim_raw_rate = math.degrees(float(policy.get_base_ang_vel()[2]))
                        sim_now = float(data.time)
                        sim_pd_error = math.degrees(angle_error(
                            sim_turn_controller.target_yaw, float(current_yaw)
                        ))
                        if not all(math.isfinite(v) for v in (
                            sim_raw_rate, sim_now, sim_pd_error
                        )):
                            sim_turn_controller.abort('EMA_NONFINITE')
                            sim_turn_command = (0.0, 0.0, 0.0)
                        elif sim_ema_rate is None:
                            sim_ema_rate = sim_raw_rate
                            sim_ema_prev_time = sim_now
                            sim_pd_wz = max(-0.12, min(
                                0.12, 0.015 * sim_pd_error - 0.001 * sim_ema_rate
                            ))
                            sim_turn_command = (0.0, 0.0, sim_pd_wz)
                        else:
                            sim_dt = sim_now - sim_ema_prev_time
                            if not (0.0 < sim_dt <= 0.10):
                                sim_turn_controller.abort('EMA_INVALID_DT')
                                sim_turn_command = (0.0, 0.0, 0.0)
                            else:
                                sim_alpha = sim_dt / (0.08 + sim_dt)
                                sim_ema_rate += sim_alpha * (
                                    sim_raw_rate - sim_ema_rate
                                )
                                sim_ema_prev_time = sim_now
                                sim_pd_wz = max(-0.12, min(
                                    0.12, 0.015 * sim_pd_error - 0.001 * sim_ema_rate
                                ))
                                sim_turn_command = (0.0, 0.0, sim_pd_wz)
                    else:
                        sim_turn_command = (0.0, 0.0, 0.0)

                        if sim_turn_policy_active:
                            policy.set_vel_cmd(0.0, 0.0, 0.0)
                            policy.deactivate_ppo_v3_turn()
                            sim_turn_policy_active = False

                    if sim_turn_controller.state is not sim_turn_last_state:
                        print(
                            f"[PHASE45_SIM] STATE "
                            f"{sim_turn_last_state.value} -> "
                            f"{sim_turn_controller.state.value} "
                            f"yaw={math.degrees(current_yaw):.2f}deg"
                        )
                        sim_turn_last_state = sim_turn_controller.state

                # When settling is complete, mark the turn-only test ready.
                if (
                    sim_turn_test_enabled
                    and sim_turn_controller.state is SimTurnState.SETTLING
                ):
                    sim_turn_controller.update(
                        current_time=float(data.time),
                        current_yaw=float(current_yaw),
                        contact=bool(obstacle_contact_previous),
                        fallen=(
                            not math.isfinite(float(data.qpos[2]))
                            or float(data.qpos[2]) < 0.09
                        ),
                    )
                sim_precision_command = None
                if sim_precision_walk_started and not sim_precision_walk_finished:
                    walk_x = float(data.qpos[qpos_adr])
                    walk_y = float(data.qpos[qpos_adr + 1])
                    walk_height = float(data.qpos[2])

                    walk_fallen = (
                        not math.isfinite(walk_height)
                        or walk_height < 0.09
                    )
                    walk_contact = bool(obstacle_contact_previous)

                    walk_reason = None
                    if nav_waypoint_index != sim_precision_walk_waypoint_index:
                        walk_reason = "WP1_REACHED"
                    elif nav_mode in ("DONE", "TIMEOUT"):
                        walk_reason = "NAV_" + nav_mode
                    elif float(data.time) - sim_precision_walk_start_time > 9.0:
                        walk_reason = "WALK_TIMEOUT"
                    elif not all(math.isfinite(v) for v in (walk_x, walk_y)):
                        walk_reason = "INVALID_POSE"
                    elif walk_fallen:
                        walk_reason = "FALL"
                    elif walk_contact:
                        walk_reason = "CONTACT"
                    else:
                        walk_target = nav_waypoints[sim_precision_walk_waypoint_index]
                        walk_heading = math.atan2(
                            walk_target[1] - walk_y,
                            walk_target[0] - walk_x,
                        )
                        walk_error_deg = math.degrees(
                            angle_error(walk_heading, float(current_yaw))
                        )
                        walk_dx = walk_target[0] - sim_precision_walk_start[0]
                        walk_dy = walk_target[1] - sim_precision_walk_start[1]
                        walk_length = math.hypot(walk_dx, walk_dy)
                        walk_cross_track = abs(
                            walk_dx * (walk_y - sim_precision_walk_start[1])
                            - walk_dy * (walk_x - sim_precision_walk_start[0])
                        ) / max(walk_length, 1e-9)

                        walk_segment_safe = fixed_segment_clear(
                            (walk_x, walk_y),
                            walk_target,
                            sim_resume_obstacles,
                            0.07,
                        )

                        walk_distance = math.hypot(
                            walk_target[0] - walk_x,
                            walk_target[1] - walk_y,
                        )
                        walk_path_heading = math.atan2(walk_dy, walk_dx)
                        walk_path_error_deg = math.degrees(
                            angle_error(walk_path_heading, float(current_yaw))
                        )
                        # Step 1214: signed cross-track correction.
                        # Adapt Step 1082's path-following geometry
                        # to the current A* waypoint segment.
                        walk_signed_cross_track = (
                            walk_dx * (walk_y - sim_precision_walk_start[1])
                            - walk_dy * (walk_x - sim_precision_walk_start[0])
                        ) / max(walk_length, 1e-9)
                        walk_track_heading = walk_path_heading + math.atan2(
                            -3.0 * walk_signed_cross_track, 1.0
                        )
                        walk_error_deg = math.degrees(
                            angle_error(walk_track_heading, float(current_yaw))
                        )

                        near_wp_heading_valid = (
                            walk_distance <= 0.10
                            and walk_cross_track <= 0.02
                            and abs(walk_path_error_deg) <= 12.0
                        )
                        if abs(walk_error_deg) > 12.0 and not near_wp_heading_valid:
                            walk_reason = "HEADING_ERROR"
                        elif walk_cross_track > 0.08:
                            walk_reason = "CROSS_TRACK_ERROR"
                        elif not walk_segment_safe:
                            walk_reason = "SEGMENT_NOT_CLEAR"

                    if walk_reason is None:
                        if sim_precision_resume.update(
                            contact=walk_contact,
                            fallen=walk_fallen,
                            path_verified=True,
                        ):
                            # Reuse the Step 858 successful goal-walk yaw controller.
                            if abs(walk_error_deg) <= 2.0:
                                walk_steer_wz = 0.0
                            else:
                                walk_steer_wz = max(
                                    -0.20,
                                    min(0.20, 1.5 * math.radians(walk_error_deg)),
                                )
                            sim_precision_command = (0.30, 0.0, walk_steer_wz)
                        else:
                            walk_reason = sim_precision_resume.stop_reason or "CONTROLLER_STOP"

                    if walk_reason in ("HEADING_ERROR", "CROSS_TRACK_ERROR"):
                        diag_target = nav_waypoints[sim_precision_walk_waypoint_index]
                        diag_heading = math.atan2(
                            diag_target[1] - walk_y,
                            diag_target[0] - walk_x,
                        )
                        diag_yaw = float(current_yaw)
                        diag_error = math.degrees(
                            angle_error(diag_heading, diag_yaw)
                        )
                        diag_distance = math.hypot(
                            diag_target[0] - walk_x,
                            diag_target[1] - walk_y,
                        )
                        print(
                            f"[SIM_WALK_STOP_DIAG] "
                            f"reason={walk_reason} "
                            f"yaw={math.degrees(diag_yaw):.2f}deg "
                            f"target_yaw={math.degrees(diag_heading):.2f}deg "
                            f"error={diag_error:.2f}deg "
                            f"distance={diag_distance:.4f}m "
                            f"cross_track={walk_cross_track:.4f}m"
                        )
                    if walk_reason is not None:
                        staging_x = walk_x
                        staging_y = walk_y + 0.06
                        staging_target = (staging_x, staging_y)

                        print(
                            f"[SIM_STAGING] "
                            f"current=({walk_x:.4f},{walk_y:.4f}) "
                            f"candidate=({staging_x:.4f},{staging_y:.4f}) "
                            f"goal=(1.0000,0.5000)"
                        )
                        if walk_reason == "WP1_REACHED":
                            sim_staging_target = (
                                walk_x + 0.06 * math.cos(float(current_yaw)),
                                walk_y + 0.06 * math.sin(float(current_yaw)),
                            )
                            sim_staging_start = (walk_x, walk_y)
                            sim_staging_start_time = float(data.time)
                            sim_staging_state = "READY"
                            print(
                                f"[SIM_STAGING_READY] "
                                f"start=({walk_x:.4f},{walk_y:.4f}) "
                                f"target=({sim_staging_target[0]:.4f},{sim_staging_target[1]:.4f})"
                            )
                        else:
                            sim_precision_resume.stop(walk_reason)
                        sim_precision_walk_finished = True
                        sim_precision_walk_status = walk_reason
                        sim_precision_command = (0.0, 0.0, 0.0)
                        print(
                            f"[SIM_PRECISION_WALK] FINISHED "
                            f"status={walk_reason} "
                            f"pos=({walk_x:.4f},{walk_y:.4f}) "
                            f"t={float(data.time) - sim_precision_walk_start_time:.2f}s"
                        )
                if sim_turn_test_enabled and sim_staging_state in ("READY", "WALKING"):
                    stage_x = float(data.qpos[qpos_adr])
                    stage_y = float(data.qpos[qpos_adr + 1])
                    stage_z = float(data.qpos[2])
                    stage_target_x, stage_target_y = sim_staging_target
                    stage_remaining = math.hypot(
                        stage_target_x - stage_x,
                        stage_target_y - stage_y,
                    )
                    stage_dx = stage_target_x - sim_staging_start[0]
                    stage_dy = stage_target_y - sim_staging_start[1]
                    stage_length = math.hypot(stage_dx, stage_dy)
                    stage_path_yaw = math.atan2(stage_dy, stage_dx)
                    stage_yaw_error = math.degrees(
                        angle_error(stage_path_yaw, float(current_yaw))
                    )
                    stage_cross_track = abs(
                        stage_dx * (stage_y - sim_staging_start[1])
                        - stage_dy * (stage_x - sim_staging_start[0])
                    ) / max(stage_length, 1e-9)
                    stage_fallen = not math.isfinite(stage_z) or stage_z < 0.09
                    stage_contact = bool(obstacle_contact_previous)
                    stage_path_clear = fixed_segment_clear(
                        (stage_x, stage_y),
                        sim_staging_target,
                        sim_resume_obstacles,
                        0.10,
                    )
                    stage_elapsed = float(data.time) - sim_staging_start_time

                    if sim_staging_state == "READY":
                        if stage_contact or stage_fallen or not stage_path_clear:
                            sim_staging_status = "START_UNSAFE"
                            sim_staging_state = "STOPPED"
                        elif abs(stage_yaw_error) > 12.0:
                            sim_staging_status = "START_HEADING_ERROR"
                            sim_staging_state = "STOPPED"
                        else:
                            sim_staging_state = "WALKING"
                            print(
                                f"[SIM_STAGING_WALK] START "
                                f"pos=({stage_x:.4f},{stage_y:.4f}) "
                                f"target=({stage_target_x:.4f},{stage_target_y:.4f})"
                            )

                    if sim_staging_state == "WALKING":
                        if stage_contact:
                            sim_staging_status = "CONTACT"
                        elif stage_fallen:
                            sim_staging_status = "FALL"
                        elif not all(math.isfinite(v) for v in (
                            stage_x, stage_y, stage_remaining,
                            stage_cross_track, stage_yaw_error,
                        )):
                            sim_staging_status = "INVALID_POSE"
                        elif not stage_path_clear:
                            sim_staging_status = "SEGMENT_NOT_CLEAR"
                        elif stage_cross_track > 0.02:
                            sim_staging_status = "CROSS_TRACK_ERROR"
                        elif abs(stage_yaw_error) > 12.0:
                            sim_staging_status = "HEADING_ERROR"
                        elif stage_remaining <= 0.02:
                            sim_staging_status = "TARGET_REACHED"
                        elif stage_elapsed > 2.0:
                            sim_staging_status = "TIMEOUT"

                        if sim_staging_status is not None:
                            sim_staging_state = "STOPPED"
                        else:
                            sim_precision_command = (0.30, 0.0, 0.0)

                    if sim_staging_state == "STOPPED":
                        if sim_staging_stop_time is None:
                            sim_staging_stop_time = float(data.time)
                            sim_staging_stop_pose = (stage_x, stage_y)
                        if sim_second_turn_state not in ("ACTIVE", "GOAL_WALK"):
                            sim_precision_command = (0.0, 0.0, 0.0)
                        if not getattr(sim_precision_resume, "_staging_stop_logged", False):
                            print(
                                f"[SIM_STAGING_WALK] FINISHED "
                                f"status={sim_staging_status} "
                                f"pos=({stage_x:.4f},{stage_y:.4f}) "
                                f"remaining={stage_remaining:.4f}m "
                                f"cross_track={stage_cross_track:.4f}m "
                                f"yaw_error={stage_yaw_error:.2f}deg"
                            )
                            sim_precision_resume._staging_stop_logged = True
                if sim_turn_test_enabled and sim_goal_walk_state == "WALKING":
                    gx = float(data.qpos[qpos_adr])
                    gy = float(data.qpos[qpos_adr + 1])
                    gz = float(data.qpos[2])

                    target_x, target_y = sim_goal_walk_target
                    distance = math.hypot(target_x - gx, target_y - gy)

                    dx = target_x - sim_goal_walk_start[0]
                    dy = target_y - sim_goal_walk_start[1]
                    length = max(math.hypot(dx, dy), 1e-9)

                    cross_track = abs(
                        dx * (gy - sim_goal_walk_start[1])
                        - dy * (gx - sim_goal_walk_start[0])
                    ) / length

                    desired_yaw = math.atan2(dy, dx)
                    yaw_error = math.degrees(
                        angle_error(desired_yaw, float(current_yaw))
                    )

                    elapsed = float(data.time) - sim_goal_walk_start_time

                    valid = all(math.isfinite(value) for value in (
                        gx, gy, gz, float(current_yaw),
                        distance, cross_track, yaw_error,
                    ))

                    path_safe = (
                        valid
                        and sim_resume_obstacles is not None
                        and fixed_segment_clear(
                            (gx, gy),
                            sim_goal_walk_target,
                            sim_resume_obstacles,
                            0.15,
                        )
                    )

                    reason = None

                    if not valid:
                        reason = "INVALID_POSE"
                    elif bool(obstacle_contact_previous):
                        reason = "CONTACT"
                    elif gz < 0.09:
                        reason = "FALL"
                    elif not path_safe:
                        reason = "SEGMENT_NOT_CLEAR"
                    elif cross_track > 0.08:
                        reason = "CROSS_TRACK_ERROR"
                    elif distance <= 0.10:
                        reason = "GOAL_REACHED"
                    elif abs(yaw_error) > 8.0:
                        reason = "HEADING_ERROR"
                    elif elapsed > 14.0:
                        reason = "TIMEOUT"

                    if reason is None:
                        goal_steer_error_rad = math.radians(yaw_error)
                        if abs(yaw_error) <= 2.0:
                            goal_steer_wz = 0.0
                        else:
                            goal_steer_wz = max(
                                -0.20,
                                min(0.20, 1.5 * goal_steer_error_rad),
                            )

                        sim_precision_command = (0.30, 0.0, goal_steer_wz)

                        if abs(yaw_error) > 2.0:
                            print(
                                f"[SIM_GOAL_STEER] "
                                f"error={yaw_error:+.2f}deg "
                                f"wz={goal_steer_wz:+.3f}rad/s "
                                f"distance={distance:.3f}m"
                            )
                    else:
                        sim_goal_walk_state = "FINISHED"
                        sim_goal_walk_status = reason
                        sim_precision_command = (0.0, 0.0, 0.0)
                        print(
                            f"[SIM_GOAL_WALK] FINISHED "
                            f"status={reason} "
                            f"pos=({gx:.4f},{gy:.4f}) "
                            f"distance={distance:.4f}m "
                            f"cross_track={cross_track:.4f}m "
                            f"yaw_error={yaw_error:.2f}deg "
                            f"elapsed={elapsed:.2f}s"
                        )
                safe_command = guarded_command(
                    replanning_controller, nav_command
                )
                # Simulation-only control route, never used in production.
                if sim_turn_command is not None:
                    safe_command = sim_turn_command
                if sim_precision_command is not None:
                    safe_command = sim_precision_command

                # Step 1254: ToF emergency stop during simulation GOAL_WALK.
                # The forward-only sensor is not a full clearance verifier.
                # Do not automatically resume or authorize a new route.
                if (
                    sim_turn_test_enabled
                    and sim_goal_walk_state == "WALKING"
                    and not sim_emergency_stop_requested
                ):
                    if phase4_sensor is None:
                        sim_emergency_stop_requested = True
                        sim_emergency_stop_reason = "SENSOR_MISSING"
                    else:
                        try:
                            sim_stop_distance = phase4_sensor.measure(model, data)

                            # Step 1257: Diagnostics only; no control changes.
                            diag_origin, diag_direction = phase4_sensor.pose(data)
                            diag_geomid = np.array([-1], dtype=np.int32)
                            diag_raw = mujoco.mj_ray(
                                model,
                                data,
                                diag_origin,
                                diag_direction,
                                phase4_sensor.geomgroup,
                                1,
                                -1,
                                diag_geomid,
                            )
                            diag_gid = int(diag_geomid[0])
                            diag_name = (
                                mujoco.mj_id2name(
                                    model, mujoco.mjtObj.mjOBJ_GEOM, diag_gid
                                )
                                if diag_gid >= 0 else None
                            )
                            diag_robot_xy = [
                                float(data.qpos[qpos_adr]),
                                float(data.qpos[qpos_adr + 1]),
                            ]
                            diag_record = {
                                "time": round(float(data.time), 4),
                                "robot_xy": diag_robot_xy,
                                "sensor_origin": np.round(
                                    diag_origin, 4
                                ).tolist(),
                                "sensor_direction": np.round(
                                    diag_direction, 4
                                ).tolist(),
                                "ray_raw": round(float(diag_raw), 4),
                                "measured": (
                                    None if sim_stop_distance is None
                                    else round(float(sim_stop_distance), 4)
                                ),
                                "geom_id": diag_gid,
                                "geom_name": diag_name,
                                "obstacle_2_center_xy": [0.7, 0.5],
                                "obstacle_2_nearest_box_xy_m": math.hypot(
                                    max(abs(diag_robot_xy[0] - 0.7) - 0.1, 0.0),
                                    max(abs(diag_robot_xy[1] - 0.5) - 0.1, 0.0),
                                ),
                            }
                            if (
                                sim_stop_distance is not None
                                and math.isfinite(float(sim_stop_distance))
                                and float(sim_stop_distance)
                                <= phase4_stop_threshold
                            ):
                                print(
                                    "[SIM_STOP_HIT] "
                                    + json.dumps(diag_record),
                                    flush=True,
                                )
                            if (
                                sim_stop_distance is not None
                                and (
                                    not math.isfinite(float(sim_stop_distance))
                                    or float(sim_stop_distance) <= phase4_stop_threshold
                                )
                            ):
                                sim_emergency_stop_requested = True
                                sim_emergency_stop_reason = "TOF_NEAR"
                                print(
                                    f"[SIM_TOF_STOP] DETECTED "
                                    f"distance={sim_stop_distance:.4f}m "
                                    f"threshold={phase4_stop_threshold:.4f}m "
                                    f"t={float(data.time):.2f}s"
                                )
                        except Exception as sim_tof_exc:
                            sim_emergency_stop_requested = True
                            sim_emergency_stop_reason = "SENSOR_EXCEPTION"
                            print(
                                f"[SIM_TOF_STOP] SENSOR_EXCEPTION "
                                f"{type(sim_tof_exc).__name__}: {sim_tof_exc}"
                            )

                    if sim_emergency_stop_requested:
                        sim_goal_walk_state = "FINISHED"
                        sim_goal_walk_status = "TOF_EMERGENCY_STOP"
                        print(
                            f"[SIM_TOF_STOP] LATCHED "
                            f"reason={sim_emergency_stop_reason}"
                        )

                # Step 1272: Diagnostic-only second replanning.
                # This cannot authorize movement or clear the stop latch.
                if (
                    sim_emergency_stop_requested
                    and sim_emergency_stop_reason == "TOF_NEAR"
                    and not step1272_shadow_attempted
                ):
                    step1272_shadow_attempted = True
                    import json as step1272_json
                    try:
                        if phase4_sensor is None:
                            raise RuntimeError("TOF_SENSOR_MISSING")
                        if sim_stop_distance is None:
                            raise RuntimeError("TOF_DISTANCE_MISSING")

                        first_count = len(obstacle_map.points)
                        sensor_origin, sensor_direction = (
                            phase4_sensor.pose(data)
                        )
                        hit = obstacle_map.register(
                            sensor_origin,
                            sensor_direction,
                            float(sim_stop_distance),
                        )
                        safety_map.add_detection(hit)

                        estimated = convert_obstacles(safety_map)
                        start_xy = (
                            float(data.qpos[0]),
                            float(data.qpos[1]),
                        )
                        final_goal = (
                            float(config["goal"]["x"]),
                            float(config["goal"]["y"]),
                        )
                        shadow_path = make_path(
                            start_xy,
                            final_goal,
                            float(current_yaw),
                            estimated,
                        )
                        if len(shadow_path) < 2:
                            raise RuntimeError("EMPTY_REPLAN")

                        shadow_result = {
                            "status": "PLAN_GENERATED_STOP_HELD",
                            "first_detection_count": first_count,
                            "total_detections": len(obstacle_map.points),
                            "hit_xy": list(hit),
                            "start_xy": list(start_xy),
                            "goal_xy": list(final_goal),
                            "path": [
                                list(p) for p in shadow_path
                            ],
                            "path_points": len(shadow_path),
                            "stop_latched": bool(
                                sim_emergency_stop_requested
                            ),
                            "resume_authorized": False,
                        }
                    except Exception as shadow_exc:
                        shadow_result = {
                            "status": "PLAN_FAILED_STOP_HELD",
                            "error": (
                                type(shadow_exc).__name__
                                + ": " + str(shadow_exc)
                            ),
                            "stop_latched": True,
                            "resume_authorized": False,
                        }
                    # Step 1279: Route task shadow only.
                    # Keep the emergency stop latched regardless of outcome.
                    if shadow_result.get("status") == "PLAN_GENERATED_STOP_HELD":
                        try:
                            from route_task_builder import (
                                build_route_tasks,
                            )

                            raw_path = shadow_result["path"]
                            if len(raw_path) < 2:
                                raise ValueError("EMPTY_SHADOW_PATH")

                            # Remove only collinear intermediate points.
                            simplified = [raw_path[0]]
                            for step1279_i in range(1, len(raw_path) - 1):
                                a = raw_path[step1279_i - 1]
                                b = raw_path[step1279_i]
                                c = raw_path[step1279_i + 1]
                                abx, aby = b[0] - a[0], b[1] - a[1]
                                bcx, bcy = c[0] - b[0], c[1] - b[1]
                                cross = abx * bcy - aby * bcx
                                dot = abx * bcx + aby * bcy
                                if abs(cross) > 1e-10 or dot <= 0:
                                    simplified.append(b)
                            simplified.append(raw_path[-1])

                            tasks = build_route_tasks(
                                simplified,
                                float(current_yaw),
                            )

                            if not tasks or any(
                                task["movement_authorized"] for task in tasks
                            ):
                                raise RuntimeError("TASK_VALIDATION_FAILED")

                            shadow_result["task_status"] = (
                                "TASKS_GENERATED_STOP_HELD"
                            )
                            shadow_result["simplified_path"] = simplified
                            shadow_result["control_tasks"] = tasks
                            shadow_result["task_count"] = len(tasks)
                            # Step 1282: Shadow sequencing only.
                            # No turn or waypoint completion events.
                            from route_execution_sequencer import (
                                ShadowRouteSequencer,
                            )
                            step1282_sequencer = ShadowRouteSequencer(tasks)
                            if not step1282_sequencer.start_shadow():
                                raise RuntimeError(
                                    "SHADOW_SEQUENCER_START_REJECTED"
                                )
                            step1282_snapshot = (
                                step1282_sequencer.snapshot()
                            )
                            if (
                                step1282_snapshot["state"] != "TURN_PENDING"
                                or step1282_snapshot["current_segment"] != 1
                                or step1282_snapshot["task_count"] != 3
                                or step1282_snapshot["events"] != [
                                    "TURN_PENDING:1"
                                ]
                                or step1282_snapshot["movement_authorized"]
                                or step1282_snapshot[
                                    "emergency_stop_release_authorized"
                                ]
                                or not sim_emergency_stop_requested
                            ):
                                raise RuntimeError(
                                    "SHADOW_SEQUENCER_INVARIANT_FAILED"
                                )
                            shadow_result["sequencer_status"] = (
                                "INITIALIZED_STOP_HELD"
                            )
                            shadow_result["sequencer_snapshot"] = (
                                step1282_snapshot
                            )

                            # Step 1288: first-turn-only simulator trial.
                            # This does not release the emergency stop.
                            step1288_task = tasks[0]
                            step1288_trial_target_yaw = float(
                                step1288_task["target_yaw_rad"]
                            )
                            step1288_trial_start_xy = (
                                float(data.qpos[qpos_adr]),
                                float(data.qpos[qpos_adr + 1]),
                            )
                            step1288_trial_start_yaw = float(current_yaw)

                            step1288_arm_ok = step1288_gate.arm(
                                emergency_latched=bool(
                                    sim_emergency_stop_requested
                                ),
                                emergency_reason=sim_emergency_stop_reason,
                                task=step1288_task,
                                now=float(data.time),
                                current_yaw=float(current_yaw),
                            )

                            if not step1288_arm_ok:
                                raise RuntimeError(
                                    "FIRST_TURN_GATE_ARM_REJECTED:"
                                    + str(step1288_gate.reason)
                                )

                            # Reuse the existing PPO V3 turn controller.
                            # Start on the NEXT simulation control tick.
                            if sim_turn_policy_active:
                                raise RuntimeError(
                                    "TURN_POLICY_ALREADY_ACTIVE"
                                )

                            sim_turn_controller = SimTurnController(
                                turn_speed=args.sim_turn_speed
                            )
                            sim_turn_last_state = SimTurnState.WAITING
                            sim_ema_rate = None
                            sim_ema_prev_time = None

                            if not sim_turn_controller.start(
                                current_time=float(data.time),
                                target_yaw=step1288_trial_target_yaw,
                            ):
                                raise RuntimeError(
                                    "FIRST_TURN_CONTROLLER_START_FAILED"
                                )

                            step1288_trial_started = True
                            step1288_trial_start_time = float(data.time)
                            shadow_result["first_turn_trial"] = {
                                "state": "ARMED",
                                "target_yaw_deg": math.degrees(
                                    step1288_trial_target_yaw
                                ),
                                "initial_yaw_deg": math.degrees(
                                    step1288_trial_start_yaw
                                ),
                                "walking_authorized": False,
                                "emergency_latch_released": False,
                            }
                            print(
                                "[SIM_FIRST_TURN] START "
                                f"t={float(data.time):.3f} "
                                f"target={math.degrees(step1288_trial_target_yaw):.2f}deg "
                                f"initial={math.degrees(step1288_trial_start_yaw):.2f}deg",
                                flush=True,
                            )

                        except Exception as step1279_exc:
                            shadow_result["task_status"] = (
                                "TASK_GENERATION_FAILED_STOP_HELD"
                            )
                            shadow_result["task_error"] = (
                                type(step1279_exc).__name__
                                + ": " + str(step1279_exc)
                            )
                            shadow_result["task_count"] = 0
                    else:
                        shadow_result["task_status"] = (
                            "PLANNER_FAILED_STOP_HELD"
                        )
                        shadow_result["task_count"] = 0

                    shadow_result["stop_latched"] = bool(
                        sim_emergency_stop_requested
                    )
                    shadow_result["resume_authorized"] = False
                    print(
                        "[SIM_ROUTE_TASK_SHADOW] "
                        + step1272_json.dumps(shadow_result),
                        flush=True,
                    )

                    print(
                        "[SIM_SECOND_REPLAN_SHADOW] "
                        + step1272_json.dumps(shadow_result),
                        flush=True,
                    )

                # Step 1288: final motor-command arbitration.
                # An active trial permits yaw ONLY. All other stop
                # cases use the unchanged Step 1252 stop semantics.
                if sim_turn_test_enabled:
                    if step1337b_state is not None:
                        safe_command = (0.0, 0.0, 0.0)

                        x = float(data.qpos[qpos_adr])
                        y = float(data.qpos[qpos_adr + 1])
                        z = float(data.qpos[2])
                        yaw = float(current_yaw)
                        now = float(data.time)

                        via_target = (0.26, 1.10)
                        distance = math.hypot(
                            via_target[0] - x,
                            via_target[1] - y
                        )
                        heading = math.atan2(
                            via_target[1] - y,
                            via_target[0] - x
                        )
                        direct_error = math.degrees(
                            angle_error(heading, yaw)
                        )
                        drift = (
                            math.hypot(
                                x - step1337b_origin[0],
                                y - step1337b_origin[1]
                            )
                            if step1337b_origin is not None
                            else float("inf")
                        )

                        pose_ok = (
                            all(math.isfinite(v) for v in (
                                x, y, z, yaw, now, distance,
                                direct_error, drift
                            ))
                            and z >= 0.09
                        )

                        latch_ok = (
                            sim_emergency_stop_requested is True
                            and sim_emergency_stop_reason == "TOF_NEAR"
                        )

                        route_ok = False
                        try:
                            obstacles = convert_obstacles(safety_map)
                            route_ok = bool(
                                len(obstacles) >= 2
                                and fixed_segment_clear(
                                    (x, y), via_target, obstacles, 0.15
                                )
                                and fixed_segment_clear(
                                    via_target, step1300_next,
                                    obstacles, 0.15
                                )
                            )
                        except Exception:
                            route_ok = False

                        # Fail closed if the existing ToF sensor cannot be read.
                        tof_valid = False
                        tof_value = None

                        try:
                            tof_measured = phase4_sensor.measure(model, data)

                            ray_origin, ray_direction = phase4_sensor.pose(data)

                            tof_geom = np.array([-1], dtype=np.int32)

                            ray_distance = float(
                                mujoco.mj_ray(
                                    model,
                                    data,
                                    ray_origin,
                                    ray_direction,
                                    phase4_sensor.geomgroup,
                                    1,
                                    -1,
                                    tof_geom,
                                )
                            )

                            sensor_max = float(phase4_sensor.sensor.max_distance)

                            if (
                                math.isfinite(ray_distance)
                                and math.isfinite(sensor_max)
                                and sensor_max > 0.0
                            ):
                                # None is acceptable ONLY for a confirmed ray miss.
                                if (
                                    tof_measured is None
                                    and ray_distance < 0.0
                                    and int(tof_geom[0]) == -1
                                ):
                                    tof_value = sensor_max
                                    tof_valid = True

                                # Some APIs report None when the hit is beyond range.
                                elif (
                                    tof_measured is None
                                    and ray_distance > sensor_max
                                    and int(tof_geom[0]) >= 0
                                ):
                                    tof_value = sensor_max
                                    tof_valid = True

                                # A measured hit must agree with the MuJoCo ray.
                                elif (
                                    tof_measured is not None
                                    and int(tof_geom[0]) >= 0
                                    and 0.0 <= ray_distance <= sensor_max
                                    and math.isfinite(float(tof_measured))
                                    and abs(float(tof_measured) - ray_distance) <= 1e-5
                                ):
                                    tof_value = float(tof_measured)
                                    tof_valid = True

                        except Exception:
                            tof_valid = False
                            tof_value = None

                        if step1337b_state == "HOLD":
                            elapsed_hold = (
                                now - step1337b_hold_time
                                if step1337b_hold_time is not None
                                else float("nan")
                            )

                            if pose_ok:
                                step1340s_max_drift = max(
                                    step1340s_max_drift, drift
                                )

                            if not pose_ok or not math.isfinite(elapsed_hold):
                                step1337b_reason = "INVALID_POSE"
                            elif not latch_ok:
                                step1337b_reason = "LATCH_CHANGED"
                            elif obstacle_contact_previous or obstacle_contact_events:
                                step1337b_reason = "CONTACT"
                            elif policy.current_policy != "standing":
                                step1337b_reason = "HOLD_POLICY_MISMATCH"
                            elif sim_turn_policy_active:
                                step1337b_reason = "TURN_POLICY_ACTIVE"
                            elif not route_ok:
                                step1337b_reason = "ROUTE_UNSAFE"
                            elif drift > 0.02 or step1340s_max_drift > 0.02:
                                step1337b_reason = "HOLD_DRIFT"
                            elif elapsed_hold >= 0.60:
                                if abs(direct_error) > 5.0:
                                    step1337b_reason = "DIRECT_HEADING_MISMATCH"
                                elif not tof_valid:
                                    step1337b_reason = "TOF_INVALID"
                                elif tof_value < 0.30:
                                    step1337b_reason = "TOF_NEAR"
                                else:
                                    # Authorization is local to the VIA trial.
                                    # The global emergency-stop latch is not cleared.
                                    step1337b_state = "WALKING"
                                    step1337b_walk_time = now
                                    step1341_walk_origin = (x, y)

                                    print(
                                        "[SIM_VIA_WALK_START] "
                                        + step1288_json.dumps({
                                            "time": now,
                                            "start_xy": [x, y],
                                            "target_xy": list(via_target),
                                            "direct_heading_error_deg": direct_error,
                                            "tof_m": tof_value,
                                            "max_hold_drift_m": step1340s_max_drift,
                                            "emergency_latched": True,
                                        }),
                                        flush=True,
                                    )

                            if step1337b_reason is not None:
                                step1337b_state = "FINISHED"

                        elif step1337b_state == "WALKING":
                            walk_elapsed = (
                                now - step1337b_walk_time
                                if step1337b_walk_time is not None
                                else float("nan")
                            )

                            if step1341_walk_origin is None:
                                step1337b_reason = "INVALID_WALK_ORIGIN"
                            else:
                                ox, oy = step1341_walk_origin
                                tx = via_target[0] - ox
                                ty = via_target[1] - oy
                                path_length = math.hypot(tx, ty)

                                if path_length > 1e-8:
                                    ux = tx / path_length
                                    uy = ty / path_length
                                    dx = x - ox
                                    dy = y - oy

                                    along = dx * ux + dy * uy
                                    cross = abs(dx * uy - dy * ux)
                                else:
                                    along = float("nan")
                                    cross = float("inf")

                                if not pose_ok or not math.isfinite(walk_elapsed):
                                    step1337b_reason = "INVALID_POSE"
                                elif not latch_ok:
                                    step1337b_reason = "LATCH_CHANGED"
                                elif obstacle_contact_previous or obstacle_contact_events:
                                    step1337b_reason = "CONTACT"
                                elif sim_turn_policy_active:
                                    step1337b_reason = "TURN_POLICY_ACTIVE"
                                elif policy.current_policy not in ("standing", "walking"):
                                    step1337b_reason = "WALK_POLICY_MISMATCH"
                                elif not route_ok:
                                    step1337b_reason = "ROUTE_UNSAFE"
                                elif not tof_valid:
                                    step1337b_reason = "TOF_INVALID"
                                elif tof_value < 0.30:
                                    step1337b_reason = "TOF_NEAR"
                                elif path_length <= 1e-8:
                                    step1337b_reason = "INVALID_PATH"
                                elif cross > 0.08:
                                    step1337b_reason = "CROSS_TRACK_LIMIT"
                                elif abs(direct_error) > 25.0:
                                    step1337b_reason = "HEADING_LIMIT"
                                elif along < -0.025:
                                    step1337b_reason = "REVERSE_DRIFT"
                                elif along > path_length + 0.05:
                                    step1337b_reason = "OVERSHOOT"
                                elif distance <= 0.05:
                                    step1337b_reason = "VIA_REACHED"
                                elif walk_elapsed >= 2.40:
                                    step1337b_reason = "SHORT_TRIAL_COMPLETE"
                                else:
                                    # Existing walking policy performs the motion.
                                    # No PPO turning is requested here.
                                    wz = max(
                                        -0.20,
                                        min(
                                            0.20,
                                            math.radians(direct_error) * 0.5
                                        )
                                    )
                                    safe_command = (0.30, 0.0, wz)

                            if (
                                walk_elapsed - step1337b_last_sample >= 0.10
                                or step1337b_reason is not None
                            ):
                                step1337b_last_sample = walk_elapsed

                                print(
                                    "[SIM_VIA_WALK_SAMPLE] "
                                    + step1288_json.dumps({
                                        "elapsed_s": walk_elapsed,
                                        "xy": [x, y],
                                        "remaining_m": distance,
                                        "heading_error_deg": direct_error,
                                        "along_m": along,
                                        "cross_m": cross,
                                        "tof_m": tof_value,
                                        "route_ok": route_ok,
                                        "command": list(safe_command),
                                        "reason": step1337b_reason,
                                    }),
                                    flush=True,
                                )

                            if step1337b_reason is not None:
                                step1337b_state = "FINISHED"

                        elif step1337b_state == "RIGHT_TURN":
                            step1349_command = (0.0, 0.0, 0.0)

                            turn_elapsed = (
                                float(data.time) - step1349_start_time
                                if step1349_start_time is not None
                                else float("nan")
                            )

                            turn_drift = (
                                math.hypot(
                                    x - step1349_origin[0],
                                    y - step1349_origin[1],
                                )
                                if step1349_origin is not None
                                else float("inf")
                            )

                            if math.isfinite(turn_drift):
                                step1349_max_drift = max(
                                    step1349_max_drift, turn_drift
                                )

                            turn_clear = False
                            try:
                                turn_clear, _, _ = evaluate_sim_turn_clearance(
                                    model,
                                    data,
                                    clearance_margin=0.10,
                                    robot_radius=0.10,
                                )
                            except Exception:
                                turn_clear = False

                            route_clear = False
                            try:
                                known = convert_obstacles(safety_map)
                                route_clear = bool(
                                    len(known) >= 2
                                    and fixed_segment_clear(
                                        (x, y), (0.95, 1.00),
                                        known, 0.15,
                                    )
                                )
                            except Exception:
                                route_clear = False

                            if not pose_ok or not math.isfinite(turn_elapsed):
                                step1349_reason = "INVALID_POSE"
                            elif not latch_ok:
                                step1349_reason = "LATCH_CHANGED"
                            elif obstacle_contact_previous or obstacle_contact_events:
                                step1349_reason = "CONTACT"
                            elif not turn_clear:
                                step1349_reason = "CLEARANCE_LOST"
                            elif not route_clear:
                                step1349_reason = "ROUTE_UNSAFE"
                            elif not tof_valid or tof_value < 0.30:
                                step1349_reason = "TOF_UNSAFE"
                            elif step1349_max_drift > 0.08:
                                step1349_reason = "TURN_DRIFT"
                            elif turn_elapsed >= 4.50:
                                step1349_reason = "TURN_TIMEOUT"
                            elif policy.current_policy not in ("standing", "ppo_v3_turn"):
                                step1349_reason = "POLICY_MISMATCH"
                            else:
                                if not sim_turn_policy_active:
                                    policy.activate_ppo_v3_turn()
                                    sim_turn_policy_active = True

                                proposed_wz = step1349_controller.update(
                                    current_time=float(data.time),
                                    current_yaw=yaw,
                                    contact=bool(obstacle_contact_previous),
                                    fallen=bool(z < 0.09),
                                )

                                if step1349_controller.state is SimTurnState.ABORTED:
                                    step1349_reason = (
                                        "CONTROLLER_ABORT:"
                                        + str(step1349_controller.abort_reason)
                                    )
                                elif (
                                    step1349_controller.state
                                    is SimTurnState.READY_TO_RESUME
                                ):
                                    final_error = math.degrees(
                                        angle_error(step1349_target_yaw, yaw)
                                    )
                                    step1349_reason = (
                                        "TURN_COMPLETE"
                                        if abs(final_error) <= 8.0
                                        else "TURN_ALIGNMENT_LOST"
                                    )
                                else:
                                    step1349_command = (
                                        0.0,
                                        0.0,
                                        max(-0.50, min(0.50, float(proposed_wz))),
                                    )

                            if (
                                turn_elapsed - step1349_last_sample >= 0.10
                                or step1349_reason is not None
                            ):
                                step1349_last_sample = turn_elapsed
                                print(
                                    "[SIM_WP2_TURN_SAMPLE] "
                                    + step1288_json.dumps({
                                        "elapsed_s": turn_elapsed,
                                        "yaw_deg": math.degrees(yaw),
                                        "target_deg": math.degrees(step1349_target_yaw),
                                        "error_deg": math.degrees(
                                            angle_error(step1349_target_yaw, yaw)
                                        ),
                                        "drift_m": turn_drift,
                                        "max_drift_m": step1349_max_drift,
                                        "tof_m": tof_value,
                                        "clearance_ok": bool(turn_clear),
                                        "route_ok": bool(route_clear),
                                        "controller_state": step1349_controller.state.value,
                                        "command": list(step1349_command),
                                        "reason": step1349_reason,
                                    }),
                                    flush=True,
                                )

                            if step1349_reason is not None:
                                step1349_command = (0.0, 0.0, 0.0)
                                safe_command = (0.0, 0.0, 0.0)
                                policy.set_vel_cmd(0.0, 0.0, 0.0)

                                if sim_turn_policy_active:
                                    policy.deactivate_ppo_v3_turn()
                                    sim_turn_policy_active = False

                                step1349_mode = (
                                    "DONE"
                                    if step1349_reason == "TURN_COMPLETE"
                                    else "ABORTED"
                                )
                                step1337b_state = "TURN_DONE"
                        elif step1337b_state == "WP2_SHORT_WALK":
                            step1356_command = (0.0, 0.0, 0.0)
                            elapsed = float(data.time) - step1356_start_time
                            walked = math.hypot(
                                x-step1356_origin[0],
                                y-step1356_origin[1],
                            )
                            step1356_max_drift = max(step1356_max_drift, walked)

                            target_distance = math.hypot(0.95-x, 1.00-y)
                            target_heading = math.atan2(1.00-y, 0.95-x)
                            heading_error = math.degrees(
                                angle_error(target_heading, yaw)
                            )

                            # Step1376: short excursions only; no PPO correction.
                            if math.isfinite(heading_error):
                                step1376_max_heading_error = max(
                                    step1376_max_heading_error,
                                    abs(heading_error),
                                )
                                if abs(heading_error) > 14.0:
                                    if step1376_heading_excursion_since is None:
                                        step1376_heading_excursion_since = float(data.time)
                                        step1376_heading_excursion_count += 1
                                else:
                                    step1376_heading_excursion_since = None

                            route_clear = False
                            body_clear = False
                            try:
                                known = convert_obstacles(safety_map)
                                route_clear = bool(
                                    len(known) >= 2
                                    and fixed_segment_clear(
                                        (x,y), (0.95,1.00), known, 0.15
                                    )
                                )
                                body_clear, _, _ = evaluate_sim_turn_clearance(
                                    model, data,
                                    clearance_margin=0.10,
                                    robot_radius=0.10,
                                )
                            except Exception:
                                route_clear = False
                                body_clear = False

                            if not pose_ok or not math.isfinite(elapsed):
                                step1356_reason = "INVALID_POSE"
                            elif not latch_ok:
                                step1356_reason = "EMERGENCY_LATCH_CHANGED"
                            elif obstacle_contact_previous or obstacle_contact_events:
                                step1356_reason = "CONTACT"
                            elif not tof_valid or tof_value is None:
                                step1356_reason = "INVALID_TOF"
                            elif tof_value < 0.60:
                                step1356_reason = "TOF_NEAR"
                            elif not route_clear:
                                step1356_reason = "ROUTE_UNSAFE"
                            elif not body_clear:
                                step1356_reason = "BODY_CLEARANCE_LOST"
                            elif not math.isfinite(heading_error):
                                step1356_reason = "INVALID_HEADING"
                            elif abs(heading_error) > 16.0:
                                step1356_reason = "HEADING_HARD_LIMIT"
                            elif (
                                abs(heading_error) > 14.0
                                and step1376_heading_excursion_since is not None
                                and float(data.time) - step1376_heading_excursion_since >= 0.12
                            ):
                                step1356_reason = "HEADING_PERSISTENT_LIMIT"
                            elif walked > 0.68:
                                step1356_reason = "DISPLACEMENT_LIMIT"
                            elif target_distance <= 0.10:
                                step1356_reason = "WP2_REACHED"
                            elif elapsed >= 7.50:
                                step1356_reason = "SHORT_WALK_COMPLETE"
                            elif sim_turn_policy_active or getattr(
                                policy, "ppo_v3_turn_active", False
                            ):
                                step1356_reason = "PPO_STILL_ACTIVE"
                            elif policy.current_policy not in ("standing", "walking"):
                                step1356_reason = "POLICY_MISMATCH"
                            else:
                                step1356_command = (
                                    (0.26 if target_distance <= 0.25 else 0.30),
                                    0.0,
                                    max(-0.12, min(
                                        0.12, 0.024 * (heading_error + 4.0)
                                    )),
                                )

                            if (
                                elapsed-step1356_last_sample >= 0.10
                                or step1356_reason is not None
                            ):
                                step1356_last_sample = elapsed
                                print(
                                    "[SIM_WP2_WALK_SAMPLE] "
                                    + step1288_json.dumps({
                                        "elapsed_s": elapsed,
                                        "xy": [x,y],
                                        "distance_to_wp2_m": target_distance,
                                        "heading_error_deg": heading_error,
                                        "tof_m": tof_value,
                                        "route_clear": route_clear,
                                        "body_clear": body_clear,
                                        "walked_m": walked,
                                        "command": list(step1356_command),
                                        "reason": step1356_reason,
                                    }),
                                    flush=True,
                                )

                            if step1356_reason is not None:
                                step1356_command = (0.0, 0.0, 0.0)
                                safe_command = (0.0, 0.0, 0.0)
                                policy.set_vel_cmd(0.0, 0.0, 0.0)
                                last_nav_command = (0.0, 0.0, 0.0)
                                step1356_state = "FINISHED"
                                step1337b_state = "WP2_WALK_DONE"
                        elif step1337b_state not in ("FINISHED", "TURN_DONE", "WP2_WALK_DONE"):
                            step1337b_reason = "INVALID_VIA_STATE"
                            step1337b_state = "FINISHED"

                        if step1337b_state == "FINISHED":
                            safe_command = (0.0, 0.0, 0.0)

                        if (
                            not all(math.isfinite(float(v)) for v in safe_command)
                            or not 0.0 <= safe_command[0] <= 0.30
                            or safe_command[1] != 0.0
                            or abs(safe_command[2]) > 0.20
                        ):
                            raise RuntimeError("STEP1341_MOTOR_INVARIANT")

                    elif (
                        step1301_started
                        and step1305_hold_until is not None
                        and float(data.time) < step1305_hold_until
                    ):
                        safe_command = (0.0, 0.0, 0.0)
                        step1301_last_motor_command = safe_command

                        if (
                            policy.current_policy != "ppo_v3_turn"
                            or not sim_turn_policy_active
                        ):
                            step1301_finished = True
                            step1301_reason = "HOLD_POLICY_MISMATCH"

                        hold_sample = {
                            "t": round(
                                float(data.time)
                                - (step1305_hold_until - 0.60),
                                3,
                            ),
                            "xy": [
                                float(data.qpos[qpos_adr]),
                                float(data.qpos[qpos_adr + 1]),
                            ],
                            "yaw_deg": round(
                                math.degrees(current_yaw), 4
                            ),
                            "contact": bool(
                                obstacle_contact_previous
                            ),
                        }
                        if (
                            not step1305_hold_samples
                            or hold_sample["t"]
                            - step1305_hold_samples[-1]["t"]
                            >= 0.099
                        ):
                            step1305_hold_samples.append(
                                hold_sample
                            )

                        if (
                            bool(obstacle_contact_previous)
                            or not math.isfinite(float(current_yaw))
                            or not math.isfinite(
                                float(data.qpos[2])
                            )
                            or float(data.qpos[2]) < 0.09
                            or sim_emergency_stop_requested is not True
                            or sim_emergency_stop_reason != "TOF_NEAR"
                        ):
                            step1301_finished = True
                            step1301_reason = "HOLD_SAFETY_FAILURE"

                    elif step1301_started:
                        # Step 1305: revalidate after the hold.
                        if not step1305_walk_revalidated:
                            step1305_walk_revalidated = True

                            sx = float(data.qpos[qpos_adr])
                            sy = float(data.qpos[qpos_adr + 1])
                            heading = math.atan2(
                                step1300_target[1] - sy,
                                step1300_target[0] - sx,
                            )
                            initial_error = math.degrees(
                                angle_error(
                                    heading, float(current_yaw)
                                )
                            )

                            try:
                                obs = convert_obstacles(
                                    safety_map
                                )
                                still_clear = (
                                    len(obs) >= 2
                                    and fixed_segment_clear(
                                        (sx, sy), step1300_target,
                                        obs, 0.20,
                                    )
                                    and fixed_segment_clear(
                                        step1300_target,
                                        step1300_next,
                                        obs, 0.20,
                                    )
                                )
                            except Exception:
                                still_clear = False

                            ready = bool(
                                all(math.isfinite(v) for v in
                                    (sx, sy, initial_error))
                                and abs(math.degrees(
                                    angle_error(
                                        step1326_bias_target_yaw,
                                        float(current_yaw),
                                    )
                                )) <= 5.0
                                and still_clear
                                and obstacle_contact_events == 0
                                and sim_emergency_stop_requested
                                and sim_emergency_stop_reason
                                    == "TOF_NEAR"
                            )

                            print(
                                "[SIM_WP1_POST_HOLD] "
                                + step1288_json.dumps({
                                    "ready": ready,
                                    "xy": [sx, sy],
                                    "yaw_deg": math.degrees(
                                        current_yaw
                                    ),
                                    "heading_error_deg":
                                        initial_error,
                                    "route_clear_020":
                                        bool(still_clear),
                                }),
                                flush=True,
                            )

                            if not (
                                policy.current_policy == "ppo_v3_turn"
                                and sim_turn_policy_active
                            ):
                                ready = False

                            if not ready:
                                step1301_finished = True
                                step1301_reason = (
                                    "POST_HOLD_NOT_READY"
                                )
                            else:
                                policy.set_vel_cmd(0.0, 0.0, 0.0)
                                policy.step1309_direct_ppo_to_walking()
                                sim_turn_policy_active = False
                                if policy.current_policy != "walking":
                                    raise RuntimeError(
                                        "STEP1309_DIRECT_SWITCH_FAILED"
                                    )
                                print(
                                    "[SIM_WP1_POLICY_TRANSITION] "
                                    + step1288_json.dumps({
                                        "from": "ppo_v3_turn",
                                        "via": "NONE",
                                        "to": "walking",
                                        "forward_cmd": 0.30,
                                        "yaw_cmd": 0.0,
                                        "heading_error_deg":
                                            initial_error,
                                        "authorized": True,
                                    }),
                                    flush=True,
                                )
                                # Reset the measured segment origin.
                                step1301_start_xy = (sx, sy)
                                ddx = step1300_target[0] - sx
                                ddy = step1300_target[1] - sy
                                length = math.hypot(ddx, ddy)
                                step1301_direction = (
                                    ddx / length,
                                    ddy / length,
                                )
                                step1301_start_time = float(
                                    data.time
                                )

                        # Step 1301: separate walk-only permission.
                        # Does not clear the emergency stop latch.
                        if step1301_finished:
                            safe_command = (0.0, 0.0, 0.0)
                            step1301_last_motor_command = (
                                0.0, 0.0, 0.0
                            )
                            policy.set_vel_cmd(0.0, 0.0, 0.0)
                            if sim_turn_policy_active:
                                policy.deactivate_ppo_v3_turn()
                                sim_turn_policy_active = False

                        wx = float(data.qpos[qpos_adr])
                        wy = float(data.qpos[qpos_adr + 1])
                        wz_height = float(data.qpos[2])
                        elapsed_walk = (
                            float(data.time) - step1301_start_time
                        )

                        ux, uy = step1301_direction
                        dx = wx - step1301_start_xy[0]
                        dy = wy - step1301_start_xy[1]
                        along = dx * ux + dy * uy
                        cross = abs(dx * uy - dy * ux)

                        step1301_travel_m = along
                        step1301_cross_track_m = cross

                        target_yaw = math.atan2(
                            step1300_target[1] - wy,
                            step1300_target[0] - wx,
                        )
                        heading_error = math.degrees(
                            angle_error(target_yaw, current_yaw)
                        )

                        sensor_ok = False
                        tof = None
                        try:
                            if phase4_sensor is not None:
                                measure = phase4_sensor.measure(model, data)
                                origin, direction = phase4_sensor.pose(data)
                                gid = np.array([-1], dtype=np.int32)
                                raw = float(mujoco.mj_ray(
                                    model, data, origin, direction,
                                    phase4_sensor.geomgroup, 1, -1, gid
                                ))
                                maximum = float(
                                    phase4_sensor.sensor.max_distance
                                )
                                if (
                                    math.isfinite(raw)
                                    and math.isfinite(maximum)
                                    and maximum > 0.0
                                ):
                                    if (
                                        measure is None and raw < 0
                                        and int(gid[0]) == -1
                                    ):
                                        tof, sensor_ok = maximum, True
                                    elif (
                                        measure is None and raw > maximum
                                        and int(gid[0]) >= 0
                                    ):
                                        tof, sensor_ok = maximum, True
                                    elif (
                                        measure is not None
                                        and 0 <= raw <= maximum
                                        and int(gid[0]) >= 0
                                        and math.isfinite(float(measure))
                                        and abs(float(measure)-raw) <= 1e-5
                                    ):
                                        tof, sensor_ok = float(measure), True
                        except Exception:
                            sensor_ok = False

                        route_ok = False
                        try:
                            observed = convert_obstacles(safety_map)
                            route_ok = bool(
                                len(observed) >= 2
                                and fixed_segment_clear(
                                    (wx, wy), step1300_target,
                                    observed, 0.15
                                )
                            )
                        except Exception:
                            route_ok = False

                        pose_ok = (
                            all(math.isfinite(float(v)) for v in (
                                wx, wy, wz_height, current_yaw,
                                along, cross, heading_error
                            ))
                            and wz_height >= 0.09
                        )

                        reason = (
                            step1301_reason
                            if step1301_finished else None
                        )
                        if reason is not None:
                            pass
                        elif not (
                            sim_emergency_stop_requested is True
                            and sim_emergency_stop_reason == "TOF_NEAR"
                        ):
                            reason = "LATCH_CHANGED"
                        elif not pose_ok:
                            reason = "INVALID_POSE"
                        elif obstacle_contact_previous:
                            reason = "CONTACT"
                        elif not sensor_ok or tof is None:
                            reason = "TOF_INVALID"
                        elif tof < 0.30:
                            reason = "TOF_NEAR"
                        elif not route_ok:
                            reason = "ROUTE_UNSAFE"
                        elif cross > 0.08:
                            reason = "CROSS_TRACK_LIMIT"
                        elif abs(heading_error) > 25.0:
                            reason = "HEADING_LIMIT"
                        elif along < -0.025:
                            reason = "REVERSE_DRIFT"
                        elif math.hypot(
                            step1300_target[0] - wx,
                            step1300_target[1] - wy,
                        ) <= 0.08:
                            reason = "WP1_REACHED"
                        elif along > math.hypot(
                            step1300_target[0] - step1301_start_xy[0],
                            step1300_target[1] - step1301_start_xy[1],
                        ) + 0.08:
                            reason = "WP1_OVERSHOOT"
                        elif elapsed_walk >= 2.0 and along < 0.03:
                            reason = "INSUFFICIENT_PROGRESS"
                        elif elapsed_walk >= 8.0:
                            reason = "TIMEOUT"

                        if reason is not None:
                            step1301_finished = True
                            step1301_reason = reason
                            safe_command = (0.0, 0.0, 0.0)
                        else:
                            # Small bounded yaw correction during walking.
                            yaw_cmd = max(-0.20, min(
                                0.20,
                                math.radians(heading_error) * 0.5,
                            ))
                            safe_command = (0.30, 0.0, yaw_cmd)

                        if (
                            not all(math.isfinite(float(v))
                                    for v in safe_command)
                            or not 0.0 <= safe_command[0] <= 0.30
                            or safe_command[1] != 0.0
                            or abs(safe_command[2]) > 0.20
                        ):
                            raise RuntimeError(
                                "STEP1301_MOTOR_COMMAND_INVARIANT"
                            )
                        step1301_last_motor_command = safe_command
                        # Step 1302: diagnostic logging only.
                        # No changes to safety-stop conditions.
                        if (
                            not hasattr(
                                step1300_aligner,
                                "_step1302_last_sample"
                            )
                            or float(data.time)
                            - step1300_aligner._step1302_last_sample
                            >= 0.099
                            or step1301_finished
                        ):
                            step1300_aligner._step1302_last_sample = (
                                float(data.time)
                            )
                            print(
                                "[SIM_WP1_WALK_SAMPLE] "
                                + step1272_json.dumps({
                                    "t": round(
                                        elapsed_walk, 3
                                    ),
                                    "xy": [wx, wy],
                                    "yaw_deg": round(
                                        math.degrees(current_yaw),
                                        3,
                                    ),
                                    "target_heading_deg": round(
                                        math.degrees(target_yaw),
                                        3,
                                    ),
                                    "heading_error_deg": round(
                                        heading_error, 3
                                    ),
                                    "along_m": round(along, 5),
                                    "cross_m": round(cross, 5),
                                    "tof_m": tof,
                                    "route_clear_020": route_ok,
                                    "command": list(safe_command),
                                    "stop_reason": reason,
                                }),
                                flush=True,
                            )

                    elif step1300_started:
                        # Simulation-only, scoped yaw permission.
                        # All navigation translation remains blocked.
                        step1300_sensor_ok = False
                        step1300_distance = None
                        try:
                            if phase4_sensor is not None:
                                measured = phase4_sensor.measure(model, data)
                                origin, direction = phase4_sensor.pose(data)
                                gid = np.array([-1], dtype=np.int32)
                                raw = float(mujoco.mj_ray(
                                    model, data, origin, direction,
                                    phase4_sensor.geomgroup, 1, -1, gid
                                ))
                                maximum = float(
                                    phase4_sensor.sensor.max_distance
                                )
                                if (
                                    math.isfinite(raw)
                                    and math.isfinite(maximum)
                                    and maximum > 0
                                ):
                                    if (
                                        measured is None
                                        and raw < 0
                                        and int(gid[0]) == -1
                                    ):
                                        step1300_distance = maximum
                                        step1300_sensor_ok = True
                                    elif (
                                        measured is None
                                        and raw > maximum
                                        and int(gid[0]) >= 0
                                    ):
                                        step1300_distance = maximum
                                        step1300_sensor_ok = True
                                    elif (
                                        measured is not None
                                        and 0 <= raw <= maximum
                                        and int(gid[0]) >= 0
                                        and math.isfinite(float(measured))
                                        and abs(float(measured)-raw) < 1e-5
                                    ):
                                        step1300_distance = float(measured)
                                        step1300_sensor_ok = True
                        except Exception as exc:
                            print(
                                "[SIM_WP1_ALIGN_SENSOR_ERROR] "
                                + type(exc).__name__,
                                flush=True
                            )

                        step1300_pose_ok = bool(
                            all(math.isfinite(float(v)) for v in (
                                data.qpos[qpos_adr],
                                data.qpos[qpos_adr+1],
                                data.qpos[2],
                                current_yaw,
                            ))
                            and float(data.qpos[2]) >= 0.09
                        )
                        step1300_clearance_ok = False
                        try:
                            step1300_clearance_ok, _, _ = (
                                evaluate_sim_turn_clearance(
                                    model, data,
                                    clearance_margin=0.10,
                                    robot_radius=0.10,
                                )
                            )
                        except Exception:
                            pass

                        step1300_cmd = step1300_aligner.command(
                            now=float(data.time),
                            current_yaw=float(current_yaw),
                            emergency_latched=bool(
                                sim_emergency_stop_requested
                            ),
                            emergency_reason=sim_emergency_stop_reason,
                            sensor_healthy=step1300_sensor_ok,
                            distance_m=step1300_distance,
                            pose_healthy=step1300_pose_ok,
                            contact=bool(obstacle_contact_previous),
                            clearance_ok=bool(step1300_clearance_ok),
                        )

                        # Last command decision before set_vel_cmd.
                        if not (
                            sim_emergency_stop_requested is True
                            and sim_emergency_stop_reason == "TOF_NEAR"
                            and step1300_aligner.state == "ALIGNING"
                            and step1300_sensor_ok
                            and step1300_pose_ok
                            and step1300_clearance_ok
                            and not bool(obstacle_contact_previous)
                        ):
                            safe_command = (0.0, 0.0, 0.0)
                        else:
                            safe_command = tuple(step1300_cmd)

                        if (
                            safe_command[0] != 0.0
                            or safe_command[1] != 0.0
                            or abs(safe_command[2]) > 0.18
                            or not all(
                                math.isfinite(float(v))
                                for v in safe_command
                            )
                        ):
                            raise RuntimeError(
                                "WP1_ALIGNMENT_COMMAND_INVARIANT"
                            )
                        step1300_last_cmd = safe_command

                    elif step1288_trial_started:
                        step1288_pose_ok = bool(
                            all(math.isfinite(float(v)) for v in (
                                data.qpos[qpos_adr],
                                data.qpos[qpos_adr + 1],
                                data.qpos[2],
                                current_yaw,
                            ))
                            and float(data.qpos[2]) >= 0.09
                        )

                        # Step 1292: distinguish a valid no-hit from
                        # a faulty or inconsistent ToF measurement.
                        # Applies ONLY to the MuJoCo first-turn trial.
                        step1288_sensor_ok = False
                        step1288_tof = None
                        step1292_tof_state = "INVALID"

                        try:
                            if phase4_sensor is not None:
                                step1292_measured = phase4_sensor.measure(
                                    model, data
                                )

                                step1292_origin, step1292_direction = (
                                    phase4_sensor.pose(data)
                                )
                                step1292_gid = np.array(
                                    [-1], dtype=np.int32
                                )

                                step1292_raw = float(mujoco.mj_ray(
                                    model,
                                    data,
                                    step1292_origin,
                                    step1292_direction,
                                    phase4_sensor.geomgroup,
                                    1,
                                    -1,
                                    step1292_gid,
                                ))

                                step1292_max = float(
                                    phase4_sensor.sensor.max_distance
                                )
                                step1292_geomid = int(step1292_gid[0])

                                if (
                                    math.isfinite(step1292_raw)
                                    and math.isfinite(step1292_max)
                                    and step1292_max > 0.0
                                    and step1292_measured is None
                                    and step1292_raw < 0.0
                                    and step1292_geomid == -1
                                ):
                                    # Valid ray, no geometry intersection.
                                    step1288_tof = step1292_max
                                    step1288_sensor_ok = True
                                    step1292_tof_state = "VALID_NO_HIT"

                                elif (
                                    math.isfinite(step1292_raw)
                                    and math.isfinite(step1292_max)
                                    and step1292_max > 0.0
                                    and step1292_measured is None
                                    and step1292_raw > step1292_max
                                    and step1292_geomid >= 0
                                ):
                                    # Geometry exists beyond sensor range.
                                    step1288_tof = step1292_max
                                    step1288_sensor_ok = True
                                    step1292_tof_state = "VALID_OUT_OF_RANGE"

                                elif (
                                    math.isfinite(step1292_raw)
                                    and 0.0 <= step1292_raw <= step1292_max
                                    and step1292_geomid >= 0
                                    and step1292_measured is not None
                                    and math.isfinite(
                                        float(step1292_measured)
                                    )
                                    and abs(
                                        float(step1292_measured)
                                        - step1292_raw
                                    ) <= 1e-5
                                ):
                                    step1288_tof = float(
                                        step1292_measured
                                    )
                                    step1288_sensor_ok = True
                                    step1292_tof_state = "VALID_HIT"

                        except Exception as step1292_exc:
                            step1288_sensor_ok = False
                            step1292_tof_state = (
                                "SENSOR_EXCEPTION:"
                                + type(step1292_exc).__name__
                            )

                        # Log transitions only, keeping output compact.
                        if (
                            step1292_tof_state
                            != getattr(
                                step1288_gate,
                                "_step1292_last_tof_state",
                                None,
                            )
                        ):
                            print(
                                "[SIM_TOF_CLASSIFY] "
                                f"t={float(data.time):.3f} "
                                f"state={step1292_tof_state} "
                                f"distance={step1288_tof} "
                                f"healthy={step1288_sensor_ok}",
                                flush=True,
                            )
                            step1288_gate._step1292_last_tof_state = (
                                step1292_tof_state
                            )

                        step1288_clearance_ok = False
                        try:
                            if step1288_pose_ok:
                                step1288_clearance_ok, _, _ = (
                                    evaluate_sim_turn_clearance(
                                        model,
                                        data,
                                        clearance_margin=0.10,
                                        robot_radius=0.10,
                                    )
                                )
                        except Exception:
                            step1288_clearance_ok = False

                        step1288_turn_complete = (
                            sim_turn_controller.state
                            is SimTurnState.READY_TO_RESUME
                        )
                        step1288_turn_aborted = (
                            sim_turn_controller.state
                            is SimTurnState.ABORTED
                        )

                        step1288_proposed_wz = (
                            float(sim_turn_command[2])
                            if sim_turn_command is not None
                            else 0.0
                        )

                        # Step 1290: Diagnostic only; do not alter gate inputs.
                        print(
                            "[SIM_GATE_DIAG] "
                            + step1272_json.dumps({
                                "time": float(data.time),
                                "sensor_healthy": bool(step1288_sensor_ok),
                                "tof_distance": (
                                    str(step1288_tof)
                                    if step1288_tof is not None else None
                                ),
                                "pose_healthy": bool(step1288_pose_ok),
                                "contact": bool(obstacle_contact_previous),
                                "clearance_ok": bool(
                                    step1288_clearance_ok
                                ),
                                "controller_state":
                                    sim_turn_controller.state.value,
                                "proposed_wz": step1288_proposed_wz,
                                "gate_state": step1288_gate.state,
                                "emergency_latched": bool(
                                    sim_emergency_stop_requested
                                ),
                            }),
                            flush=True,
                        )

                        step1288_proposal = step1288_gate.command(
                            now=float(data.time),
                            proposed_yaw_rate=step1288_proposed_wz,
                            emergency_latched=bool(
                                sim_emergency_stop_requested
                            ),
                            emergency_reason=sim_emergency_stop_reason,
                            sensor_healthy=step1288_sensor_ok,
                            tof_distance_m=step1288_tof,
                            pose_healthy=step1288_pose_ok,
                            contact=bool(obstacle_contact_previous),
                            clearance_ok=bool(step1288_clearance_ok),
                            turn_complete=step1288_turn_complete,
                            turn_aborted=step1288_turn_aborted,
                        )

                        safe_command = arbitrate_first_turn(
                            normal_command=safe_command,
                            emergency_latched=bool(
                                sim_emergency_stop_requested
                            ),
                            emergency_reason=sim_emergency_stop_reason,
                            trial_enabled=True,
                            gate=step1288_gate,
                            gate_proposal=step1288_proposal,
                            explicit_trial_permission=bool(
                                step1288_trial_started
                                and not step1288_trial_finished
                            ),
                        )

                        if safe_command[:2] != (0.0, 0.0):
                            raise RuntimeError(
                                "FIRST_TURN_TRANSLATION_FORBIDDEN"
                            )
                        if abs(float(safe_command[2])) > 0.50:
                            raise RuntimeError(
                                "FIRST_TURN_YAW_LIMIT_FAILED"
                            )

                        if step1288_gate.state in ("COMPLETE", "BLOCKED"):
                            step1288_trial_finished = True
                            step1288_trial_reason = (
                                "TURN_COMPLETE"
                                if step1288_gate.state == "COMPLETE"
                                else str(step1288_gate.reason)
                            )
                            safe_command = (0.0, 0.0, 0.0)

                    else:
                        safe_command = apply_sim_final_stop_gate(
                            safe_command,
                            emergency_stop=sim_emergency_stop_requested,
                        )
                        if sim_emergency_stop_requested:
                            if safe_command != (0.0, 0.0, 0.0):
                                raise RuntimeError(
                                    "Final stop gate failed to zero velocity"
                                )
                            if not getattr(
                                sim_turn_controller,
                                "_stop_gate_logged",
                                False,
                            ):
                                print(
                                    "[SIM_FINAL_GATE] STOP_APPLIED "
                                    f"command={safe_command}"
                                )
                                sim_turn_controller._stop_gate_logged = True

                # Step1349-R: separate yaw-only simulation trial gate.
                if step1349_mode == "TURNING":
                    if step1337b_state == "RIGHT_TURN":
                        permitted = (
                            sim_emergency_stop_requested is True
                            and sim_emergency_stop_reason == "TOF_NEAR"
                            and not obstacle_contact_previous
                            and obstacle_contact_events == 0
                            and sim_turn_policy_active
                            and sim_turn_controller.state not in (
                                SimTurnState.TURNING,
                                SimTurnState.SETTLING,
                            )
                            and policy.current_policy == "ppo_v3_turn"
                            and all(
                                math.isfinite(float(v))
                                for v in step1349_command
                            )
                            and step1349_command[:2] == (0.0, 0.0)
                            and abs(step1349_command[2]) <= 0.50
                        )
                        if permitted:
                            safe_command = tuple(step1349_command)
                        else:
                            safe_command = (0.0, 0.0, 0.0)
                            step1349_reason = "FINAL_GATE_BLOCKED"
                            step1349_mode = "ABORTED"
                            step1337b_state = "TURN_DONE"
                            policy.set_vel_cmd(0.0, 0.0, 0.0)
                            last_nav_command = (0.0, 0.0, 0.0)
                            if sim_turn_policy_active:
                                policy.deactivate_ppo_v3_turn()
                                sim_turn_policy_active = False
                    else:
                        safe_command = (0.0, 0.0, 0.0)

                if step1349_mode in ("DONE", "ABORTED"):
                    safe_command = (0.0, 0.0, 0.0)

                if step1356_state == "RUNNING":
                    walking_permitted = (
                        step1337b_state == "WP2_SHORT_WALK"
                        and sim_emergency_stop_requested is True
                        and sim_emergency_stop_reason == "TOF_NEAR"
                        and not obstacle_contact_previous
                        and obstacle_contact_events == 0
                        and not sim_turn_policy_active
                        and not getattr(
                            policy, "ppo_v3_turn_active", False
                        )
                        and policy.current_policy in (
                            "standing", "walking"
                        )
                        and all(
                            math.isfinite(float(v))
                            for v in step1356_command
                        )
                        and 0.0 <= step1356_command[0] <= 0.30
                        and step1356_command[1] == 0.0
                        and abs(step1356_command[2]) <= 0.12
                    )

                    if walking_permitted:
                        safe_command = tuple(step1356_command)
                    else:
                        safe_command = (0.0,0.0,0.0)
                        policy.set_vel_cmd(0.0,0.0,0.0)
                        last_nav_command = (0.0,0.0,0.0)
                        step1356_reason = "FINAL_GATE_BLOCKED"
                        step1356_state = "FINISHED"
                        step1337b_state = "WP2_WALK_DONE"

                if step1356_state == "FINISHED":
                    safe_command = (0.0,0.0,0.0)

                if safe_command != last_nav_command:
                    policy.set_vel_cmd(*safe_command)
                    last_nav_command = safe_command

                    print(
                        f"[NAV] t={sim_elapsed:.2f}s "
                        f"Mode={nav_mode} "
                        f"Distance={distance:.3f}m "
                        f"Error={error_deg:.1f}deg "
                        f"Cmd={nav_command}"
                    )

                # === AUTO NAVIGATION END ===

                # === POSE LOG START ===
                if data.time - last_pose_log_time >= 1.0:
                    x = float(data.qpos[qpos_adr])
                    y = float(data.qpos[qpos_adr + 1])

                    quat = data.qpos[qpos_adr + 3:qpos_adr + 7]
                    qw, qx, qy, qz = quat

                    yaw = math.atan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz)
                    )

                    print(
                        f"[POSE] t={data.time - auto_nav_start_time:.2f}s "
                        f"X={x:.3f} Y={y:.3f} "
                        f"Yaw={math.degrees(yaw):.1f}°"
                    )


                    # === GOAL CALCULATION START ===
                    goal_x = 1.0
                    goal_y = 0.5

                    dx = goal_x - x
                    dy = goal_y - y

                    goal_distance = math.hypot(dx, dy)
                    goal_heading = math.atan2(dy, dx)

                    heading_error = math.atan2(
                        math.sin(goal_heading - yaw),
                        math.cos(goal_heading - yaw)
                    )

                    print(
                        f"[GOAL] Target=({goal_x:.2f}, {goal_y:.2f}) "
                        f"Distance={goal_distance:.3f}m "
                        f"Heading={math.degrees(goal_heading):.1f}deg "
                        f"Error={math.degrees(heading_error):.1f}deg"
                    )
                    # === GOAL CALCULATION END ===

                    last_pose_log_time = data.time
                # === POSE LOG END ===

                if not policy_enabled and policy_enable_time is not None:
                    if step_start >= policy_enable_time:
                        policy_enabled = True
                        if original_kp is not None:
                            set_standby_gains(False)
                            print("Policy inference enabled (after 1s standby)")
                            print(f"  Restored original kp gains (range: [{original_kp.min():.2f}, {original_kp.max():.2f}])")

                actual_dt = step_start - prev_step_time
                prev_step_time = step_start

                policy.update_ground_pick_phase(actual_dt)
                policy.update_behavior(actual_dt)

                if policy_enabled:
                    action = policy.infer()
                    policy.apply_action(action)
                else:
                    # Paused: keep last ctrl, don't query the policy. Motors
                    # hold position. Use a zero action just so downstream
                    # logging (csv/debug) sees something consistent.
                    action = np.zeros(policy.n_joints, dtype=np.float32)

                control_step_count += 1

                # Track BODY-frame forward/lateral velocity + yaw rate, print the
                # 1-second moving average once per second vs the commanded values.
                # Body frame so "forward" / "turn" are directly comparable to the
                # command (which is in the robot frame): lets us see if the policy
                # actually achieves commanded forward speed and turn rate.
                quat = data.qpos[qpos_adr + 3:qpos_adr + 7].astype(np.float32)
                v_world = np.array([
                    data.qvel[_trunk_qvel_adr + 0],
                    data.qvel[_trunk_qvel_adr + 1],
                    data.qvel[_trunk_qvel_adr + 2],
                ], dtype=np.float32)
                v_body = policy.quat_rotate_inverse(quat, v_world)
                yaw_rate = float(data.qvel[_trunk_qvel_adr + 5])  # body-frame wz
                vel_history.append((float(v_body[0]), float(v_body[1]), yaw_rate))
                if control_step_count % _vel_window_steps == 0 and len(vel_history) > 0:
                    n = len(vel_history)
                    avg_fwd = sum(v[0] for v in vel_history) / n
                    avg_lat = sum(v[1] for v in vel_history) / n
                    avg_yaw = sum(v[2] for v in vel_history) / n
                    cmd_x, cmd_y, cmd_yaw = policy.vel_cmd[0], policy.vel_cmd[1], policy.vel_cmd[2]
                    trunk_z = float(data.qpos[qpos_adr + 2])
                    print(
                        f"[vel 1s avg] achieved/cmd  fwd={avg_fwd:+.2f}/{cmd_x:+.2f}  "
                        f"lat={avg_lat:+.2f}/{cmd_y:+.2f} m/s  "
                        f"yaw={avg_yaw:+.2f}/{cmd_yaw:+.2f} rad/s   "
                        f"trunk_z={trunk_z*1000:.1f} mm"
                    )

                if csv_data is not None:
                    obs = policy.get_observations()
                    row = {'step': control_step_count, 'time': control_step_count * control_dt}
                    for i in range(obs.size):
                        row[f'obs_{i}'] = obs[i]
                    for i in range(action.size):
                        row[f'action_{i}'] = action[i]
                    csv_data.append(row)

                if recorded_observations is not None:
                    obs = policy.get_observations()
                    timestamp = time.time() - start_time
                    recorded_observations.append({'timestamp': timestamp, 'observation': obs.tolist()})

                if args.debug:
                    should_print = control_step_count <= 10 or control_step_count % 50 == 0
                    if should_print:
                        obs = policy.get_observations()
                        pos = data.qpos[qpos_adr:qpos_adr + 3]
                        quat = data.qpos[qpos_adr + 3:qpos_adr + 7]
                        com_height = pos[2]

                        print(f"\n{'='*70}")
                        print(f"Step {control_step_count} DEBUG:")
                        print(f"{'='*70}")
                        print(f"Active policy: {policy.current_policy}")
                        print(f"Base state:")
                        print(f"  Position: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
                        print(f"  CoM height: {com_height:7.4f}")
                        print(f"  Quaternion: [{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                        print(f"\nObservation (shape {obs.shape}, total {obs.size}):")
                        print(f"  Ang vel [0:3]:        {obs[0:3]}")
                        print(f"  Proj grav [3:6]:      {obs[3:6]}")
                        print(f"  Joint pos [6:{6+policy.n_joints}]:     {obs[6:6+policy.n_joints]}")
                        print(f"  Joint vel [{6+policy.n_joints}:{6+2*policy.n_joints}]:    {obs[6+policy.n_joints:6+2*policy.n_joints]}")
                        print(f"  Last action [{6+2*policy.n_joints}:{6+3*policy.n_joints}]:  {obs[6+2*policy.n_joints:6+3*policy.n_joints]}")
                        cmd_end = 6+3*policy.n_joints+3
                        print(f"  Command [{6+3*policy.n_joints}:{cmd_end}]:      {obs[6+3*policy.n_joints:cmd_end]}")
                        if policy.current_policy == "standing":
                            print(f"  Body cmd (raw): z={policy.body_cmd[0]*1000:.1f}mm  pitch={math.degrees(policy.body_cmd[1]):.1f}°  roll={math.degrees(policy.body_cmd[2]):.1f}°")
                        print(f"\nAction output:")
                        print(f"  Raw action: {action}")
                        print(f"  Action min/max: [{action.min():.4f}, {action.max():.4f}]")
                        if policy.use_delay:
                            print(f"  Delay: {policy.current_lag} timesteps (buffered)")
                        ctrl_kind = "torque [Nm]" if bam_ctrl is not None else "position target"
                        print(f"  Applied ctrl ({ctrl_kind}, first 5): {data.ctrl[:5]}")
                        print(f"  Applied ctrl ({ctrl_kind}, last 5):  {data.ctrl[-5:]}")

                for _ in range(decimation):
                    if bam_ctrl is not None:
                        # BAM owns control/torque/friction: update() runs the
                        # firmware P-loop + DC-motor equation, writes the torque
                        # to data.ctrl and pushes the friction budget onto the
                        # dofs so MuJoCo's solver applies it on this step.
                        bam_ctrl.update()
                    mujoco.mj_step(model, data)
                    if (
                        sim_turn_test_enabled
                        and sim_turn_start_pose is not None
                        and not sim_turn_finish_logged
                    ):
                        sim_turn_trajectory.append((
                            float(data.time),
                            float(data.qpos[qpos_adr]),
                            float(data.qpos[qpos_adr + 1]),
                        ))

                    # Step 9-7-10: Obstacle contact detection
                    obstacle_contact_now = False

                    if obstacle_geom_ids:
                        for ci in range(data.ncon):
                            contact = data.contact[ci]
                            geom1 = int(contact.geom1)
                            geom2 = int(contact.geom2)
                            if (
                                (
                                    geom1 in obstacle_geom_ids
                                    and geom2 in robot_geom_ids
                                )
                                or (
                                    geom2 in obstacle_geom_ids
                                    and geom1 in robot_geom_ids
                                )
                            ):
                                obstacle_contact_now = True
                                break

                    if obstacle_contact_now:
                        obstacle_contact_steps += 1

                        if not obstacle_contact_previous:
                            obstacle_contact_events += 1
                            if obstacle_first_contact_time is None:
                                obstacle_first_contact_time = data.time
                            print(
                                f"[CONTACT] Obstacle contact detected! "
                                f"t={data.time:.2f}s "
                                f"events={obstacle_contact_events}"
                            )

                    obstacle_contact_previous = obstacle_contact_now

                if odom_compare is not None:
                    odom_compare.step(data)
                    if control_step_count % 50 == 0:
                        odom_compare.print_status(data)
                if odom_overlay is not None and viewer is not None:
                    odom_overlay.draw(data, viewer.user_scn, odom_compare)
                if viewer is not None:
                    viewer.sync()

                # Phase 2: automatic termination in headless mode.
                # Contact detection and state updates have completed.
                if args.headless:
                    # Verify the current-to-waypoint segment before walking.
                    if (
                        sim_turn_test_enabled
                        and sim_turn_controller.state is SimTurnState.READY_TO_RESUME
                        and not sim_resume_path_checked
                    ):
                        sim_resume_path_checked = True
                        sim_resume_path_safe = False

                        if (
                            phase4_replan_ready
                            and sim_resume_obstacles is not None
                            and nav_waypoint_index < len(nav_waypoints)
                        ):
                            resume_x = float(data.qpos[qpos_adr])
                            resume_y = float(data.qpos[qpos_adr + 1])
                            resume_target_x, resume_target_y = nav_waypoints[
                                nav_waypoint_index
                            ]

                            if all(math.isfinite(v) for v in (
                                resume_x,
                                resume_y,
                                resume_target_x,
                                resume_target_y,
                            )):
                                sim_precision_path_applied = False
                                if sim_resume_obstacles is not None:
                                    from sim_path_feasibility import evaluate_path_feasibility

                                    precision_start = (resume_x, resume_y)
                                    precision_wp = (0.10, 0.50)
                                    precision_goal = (
                                        float(config["goal"]["x"]),
                                        float(config["goal"]["y"]),
                                    )
                                    precision_path = [
                                        precision_start,
                                        precision_wp,
                                        precision_goal,
                                    ]

                                    precision_feasible, precision_reason = evaluate_path_feasibility(
                                        precision_path,
                                        sim_resume_obstacles,
                                        margin=0.07,
                                        turn_drift_budget=0.08,
                                        initial_yaw=float(current_yaw),
                                        turn_threshold_deg=8.0,
                                    )

                                    print(
                                        f"[SIM_PRECISION_PATH] "
                                        f"feasible={precision_feasible} "
                                        f"reason={precision_reason} "
                                        f"start={precision_start} "
                                        f"wp={precision_wp} "
                                        f"goal={precision_goal}"
                                    )
                                    if precision_feasible:
                                        precision_heading = math.atan2(
                                            precision_wp[1] - resume_y,
                                            precision_wp[0] - resume_x,
                                        )
                                        precision_heading_error = math.degrees(
                                            angle_error(precision_heading, float(current_yaw))
                                        )

                                        if abs(precision_heading_error) <= 8.0:
                                            nav_waypoints = [precision_wp, precision_goal]
                                            nav_waypoint_index = 0
                                            nav_goal_x, nav_goal_y = precision_wp
                                            sim_precision_path_applied = True
                                            nav_mode = "DRIVE"

                                            print(
                                                f"[SIM_PRECISION_APPLY] "
                                                f"waypoints={nav_waypoints} "
                                                f"nav_mode={nav_mode} "
                                                f"heading_error={precision_heading_error:.2f}deg"
                                            )
                                        else:
                                            print(
                                                f"[SIM_PRECISION_APPLY] BLOCKED "
                                                f"heading_error={precision_heading_error:.2f}deg"
                                            )
                                    else:
                                        print(
                                            f"[SIM_PRECISION_APPLY] BLOCKED "
                                            f"reason={precision_reason}"
                                        )
                                print(
                                    f"[SIM_RESUME_DIAG] "
                                    f"start=({resume_x:.4f},{resume_y:.4f}) "
                                    f"target=({resume_target_x:.4f},{resume_target_y:.4f}) "
                                    f"margin={margin!r} "
                                    f"obstacles={sim_resume_obstacles!r}"
                                )
                                resume_check_margin = (
                                    0.10 if sim_precision_path_applied else margin
                                )
                                if sim_precision_path_applied:
                                    resume_target_x, resume_target_y = nav_waypoints[
                                        nav_waypoint_index
                                    ]
                                print(
                                    f"[SIM_PRECISION_CHECK] "
                                    f"applied={sim_precision_path_applied} "
                                    f"margin={resume_check_margin} "
                                    f"target=({resume_target_x:.3f},{resume_target_y:.3f})"
                                )
                                sim_resume_path_safe = bool(
                                    fixed_segment_clear(
                                        (resume_x, resume_y),
                                        (resume_target_x, resume_target_y),
                                        sim_resume_obstacles,
                                        resume_check_margin,
                                    )
                                )

                                resume_heading = math.atan2(
                                    resume_target_y - resume_y,
                                    resume_target_x - resume_x,
                                )

                                resume_error = math.degrees(
                                    angle_error(
                                        resume_heading,
                                        float(current_yaw),
                                    )
                                )

                                print(
                                    f"[SIM_RESUME] PATH CHECK "
                                    f"safe={sim_resume_path_safe} "
                                    f"heading_error={resume_error:.2f}deg "
                                    f"target=({resume_target_x:.3f},"
                                    f"{resume_target_y:.3f})"
                                )
                                if (
                                    sim_precision_path_applied
                                    and sim_resume_path_safe
                                    and sim_turn_controller.state is SimTurnState.READY_TO_RESUME
                                    and not sim_precision_walk_started
                                ):
                                    if sim_precision_resume.begin_validation():
                                        resume_fallen = (
                                            not math.isfinite(float(data.qpos[2]))
                                            or float(data.qpos[2]) < 0.09
                                        )
                                        resume_allowed = sim_precision_resume.validate(
                                            pose=(resume_x, resume_y, float(current_yaw)),
                                            heading_error_deg=resume_error,
                                            path_verified=True,
                                            contact=bool(obstacle_contact_previous),
                                            fallen=resume_fallen,
                                        )
                                        if resume_allowed:
                                            sim_precision_walk_started = True
                                            nav_goal_tolerance = 0.03
                                            print(
                                                f"[SIM_WP_TOLERANCE] "
                                                f"value={nav_goal_tolerance:.3f}m "
                                                f"target=({nav_goal_x:.3f},{nav_goal_y:.3f})"
                                            )
                                            sim_precision_walk_start = (resume_x, resume_y)
                                            sim_precision_walk_waypoint_index = nav_waypoint_index
                                            sim_precision_walk_start_time = float(data.time)
                                            print(
                                                f"[SIM_PRECISION_WALK] START "
                                                f"target=({nav_goal_x:.3f},{nav_goal_y:.3f}) "
                                                f"cmd=(0.30,0.00,0.00)"
                                            )
                                        else:
                                            print(
                                                f"[SIM_PRECISION_WALK] BLOCKED "
                                                f"reason={sim_precision_resume.stop_reason}"
                                            )

                        if not sim_resume_path_safe:
                            print("[SIM_RESUME] BLOCKED: path not verified")
                    # Step 1337-B: VIA termination has priority.
                    if step1337b_state is not None:
                        if (
                            step1337b_state == "FINISHED"
                            and step1337b_reason == "VIA_REACHED"
                            and step1349_mode == "WAITING"
                        ):
                            px = float(data.qpos[qpos_adr])
                            py = float(data.qpos[qpos_adr + 1])
                            yaw = float(current_yaw)
                            target = math.atan2(1.00 - py, 0.95 - px)

                            prestart_ok = (
                                sim_emergency_stop_requested is True
                                and sim_emergency_stop_reason == "TOF_NEAR"
                                and obstacle_contact_events == 0
                                and not obstacle_contact_previous
                                and policy.current_policy == "standing"
                                and not sim_turn_policy_active
                                and sim_turn_controller.state not in (
                                    SimTurnState.TURNING,
                                    SimTurnState.SETTLING,
                                )
                                and math.isfinite(yaw)
                                and float(data.qpos[2]) >= 0.09
                            )

                            clearance_ok = False
                            route_ok = False

                            try:
                                clearance_ok, _, _ = evaluate_sim_turn_clearance(
                                    model, data,
                                    clearance_margin=0.10,
                                    robot_radius=0.10,
                                )
                                known = convert_obstacles(safety_map)
                                route_ok = bool(
                                    len(known) >= 2
                                    and fixed_segment_clear(
                                        (px, py), (0.95, 1.00),
                                        known, 0.15,
                                    )
                                )
                            except Exception:
                                clearance_ok = False
                                route_ok = False

                            if prestart_ok and clearance_ok and route_ok:
                                started = step1349_controller.start(
                                    current_time=float(data.time),
                                    target_yaw=target,
                                )

                                if started:
                                    step1349_origin = (px, py)
                                    step1349_target_yaw = target
                                    step1349_start_time = float(data.time)
                                    step1349_mode = "TURNING"
                                    step1337b_state = "RIGHT_TURN"

                                    print(
                                        "[SIM_WP2_TURN_START] "
                                        + step1288_json.dumps({
                                            "xy": [px, py],
                                            "yaw_deg": math.degrees(yaw),
                                            "target_deg": math.degrees(target),
                                            "required_deg": math.degrees(
                                                angle_error(target, yaw)
                                            ),
                                            "clearance_ok": True,
                                            "route_ok": True,
                                        }),
                                        flush=True,
                                    )
                                    continue

                            step1349_mode = "ABORTED"
                            step1349_reason = "PRESTART_GATE_BLOCKED"
                            step1337b_state = "TURN_DONE"

                        if step1337b_state == "TURN_DONE":
                            policy.set_vel_cmd(0.0, 0.0, 0.0)

                            print(
                                "[SIM_WP2_TURN_RESULT] "
                                + step1288_json.dumps({
                                    "reason": step1349_reason,
                                    "xy": [
                                        float(data.qpos[qpos_adr]),
                                        float(data.qpos[qpos_adr + 1]),
                                    ],
                                    "yaw_deg": math.degrees(current_yaw),
                                    "target_deg": (
                                        math.degrees(step1349_target_yaw)
                                        if step1349_target_yaw is not None
                                        else None
                                    ),
                                    "max_drift_m": step1349_max_drift,
                                    "contacts": obstacle_contact_events,
                                    "final_command": [0.0, 0.0, 0.0],
                                    "emergency_latched": bool(
                                        sim_emergency_stop_requested
                                    ),
                                    "wp2_started": False,
                                }),
                                flush=True,
                            )
                            if (
                                step1349_reason == "TURN_COMPLETE"
                                and step1349_mode == "DONE"
                                and step1356_state == "WAITING"
                                and obstacle_contact_events == 0
                                and sim_emergency_stop_requested is True
                                and sim_emergency_stop_reason == "TOF_NEAR"
                                and not sim_turn_policy_active
                                and not getattr(policy, "ppo_v3_turn_active", False)
                                and policy.current_policy in ("standing", "walking")
                            ):
                                step1356_state = "RUNNING"
                                step1356_start_time = float(data.time)
                                step1356_origin = (
                                    float(data.qpos[qpos_adr]),
                                    float(data.qpos[qpos_adr+1]),
                                )
                                step1337b_state = "WP2_SHORT_WALK"
                                last_nav_command = (0.0,0.0,0.0)
                                print(
                                    "[SIM_WP2_WALK_START] "
                                    + step1288_json.dumps({
                                        "xy": list(step1356_origin),
                                        "yaw_deg": math.degrees(current_yaw),
                                        "target_xy": [0.95,1.00],
                                        "duration_limit_s": 7.50,
                                        "cmd_vx_limit": 0.30,
                                    }),
                                    flush=True,
                                )
                                continue
                            break
                        if step1337b_state == "WP2_WALK_DONE":
                            policy.set_vel_cmd(0.0,0.0,0.0)
                            last_nav_command = (0.0,0.0,0.0)
                            ex = float(data.qpos[qpos_adr])
                            ey = float(data.qpos[qpos_adr+1])
                            print(
                                "[SIM_WP2_WALK_RESULT] "
                                + step1288_json.dumps({
                                    "reason": step1356_reason,
                                    "end_xy": [ex,ey],
                                    "walked_m": math.hypot(ex-step1356_origin[0], ey-step1356_origin[1]),
                                    "remaining_m": math.hypot(0.95-ex,1.00-ey),
                                    "contacts": obstacle_contact_events,
                                    "emergency_latched": bool(sim_emergency_stop_requested),
                                    "final_command": [0.0,0.0,0.0],
                                    "wp2_reached": step1356_reason == "WP2_REACHED",
                                }), flush=True,
                            )
                            break
                        if step1337b_state == "FINISHED":
                            policy.set_vel_cmd(0.0, 0.0, 0.0)
                            last_nav_command = (0.0, 0.0, 0.0)
                            px = float(data.qpos[qpos_adr])
                            py = float(data.qpos[qpos_adr + 1])
                            print(
                                "[SIM_VIA_WALK_RESULT] "
                                + step1288_json.dumps({
                                    "reason": step1337b_reason,
                                    "end_xy": [px, py],
                                    "remaining_m": math.hypot(
                                        0.26-px, 1.10-py
                                    ),
                                    "contacts": obstacle_contact_events,
                                    "final_command": [0.0, 0.0, 0.0],
                                    "emergency_latched": bool(
                                        sim_emergency_stop_requested
                                    ),
                                    "wp2_started": False,
                                }), flush=True
                            )
                            print(
                                "[HEADLESS] FINISHED "
                                "status=SIM_VIA_SHORT_TRIAL_END",
                                flush=True,
                            )
                            break
                        continue

                    # Step 1288: first-turn-only headless exit.
                    if step1301_started:
                        if step1301_finished:
                            policy.set_vel_cmd(0.0, 0.0, 0.0)
                            last_nav_command = (0.0, 0.0, 0.0)
                            end_xy = [
                                float(data.qpos[qpos_adr]),
                                float(data.qpos[qpos_adr + 1]),
                            ]
                            distance_wp1 = math.hypot(
                                step1300_target[0] - end_xy[0],
                                step1300_target[1] - end_xy[1],
                            )
                            print(
                                "[SIM_WP1_SHORT_RESULT] "
                                + step1288_json.dumps({
                                    "reason": step1301_reason,
                                    "start_xy": list(
                                        step1301_start_xy
                                    ),
                                    "end_xy": end_xy,
                                    "along_track_m":
                                        step1301_travel_m,
                                    "cross_track_m":
                                        step1301_cross_track_m,
                                    "remaining_wp1_m": distance_wp1,
                                    "elapsed_s": (
                                        float(data.time)
                                        - step1301_start_time
                                    ),
                                    "contacts":
                                        obstacle_contact_events,
                                    "emergency_latched": bool(
                                        sim_emergency_stop_requested
                                    ),
                                    "final_command":
                                        [0.0, 0.0, 0.0],
                                    "wp2_started": False,
                                }),
                                flush=True,
                            )
                            # Step 1337-B: authorize VIA HOLD only
                            # after verified WP1 success and zero command.
                            via_start_allowed = bool(
                                step1301_reason == "WP1_REACHED"
                                and distance_wp1 <= 0.08
                                and obstacle_contact_events == 0
                                and sim_emergency_stop_requested is True
                                and sim_emergency_stop_reason == "TOF_NEAR"
                                and policy.current_policy in ("standing", "walking")
                                and not sim_turn_policy_active
                                and last_nav_command
                                    == (0.0, 0.0, 0.0)
                                and all(math.isfinite(v) for v in (
                                    end_xy[0], end_xy[1],
                                    float(data.qpos[2]),
                                    float(current_yaw),
                                ))
                                and float(data.qpos[2]) >= 0.09
                            )

                            if via_start_allowed:
                                step1337b_state = "HOLD"
                                step1337b_hold_time = float(data.time)
                                step1337b_origin = tuple(end_xy)
                                step1337b_reason = None
                                print(
                                    "[SIM_VIA_HOLD_START] "
                                    + step1288_json.dumps({
                                        "start_xy": end_xy,
                                        "target_xy": [0.26, 1.10],
                                        "hold_s": 0.60,
                                    }), flush=True
                                )
                                continue

                            print(
                                "[HEADLESS] FINISHED "
                                "status=SIM_WP1_SHORT_TRIAL_END",
                                flush=True,
                            )
                            break
                        continue

                    if step1300_started:
                        if step1300_aligner.state in (
                            "COMPLETE", "BLOCKED"
                        ):
                            policy.set_vel_cmd(0.0, 0.0, 0.0)
                            last_nav_command = (0.0, 0.0, 0.0)
                            if sim_turn_policy_active:
                                policy.deactivate_ppo_v3_turn()
                                sim_turn_policy_active = False

                            x = float(data.qpos[qpos_adr])
                            y = float(data.qpos[qpos_adr + 1])
                            yaw = float(current_yaw)
                            heading = math.atan2(
                                step1300_target[1]-y,
                                step1300_target[0]-x,
                            )
                            err = math.degrees(
                                angle_error(heading, yaw)
                            )
                            drift = math.hypot(
                                x-step1300_start_xy[0],
                                y-step1300_start_xy[1],
                            )
                            print(
                                "[SIM_WP1_ALIGN_RESULT] "
                                + step1288_json.dumps({
                                    "state": step1300_aligner.state,
                                    "reason": step1300_aligner.reason,
                                    "post_align_xy": [x, y],
                                    "post_align_yaw_deg":
                                        math.degrees(yaw),
                                    "heading_error_deg": err,
                                    "drift_m": drift,
                                    "contacts": obstacle_contact_events,
                                    "stop_latched": bool(
                                        sim_emergency_stop_requested
                                    ),
                                    "final_command": [0.0, 0.0, 0.0],
                                    "wp1_walking_started": False,
                                }),
                                flush=True,
                            )
                            # Transition only after confirmed alignment.
                            walk_ready = False
                            if step1300_aligner.state == "COMPLETE":
                                try:
                                    observed = convert_obstacles(safety_map)
                                    walk_heading = math.atan2(
                                        step1300_target[1] - y,
                                        step1300_target[0] - x,
                                    )
                                    walk_ready = bool(
                                        sim_emergency_stop_requested
                                        and sim_emergency_stop_reason
                                        == "TOF_NEAR"
                                        and obstacle_contact_events == 0
                                        and abs(math.degrees(
                                            angle_error(
                                                step1326_bias_target_yaw, yaw
                                            )
                                        )) <= 5.0
                                        and len(observed) >= 2
                                        and fixed_segment_clear(
                                            (x, y), step1300_target,
                                            observed, 0.20
                                        )
                                        and fixed_segment_clear(
                                            step1300_target,
                                            step1300_next,
                                            observed, 0.20
                                        )
                                    )
                                except Exception:
                                    walk_ready = False

                            if walk_ready:
                                distance_target = math.hypot(
                                    step1300_target[0] - x,
                                    step1300_target[1] - y,
                                )
                                if distance_target <= 0.15:
                                    walk_ready = False

                            print(
                                "[SIM_WP1_SHORT_START] "
                                + step1288_json.dumps({
                                    "authorized": walk_ready,
                                    "start_xy": [x, y],
                                    "target_xy":
                                        list(step1300_target),
                                    "heading_error_deg": err,
                                    "margin_m": 0.20,
                                    "distance_cap_m": 0.10,
                                    "time_cap_s": 3.0,
                                }),
                                flush=True,
                            )

                            if not walk_ready:
                                print(
                                    "[HEADLESS] FINISHED "
                                    "status=SIM_WP1_SHORT_NOT_AUTHORIZED",
                                    flush=True,
                                )
                                break

                            step1301_start_xy = (x, y)
                            segment_dx = (
                                step1300_target[0] - x
                            )
                            segment_dy = (
                                step1300_target[1] - y
                            )
                            segment_length = math.hypot(
                                segment_dx, segment_dy
                            )
                            step1301_direction = (
                                segment_dx / segment_length,
                                segment_dy / segment_length,
                            )
                            step1301_start_time = float(data.time)
                            policy.set_vel_cmd(0.0, 0.0, 0.0)
                            if policy.current_policy != "standing":
                                raise RuntimeError(
                                    "STEP1307_EXPECTED_STANDING"
                                )
                            policy.activate_ppo_v3_turn()
                            sim_turn_policy_active = True
                            step1305_hold_until = (
                                float(data.time) + 0.60
                            )
                            step1305_hold_start_xy = (x, y)
                            step1305_hold_start_yaw = float(
                                current_yaw
                            )
                            step1301_started = True
                            step1300_started = False
                            print(
                                "[SIM_WP1_HOLD_START] "
                                + step1288_json.dumps({
                                    "duration_s": 0.60,
                                    "policy": policy.current_policy,
                                    "start_xy": [x, y],
                                    "yaw_deg": math.degrees(
                                        current_yaw
                                    ),
                                    "walking_command":
                                        [0.0, 0.0, 0.0],
                                }),
                                flush=True,
                            )
                            continue
                        continue

                    if step1288_trial_started:
                        step1288_elapsed = (
                            float(data.time)
                            - step1288_trial_start_time
                        )
                        step1288_exit = (
                            step1288_trial_finished
                            or step1288_elapsed >= 4.10
                        )
                        if step1288_exit:
                            if not step1288_trial_finished:
                                step1288_trial_reason = (
                                    "OUTER_TURN_TIMEOUT"
                                )
                                step1288_gate.block(
                                    step1288_trial_reason
                                )

                            # No waypoint walking is allowed.
                            policy.set_vel_cmd(0.0, 0.0, 0.0)
                            step1288_trial_finished = True

                            if sim_turn_policy_active:
                                policy.deactivate_ppo_v3_turn()
                                sim_turn_policy_active = False

                            step1288_end_xy = (
                                float(data.qpos[qpos_adr]),
                                float(data.qpos[qpos_adr + 1]),
                            )
                            step1288_end_yaw = float(current_yaw)
                            step1288_error = math.degrees(
                                angle_error(
                                    step1288_trial_target_yaw,
                                    step1288_end_yaw,
                                )
                            )
                            step1288_drift = math.hypot(
                                step1288_end_xy[0]
                                - step1288_trial_start_xy[0],
                                step1288_end_xy[1]
                                - step1288_trial_start_xy[1],
                            )

                            import json as step1288_json
                            # Step 1295: read-only WP1 approach diagnosis.
                            # Do not start walking or change the stop latch.
                            step1295_wp1 = (0.25, 0.95)
                            step1295_dx = (
                                step1295_wp1[0] - step1288_end_xy[0]
                            )
                            step1295_dy = (
                                step1295_wp1[1] - step1288_end_xy[1]
                            )
                            step1295_distance = math.hypot(
                                step1295_dx, step1295_dy
                            )
                            step1295_heading = math.atan2(
                                step1295_dy, step1295_dx
                            )
                            step1295_error_deg = math.degrees(
                                angle_error(
                                    step1295_heading, step1288_end_yaw
                                )
                            )
                            # Step 1297: read-only safety-map inspection.
                            # Never authorize translation in this probe.
                            step1297_obstacles = convert_obstacles(
                                safety_map
                            )
                            step1297_margins = (0.10, 0.15, 0.20)
                            step1297_checks = {}

                            for step1297_margin in step1297_margins:
                                step1297_key = (
                                    f"{step1297_margin:.2f}"
                                )
                                try:
                                    step1297_checks[step1297_key] = {
                                        "clear": bool(
                                            fixed_segment_clear(
                                                step1288_end_xy,
                                                step1295_wp1,
                                                step1297_obstacles,
                                                step1297_margin,
                                            )
                                        ),
                                        "error": None,
                                    }
                                except Exception as step1297_exc:
                                    step1297_checks[step1297_key] = {
                                        "clear": False,
                                        "error": (
                                            type(step1297_exc).__name__
                                            + ": " + str(step1297_exc)
                                        ),
                                    }

                            # Step 1298: read-only candidate search.
                            # Use detected obstacles only.
                            # Maintain 0.20 m as the exclusion margin.
                            step1298_margin = 0.20
                            step1298_candidates = []
                            step1298_wp2_options = [
                                (0.95, 0.95),
                                (0.95, 1.00),
                                (0.95, 1.05),
                                (0.95, 1.10),
                                (0.95, 1.15),
                            ]

                            for step1298_ix in range(8, 19):
                                step1298_x = round(
                                    step1298_ix * 0.025, 4
                                )
                                for step1298_iy in range(36, 49):
                                    step1298_y = round(
                                        step1298_iy * 0.025, 4
                                    )
                                    step1298_wp = (
                                        step1298_x, step1298_y
                                    )

                                    try:
                                        step1298_first_clear = bool(
                                            fixed_segment_clear(
                                                step1288_end_xy,
                                                step1298_wp,
                                                step1297_obstacles,
                                                step1298_margin,
                                            )
                                        )
                                    except Exception:
                                        step1298_first_clear = False

                                    if not step1298_first_clear:
                                        continue

                                    step1298_next = []
                                    for step1298_wp2 in (
                                        step1298_wp2_options
                                    ):
                                        try:
                                            step1298_second_clear = bool(
                                                fixed_segment_clear(
                                                    step1298_wp,
                                                    step1298_wp2,
                                                    step1297_obstacles,
                                                    step1298_margin,
                                                )
                                            )
                                        except Exception:
                                            step1298_second_clear = False

                                        if step1298_second_clear:
                                            step1298_next.append(
                                                list(step1298_wp2)
                                            )

                                    step1298_dx = (
                                        step1298_x
                                        - step1288_end_xy[0]
                                    )
                                    step1298_dy = (
                                        step1298_y
                                        - step1288_end_xy[1]
                                    )
                                    step1298_distance = math.hypot(
                                        step1298_dx, step1298_dy
                                    )
                                    step1298_heading = math.atan2(
                                        step1298_dy, step1298_dx
                                    )
                                    step1298_heading_error = (
                                        math.degrees(angle_error(
                                            step1298_heading,
                                            step1288_end_yaw,
                                        ))
                                    )

                                    step1298_candidates.append({
                                        "wp1": list(step1298_wp),
                                        "distance_m": round(
                                            step1298_distance, 5
                                        ),
                                        "heading_error_deg": round(
                                            step1298_heading_error, 3
                                        ),
                                        "heading_within_8deg": (
                                            abs(step1298_heading_error)
                                            <= 8.0
                                        ),
                                        "segment_1_clear_020": True,
                                        "compatible_wp2": step1298_next,
                                        "two_segments_clear_020": bool(
                                            step1298_next
                                        ),
                                    })

                            step1298_candidates.sort(
                                key=lambda item: (
                                    not item[
                                        "two_segments_clear_020"
                                    ],
                                    not item[
                                        "heading_within_8deg"
                                    ],
                                    item["distance_m"],
                                )
                            )

                            print(
                                "[SIM_WP1_CANDIDATE_SEARCH] "
                                + step1288_json.dumps({
                                    "margin": step1298_margin,
                                    "source": "SENSOR_DERIVED_SAFETY_MAP",
                                    "original_wp1":
                                        list(step1295_wp1),
                                    "post_turn_xy":
                                        list(step1288_end_xy),
                                    "original_segment_clear_020":
                                        step1297_checks["0.20"]["clear"],
                                    "searched_candidates": 143,
                                    "segment1_clear_count":
                                        len(step1298_candidates),
                                    "two_segment_clear_count": sum(
                                        bool(c["compatible_wp2"])
                                        for c in step1298_candidates
                                    ),
                                    "direct_walk_compatible_count": sum(
                                        c["heading_within_8deg"]
                                        for c in step1298_candidates
                                    ),
                                    "top_candidates":
                                        step1298_candidates[:20],
                                    "walking_authorized": False,
                                }),
                                flush=True,
                            )
                            print(
                                "[SIM_WP1_CLEARANCE] "
                                + step1288_json.dumps({
                                    "start_xy": list(step1288_end_xy),
                                    "wp1_xy": list(step1295_wp1),
                                    "heading_error_deg":
                                        step1295_error_deg,
                                    "detection_count":
                                        len(obstacle_map.points),
                                    "estimated_obstacle_count":
                                        len(step1297_obstacles),
                                    "estimated_obstacles": [
                                        repr(o)
                                        for o in step1297_obstacles
                                    ],
                                    "segment_checks":
                                        step1297_checks,
                                    "contacts":
                                        obstacle_contact_events,
                                    "stop_latched": bool(
                                        sim_emergency_stop_requested
                                    ),
                                    "walking_authorized": False,
                                    "walking_started": False,
                                }),
                                flush=True,
                            )
                            print(
                                "[SIM_WP1_READINESS] "
                                + step1288_json.dumps({
                                    "post_turn_xy": list(
                                        step1288_end_xy
                                    ),
                                    "post_turn_yaw_deg": math.degrees(
                                        step1288_end_yaw
                                    ),
                                    "wp1_xy": list(step1295_wp1),
                                    "distance_m": step1295_distance,
                                    "heading_deg": math.degrees(
                                        step1295_heading
                                    ),
                                    "heading_error_deg":
                                        step1295_error_deg,
                                    "heading_within_8deg":
                                        abs(step1295_error_deg) <= 8.0,
                                    "first_turn_completed":
                                        step1288_trial_reason
                                        == "TURN_COMPLETE",
                                    "contact_events":
                                        obstacle_contact_events,
                                    "stop_latched": bool(
                                        sim_emergency_stop_requested
                                    ),
                                    "walk_authorized": False,
                                    "walk_started": False
                                }),
                                flush=True,
                            )
                            print(
                                "[SIM_FIRST_TURN_RESULT] "
                                + step1288_json.dumps({
                                    "reason": step1288_trial_reason,
                                    "controller_state":
                                        sim_turn_controller.state.value,
                                    "gate_state":
                                        step1288_gate.state,
                                    "gate_reason":
                                        step1288_gate.reason,
                                    "target_yaw_deg":
                                        math.degrees(
                                            step1288_trial_target_yaw
                                        ),
                                    "initial_yaw_deg":
                                        math.degrees(
                                            step1288_trial_start_yaw
                                        ),
                                    "final_yaw_deg":
                                        math.degrees(
                                            step1288_end_yaw
                                        ),
                                    "final_error_deg":
                                        step1288_error,
                                    "drift_m": step1288_drift,
                                    "elapsed_s": step1288_elapsed,
                                    "contacts":
                                        obstacle_contact_events,
                                    "final_command": [0.0, 0.0, 0.0],
                                    "emergency_latched": bool(
                                        sim_emergency_stop_requested
                                    ),
                                    "walking_started": False,
                                }),
                                flush=True,
                            )
                            # Step 1300: only after verified first turn.
                            step1300_eligible = bool(
                                step1288_trial_reason == "TURN_COMPLETE"
                                and step1288_gate.state == "COMPLETE"
                                and sim_turn_controller.state
                                is SimTurnState.READY_TO_RESUME
                                and sim_emergency_stop_requested
                                and sim_emergency_stop_reason == "TOF_NEAR"
                                and obstacle_contact_events == 0
                                and abs(step1288_error) <= 8.0
                                and len(obstacle_map.points) >= 2
                            )

                            step1300_route_clear = False
                            try:
                                observed = convert_obstacles(safety_map)
                                step1300_route_clear = bool(
                                    len(observed) >= 2
                                    and fixed_segment_clear(
                                        step1288_end_xy,
                                        step1300_target,
                                        observed, 0.20,
                                    )
                                    and fixed_segment_clear(
                                        step1300_target,
                                        step1300_next,
                                        observed, 0.20,
                                    )
                                )
                            except Exception:
                                step1300_route_clear = False

                            dx = step1300_target[0] - step1288_end_xy[0]
                            dy = step1300_target[1] - step1288_end_xy[1]
                            step1300_target_yaw = math.atan2(dy, dx)
                            # Step 1326: left-biased alignment target only.
                            # Real WP1 target remains unchanged.
                            step1326_bias_target_yaw = (
                                step1300_target_yaw + math.radians(10.0)
                            )
                            step1300_start_xy = step1288_end_xy
                            step1300_start_yaw = step1288_end_yaw

                            step1300_arm = False
                            if step1300_eligible:
                                step1300_arm = step1300_aligner.start(
                                    now=float(data.time),
                                    current_yaw=step1300_start_yaw,
                                    target_yaw=step1326_bias_target_yaw,
                                    emergency_latched=bool(
                                        sim_emergency_stop_requested
                                    ),
                                    emergency_reason=sim_emergency_stop_reason,
                                    route_clear=step1300_route_clear,
                                )

                            if step1300_arm:
                                policy.activate_ppo_v3_turn()
                                sim_turn_policy_active = True
                                step1300_started = True

                            print(
                                "[SIM_WP1_ALIGN_START] "
                                + step1288_json.dumps({
                                    "armed": bool(step1300_arm),
                                    "eligible": step1300_eligible,
                                    "route_clear_020": step1300_route_clear,
                                    "start_xy": list(step1300_start_xy),
                                    "wp1": list(step1300_target),
                                    "start_yaw_deg": math.degrees(
                                        step1300_start_yaw
                                    ),
                                    "target_yaw_deg": math.degrees(
                                        step1300_target_yaw
                                    ),
                                    "state": step1300_aligner.state,
                                    "reason": step1300_aligner.reason,
                                }),
                                flush=True,
                            )
                            if not step1300_arm:
                                print(
                                    "[HEADLESS] FINISHED "
                                    "status=SIM_WP1_ALIGN_NOT_ARMED",
                                    flush=True
                                )
                                break
                            continue

                    if (sim_goal_walk_state == "FINISHED"
                            and not step1288_trial_started):
                        if sim_emergency_stop_requested:
                            hold_t = float(data.time)
                            hold_xy = (
                                float(data.qpos[qpos_adr]),
                                float(data.qpos[qpos_adr + 1]),
                            )

                            if sim_stop_hold_start_time is None:
                                sim_stop_hold_start_time = hold_t
                                sim_stop_hold_start_xy = hold_xy
                                print(
                                    f"[SIM_STOP_HOLD] START t={hold_t:.3f} "
                                    f"xy={hold_xy}"
                                )

                            elapsed = hold_t - sim_stop_hold_start_time

                            displacement = math.hypot(
                                hold_xy[0] - sim_stop_hold_start_xy[0],
                                hold_xy[1] - sim_stop_hold_start_xy[1],
                            )

                            sim_stop_hold_max_displacement = max(
                                sim_stop_hold_max_displacement,
                                displacement,
                            )

                            # Point-to-box distance, not robot clearance.
                            box_distance = math.hypot(
                                max(abs(hold_xy[0] - 0.7) - 0.1, 0.0),
                                max(abs(hold_xy[1] - 0.5) - 0.1, 0.0),
                            )

                            sim_stop_hold_min_point_box_distance = min(
                                sim_stop_hold_min_point_box_distance,
                                box_distance,
                            )

                            sample = min(int((elapsed + 1e-8) / 0.5), 4)

                            if sample > sim_stop_hold_last_sample:
                                sim_stop_hold_last_sample = sample
                                print(
                                    "[SIM_STOP_HOLD] SAMPLE "
                                    f"elapsed={elapsed:.3f} "
                                    f"xy={hold_xy} "
                                    f"displacement={displacement:.6f} "
                                    f"point_box_distance={box_distance:.6f} "
                                    f"contact={bool(obstacle_contact_previous)}"
                                )

                            if elapsed >= 2.0:
                                print(
                                    "[SIM_STOP_HOLD] COMPLETE "
                                    f"elapsed={elapsed:.3f} "
                                    f"final_displacement={displacement:.6f} "
                                    f"max_displacement={sim_stop_hold_max_displacement:.6f} "
                                    f"min_point_box_distance="
                                    f"{sim_stop_hold_min_point_box_distance:.6f} "
                                    f"contact_events={obstacle_contact_events}"
                                )
                                print(
                                    "[HEADLESS] FINISHED "
                                    "status=SIM_GOAL_TOF_STOP_HOLD_COMPLETE"
                                )
                                break
                        else:
                            print(
                                f"[HEADLESS] FINISHED status=SIM_GOAL_{sim_goal_walk_status}"
                            )
                            break
                    if sim_staging_state == "STOPPED":
                        if sim_staging_status != "TARGET_REACHED":
                            print(
                                f"[HEADLESS] FINISHED status=SIM_STAGING_{sim_staging_status}"
                            )
                            break

                        if sim_second_turn_state == "WAITING":
                            settle_elapsed = float(data.time) - sim_staging_stop_time
                            if settle_elapsed >= 0.5:
                                second_x = float(data.qpos[qpos_adr])
                                second_y = float(data.qpos[qpos_adr + 1])
                                second_z = float(data.qpos[2])
                                second_pose = (second_x, second_y)

                                second_goal = (
                                    float(config["goal"]["x"]),
                                    float(config["goal"]["y"]),
                                )
                                second_yaw_target = math.atan2(
                                    second_goal[1] - second_y,
                                    second_goal[0] - second_x,
                                )

                                second_heading_error = math.degrees(
                                    angle_error(second_yaw_target, float(current_yaw))
                                )

                                from path_planner_safety_fixed import segment_clear
                                from sim_turn_drift_safety import evaluate_turn_drift_clearance

                                second_pose_valid = all(
                                    math.isfinite(value)
                                    for value in (second_x, second_y, second_z, float(current_yaw))
                                )
                                second_pose_stable = (
                                    second_pose_valid
                                    and math.hypot(
                                        second_x - sim_staging_stop_pose[0],
                                        second_y - sim_staging_stop_pose[1],
                                    ) <= 0.02
                                )
                                second_contact = bool(obstacle_contact_previous)
                                second_fallen = not math.isfinite(second_z) or second_z < 0.09

                                second_path_safe = (
                                    second_pose_valid
                                    and sim_resume_obstacles is not None
                                    and segment_clear(
                                        second_pose,
                                        second_goal,
                                        sim_resume_obstacles,
                                        0.15,
                                    )
                                )

                                if second_pose_valid and sim_resume_obstacles is not None:
                                    second_drift_result = evaluate_turn_drift_clearance(
                                        second_pose,
                                        sim_resume_obstacles,
                                        margin=0.10,
                                        drift_budget=0.08,
                                    )
                                    second_dynamic_safe, _, _ = evaluate_sim_turn_clearance(
                                        model,
                                        data,
                                        clearance_margin=0.10,
                                        robot_radius=0.10,
                                    )
                                else:
                                    second_drift_result = (False, None, "INVALID_POSE_OR_MAP")
                                    second_dynamic_safe = False

                                second_drift_safe = (
                                    isinstance(second_drift_result, tuple)
                                    and len(second_drift_result) >= 3
                                    and second_drift_result[0] is True
                                    and second_drift_result[2] == "DRIFT_CLEARANCE_OK"
                                )

                                second_start_allowed = all((
                                    second_pose_valid,
                                    second_pose_stable,
                                    not second_contact,
                                    not second_fallen,
                                    second_path_safe,
                                    second_drift_safe,
                                    second_dynamic_safe,
                                    not sim_turn_policy_active,
                                ))

                                print(
                                    f"[SIM_SECOND_TURN] VALIDATE "
                                    f"pos=({second_x:.4f},{second_y:.4f}) "
                                    f"yaw={math.degrees(current_yaw):.2f}deg "
                                    f"target={math.degrees(second_yaw_target):.2f}deg "
                                    f"error={second_heading_error:.2f}deg "
                                    f"stable={second_pose_stable} "
                                    f"path={second_path_safe} "
                                    f"drift={second_drift_result} "
                                    f"dynamic={second_dynamic_safe} "
                                    f"contact={second_contact} fallen={second_fallen}"
                                )

                                if not second_start_allowed:
                                    sim_second_turn_state = "BLOCKED"
                                    print("[SIM_SECOND_TURN] BLOCKED: safety validation failed")
                                else:
                                    sim_turn_controller = SimTurnController(
                                        turn_speed=args.sim_turn_speed
                                    )
                                    sim_turn_last_state = SimTurnState.WAITING
                                    sim_ema_rate = None
                                    sim_ema_prev_time = None
                                    sim_turn_finish_logged = False
                                    sim_turn_start_pose = None
                                    sim_turn_start_yaw = None
                                    sim_turn_trajectory = []

                                    second_started = sim_turn_controller.start(
                                        current_time=float(data.time),
                                        target_yaw=second_yaw_target,
                                    )
                                    if second_started:
                                        sim_second_turn_state = "ACTIVE"
                                        print(
                                            f"[SIM_SECOND_TURN] START "
                                            f"target_yaw={math.degrees(second_yaw_target):.2f}deg"
                                        )
                                    else:
                                        sim_second_turn_state = "BLOCKED"
                                        print("[SIM_SECOND_TURN] BLOCKED: controller rejected start")

                        if sim_second_turn_state == "ACTIVE":
                            if sim_turn_controller.state in (
                                SimTurnState.READY_TO_RESUME,
                                SimTurnState.ABORTED,
                            ):
                                if sim_turn_policy_active:
                                    policy.set_vel_cmd(0.0, 0.0, 0.0)
                                    policy.deactivate_ppo_v3_turn()
                                    sim_turn_policy_active = False
                                second_final_error = math.degrees(
                                    angle_error(
                                        sim_turn_controller.target_yaw,
                                        float(current_yaw),
                                    )
                                )
                                second_end_x = float(data.qpos[qpos_adr])
                                second_end_y = float(data.qpos[qpos_adr + 1])
                                second_end_pose = (second_end_x, second_end_y)

                                second_drift = math.hypot(
                                    second_end_x - sim_turn_start_pose[0],
                                    second_end_y - sim_turn_start_pose[1],
                                ) if sim_turn_start_pose is not None else float("nan")

                                second_goal_distance = math.hypot(
                                    second_goal[0] - second_end_x,
                                    second_goal[1] - second_end_y,
                                )

                                second_goal_path_safe = (
                                    sim_resume_obstacles is not None
                                    and fixed_segment_clear(
                                        second_end_pose,
                                        second_goal,
                                        sim_resume_obstacles,
                                        0.15,
                                    )
                                )

                                print(
                                    f"[SIM_SECOND_TURN_POSE] "
                                    f"start={sim_turn_start_pose} "
                                    f"end=({second_end_x:.4f},{second_end_y:.4f}) "
                                    f"drift={second_drift:.4f}m "
                                    f"goal_distance={second_goal_distance:.4f}m "
                                    f"goal_path_safe={second_goal_path_safe}"
                                )
                                print(
                                    f"[SIM_SECOND_TURN] FINISHED "
                                    f"state={sim_turn_controller.state.value} "
                                    f"yaw={math.degrees(current_yaw):.2f}deg "
                                    f"error={second_final_error:.2f}deg "
                                    f"abort={sim_turn_controller.abort_reason}"
                                )
                                if sim_turn_controller.state is SimTurnState.ABORTED:
                                    print("[SIM_GOAL_WALK] BLOCKED reason=SECOND_TURN_ABORTED")
                                    break

                                sim_goal_walk_start = (
                                    float(data.qpos[qpos_adr]),
                                    float(data.qpos[qpos_adr + 1]),
                                )
                                sim_goal_walk_target = (
                                    float(config["goal"]["x"]),
                                    float(config["goal"]["y"]),
                                )

                                sx, sy = sim_goal_walk_start
                                sz = float(data.qpos[2])
                                goal_yaw = math.atan2(
                                    sim_goal_walk_target[1] - sy,
                                    sim_goal_walk_target[0] - sx,
                                )
                                initial_error = math.degrees(
                                    angle_error(goal_yaw, float(current_yaw))
                                )

                                start_safe = (
                                    all(math.isfinite(value) for value in (
                                        sx, sy, sz, float(current_yaw), initial_error
                                    ))
                                    and sz >= 0.09
                                    and not bool(obstacle_contact_previous)
                                    and sim_resume_obstacles is not None
                                    and fixed_segment_clear(
                                        sim_goal_walk_start,
                                        sim_goal_walk_target,
                                        sim_resume_obstacles,
                                        0.15,
                                    )
                                    and abs(initial_error) <= 8.0
                                )

                                sim_second_turn_state = "GOAL_WALK"

                                if start_safe:
                                    sim_goal_walk_state = "WALKING"
                                    sim_goal_walk_start_time = float(data.time)
                                    print(
                                        f"[SIM_GOAL_WALK] START "
                                        f"pos=({sx:.4f},{sy:.4f}) "
                                        f"target={sim_goal_walk_target} "
                                        f"heading_error={initial_error:.2f}deg"
                                    )
                                else:
                                    sim_goal_walk_state = "FINISHED"
                                    sim_goal_walk_status = "START_UNSAFE"
                                    print(
                                        f"[SIM_GOAL_WALK] BLOCKED "
                                        f"reason=START_UNSAFE "
                                        f"heading_error={initial_error:.2f}deg"
                                    )

                        if sim_second_turn_state == "BLOCKED":
                            print("[HEADLESS] FINISHED status=SIM_SECOND_TURN_BLOCKED")
                            break
                    if sim_precision_walk_finished and sim_staging_state == "WAITING":
                        print(
                            f"[HEADLESS] FINISHED "
                            f"status=SIM_PRECISION_{sim_precision_walk_status}"
                        )
                        break
                    if (
                        sim_turn_test_enabled
                        and not sim_precision_walk_started
                        and sim_turn_controller.state in (
                            SimTurnState.READY_TO_RESUME,
                            SimTurnState.ABORTED,
                        )
                    ):
                        policy.set_vel_cmd(0.0, 0.0, 0.0)

                        if sim_turn_policy_active:
                            policy.deactivate_ppo_v3_turn()
                            sim_turn_policy_active = False

                        sim_turn_final_error = (
                            math.degrees(
                                angle_error(
                                    sim_turn_controller.target_yaw,
                                    float(current_yaw),
                                )
                            )
                            if sim_turn_controller.target_yaw is not None
                            else None
                        )

                        print(
                            f"[PHASE45_SIM] TURN RESULT "
                            f"state={sim_turn_controller.state.value} "
                            f"yaw={math.degrees(current_yaw):.2f}deg "
                            f"error={sim_turn_final_error} "
                            f"abort={sim_turn_controller.abort_reason}"
                        )
                        print(
                            f"[SIM_RESUME_STATE] "
                            f"nav_mode={nav_mode} "
                            f"waypoint_index={nav_waypoint_index} "
                            f"waypoint_count={len(nav_waypoints)} "
                            f"target=({nav_goal_x:.3f},{nav_goal_y:.3f}) "
                            f"replan_ready={phase4_replan_ready} "
                            f"path_checked={sim_resume_path_checked} "
                            f"path_safe={sim_resume_path_safe} "
                            f"turn_state={sim_turn_controller.state.value}"
                        )
                        if sim_turn_start_pose is not None:
                            sim_turn_end_pose = (
                                float(data.qpos[qpos_adr]),
                                float(data.qpos[qpos_adr + 1]),
                            )
                            sim_turn_drift = math.dist(
                                sim_turn_start_pose,
                                sim_turn_end_pose,
                            )
                            print(
                                f"[SIM_TURN_DRIFT] RESULT "
                                f"start={sim_turn_start_pose} "
                                f"end={sim_turn_end_pose} "
                                f"drift={sim_turn_drift:.4f}m"
                            )
                            if sim_turn_trajectory:
                                from pathlib import Path as _TurnPath
                                import csv as _turn_csv
                            
                                _turn_samples = [
                                    (
                                        t, x, y,
                                        math.hypot(
                                            x - sim_turn_start_pose[0],
                                            y - sim_turn_start_pose[1],
                                        ),
                                    )
                                    for t, x, y in sim_turn_trajectory
                                ]
                            
                                _turn_max = max(
                                    _turn_samples,
                                    key=lambda item: item[3],
                                )
                            
                                print(
                                    f"[SIM_TURN_TRAJECTORY] "
                                    f"samples={len(_turn_samples)} "
                                    f"max_drift={_turn_max[3]:.4f}m "
                                    f"at_time={_turn_max[0]:.3f}s "
                                    f"pos=({_turn_max[1]:.4f},{_turn_max[2]:.4f})"
                                )
                            
                                _turn_csv_path = (
                                    _TurnPath(__file__).resolve().parent
                                    / "turn_trajectory.csv"
                                )
                            
                                with _turn_csv_path.open("w", newline="") as _turn_file:
                                    _turn_writer = _turn_csv.writer(_turn_file)
                                    _turn_writer.writerow(
                                        ["time_s", "x_m", "y_m", "drift_m"]
                                    )
                                    _turn_writer.writerows(_turn_samples)
                            
                                print(
                                    f"[SIM_TURN_TRAJECTORY] csv={_turn_csv_path}"
                                )

                        print(
                            f"[HEADLESS] FINISHED "
                            f"status=SIM_TURN_{sim_turn_controller.state.value}"
                        )
                        break
                    sim_elapsed_end = data.time - auto_nav_start_time

                    if (
                        not sim_turn_test_enabled
                        and                         phase4_sensor is not None
                        and phase4_stop_time is not None
                        and data.time - phase4_stop_time >= 1.0
                    ):
                        stop_dx = float(data.qpos[0]) - phase4_stop_origin[0]
                        stop_dy = float(data.qpos[1]) - phase4_stop_origin[1]
                        stop_drift = math.hypot(stop_dx, stop_dy)

                        print(
                            f"[PHASE4] Stop observation: "
                            f"drift={stop_drift:.4f}m"
                        )
                        print(
                            f"[HEADLESS] FINISHED "
                            f"status=STOPPED "
                            f"t={sim_elapsed_end:.2f}s"
                        )
                        break

                    if nav_mode in ("DONE", "TIMEOUT"):
                        print(
                            f"[HEADLESS] FINISHED "
                            f"status={nav_mode} "
                            f"t={sim_elapsed_end:.2f}s"
                        )
                        break

                    if sim_elapsed_end >= 100.0:
                        print(
                            f"[HEADLESS] SAFETY TIMEOUT "
                            f"t={sim_elapsed_end:.2f}s"
                        )
                        break

                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt received (Ctrl+C). Saving data...")

    # Step 9-7-10: Obstacle contact summary
    print("[CONTACT] ===== Obstacle Contact Summary =====")
    print(f"[CONTACT] Events: {obstacle_contact_events}")
    print(f"[CONTACT] Contact steps: {obstacle_contact_steps}")

    if obstacle_first_contact_time is None:
        print("[CONTACT] First contact time: None")
    else:
        print(
            f"[CONTACT] First contact time: "
            f"{obstacle_first_contact_time:.2f}s"
        )

    if not obstacle_geom_ids:
        print("[CONTACT] Result: OBSTACLE NOT FOUND")
    elif obstacle_contact_steps == 0:
        print("[CONTACT] Result: NO CONTACT")
    else:
        print("[CONTACT] Result: CONTACT DETECTED")

    print("\nInference stopped.")
    if odom_compare is not None:
        odom_compare.print_summary(data)

    if csv_data is not None and len(csv_data) > 0:
        print(f"\nSaving {len(csv_data)} steps to: {args.save_csv}")
        with open(args.save_csv, 'w', newline='') as csvfile:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"CSV file saved successfully!")
        print(f"  Columns: {len(fieldnames)}")
        print(f"  Rows: {len(csv_data)}")

    if recorded_observations is not None and len(recorded_observations) > 0:
        print(f"\nSaving {len(recorded_observations)} recorded observations to: {args.record}")
        with open(args.record, 'wb') as f:
            pickle.dump(recorded_observations, f)
        print(f"Recorded observations saved to {args.record}")
        print(f"  Observations: {len(recorded_observations)}")
        print(f"  Duration: {recorded_observations[-1]['timestamp']:.2f}s")


if __name__ == "__main__":
    main()
