# MicroDuck Autonomous Navigation — Final Project

## PPO V3 Model Availability and Reproducibility

**The trained PPO V3 turning model (`model_99.onnx`) is not included in this repository.**

The model was developed through additional training based on an existing
policy checkpoint. Redistribution conditions for the pretrained weights
are being clarified with Pollen Robotics:

https://github.com/pollen-robotics/microduck_rl/issues/88

If redistribution is confirmed to be permitted, the validated PPO V3
model is intended to be provided through a future GitHub Release,
subject to applicable license conditions.

In the meantime:

- Download the verified official walking and standing policies using
  `python3 scripts/setup_official_models.py`.
- Refer to `microduck_turn_training/` for PPO training code and configuration.
- Refer to `MODEL_SETUP.md` for the expected PPO V3 model path,
  interface, and SHA-256.
- Reproduction of the exact PPO V3 model from the published training
  materials alone has not been verified.
- A fresh clone cannot fully reproduce the validated PPO V3 navigation
  run until the required model is available.

The validated Scenario A run reached the goal without obstacle contact.
Automatic forward resumption after the STOPPED state was not demonstrated.



An educational MicroDuck autonomous-navigation project
developed and evaluated in MuJoCo.

This repository is intended for final-course submission,
personal learning, and future experimentation with a
physical MicroDuck robot. It is not a commercial product.

## What this project contains

- Autonomous navigation and waypoint control
- Obstacle detection, safety checks, and replanning components
- PPO V3 turning-policy integration
- Scenario A simulation configuration
- PPO training code and MuJoCo robot assets
- Validation results from a local headless simulation

## Repository layout

- `navigation/` — autonomous navigation and safety logic
- `navigation/scenarios/` — Scenario A XML and JSON
- `microduck_turn_training/` — PPO training package,
  configuration, and MuJoCo assets
- `VALIDATION.md` — measured results and limitations

## Verified result

In a single Scenario A headless simulation, the release
candidate reached the goal with a final goal distance of
0.0979 m and zero recorded obstacle-contact events.

The run also reported WP1 arrival and a replanning event.
It did not establish that every stop/resume scenario works.
See `VALIDATION.md` for the exact scope of the evidence.

## Reproduction status

A local startup check and MuJoCo model-loading check passed.
The release candidate also completed one Scenario A run.

A clean-clone installation and independent reproduction
have not yet been verified.

Pretrained walking/standing policies and the PPO V3
turning policy are required for the tested configuration.
Model download instructions and distribution conditions
are still being finalized.

## Future hardware work

The current results are simulation-only. Physical sensor
interfaces, timing, robot safety, and hardware-specific
controllers must be evaluated before use on a real robot.

## Credits and licenses

This project incorporates material originating from the
Pollen Robotics MicroDuck ecosystem.

Python code, robot meshes, and learned policies may have
different licenses. Consult the included source-license
documents and asset-specific notices. The intended
educational use of this project does not override upstream
license conditions.

**Status:** Educational research project; ONNX policies are not distributed.
for Public GitHub publication.

## Running Scenario A locally

### Prerequisites

- macOS environment with MuJoCo and the Python dependencies installed
- `uv` and the compatible Python environment
- The three ONNX policies described in `MODEL_SETUP.md`

The tested development environment used MuJoCo 3.10.0.
A clean-clone dependency installation has **not** yet
been verified.

The commands below describe the existing working local
environment. They are not a guarantee that a newly
cloned repository can be executed without additional setup.

### Required policies

The tested Scenario A run used:

- Walking: `velstand.onnx` (seed-v7)
- Standing: `alpha_stand.onnx` (seed-v7)
- Turning: protected PPO V3 `model_99.onnx`

The walking and standing policy paths in the example
below use the developer's existing policy cache.

The turning policy must be present at:

`navigation/phase5_turn_ppo/protected/v3_model99/model_99.onnx`

ONNX files are excluded from Git by `.gitignore`.
They are **not distributed in this repository**.

See `MODEL_SETUP.md` for the exact SHA-256 hashes and
current distribution limitations.

### Example command

From the repository root, in an environment where the
required dependencies and policy files have already
been installed:

```bash
REPO="$(pwd)"
cd "$REPO/microduck_turn_training"

export PYTHONPATH="$REPO/navigation:$REPO/microduck_turn_training/src${PYTHONPATH:+:$PYTHONPATH}"

uv run --no-sync --offline mjpython \
  "$REPO/navigation/autonomous_navigation.py" \
  --walking "$HOME/.cache/duck-sim/policies/current/velstand.onnx" \
  --standing "$HOME/.cache/duck-sim/policies/current/alpha_stand.onnx" \
  --scene "$REPO/navigation/scenarios/scene_generated_scenario_a.xml" \
  --config "$REPO/navigation/scenarios/test_config_scenario_a_unknown.json" \
  --new-cmd-obs \
  --headless \
  --sim-ppo-v3-turn
```

The `--no-sync --offline` options assume a previously
prepared local environment. Dependency installation
from a clean checkout is a separate outstanding check.

### Expected successful result

The locally validated Scenario A run produced:

- `[SIM_GOAL_WALK] FINISHED status=GOAL_REACHED`
- `[HEADLESS] FINISHED status=SIM_GOAL_GOAL_REACHED`
- `[CONTACT] Events: 0`
- `[CONTACT] Result: NO CONTACT`

The final distance to the target was 0.0979 m.

This is evidence from one local simulation run, not
a claim of guaranteed success for every execution.

A replanning event was recorded, but automatic resume
after that stop was not independently established.

### Important limitations

The run uses simulation-specific conditions and policies.
A physical MicroDuck requires separately verified
sensor interfaces, actuator commands, timing,
and hardware safety controls.

Do not deploy this navigation program directly to
physical hardware without suitable hardware validation.

## Third-party attribution and model rights

See `THIRD_PARTY_NOTICES.md` for upstream code and
3D-mesh licensing information.

See `MODEL_SETUP.md` for the ONNX policy requirements,
hashes, and redistribution limitations.

The project's educational and noncommercial purpose
does not change the licenses of upstream material.
