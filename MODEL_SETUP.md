# Model setup and reproducibility

## Model requirements

The locally validated Scenario A run used three policies:

| Role | Filename | SHA-256 |
|---|---|---|
| Walking policy | `velstand.onnx` | `ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a` |
| Standing policy | `alpha_stand.onnx` | `1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b` |
| PPO V3 turning policy | `model_99.onnx` | `70afbeacdf6d520c6c862fbeb2851ceba01e3e661b30304d8d6b7123a3aa6ac1` |

## Distribution status

The intended Public repository does not bundle ONNX files.
The local validation used models already available on the
developer's machine. Access to matching models is required
to reproduce the exact tested run.

The official MicroDuck project is:
https://github.com/pollen-robotics/microduck_rl

The availability, retrieval procedure, and redistribution
rights for each policy must be verified separately.
The original PPO V3 training logs must also be checked
to establish whether pretrained weights were used.

Do not assume that the PPO V3 turning model is freely
redistributable solely because it was trained for this project.

## Local validation

The candidate completed one Scenario A headless run:

- Status: `SIM_GOAL_GOAL_REACHED`
- Final goal distance: 0.0979 m
- Obstacle contact events: 0

See `VALIDATION.md` for limitations.

## Reproducibility limitation

A clean clone with freshly acquired policies has not yet
been validated. The model hashes above document the
specific local assets, not public download availability.

## Verified model setup (October 2026)

### 1. Obtain the official walking and standing models

Run from the repository root:

    python3 scripts/setup_official_models.py

The script downloads the Pollen Robotics v7 policies verified
for this project.

- Walking model: `velstand.onnx`
- Standing model: `alpha_stand.onnx`

Official source:

https://huggingface.co/pollen-robotics/microduck-policies

The expected SHA-256 hashes are:

Walking:

    ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a

Standing:

    1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b

The script verifies SHA-256 and installs both models into:

    ~/.cache/duck-sim/policies/current/

Existing files with different hashes are not overwritten.

### 2. Prepare the PPO V3 turning model

The validated turning model is:

    model_99.onnx

Expected destination relative to the repository:

    navigation/phase5_turn_ppo/protected/v3_model99/model_99.onnx

Expected SHA-256:

    70afbeacdf6d520c6c862fbeb2851ceba01e3e661b30304d8d6b7123a3aa6ac1

ONNX Runtime interface:

- Input: `obs`, shape `[1, 61]`
- Output: `actions`, shape `[1, 14]`

The PPO V3 model is not currently distributed with this repository.

Its initialization provenance and redistribution conditions
are still under review.

A fresh clone cannot fully reproduce the PPO V3 navigation run
without separately obtaining this validated model.

There is currently no public download URL for this
project-specific model.

Do not substitute an arbitrary ONNX model merely because
its input and output dimensions match.

### 3. Reproducibility status

- Official walking policy: available and SHA-256 verified.
- Official standing policy: available and SHA-256 verified.
- PPO V3 turning policy: locally validated but not publicly available.
- Clean-clone end-to-end navigation run: not yet verified.

The last successful Scenario A run reached the goal without
obstacle contact.

It did not establish automatic forward resumption after
the STOPPED state.
