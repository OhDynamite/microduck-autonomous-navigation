# Third-party notices

## Project purpose

This repository is prepared for an educational final project,
personal research, and future MicroDuck hardware experiments.

The project is not intended for commercialization.
This statement describes the project's intended use;
it does not replace or modify third-party license conditions.

## Pollen Robotics / MicroDuck

Upstream project:

https://github.com/pollen-robotics/microduck_rl

### Code and MJCF/XML assets

The upstream project's license documents identify code
and MJCF/XML robot descriptions as covered by Apache-2.0.

The upstream Apache-2.0 license text is retained at:

`microduck_turn_training/LICENSE`

Retain upstream copyright, attribution, and applicable
license notices. Modifications and original additions must
be distinguished from upstream material.

### STL meshes

The files under:

`microduck_turn_training/src/mjlab_microduck/robot/microduck/assets/`

are attributed to Pollen Robotics and covered by
CC BY-NC-SA 4.0 according to the upstream license notice.

Original notice retained at:

`microduck_turn_training/LICENSE-MODELS.md`

License:

https://creativecommons.org/licenses/by-nc-sa/4.0/

The STL meshes were copied without intentional geometry
modification for this repository's simulation setup.
This does not imply that other project files are unchanged.

Any subsequent sharing or modification must comply with
the applicable attribution, noncommercial, and share-alike
terms.

## Learned policies

No ONNX weights are included in the intended Public release
at this stage.

Walking, standing, and transfer-trained turning policies
may have different rights or provenance from code and
STL assets. Their redistribution conditions are not yet
verified.

See `MODEL_SETUP.md`.

## Other dependencies

External Python packages retain their own licenses.
The presence of an upstream Apache-2.0 LICENSE does not
automatically apply that license to every dependency
or learned policy.

## Source provenance comparison

The Python training package was compared against the
local `microduck_rl` upstream source snapshot.
This comparison is based on relative paths and SHA-256
hashes, not on a legal authorship determination.

- Byte-identical upstream Python files: 24
- Different files at matching upstream paths: 2
- Python files without matching upstream relative paths: 10
- Navigation Python files separately developed or adapted: 29

### Files differing from corresponding upstream files

- `microduck_turn_training/src/mjlab_microduck/hf_jobs.py`
- `microduck_turn_training/src/mjlab_microduck/tasks/__init__.py`

These files differ from the local upstream snapshot.
The original upstream license and attribution obligations
must be preserved for any upstream-derived portions.

### Additional turn-PPO files

- `microduck_turn_training/src/mjlab_microduck/turn_ppo/__init__.py`
- `microduck_turn_training/src/mjlab_microduck/turn_ppo/microduck_turn_env_cfg.py`
- `microduck_turn_training/src/mjlab_microduck/turn_ppo/microduck_turn_train_env_cfg.py`
- `microduck_turn_training/src/mjlab_microduck/turn_ppo/microduck_turn_train_env_cfg_v2.py`
- `microduck_turn_training/src/mjlab_microduck/turn_ppo/microduck_turn_train_env_cfg_v3.py`
- `microduck_turn_training/src/mjlab_microduck/turn_ppo/register_transfer_v3_task.py`
- `microduck_turn_training/src/mjlab_microduck/turn_ppo/register_turn_task.py`
- `microduck_turn_training/src/mjlab_microduck/turn_ppo/train_transfer_ppo_v3.py`
- `microduck_turn_training/src/mjlab_microduck/turn_ppo/transfer_runner.py`
- `microduck_turn_training/src/mjlab_microduck/turn_ppo/turn_v3_rewards.py`

These files have no same-path counterpart in the local
upstream snapshot. This does not by itself prove that
all their contents are independently authored.

### Navigation files

The 29 files under `navigation/` have no byte-identical
counterpart among the scanned upstream Python files.
Some navigation code may adapt upstream architecture,
APIs, or implementations. A hash comparison alone does
not establish ownership of every contribution.

### Licensing scope

The Apache-2.0 notice retained for upstream code and
the CC BY-NC-SA 4.0 notice retained for STL meshes
remain applicable to the respective upstream materials.
This provenance note does not relicense third-party
material or grant a new license for independent additions.
