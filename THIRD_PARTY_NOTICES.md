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
