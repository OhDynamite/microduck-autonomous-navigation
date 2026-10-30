# Simulation validation

## Final project validation (submitted report)

The submitted final-project experiment followed START,
WP1 (0.10, 0.50 m), and WP2 (0.95, 1.00 m).

| Evaluation item | Final reported result |
| --- | --- |
| WP2 / GOAL | Reached in 3 of 3 runs |
| Final goal distance | Approximately 0.098 m |
| Obstacle contacts | 0 in each run |
| Initial obstacle detection and stopping | Completed |
| Initial A* replanning | Applied to navigation |
| PPO V3 turning and walking resumption | Completed |
| Later additional A* candidate | Generated and evaluated, not applied |

Source: submitted final-project report, sections 4, 6, and 8.

This is the later submitted integrated experiment, not the
separate Scenario A release-candidate test below.
The result does not establish safety for all obstacle layouts
or real-hardware operation.

The exact correspondence between the final submitted
experiment and the GitHub release source files has not
been independently verified.

## Separate Scenario A verified run

- Test: Scenario A, GitHub release candidate
- Entry point: `navigation/autonomous_navigation.py`
- Environment: MuJoCo 3.10.0, existing local Python environment
- Execution mode: headless, PPO V3 turn enabled
- Final status: `SIM_GOAL_GOAL_REACHED`
- Final goal distance: 0.0979 m
- Obstacle contact events: 0
- WP1 reached: yes
- Replanning event logged: yes
- Replanning-to-resume behavior: not established
- Repeated-run reliability: not evaluated

This evidence comes from a single local simulation run.
It does not establish success on physical MicroDuck hardware.

## Model dependencies

The tested local environment used:

- Walking: `velstand.onnx` from seed-v7
- Standing: `alpha_stand.onnx` from seed-v7
- Turn: protected PPO V3 `model_99.onnx`

These policy files are not included in the intended Public
repository until their respective redistribution conditions
are resolved. Reproducing the exact run therefore currently
requires access to the appropriate models.

## Limitations

The existing development environment was used for this test.
A clean installation from GitHub has not yet been verified.

## Final release-candidate validation (Step 55)

The final local Scenario A run, after the XML mesh-path and
CSV-output-name changes, completed successfully.

- Process exit code: 0
- Headless result: `SIM_GOAL_GOAL_REACHED`
- Final goal distance: 0.0979 m
- Cross-track error: 0.0122 m
- Final yaw error: 5.38 degrees
- Waypoint arrivals: 1
- Replanning-ready events: 1
- Obstacle contact events: 0
- Explicit failure indicators: 0
- Wall-clock time: 20.92 seconds

**Scope:** This validates goal completion and no observed
obstacle contact in the tested Scenario A. The log explicitly
reported `STOPPED (resume disabled)`, so it does not establish
fully autonomous resume after stopping.

This is a local simulation result, not a validation on physical
MicroDuck hardware. A clean checkout without the separately
obtained ONNX policies has not been validated.

The raw run log was retained outside this Public-release candidate.
