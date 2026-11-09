# Simulator review notes

What was reviewed in `env.py` (PyBullet scene, UR5e + pushing blade/suction cup, cabinet, objects), what was fixed, what is
opt-in, and what is still open. Evidence from `profile_sim.py` runs is marked **[measured]**; everything else was found by reading
the code and **has not been validated on the real robot** (the opt-in options also not yet against the legacy behaviour).

## Fixed in the default path (errors)
| issue | fix |
|---|---|
| `check_sim()` compared a list with a float (`TypeError`, `reset()` crashed) | numpy arrays |
| `resetSimulation()` silently restores the 1/240 s time step | `setTimeStep` after every reset |
| `move_joints` / `wait_static` timed out on wall-clock time (success depended on CPU speed) | simulation-step budgets |
| IK accuracy used the max of the *signed* error | error norm |
| `detect_contact` compared a body id with a link index | ignore contacts with the robot itself |
| cabinet walls were 100 kg dynamic bodies pinned by constraints; objects spawned half inside the floor | static walls; spawn on the floor |
| replay buffer stored raw 480x640 camera images (~4 MB per observation) | compact observations |
| hindsight relabelling mutated shared arrays and mixed padded / un-padded maps | copies, aligned windows |

## Found, opt-in (`--sim_quality improved|modern`, see README)
1. **Joint control law.** `move_joints` sets the target to `current + 100*speed*error`: for `speed=0.01` the whole goal at once, for
   `0.02` twice the error. `speed` is not a speed; the PD controller closes the gap in a few steps (large accelerations).
   `controller='pd'` advances a command at most `v_max*dt` per step (URDF velocity limits), starting from the current state.
2. **Force check during free-space motion [measured].** `start_sensing = 0.5 m` is larger than every tool x on the approach
   (<= 0.35), so the 300 N check runs from the first waypoint: about half of all actions (19-23 of 40) were aborted, at 300-1700 N,
   in every configuration. `--force_sense_x 0.27` limits the check to the cabinet; `profile_sim.py` now prints how many stops
   happened outside the cabinet front.
3. **Raw reaction forces.** `getJointState()[2]` includes the tool weight and changes with the pose: tared, median-filtered,
   debounced monitor (`force_sensing.py`) -- it did not reduce the stops [measured], i.e. they are sustained, not spikes.
4. **`place()` forced `success = True`** after a failed approach (retry result thrown away). `strict_place` keeps the retry result.
5. **IK without limits / rest pose.** PyBullet ignores URDF joint limits unless they are passed. Limits + rest pose are passed
   with `ik_limits` (the 100-iteration setting was removed: 7x slower IK [measured]).
6. **Suction = rigid 6-DoF weld, 1000 N.** A real cup holds ~tens of N and breaks at a shear / axial limit. `suction_verify`
   (still held at the grasp offset after the lift), `suction_min_alignment`, and `suction_max_force` (paper preset: 50 N) exist.
   Isaac Sim's surface gripper models this with separate shear- and coaxial-force break limits and a compliant axis
   (see sources in the PR description); a force-limit based release is the natural next step.
7. **Solver.** PyBullet's default is 50 solver iterations; `solver_iterations=100` and `deterministicOverlappingPairs` (identical
   runs for identical seeds) are set by `modern`. Tune one setting at a time against a fixed seed.

## Still open (not changed)
* `activate()`: in the closest-point fallback the constraint is created from the *last* contact point's body even if it is not a
  rigid object (stale `obj_to_body`).
* `getCameraImage` segmentation is cast to `uint8`: object ids are truncated to 8 bits and link indices dropped (fine below 256 bodies).
* Force sensors are enabled on hard-coded joint indices 10 and 11 (look them up by name).
* Motor forces are not passed (`pd_motor_forces='urdf'` uses the URDF efforts; the base slide declares 1 N, a placeholder).
* Cabinet is 0.5 x 0.35 m vs 40 x 60 x 27 cm in the paper (`--physics paper`), blocks weigh 1 kg vs 0.2-0.4 kg YCB objects.
* The base slide (prismatic joint along y) makes this a gantry-like UR5e; the paper describes a UR5.
