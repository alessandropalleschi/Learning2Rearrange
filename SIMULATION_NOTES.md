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

## Second audit: rendering, observation and networks
Found and fixed:
| issue | fix |
|---|---|
| `get_observation` relabelled the segmentation image in place while still looping over the ids found in it (labels 1/3/5 are also valid object ids) | separate label map, logic moved to `_class_masks` (unit-tested with colliding ids) |
| segmentation was cast to `uint8`: object ids truncated to 8 bits and link indices dropped | decode the 24-bit object uid (`& 0xFFFFFF`) |
| `ndimage.interpolation.shift` (deprecated namespace) | `ndimage.shift` |
| depth noise of 3 mm vs. the 8 mm "object" threshold of the heightmap: ~0.4% false object cells with `--randomize` | `depth_noise_std = 1.5 mm` |
| push orientation quaternion `(0.7, 0, -0.7, 0)` has norm 0.99 | exact unit quaternion |
| dead code: unused methods (`step`, `seed`, `log_video`, ...), unused constants, unreachable branches, unused attributes / parameters | removed (`vulture` clean at 100% confidence) |

Checked, no error found (kept): camera model and depth conversion, point-cloud transform, z-buffered heightmap, padding / crop
offsets of all maps (colormap and depthmap are padded by 4 px, idmap / classmap / freespace are not), the 180 degree flip (the 160x160 canvas is
symmetric around the cabinet, tested for equality with the explicit forward pass), the double-DQN target (policy net selects, target
net evaluates, masked actions, clipped to the reachable range), prioritised replay (sampling with replacement, importance weights,
priority update order), the GroupNorm / residual blocks, hindsight relabelling.

Open (design limits, not bugs):
* The global-context pooling includes the zero-padded border (about half of the 160x160 canvas); a masked pooling would be cleaner.
* A greedy push is not checked for feasibility before it is executed (exploration is); an infeasible push is a no-op with reward -1
  and the network learns a low value for it.
* `classmap` (exact object pixels) is used by the environment to validate a grasp while the policy mask uses `height > 8 mm`.
* Weight decay is also applied to normalisation parameters and biases.

## Rebuild of the simulator (package `l2r.sim`)
`env.py` (one 1400-line class with module-level globals) was replaced by cohesive components: `Workspace`, `Scene`, `Camera`,
`vision`, `Arm`, `SuctionCup`, `Manipulator`, wired by `Environment` (the PyBullet client is injected, so every component is
testable with a fake backend). The port was verified against a **deterministic scripted scenario on the real simulator**
(`tests/sim_scenario.py`): the rebuilt code reproduced the recorded results of the old code in both the legacy and the modern mode
before any behaviour was changed.

Bugs found while porting (fixed afterwards; `tests/golden/*.json` were re-recorded as the new baseline):
| bug | fix |
|---|---|
| `ForceMonitor` sampled every reading twice (`update()` then `force()`), halving the median window | `update()` computes the force once and exposes `last_force` |
| `place()` returned the success of the *retreat*, so a place aborted by the force limit still counted as executed | success = reached and placed and retreated and valid |
| a dead `np.random.randint(0, 1)` draw (one candidate set) | removed |
| the target of the heuristic placement ignores the object's extent: it is placed up to 8 columns (4 cm) from the back wall while the object body reaches past the tool **[measured]**: the force limit stops the approach at the back wall | open: erode the free-space map by the object's half-depth |
| per-event `print`s from the simulator | `logging` (`DEBUG` for force stops, `WARNING` for a failed reset) |

Removed: the unused `ontargetmap`, the raw `color` / `depth` / `mask` images in the observation, class entries other than
blue / green / yellow (`DIMENSIONS` held colour tuples for the unused colours), the unused `fixed` object category, and the
module-level mutable `BOUNDS` / `PIXEL_SIZE` / `PLACE_TARGETS`.
