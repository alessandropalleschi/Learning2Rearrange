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
| the target of the heuristic placement ignores the object's extent: it is placed up to 8 columns (4 cm) from the back wall while the object body reaches past the tool **[measured]**: the force limit stops the approach at the back wall | `--footprint_placement` clears the free-space map of the cells closer to the back wall than the object is deep: on 8 seeded scenes, completed pick-and-place 2/8 -> 5/8 and force-limit stops 5 -> 2 (same objects reached their target region: sorted gain 1.65 in both). Small sample, off by default |
| per-event `print`s from the simulator | `logging` (`DEBUG` for force stops, `WARNING` for a failed reset) |

Removed: the unused `ontargetmap`, the raw `color` / `depth` / `mask` images in the observation, class entries other than
blue / green / yellow (`DIMENSIONS` held colour tuples for the unused colours), the unused `fixed` object category, and the
module-level mutable `BOUNDS` / `PIXEL_SIZE` / `PLACE_TARGETS`.

## Architecture and observation processing: review with measurements
Measured with the real simulator (8 objects) and the network on 4 CPU threads.

**Observation pipeline** (358 ms per observation before, 252 ms now):
| stage | before | now | note |
|---|---|---|---|
| camera render (TinyRenderer, 480x640) | 196 ms | 204 ms | 80% of the time now; `--camera_scale 0.5` renders 2.7x faster (sparser maps, check the occluded fraction in `profile_sim.py`) |
| class masks | 22 ms | 6 ms | the per-object full-size mask images are gone |
| point cloud + camera -> world | 35 ms | ~10 ms | pixel offsets cached per camera, one matrix product |
| per-object maps | 91 ms (grows with the number of objects: each object re-projected the whole point cloud) | ~15 ms | one projection of the object-id image: the id of the highest point of every cell; verified equal to the per-object reference (`tests/reference_vision.py`) on random scenes and on the real scenario |
| top-down maps, free space, padding | 14 ms | 14 ms | |

Observations that are *not* changed (open): the 8 mm floor threshold, the free-space dilation (rows {-2, 0, +2}, +-4 columns, 5 / 8 px
borders) and the "misplaced objects block their whole row" rule are inline constants of `vision.py`; the occluded cells are marked by the
grey value 105 inside the colour channels (an explicit mask channel would be cleaner: ablation); heights are float32 (uint16 millimetres
would halve the replay buffer: 8000 transitions x 2 maps x 53 KB).

**Network** (8.3 M parameters: encoder 4.8 M, decoder 2.5 M, global context 0.8 M, heads 0.2 M). The cost is in the full-resolution layers, not
the parameters: per pass 21 ms encoder, 11 ms decoder, 21 ms for the three heads (full-resolution residual blocks).
| change | forward (batch 1) | train step (batch 1) | forward (batch 8) | train step (batch 8) |
|---|---|---|---|---|
| before: 160x160 padded canvas, all heads | 174 ms | 442 ms | 1491 ms | 4198 ms |
| now: native 128x104 canvas, place head only when placement is learned | 99 ms | 236 ms | 527 ms | 1583 ms |

* The maps (128 x 104) are already multiples of 8, so the network runs on them directly: no padding to 160 x 160 (46% empty compute), the global
  pooling no longer averages over padding, and the 180 degree flip is exact on a rectangle too.
* Both push directions go through the encoder in one batched call; `cudnn.benchmark` is on (fixed input size).
* Open: lighter heads (they cost as much as the encoder; a 1x1 / half-resolution head risks the 5 mm precision of the Q-maps, test it),
  `channels_last` + bf16, `torch.compile` of `QNet.features` on GPU, an asynchronous actor / learner split (simulation, ~1 s per action,
  dominates the wall-clock; the learner is idle while the workers act).

## Learned perception instead of the simulator's ground-truth masks
`--perception network` replaces the PyBullet segmentation buffer by a small U-Net (0.39 M parameters, `l2r/perception`).
Design points:
* An object's class is part of the task (its target region), not of its appearance. The network predicts the object **type** (4 YCB types +
  background); the episode's specification (type -> class) maps types to classes, types that are not part of the episode are ignored.
* Instances are the connected components of every type, **split in the top view** (`vision.split_components`): in the image, objects in
  front of / behind each other overlap and merge, in 3D they do not.
* The policy only sees what the network produced. The ground truth survives as *privileged* information for the learning signals that need
  it (`gt_idmap`, `instance_to_object`: hindsight relabelling, the belief's id memory), never as policy input.

Measured (300 training scenes, camera pose jitter + sensor noise, 9 minutes of CPU training; evaluated on 25 fresh
scenes with 2-8 objects):
| | value |
|---|---|
| pixel IoU of the object types | 0.948 / 0.963 / 0.969 / 0.965 (background 0.997) |
| instance recall / precision, connected components in the image | 0.844 / 0.945 |
| instance recall / precision, split in the top view (what the policy sees) | **0.959 / 0.860** |
| class-map agreement with the ground-truth observation (object cells) | 0.886 -> **0.928** with the top-view split |
| free-space map IoU with the ground-truth observation | 0.757 -> **0.969** with the top-view split |
| inference | 63 ms per image on 4 CPU threads (the observation grows from ~250 to ~320 ms) |

Limits (open): only the 4 YCB types of the simulator were seen (no "unknown objects": that needs a class-agnostic instance
segmentation, e.g. the methods the paper cites); the training scenes are initial arrangements (no arm in motion, no objects pressed against
the walls); touching objects of one type stay merged; precision 0.86 comes from objects visible in several pieces. Whether a policy
trained on the network's observations matches one trained on the ground truth has **not** been measured (it needs a full training run each);
`tests/test_sim_real.py::test_learned_perception_path_with_an_oracle` and a scripted run with both placement modes, the spatial belief and
the relabelling check that the whole stack works with the network as the only source of masks.

The trained weights are in `assets/perception/segnet.pth` (1.5 MB, legacy cabinet / camera; retrain with `train_perception.py` after
changing the scene, e.g. `--physics paper`, or the camera).
