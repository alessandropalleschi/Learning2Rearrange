# Keeping a move to Isaac Sim / Isaac Lab open

Status: a plan, not measured. Nothing here was run on Isaac. It records what a migration would involve so that today's changes do not
make it harder, and when it would pay off.

## Why it could pay off
Simulation is the bottleneck: ~1 action/s per PyBullet process (thousands of 1/480 s physics steps per action), so the learner idles.
Isaac's gain comes from running many environments in parallel on the GPU and from better contact / suction / deformable models, not from a
faster single environment. The speed options already in the repo (`--adaptive_step`, `--joint_tolerance`, more `--num_envs`) give
roughly 2.4x x ~3x on PyBullet; Isaac is worth its cost when training needs millions of steps, or when sim-to-real needs a better gripper model.

## The boundary (done): `l2r/sim/backend.py`
Everything above the simulator uses only `SimBackend`: reset, observe, push, pick, place and a few task-state queries. The observation
maps (`vision.py`) are pure numpy and perception, belief, replay, rewards and the agents never touch the engine. A backend that returns
the same observation dict is a drop-in; `tests/fakes.py::FakeSim` already is one.

## What a port consists of, in order of risk
1. **Scene**: cabinet (static boxes), YCB objects (USD conversion of the meshes, convex decomposition for collisions), UR5e on the lateral
   slide (Isaac ships a UR5e; the prismatic base joint is custom).
2. **Camera -> maps**: render RGB-D + instance ids (Isaac provides instance segmentation); `vision.build_observation` can be reused unchanged
   because it only needs colour, metric depth, an id image and the camera pose.
3. **Suction**: replace the constraint model by Isaac's surface gripper (shear / coaxial force limits). This is where fidelity improves.
4. **Primitives**: the real work. Today they are sequential scripts (IK waypoint, settle, check force). For many parallel environments they must become
   batched controllers (differential IK / operational-space control over all environments, per-environment termination masks). A scripted
   per-environment loop would give none of the GPU speed-up.
5. **Vectorised environment**: `l2r/envs/vec.py` runs one process per simulator; Isaac steps all environments in one process, so the batch
   interface (`step` over N actions) becomes native and the worker processes disappear.

## What stays valid / what does not
Valid: network, trainer, replay, belief, perception code, policy selection, reward variants, the validation curriculum, the experiment runner.
Not valid: golden regression results (`tests/golden`), the force-limit and timing numbers, any learned perception weights (new renderer), and
comparisons with PyBullet runs. Re-record the goldens and retrain perception on the new backend.

## Suggested path if it is pursued
1. Prove the policy works on PyBullet first (this is the current work): an architecture that does not learn here will not learn faster on Isaac.
2. Port the scene + camera and compare `observe()` against PyBullet on identical arrangements (heightmaps should agree within a cell or two).
3. Port the primitives for one environment with the same interface, check `executed` rates, then batch them.
4. Only then scale the number of environments.
