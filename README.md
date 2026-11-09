# Learning to Rearrange Objects in Confined Environments

Code for the paper's pushing + pick-and-place rearrangement policy (double DQN with fully convolutional
networks, PyBullet UR5 simulation). Project page: https://alessandropalleschi.github.io/Learning2Rearrange/

## Run in a dev container (recommended)
1. Install Docker and VS Code with the *Dev Containers* extension (or use GitHub Codespaces).
   For GPU training also install the NVIDIA Container Toolkit; without a GPU delete `"--gpus=all"` from
   `.devcontainer/devcontainer.json` (training then runs on the CPU, slowly).
2. "Reopen in Container", then in the terminal:

```bash
python train.py                 # headless training, curriculum 1 -> 8 objects
xvfb-run -a python train.py --gui   # PyBullet GUI inside the container (virtual display)
python train.py --is_testing --load_snapshot --snapshot logs/train/models/snapshot-000500.pth
```

Useful flags:

| flag | meaning |
|---|---|
| `--num_envs N` | N parallel simulators (training is simulation-bound) |
| `--num_obj` / `--max_obj` | curriculum start / end (paper: 1 -> 8 objects) |
| `--max_actions` | actions per episode |
| `--physics paper` | 40x60x27 cm cabinet, weaker suction, lighter objects (unvalidated; default `legacy`) |
| `--push_lengths 0.04 0.08 0.12` | learned push: the policy also chooses the push length (default: fixed 0.08 m as in the paper) |
| `--footprint_placement` | keep placement targets clear of the back wall by the held object's depth (fewer aborted places; see `SIMULATION_NOTES.md`) |
| `--placement heuristic\|learned` | where the held object is placed: rule from the paper, or a learned place head (see below) |
| `--belief none\|spatial\|recurrent` | belief-state ablation (see below); `evaluate.py` needs the same value |
| `--sim_quality improved` | opt-in simulation improvements: IK with joint limits, tared/filtered/debounced F/T contact detection, verified suction grasps (see below) |
| `--randomize` | domain randomisation: object friction/mass, camera pose, sensor noise |
| `--gui` | PyBullet GUI for the first simulator |

### Simulation quality (`--sim_quality improved`, opt-in)
The default reproduces the original simulator behaviour. `improved` switches on (all unvalidated against the legacy behaviour, so
compare success rates and `profile_sim.py` timings before trusting it):
* **IK**: `calculateInverseKinematics` gets the URDF joint limits (base slide limited to the workspace), the home pose as rest pose
  and tighter solver tolerances (PyBullet ignores joint limits unless they are passed explicitly).
* **Contact detection** (`force_sensing.py`): the joint F/T reaction wrench is tared at the start of each motion, median filtered and
  debounced before it stops a motion. The raw reading contains the tool weight, differs per arm pose and spikes for single steps.
* **Suction**: seal needs a surface normal within ~37 degrees of the cup axis (legacy: 60), and after the lift the object must still be
  at its grasp offset, otherwise the grasp counts as lost (`suction_holds`).
`--sim_quality modern` adds the rate-limited joint controller (`arm_control.py`), 100 solver iterations, deterministic contacts,
strict `place()` errors, adaptive (coarser) retreat waypoints and the force check only near the cabinet. See `SIMULATION_NOTES.md`
for the review of the simulator (errors fixed, findings, open issues). Compare modes with the same seed:
`python profile_sim.py --actions 100 --seed 1 [--sim_quality modern]`.
Not changed (needs tuning on your setup): explicit motor forces / velocity limits of the joint controller, and a force-limited
vacuum model instead of the fixed constraint (`suction_max_force` is already a parameter; the `paper` preset uses 50 N).

See **ABLATIONS.md** for the transformer backbone, the hierarchical agent, the reward variants (with measurements) and the decision on ACT-style models.

### Perception: a segmentation network instead of the simulator's ground truth (`--perception network`)
The policy sees colour-coded class maps built from object masks. By default the masks are PyBullet's ground truth; with
`--perception network` a trained U-Net (`l2r/perception`) provides them. An object's *class* (which target region it belongs to) is part
of the task, not of its appearance, so the network predicts the object **type** and the episode's specification (type -> class) turns it
into a class; instances are the connected components of each type. Ground truth stays available only as *privileged* information for
the learning signals that need it (rewards, hindsight relabelling) -- the policy never sees it.
```bash
python train_perception.py collect  --scenes 300 --out data/perception      # random scenes from the simulator (camera jitter + noise)
python train_perception.py train    --data data/perception --minutes 9      # -> assets/perception/segnet.pth
python train_perception.py evaluate --weights assets/perception/segnet.pth    # IoU, instance recall / precision, effect on the observation
python train.py --perception network --perception_weights assets/perception/segnet.pth ...
```

### Learned push (`--push_lengths`)
The push primitive has two directions (left/right, the arm only has lateral access) and, by default, a fixed 8 cm length. With
several lengths (e.g. `--push_lengths 0.04 0.08 0.12`) the push head outputs one Q-map per length for each direction
(`2L` maps, action index `direction * L + length`), and the feasibility mask of a push grows with its length (a longer push can start
farther from an object). Direction handling (180 degree flip), the Q-learning target, the recurrent action encoding and the hindsight
replay all use the same index. Not learned: the push height (5 cm, chosen to avoid toppling) and finer angles.
Train and evaluate with the same `--push_lengths`.

### Placement ablation
* `heuristic` - pick-and-place is one action; the place pixel is chosen by a hand-written rule (paper).
* `learned` - pick and place are two decisions. After a successful pick the environment returns the observation with the
  object lifted (`held_class`, free cells, the heuristic proposal) and reward 0; the policy then selects the place pixel
  with a third Q-head conditioned on the held class and restricted to free cells. The reward (-1 / 0 / -10) is given after the
  place, the pick is bootstrapped from the best place Q-value without discount. While exploring, the placement is the
  heuristic one with probability `place_heuristic_mix` (0.5), otherwise a random free cell. A failed pick ends the action
  with the usual step penalty. Hindsight relabelling also recolours the held class.

### Belief-state ablation
* `none` - memoryless policy on the current observation (the paper).
* `spatial` - `belief.py` keeps a per-episode memory map (last seen colour/height/object id of every cell, its age, and the
  location of the last action) and feeds it as 7 extra input channels. Occluded cells are the grey ones of the colormap.
* `recurrent` - a ConvGRU at the network bottleneck keeps a latent belief, updated from the encoded latest observation and the
  encoded latest action. The hidden state is stored with every transition (stored-state replay, one-step gradient): it can be
  stale w.r.t. the current weights, and hindsight-relabelled transitions reuse the original hidden state.

Logs go to `logs/<logging_directory>/` (`episodes.csv`, TensorBoard in `tb/`, snapshots in `models/`):

```bash
tensorboard --logdir logs
```

Speed: simulation is the bottleneck. Use one worker per CPU core (`nproc`), e.g. `--num_envs 8`, and see where the time
goes with `python profile_sim.py --actions 20` (compare `--step_distance 0.004` / `--move_budget 400`: coarser IK waypoints
and a smaller joint-move budget are faster but change the motion slightly, so re-check the success rate).

Evaluate a snapshot on fixed seeds (success rate and mean number of actions per object count):

```bash
python evaluate.py --snapshot logs/train/models/snapshot-0010000.pth --objects 3 5 8 --episodes 20 --num_envs 4
```

Tests (no PyBullet needed, uses a fake simulator): `pytest -q`; lint: `ruff check .`. CI runs both.

## Without a container
`pip install torch` (CUDA build if you have a GPU) and `pip install -r requirements.txt`.

## Code layout
```
train.py  evaluate.py  profile_sim.py  demo_sim.py      thin entry points (logic lives in l2r/)
l2r/
  config.py        Config (learning) and SimParams (physics, controller, camera options, presets)
  training.py      TrainingLoop + Curriculum: parallel simulators, replay, hindsight relabelling, snapshots
  evaluation.py    fixed-seed greedy evaluation per object count
  profiling.py     timing and behaviour breakdown of the simulator
  logger.py utils.py
  perception/
    base.py        Perception interface: ground truth (PyBullet) or network; Segmentation (instances, names, privileged match)
    unet.py        small U-Net (0.4 M parameters) for the object types
    network.py     NetworkPerception: types -> classes (task specification) -> instances
    training.py    collect data, train, evaluate (also the effect on the policy's observation)
  envs/
    rearrange.py   RearrangeEnv: Gym-style reset()/step(): action execution, reward, termination, relabelling
    vec.py         parallel simulators in worker processes
  learning/
    models.py      QNet: 4-channel input, GroupNorm, global context, shared backbone, push/pick/place heads
    trainer.py     batched double DQN (prioritized replay, Huber loss on the taken pixel, clipped targets, AMP)
    policy.py      epsilon-greedy action selection, recurrent-belief bookkeeping
    memory.py belief.py
  sim/
    environment.py Environment: wires the parts below to one PyBullet client (injectable for tests)
    workspace.py   Workspace: cabinet bounds, heightmap grid, pixel <-> metre conversions (no global state)
    constants.py   classes, colours, object catalogue, target-region candidates
    scene.py       Scene: cabinet, objects, target regions, goal / invalid-state checks
    camera.py      Camera: RGB-D rendering and its randomisation
    vision.py      point cloud -> heightmaps, object masks, free-space map (pure numpy)
    arm.py         Arm: joint control (legacy / rate-limited), IK, straight moves, force-limit stop
    suction.py     SuctionCup: constraint-based vacuum model
    primitives.py  Manipulator: push / pick / place
    arm_control.py force_sensing.py   rate limiter and URDF limits; tared/filtered contact detection
tests/             pytest: components against fakes, pipeline with a fake simulator, real-simulator regression
SIMULATION_NOTES.md  review of the simulator, rendering and networks: errors fixed, findings, open issues
```

Tests: `pytest -q` (no PyBullet needed). With PyBullet and a recorded golden file, `L2R_GOLDEN=1 pytest tests/test_sim_regression.py`
replays a seeded, scripted scenario on the real simulator (`python tests/sim_scenario.py record|check`).

Snapshots saved by older versions (two separate BatchNorm networks, 6 input channels) are not compatible.
