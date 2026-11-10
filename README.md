# Learning to Rearrange Objects in Confined Environments

A robot arm sorts objects inside a cabinet it can only reach from the front. It sees the cabinet from one RGB-D camera, and chooses
between **pushing** (a blade) and **pick-and-place** (a suction cup or a parallel-jaw gripper) at every step. The policy is a
fully convolutional network that outputs a dense Q-map over the top-down map of the cabinet, trained with double deep Q-learning in a
PyBullet simulation. Project page: <https://alessandropalleschi.github.io/Learning2Rearrange/>

The repository started as the code of the paper and is now a research platform around it: every extension is an opt-in flag, so the
paper's method is one configuration among several. **Nothing here has been trained to convergence; the numbers in the docs measure mechanics (speed, signals, expert
success, mask effects, imitation, RL plateaus and their causes), not final success rates.** See [Status and results](#status-and-results).

| document | content |
|---|---|
| this file | what the code does, how it is organised, the variants and how to run them |
| [`ABLATIONS.md`](ABLATIONS.md) | what each flag changes, why, and what was decided |
| [`SIMULATION_NOTES.md`](SIMULATION_NOTES.md) | review of the simulator, rendering and networks: bugs fixed, measurements, open issues |
| [`ISAAC_MIGRATION.md`](ISAAC_MIGRATION.md) | how a move to Isaac Sim / Lab would work (a plan, not measured) |

## Contents
1. [Quick start](#quick-start)
2. [The task and the pipeline](#the-task-and-the-pipeline)
3. [Code layout](#code-layout)
4. [Observation processing](#observation-processing)
5. [The agents and networks](#the-agents-and-networks)
6. [Actions and primitives](#actions-and-primitives)
7. [Training](#training)
8. [Variants](#variants)
9. [Tools and experiments](#tools-and-experiments)
10. [Status and results](#status-and-results)

---

## Quick start

**Dev container (recommended).** Install Docker and the VS Code *Dev Containers* extension (or use Codespaces); for GPU training also the
NVIDIA Container Toolkit (without a GPU delete `"--gpus=all"` from `.devcontainer/devcontainer.json`; training then runs on the CPU, slowly).
"Reopen in Container", then:

```bash
pytest -q                                     # unit + pipeline tests (no PyBullet needed for most)
python train.py --num_envs 8 --max_obj 5      # headless training, curriculum 1 -> 5 objects
python evaluate.py --snapshot logs/train/models/snapshot-0010000.pth --objects 3 4 5 --episodes 20 --num_envs 4
xvfb-run -a python train.py --gui             # PyBullet GUI inside the container (virtual display)
```

Model checkpoints (`*.pth`, e.g. the perception networks in `assets/perception/`) are stored with **Git LFS**: run `git lfs install && git lfs pull`
after cloning (the dev container has it installed).

Without a container: `pip install torch` (CUDA build for a GPU) and `pip install -r requirements.txt`.
Logs go to `logs/<logging_directory>/` (`episodes.csv`, `validation.csv`, TensorBoard in `tb/`, snapshots in `models/`).

The flags that define the **model and agent** (`--belief`, `--placement`, `--backbone`, `--hierarchy`, `--push_lengths`, `--mask_level`,
`--end_effector`, ...) must be identical for training and `evaluate.py`; `python train.py --help` lists everything.

---

## The task and the pipeline

Objects of three classes (blue / green / yellow) stand on the floor of a cabinet; each class has a **target region** (a band across the
cabinet width). The episode ends when every object is inside the region of its class (goal), an object falls or topples (invalid), or
the action budget is used up. The robot reaches in from the front only, so objects hide behind each other.

![scene](docs/figures/scene.png)
*The UR5e with the suction tool in front of the cabinet, the robot's own RGB-D camera, and a view into the cabinet. YCB objects (cracker
boxes, chips cans, meat cans, mustard bottles) are assigned to the three classes per episode.*

![episode](docs/figures/episode.png)
*Robot state (top) and the observation the agent receives (bottom) along an episode of the scripted expert with learned placement:
a grasp lifts the object (the observation then shows the scene without it), the place puts it into its region. In step 3 the grasp
tips the chips can over: the episode ends as invalid, the most common failure with many objects.*

```mermaid
flowchart LR
    subgraph SIM["Simulator (PyBullet) - l2r/sim"]
        direction TB
        S1[Scene: cabinet, objects,<br/>target regions] --> C1[Camera RGB-D]
        A1[Arm + tool<br/>suction / gripper] --- S1
    end
    C1 -->|colour, depth, object ids| P["Perception<br/>ground truth | type network | agnostic network"]
    P -->|instance masks + classes| V["Vision: point cloud -> top-down maps<br/>l2r/sim/vision.py"]
    V -->|"colormap + height map<br/>(+ belief)"| N["Q-network<br/>l2r/learning/models.py"]
    N -->|"dense Q-maps<br/>push / pick / place"| M["Masks + action selection<br/>l2r/learning/masks.py, policy.py"]
    M -->|"primitive + pixel (+ length)"| E["RearrangeEnv<br/>l2r/envs/rearrange.py"]
    E -->|"push / pick / place"| A1
    E -->|"reward, done, next observation"| R[(Replay buffer<br/>+ demonstrations)]
    R --> T["Double-DQN trainer<br/>l2r/learning/trainer.py"]
    T --> N
```

`RearrangeEnv` is the Gym-style boundary (`reset`, `step`): it executes the chosen primitive in the simulator, computes the reward, detects the
end of the episode, produces the next observation and the **hindsight relabelling** (a state where every object sits in *some* region is
turned into a sorted state by reassigning the classes). The simulator behind it is replaceable: `l2r/sim/backend.py` lists the
~15 calls the learning code needs (see [`ISAAC_MIGRATION.md`](ISAAC_MIGRATION.md)).

---

## Code layout

```
train.py  evaluate.py  profile_sim.py  analyze_rewards.py  train_perception.py  demo_sim.py     thin entry points
scripts/   run_experiments.py (suites, seeds, summaries)  collect_demos.py  merge_demos.py  speed_ablation.py  grasp_profile.py
           arch_benchmark.py (supervised architecture benchmark)  dagger.py (DAgger with the scripted expert)
l2r/
  config.py          Config (learning, agent variants) and SimParams (physics, controller, camera, presets); all CLI flags
  training.py        TrainingLoop + Curriculum: parallel simulators, replay, relabelling, validation-based curriculum, snapshots
  evaluation.py      fixed-seed greedy evaluation per object count
  demos.py           scripted expert, demonstration recording, demo adaptation (push-length encoding, belief stripping)
  arch_benchmark.py  network variants trained on demonstrations (cross-entropy), held-out metrics, closed-loop rollouts
  dagger.py          DAgger: learner drives, expert labels the visited states, aggregate, retrain
  profiling.py  reward_analysis.py  logger.py  utils.py
  envs/
    rearrange.py     RearrangeEnv: action execution, reward variants, termination, relabelling, placement helpers, outcome record
    approach.py      collision-aware approach lines and placement corridors from the height map
    vec.py           parallel simulators in worker processes
  learning/
    factory.py       make_trainer(cfg): the agent by --agent (dense | object | policy_critic)
    models.py        dense QNet: encoders (residual | ConvNeXt | ResNet-18), decoders (plain | U-Net | FPN | attention | HRNet),
                     bottleneck context (global | transformer | PPM | ASPP | ConvGRU), push/pick/place heads, hierarchy
    trainer.py       batched double DQN: prioritized replay, Huber loss on the taken pixel, DQfD margin loss, demo annealing, AMP
    object_centric.py  object-centric agent: instances / ghosts / place candidates from the observation, token transformer
    object_trainer.py  double DQN over object-action slots and place candidates (same interface as the dense trainer)
    policy_critic.py   policy heatmaps trained by behaviour cloning + Q critic over their top-K proposals
    iql_trainer.py     IQL for the object agent (expectile value, advantage-weighted policy)
    skill_params.py    parameterised skills (approach, heading, length) with a flow-matching parameter policy
    nstep.py           n-step returns for the replay
    policy.py        epsilon-greedy / hierarchical action selection, recurrent-belief bookkeeping
    masks.py         structural action masks (front-face grasps, blade-strip pushes)
    belief.py        spatial belief tracker (memory map)
    memory.py        uniform and prioritized replay
  perception/
    base.py          Perception interface: ground truth | network (Segmentation: instances, names, privileged id match)
    unet.py network.py agnostic.py training.py     type U-Net, object-agnostic U-Net, data collection / training / evaluation
  sim/
    environment.py   Environment: wires the parts below to one PyBullet client (injectable for tests)
    backend.py       SimBackend: the simulator boundary used by the learning code
    workspace.py     cabinet bounds, heightmap grid, pixel <-> metre conversions (no global state)
    scene.py         cabinet, objects, target regions, goal and invalid-state checks
    camera.py vision.py       RGB-D rendering; point cloud -> top-down maps, object maps, free-space map (pure numpy)
    arm.py arm_control.py force_sensing.py     UR5e: joint control (legacy / rate-limited), IK, straight moves, force-limit stop
    suction.py gripper.py primitives.py        suction cup (constraint model), parallel-jaw gripper, push / pick / place
    constants.py     classes, colours, object catalogue, target-region candidates
tests/               pytest: components against fakes, learning, pipeline with a fake simulator, real-simulator tests + regression
```

---

## Observation processing

The policy never sees the camera image directly: it sees a **top-down map of the cabinet** in which every object is recoloured by its
class, so appearance (texture, lighting, object shape) is abstracted away.

```mermaid
flowchart TB
    R["Camera render<br/>RGB, metric depth, object-id image"] --> SEG["Perception: instance masks<br/>(+ class per instance)"]
    SEG --> RC["Recolour by class<br/>background white, object = class colour"]
    R --> PC["Depth -> point cloud -> world frame"]
    RC --> TD
    PC --> TD["Top-down projection (z-buffer, 5 mm cells)"]
    TD --> HM["Height map (m)"]
    TD --> CM["Colour map<br/>grey = occluded cell, target regions painted"]
    SEG --> OM["Object maps from ONE projection of the id image:<br/>id map, class map, on-target cells"]
    OM --> FS["Free-space map<br/>(placement candidates)"]
    HM --> PAD["Pad 4 px"] --> OBS(("Observation<br/>colormap + height map<br/>128 x 104"))
    CM --> PAD
```

![observation](docs/figures/observation.png)
*The observation: colour map (class colours, grey = occluded cells, target regions as coloured bands), height map, the age channel
of the spatial belief, and, while an object is held, the free cells it fits in (the learned placement's candidates).*

* The network input has **4 channels** (RGB colour map + height map), or 11 with the spatial belief.
* Occluded cells are marked grey inside the colour channels (the camera sees the cabinet from the front, so everything behind an
  object is unobserved).
* With a learned perception, instances are split into connected pieces **in the top view** (objects that overlap in the image
  separate in 3D). Ground truth then survives only as *privileged* information for the learning signals that need it
  (hindsight relabelling, the belief's id memory) and never as policy input.
* The replay buffer stores compact observations only (colour map, height map, belief): a few hundred KB instead of the raw images.

---

## The agents and networks

Three agents share the environment, the replay, the curriculum and the trainer interface (`--agent`, built by
`l2r/learning/factory.py`): every agent returns the same dense Q-maps, so action selection, masks and evaluation are common code.

| `--agent` | decides over | network |
|---|---|---|
| `dense` (default, paper) | every pixel of the push / pick / place maps | fully convolutional encoder-decoder |
| `object` | (visible object, skill) slots + place candidates | transformer over object tokens |
| `policy_critic` | top-K cells of learned policy heatmaps, ranked by the Q critic | dense network with policy heads |

### Dense Q-network (`--agent dense`)

One shared fully convolutional backbone with a dense head per primitive. The second push direction is the same head applied to the
input rotated by 180 degrees (an exact flip), so the network needs no learned rotation. Encoder, bottleneck context and decoder are
interchangeable:

```mermaid
flowchart LR
    X(("input<br/>4 x 128 x 104<br/>(+7 belief, +5 geometry)")) --> STEM["stem 3x3, 64"]
    FX["180 deg flip<br/>(2nd push direction)"] -.-> STEM
    STEM --> E1["stage 1 / 2<br/>128"] --> E2["stage 2 / 4<br/>256"] --> E3["stage 3 / 8<br/>512"]
    E3 --> CTX{{"bottleneck context<br/>512 x 16 x 13"}}
    HELD["held class<br/>(learned placement)"] -.-> CTX
    CTX --> D3["up + block<br/>256"] --> D2["up + block<br/>128"] --> D1["up + block<br/>64"]
    E2 -. "skip (U-Net)" .-> D3
    E1 -. "skip" .-> D2
    STEM -. "skip" .-> D1
    D1 --> PH["push head<br/>2 directions x L lengths"]
    D1 --> KH["pick head<br/>suction: 1 / gripper: 1 per jaw yaw"]
    D1 --> LH["place head"]
    D1 -.-> AH["hierarchy: length actor,<br/>primitive values"]
```

| part | flag | options |
|---|---|---|
| encoder blocks | `--arch_encoder` | `resnet` (residual 3x3, GroupNorm), `convnext` (7x7 depthwise ConvNeXt blocks), `resnet18_imagenet` / `resnet18_scratch` (torchvision stages) |
| decoder | `--arch_decoder`, `--arch_skips` | `plain` (paper: no skips), `unet` (skips), `fpn`, `attn_unet` (attention-gated skips), `unet_attn4` (+ global attention at 1/4), `hrnet` (parallel resolutions) |
| bottleneck context | `--backbone` | `conv` (pooled global context), `hybrid` (transformer, 208 tokens), `ppm` (pyramid pooling 1/2/4), `aspp` (dilations 1/2/4 + global pool), `ppm_aspp` |
| bottleneck size | `--arch_stride4` | bottleneck at 1/4 instead of 1/8 resolution (U-Net decoders) |
| memory | `--belief recurrent` | ConvGRU at the bottleneck, from the encoded observation and the last action |

![dense Q-maps](docs/figures/qmaps_dense.png)
*Q-maps of the dense U-Net + ConvNeXt agent (RL with the curriculum, 5,000 steps, still training) on one scene with five objects.
Q-values are shown only on feasible cells: pick on the objects' front faces, push from cells whose blade path meets an object (best of
the three lengths per cell; the arrow is the greedy push and its length), place on the free cells the held object fits in. The coloured
bars on the left are the target regions; light grey is occluded. The greedy action pushes the blue object standing in the yellow region
up into the blue one.*

The context modules start as the identity (zero-initialised residual output). Which combination works best is measured in
[Status and results](#status-and-results): the U-Net decoder with the ConvNeXt encoder (`--arch_skips --arch_encoder convnext`).

### Object-centric agent (`--agent object`)

The scene is described by tokens extracted from the observation alone (no privileged ids): one per visible object (a connected piece
of one class colour on raised cells), one per remembered but hidden object (from the spatial belief), one per target region and one
global token. A transformer contextualises them; each object token gets a value per skill. Candidate pixels are geometric: a grasp just
inside the object's front face, a push with the blade beside the object.

```mermaid
flowchart LR
    OBS(("observation<br/>colour + height<br/>(+ belief)")) --> EXT["extract tokens<br/>(object_centric.py)"]
    EXT --> G["global token<br/>objects, misplaced share"]
    EXT --> RG["3 region tokens<br/>class, rows"]
    EXT --> OB["N object tokens<br/>class, position, size, height,<br/>in region, occluded behind, clutter ahead"]
    EXT --> GH["up to 8 ghost tokens<br/>remembered hidden objects + age<br/>(--belief spatial)"]
    G & RG & OB & GH --> TF["transformer encoder<br/>2 layers, 64-d, type embeddings"]
    TF --> PICK["per object: pick value"]
    TF --> PUSH["per object and direction:<br/>push value (+ length actor)"]
    TF --> XA
    PC["place candidates<br/>free cells the held object fits in<br/>(grid, at most 512)"] --> XA["cross-attention<br/>candidates -> scene tokens"]
    XA --> PLACE["place value per candidate"]
    PICK & PUSH & PLACE --> RAS["rasterise into the dense<br/>Q-map interface"]
```

![object-centric agent](docs/figures/qmaps_object.png)
*The object-centric agent (RL, 8,000 steps, still training) on the same scene: the object tokens it extracts (numbered) and a ghost
token for a remembered, now hidden object (dashed); the Q-value of each skill on each object (n/a: infeasible, e.g. no room for the
blade); and the values of the place candidates while an object is held. It makes the same choice as the dense agent: push object 3 up,
with its own continuous length.*

* **Fixed or continuous push length.** Without `--hierarchy` every (direction, length) is a slot. With `--hierarchy` an actor proposes
  a length per object and direction, and a length-conditioned critic (Fourier features of the length) scores it. RL trains the actor
  DDPG-style at a tenth of the critic's learning rate; imitation regresses it to the expert's length.
* **Learned placement** (`--placement learned`): while an object is held, the free cells it fits in (footprint-aware) become candidate
  tokens on an adaptive grid; each attends to the scene and gets a value. RL backs the grasp up from the best placement without discount.
* **Ghost tokens** carry the spatial belief: objects that were seen and are now hidden take part in the attention but are not actionable.
* **What it knows about its last action**: every observation carries the last primitive, whether it executed, how much of the height
  map changed and whether the scene became invalid (`RearrangeEnv._record_last`, no privileged information). Object tokens flag "the
  last pick / place was on me" and "the last push started next to me"; the global token holds the rest. `--history_tokens` adds the last
  4 steps as memory tokens (object class at the action cell, primitive, outcome, parameters, age) next to the ghosts.
* **Outcome heads** (`--outcome_weight`): per candidate P(executed), P(scene invalid) and the expected height-map change, trained from
  those free labels. `--outcome_mask tau` drops candidates predicted to fail or to change nothing (when acting and in the TD targets),
  `--outcome_penalty` subtracts the predicted invalid risk from Q.
* **Learning algorithms** (`--object_algo`): double DQN (default) with optional clipped double Q (`--twin_q`) and n-step returns
  (`--n_step`); or **IQL** (`--object_algo iql`): V(s) regresses an upper expectile of the target Q of the actions actually taken, Q
  regresses r + g V(s'), and the policy (logits over the object-skill slots and the place candidates) is fitted by advantage-weighted
  cross-entropy. No max over unseen actions anywhere, which is where the DQN agent overestimated (see the results).
* **Parameterised skills** (`--skill_params`, with IQL): grasp with a continuous approach angle, push with a continuous heading and
  length (the start cell follows from the heading). A parameter-conditioned twin critic scores them; the parameters are modelled by
  advantage-weighted **conditional flow matching**; acting takes the best of 16 flow samples under the critic (IDQL-style).
* All of this is tested by bandits in `tests/test_object_centric.py` and `tests/test_skill_params.py`; the agent is about 0.1 M
  parameters (the dense ConvNeXt network has 9.8 M).

### Policy-critic agent (`--agent policy_critic`)
The dense network gets policy heads (logits per cell) trained by behaviour cloning on the demonstrations; the action is chosen among the
top-K cells of the policy by the Q critic (`--policy_topk`). It keeps the dense maps but restricts the argmax to plausible cells.

---

## Actions and primitives

| primitive | action (pixel in the Q-map) | execution |
|---|---|---|
| **push** | `(direction x length index, row, col)`: start of the pusher, direction left / right; or `(0, row, col, length, heading)` with any heading away from the robot | the pusher goes in 5 cm above the floor, pushes along the cabinet width (or the heading), retreats |
| **pick** | `(row, col)` (suction) or `(jaw yaw, row, col)` (gripper); optionally `(0, row, col, approach)` | approach until contact (straight in, or along a line rotated by `approach`), vacuum on / jaws close, lift out |
| **place** | free-space pixel | carry the object to the cell, release, retreat |
| **pull** | `{'primitive': 'pull', 'pix': ..., 'heading', 'length', 'approach'}` | attach (cup or jaws), drag along the floor (default towards the robot), release |

Primitive options (all opt-in; the defaults are the original motions):

| flag | what it changes | measured |
|---|---|---|
| `--pusher paddle` | a T-shaped plate fixed to the tool during pushes: an 8 cm side plate for pushes along the width, a 6 cm front plate only for pushes towards the back | an 8 cm push moves objects 4.8 cm on average instead of 2.5 cm; 70 % instead of 38 % of pushes move the object by >= 3 cm |
| `--grasp_approach clear` | the env picks the grasp's approach line (0, +-15, +-30, +-45 deg) with the fewest other objects in the corridor the tool and the held object sweep (height map only) | grasps whose straight corridor is blocked: held 0.78 -> 0.97, scene invalid 0.14 -> 0.03 |
| `--place_corridor` | the heuristic placement prefers cells whose carry-in corridor is free; the learned placement gets the corridor as a candidate feature | helps the expert under `modern` physics, not under `legacy` (see the results) |

![paddle](docs/figures/paddle.png)
![primitives](docs/figures/primitives.png)
*Left: displacement of an 8 cm push at 5 cm height, tool tip vs paddle (`scripts/physics_sanity.py`). Right: grasps whose straight
corridor is blocked by other objects, straight in vs the clearest approach line (`scripts/approach_profile.py`).*

```mermaid
sequenceDiagram
    participant Pi as Policy
    participant Env as RearrangeEnv
    participant Sim as Simulator
    Note over Pi,Sim: placement = heuristic (paper): pick-and-place is ONE action
    Pi->>Env: grasp at pixel p
    Env->>Sim: pick(p)
    Env->>Env: heuristic place pixel (target region, free space)
    Env->>Sim: place(q)
    Env-->>Pi: reward (-1 / 0 / -10), next observation
    Note over Pi,Sim: placement = learned: TWO decisions, reward after the second
    Pi->>Env: grasp at pixel p
    Env->>Sim: pick(p)
    Env-->>Pi: observation with the object lifted (held class, free cells), reward 0
    Pi->>Env: place at pixel q (place head, restricted to free cells)
    Env->>Sim: place(q)
    Env-->>Pi: reward, next observation
```

The pick is bootstrapped from the best place value without discount (one pick-and-place counts as one decision). Pushing is not a "fallback":
it exposes occluded objects, moves several objects at once and finishes objects that are one push from their region.

---

## Training

```mermaid
flowchart TB
    subgraph CUR["Curriculum"]
        direction LR
        K1["1 object"] --> K2["2"] --> K3["..."] --> K5["max_obj"]
    end
    CUR -->|object count| W["N parallel simulators<br/>(one process each)"]
    W -->|"observations"| NET["Q-network (batched forward)"]
    NET -->|"epsilon-greedy over masked Q-maps"| W
    W -->|"transition + hindsight-relabelled copy"| RB[("Prioritized replay")]
    DEMO[("Expert demonstrations")] -.-> TR
    RB --> TR["Double DQN: Huber loss on the taken pixel<br/>target net, Polyak averaging"]
    DEMO -.->|"25% of each batch + margin loss"| TR
    TR --> NET
    NET -. "every N episodes: 20 greedy episodes<br/>on fresh configurations" .-> VAL{"success >= 0.9?"}
    VAL -->|yes| CUR
```

* **Reward** (paper): -1 per action, 0 on the goal, -10 for an invalid state; variants below.
* **Exploration**: epsilon-greedy, `--eps_start` (default 0.8) annealed to `--eps_end` (default 0.01) **at every curriculum stage**,
  over the *masked* maps. For demonstration-initialised agents use `--eps_start 0.2 --eps_end 0.05`: restarting at 0.8 floods the
  replay with random actions exactly when a new stage begins, and that made the IQL agent collapse (see the results).
* **Targets**: one-step double DQN by default; `--n_step N` stores N-step returns (the grasp -> place step stays undiscounted),
  `--twin_q` (object agent) takes the minimum of two value heads.
* **Validation noise**: `--curriculum_window K --success_threshold 0.85` advances on the success pooled over the last K validations;
  `--fixed_validation_seeds` validates a stage on the same scenes every time.
* **Curriculum**: the object count grows when the recent success rate is high. The default rule uses the epsilon-greedy training window
  (as in the paper); `--curriculum_eval 20` instead advances only after a **greedy validation** on 20 fresh random configurations.
  `--stage_mix p` draws earlier stages with probability p to keep easy states in the buffer.
* **Hindsight relabelling** recolours both observations of the transition with the post-hoc class assignment and gives it the goal reward.
* Parallel simulation is the bottleneck (about 1 action/s per process); `--adaptive_step --joint_tolerance 3e-3` cuts the physics steps per
  action by ~2.4x with the same executed fraction (`scripts/speed_ablation.py`).

---

## Variants

Every variant is a flag; they combine. The baseline of the experiment runner is `--placement learned --footprint_placement`.

### Agent and architecture
| flag | variant | what it is |
|---|---|---|
| `--belief none\|spatial\|recurrent` | belief state | memoryless (paper) / a memory map of what was seen (last colour, height, age, last action; +7 input channels) / a ConvGRU latent updated from the encoded observation and action |
| `--placement heuristic\|learned` | placement | rule from the paper / a place head over free cells, conditioned on the held class |
| `--push_lengths 0.04 0.08 0.12` | learned push length | one Q-map per (direction, length) |
| `--hierarchy` | hierarchical agent | level 1 picks the primitive from pooled features; level 2 the pixel and, for pushes, a **continuous length** (DDPG-style actor on a length-conditioned critic) |
| `--backbone hybrid` | transformer | see [The agents and networks](#the-agents-and-networks) |
| `--end_effector suction\|gripper` | tool | suction cup (constraint model) / wide-stroke parallel-jaw gripper (friction, one Q channel per jaw yaw, `--jaw_yaws`) |

```mermaid
flowchart LR
    subgraph FLAT["Flat agent (default)"]
        F1["dense Q-maps:<br/>push, pick, place"] --> F2["argmax over all<br/>feasible actions"]
    end
    subgraph HIER["--hierarchy"]
        H1["level 1: value of each primitive<br/>(pooled features)"] --> H2["choose primitive"]
        H2 --> H3["level 2: pixel from that primitive's Q-map"]
        H3 --> H4["push only: continuous length<br/>(actor head)"]
    end
```

### Observation and perception
| flag | variant |
|---|---|
| `--perception gt` | masks from PyBullet's ground truth |
| `--perception network` | a small U-Net predicts the object **type**; the episode's type -> class specification gives the class (0.96 pixel IoU, top-view instance recall 0.97 on unseen scenes) |
| `--perception agnostic` | an **object-agnostic** U-Net (background / interior / touching-boundary, no type catalogue); instances are connected components of the interior, so it should transfer to unseen object types (`--object_types` holds types out) |

An object's class is part of the task (its target region), not of its appearance, which is why the type network needs the episode specification.
Train and evaluate: `python train_perception.py collect|train|evaluate [--task agnostic]`.

### Rewards, masks, placement
| flag | variant |
|---|---|
| `--reward_mode sparse` | the paper's reward |
| `--reward_mode progress\|distance` | potential-based shaping (policy-invariant) by sorted fraction / by distance to the target region |
| `--reward_mode noop` | sparse + a penalty for actions that moved nothing (changes the optimum) |
| `--reward_mode combined --reward_scale 3 --noop_penalty 0.5` | progress + distance potential, scaled, plus the no-op penalty: a stronger placement signal (+0.6 per sorted object at 5 objects instead of +0.2) |
| `--mask_level basic` | grasp on any object cell, push near objects (original) |
| `--mask_level structured` | grasp only just inside an object's **front face**, push only with an object in the **blade's own strip**: 703 -> 193 grasp cells per scene; a random push moves an object 88% of the time instead of 62% |
| `--footprint_placement` | keep the placement target clear of the back wall by the held object's depth |
| `--placement_space region` | additionally fit the object's whole footprint and offer only cells inside its own target region |

### Demonstrations, DAgger and RL
```mermaid
flowchart LR
    X["Scripted expert<br/>(privileged ground truth)"] --> D[("Demonstrations<br/>pick, place, push<br/>1-5 objects")]
    D --> PT["DQfD pre-training<br/>TD + large-margin loss"]
    PT --> RL["RL with the curriculum<br/>demo share 25% -> 0<br/>(--demo_anneal_steps)"]
    D --> RL
    D --> BC["behaviour cloning<br/>(arch_benchmark)"]
    BC --> DAG["DAgger rounds:<br/>learner drives,<br/>expert labels"]
    DAG -->|aggregate| BC
```
* **DQfD**: `--demos <file> --pretrain_steps 3000` pre-trains on the demonstrations, then 25% of each RL batch comes from them with the
  margin loss on the expert action. That loss pins the policy to the expert; `--demo_anneal_steps N` fades the share to 0 over N RL
  gradient steps so the demonstrations bootstrap the agent without capping it.
* **DAgger** (`scripts/dagger.py --variant <name>`): the learner drives the simulator, the expert labels every visited state, the labels
  are aggregated with the demonstrations and the learner is retrained; the held-out evaluation runs after behaviour cloning and after DAgger.
* Demonstrations recorded with a continuous length (`--hierarchy`) are mapped to the nearest selectable length for agents with discrete
  lengths; the recorded belief is stripped for agents without one (`demos.adapt_demos`).
The expert picks the front-most misplaced object (aiming a few millimetres *inside* it: the cup stops on contact and a front-most
aim often misses), places it by the footprint-aware heuristic, pushes objects that are one push from their region, and pushes a front
object aside when every misplaced object is hidden. A margin loss keeps the network a **Q**-network (plain behaviour-cloning logits
would not be Q-values); a generative policy (diffusion / ACT) is not used: a dense pixel map already represents several equally good picks as several peaks.
`python scripts/collect_demos.py ...` records them, `python train.py --demos data/demos/expert.pkl --pretrain_steps 3000` uses them.

### Simulator
| flag | variant |
|---|---|
| `--sim_quality legacy\|improved\|modern` | IK with joint limits, tared / filtered force contact detection, verified suction grasps; `modern` adds a rate-limited joint controller, more solver iterations, deterministic contacts |
| `--physics paper` | the paper's cabinet dimensions, weaker suction, lighter objects (unvalidated) |
| `--adaptive_step`, `--joint_tolerance`, `--camera_scale`, `--step_distance` | speed options (measured in `SIMULATION_NOTES.md`) |
| `--randomize` | domain randomisation (friction, mass, camera pose, sensor noise) |

---

## Tools and experiments

| tool | use |
|---|---|
| `scripts/run_experiments.py list\|run\|summarize` | named suites (`core`, `belief`, `perception`, `end_effector`, `physics`, `smoke`) x seeds -> `summary.md`; `--max_obj`, `--num_obj`, `--max_episodes`, `--extra <train flags>` |
| `evaluate.py` | greedy evaluation on fixed seeds per object count (success rate, mean actions, return) |
| `profile_sim.py` | where the simulation time goes, executed / attempted actions, force-limit stops |
| `analyze_rewards.py` | the reward signal and scene volatility under random actions |
| `scripts/speed_ablation.py` | physics steps per action and executed fraction for the speed options |
| `scripts/collect_demos.py`, `scripts/merge_demos.py` | record expert demonstrations (in parallel parts) and merge them |
| `scripts/arch_benchmark.py --variants ... --seeds 3 [--rollouts N]` | supervised benchmark of network variants on the demonstrations: held-out primitive / in-class / sensible / exact accuracy, optional closed-loop rollouts |
| `scripts/dagger.py --variant ... --objects 2 5 --eval_objects 4 5` | DAgger with the scripted expert; held-out success after behaviour cloning and after DAgger |
| `scripts/failure_analysis.py --snapshot ... \| --expert -- <run flags>` | greedy episodes on fresh seeds: outcome per episode (sorted / invalid by which primitive and which object / looping / hidden / ...), per-action effects, placement quality, predicted Q vs realised return |
| `scripts/primitive_profile.py`, `scripts/approach_profile.py`, `scripts/grasp_height_sweep.py`, `scripts/place_profile.py` | oracle tests of the primitives: headings, pulls, approach lines, contact height, release tilt / drift of placements |
| `scripts/physics_sanity.py --sim_quality ... --pusher ...` | rest stability, slide-or-tip under pushes vs the rigid-body prediction, contact forces, object speeds |
| `scripts/make_figures.py`, `scripts/make_result_figures.py` | the figures of this README (scene, episode, observation, Q-maps; learning curves, failures, overestimation, primitives) |
| `tests/sim_scenario.py record\|check` | deterministic scripted scenario on the real simulator (golden results for refactoring) |

Tests: `pytest -q` (the real-simulator regression runs with `L2R_GOLDEN=1`); lint: `ruff check .`.

---

## Status and results

All numbers are from simulation, on the `legacy` physics with the speed options unless stated otherwise. The first tables measure the
networks by imitation (supervised on expert demonstrations, then closed loop); the RL results follow.

**Scripted expert** (privileged): 93 / 87 / 87 / 73 % of scenes sorted with 2 / 3 / 4 / 5 objects (15 episodes each, about +-10
points); 0.81-0.87 over 1-5 objects in the recorded demonstration sets. It is the ceiling of everything trained by imitation.

**Architecture benchmark** (`scripts/arch_benchmark.py`): dense networks trained by cross-entropy over the masked action space on about
1,400 expert transitions, evaluated on 340 held-out ones; in-class = the chosen cell lies on the same object/equivalence class as the
expert's; 1,500 steps, 3 seeds, mean +- std.

| network | held-out in-class | exact | train in-class |
|---|---|---|---|
| plain hourglass (paper) | 0.35 +- 0.04 (0.70 at 4,500 steps) | 0.10 | 0.52 |
| U-Net skips | 0.75 +- 0.06 (0.83 at 4,500 steps) | 0.48 | |
| U-Net + PPM | 0.80 +- 0.00 | 0.55 | 0.97 |
| **U-Net + ConvNeXt encoder** | **0.83 +- 0.02** | **0.60** | 0.97 |
| U-Net + ResNet-18 ImageNet / scratch | 0.79 +- 0.01 / 0.55 +- 0.01 | 0.53 / 0.30 | 0.99 / 0.95 |
| U-Net + ConvNeXt + PPM / + ASPP | 0.82 +- 0.04 / 0.79 +- 0.03 | 0.58 / 0.55 | 0.97 |
| U-Net + transformer / ASPP / PPM+ASPP | 0.64 / 0.73 / 0.73 (+- 0.07-0.14) | | |
| FPN / attention U-Net / HRNet | 0.53 / 0.62 / 0.50 | | 0.87 |

The plain network learns slowly but is not capacity-limited; skips mainly buy sample efficiency; most variants split into good
(~0.8) and bad (~0.5) seeds at this budget, ConvNeXt and PPM do not; pretraining matters for ResNet-18, not for the rest.

**Closed loop** (`scripts/dagger.py`, 6 rounds x 20 episodes on 2-5 objects; success on 50 held-out scenes each, +-0.07):

| agent | BC only, 4 / 5 objects | after DAgger, 4 / 5 objects |
|---|---|---|
| dense U-Net skips (3 seeds) | 0.60 / 0.10 / 0.40 ; 0.46 / 0.10 / 0.30 | 0.79 ; 0.62 |
| dense U-Net + ConvNeXt (3 seeds) | 0.77 ; 0.58 | 0.73 ; 0.62 |
| object, heuristic placement | 0.74 ; 0.62 | 0.80 ; 0.62 |
| object + learned placement | 0.76 ; 0.66 | 0.76 ; 0.64 |
| object + continuous push length | 0.76 ; 0.60 | 0.72 ; 0.64 |
| object + both (learned) | 0.80 ; 0.70 | 0.72 ; 0.60 |

DAgger rescues unstable policies (skips: 0.10 -> 0.78 for the worst seed) but does not lift stable ones: every agent ends near the
expert's own success, which is what imitation can reach. The object agent matches the dense ConvNeXt network with about 1 % of its
parameters.

### RL with the curriculum (1 -> 10 objects, DQfD initialisation from 1-5 object demonstrations)

![learning curves](docs/figures/learning_curves.png)

![failure breakdown](docs/figures/failure_breakdown.png)
*50 greedy episodes on the same fresh 5-object scenes (`scripts/failure_analysis.py`): how episodes end, and how often the agent grasps
an object that is already in its region (the "re-grasp loop").*

What the runs and diagnostics showed (one seed per configuration; 50-episode diagnostics have about +-0.07 standard error):

* **Every agent reaches about the expert's level and stops there.** The scripted expert sorts 0.66 of 5-object scenes; the RL agents
  measure 0.58-0.70 on the same 5-object scenes, and the dense ConvNeXt agent reached 6 objects at about 0.68 validation success. No configuration has improved clearly
  beyond imitation yet.
* **The object agent's double DQN overestimates.** Its predicted Q of the chosen actions is 4.5-4.8 above the return that followed
  (the dense agent: 0.6). The optimism concentrates on grasps of objects already in their region (+6.4), and with a stronger shaped
  reward the agent loops on them: re-grasp, place, re-grasp.
* **3-step returns stop the loop** (H, I: 86-99 % of grasps on misplaced objects), clipped double Q alone does not change much.
  **IQL** starts strong (its offline phase is advantage-weighted cloning) but drifted into the same loop after the epsilon reset at
  5 objects; **with `--eps_start 0.2`** it no longer collapses: J' (IQL + twin) averages 0.67 over its last 8 validations at 6,600
  steps, where J without it fell to 0.25-0.40 by 3,000. J'' (+ 3-step) averages 0.66, but its latest validation dropped to 0.35, so
  its stability is still open.
* **Where episodes fail**: about a quarter end in an invalid scene, almost always because a **neighbour** is knocked over during a
  grasp or the **placed** object tips; the rest run out of actions (loops, pushes that move nothing).

![overestimation](docs/figures/overestimation.png)

### Physics and primitives

* **The physics is plausible** (`scripts/physics_sanity.py`): resting scenes do not drift (< 0.1 mm in 10 s), contact forces during
  pushes are tens of newtons, tall objects tip under high pushes about as the rigid-body condition predicts. The suction is
  idealised (a 1000 N constraint sealing up to 60 deg off the cup axis in `legacy`); objects do not slip from the cup (about 1 deg tilt
  and 1 mm drift at release in clean placements).
* **The tool tip was a weak pusher** (objects moved a third of the push); the paddle doubles the transfer.
* **Primitive fixes depend on the physics and on the scene density**: for the scripted expert (50 episodes per cell) the clear
  approach helps at 5 objects (0.66 -> 0.80) but not at 6 (0.74 -> 0.60) under `legacy`; clear approach + corridor placement helps
  under `modern` (0.66 -> 0.78). A 100-episode comparison under `modern` (straight / clear + corridor / + paddle, 5 and 6 objects) is
  running and will decide the configuration of the next demonstrations.

**Known limits:** the expert is pick-and-place first and pushes only in the situations above; above 5 objects scenes often become
invalid within a few actions (objects pushed or dropped out of place), which the RL curriculum will meet; PyBullet is slow (about 1 action/s
per process; a move to Isaac is kept open in [`ISAAC_MIGRATION.md`](ISAAC_MIGRATION.md)).

Snapshots saved by older versions of this code are not compatible with the current networks.
