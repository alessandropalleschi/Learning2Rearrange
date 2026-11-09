# Ablations: what each flag does, and what was decided

Every ablation is opt-in and combines with the others (train and evaluate with the same model flags). None of them has been
trained to convergence here: the numbers below are measurements of the mechanics (speed, signals, learning of a toy problem), not of final success rates.

| flag | what it changes |
|---|---|
| `--backbone hybrid` | transformer encoder on the 1/8-resolution bottleneck of the conv network |
| `--hierarchy` | level 1 selects the primitive, level 2 its parameters, incl. a *continuous* push length (actor-critic) |
| `--reward_mode progress\|distance\|noop` | potential-based shaping / no-op penalty on top of the paper's sparse reward |
| `--belief spatial\|recurrent` | belief state from a memory map / a ConvGRU |
| `--placement learned` | a Q-head chooses where to place the held object |
| `--push_lengths 0.04 0.08 0.12` | discrete learned push length |
| `--perception network` | object masks from a segmentation network instead of PyBullet's ground truth |

## Transformer backbone: hybrid, not Swin
Decision: keep the convolutional encoder / decoder and add global self-attention on the bottleneck (`TransformerBottleneck`: 192-d, 3 pre-norm
layers, fixed 2D sine-cosine positions, zero-initialised output projection, so it starts as the conv network). The bottleneck grid of the
128 x 104 maps is 16 x 13 = 208 tokens, so full attention is cheap: +0.75 M parameters, +3 ms per forward pass (91 -> 94 ms on 4 CPU threads).
Why not Swin: (1) window attention (7 x 7) does not divide the 16 x 13 grid and restricts exactly the global reasoning (relations between
objects, targets, occlusions) a transformer is meant to add; (2) its 4x patch embedding at the input discards the fine detail the 5 mm Q-maps
need unless a conv decoder is added anyway, which is what the hybrid already is; (3) ImageNet weights do not transfer to these 4-channel top-down maps,
and without them a Swin is a large model to train from scratch with RL signals; (4) shifted-window masks and relative-position tables for no gain at this resolution.
A ViT-style model without the conv encoder would lose the same detail. Worth trying later if the hybrid helps: more layers / a conditioning token for the held object.

## Hierarchical agent
`--hierarchy` implements a parameterised-action agent (P-DQN style with a dense actor):
* **level 1** `prim_head` (pooled decoder features -> 3 values: push, pick, place) is regressed to the best Q of each primitive at the actor's
  parameters over its feasible pixels (only for primitives available in the state). At acting time it picks the primitive *before* looking at the
  pixel maxima; exploration picks a random feasible one.
* **level 2**: the pixel / direction from the dense Q-map of that primitive, and for a push the **continuous length** (`--push_length_range`, default
  3-15 cm): a per-pixel actor `length_head`; the push critic gets the length as an input channel. The critic is trained by TD at the *executed*
  length (actor + Gaussian noise ~ epsilon, or uniform when exploring); the actor follows the critic's gradient (DDPG), and only the actor's
  parameters receive it. The TD target uses the critic at the (target) actor's lengths.
* verified: on a bandit whose reward depends only on the push length (optimum at 0.8 of the range) the critic learns the reward curve and the actor
  moves to the optimum (`tests/test_hierarchy.py`); the level-1 head ranks primitives by the best Q; gradients reach only the intended parameters.
* cost: the training step runs both push directions through the network (about 2x the conv network's training step).
Not included: continuous *locations* (the 5 mm pixel grid is finer than the 17 mm cup), continuous direction (the arm only reaches +-y).

## Rewards
The paper's reward is deliberately sparse: -1 per action, 0 on the goal, -10 for an invalid state. Measured with random actions on the real simulator
(120 actions, 5 objects; `python analyze_rewards.py`):
| measurement | value |
|---|---|
| actions that change the number of correctly placed objects | 32.5% (+25.8% / -6.7%) |
| actions that move objects closer to / farther from their target region | 37.5% / 10.0% (mean +7 mm) |
| actions that move nothing | 20% (all of them *unexecuted* actions: failed grasps, aborted motions, infeasible pushes) |
| objects moved per action | 1.29; 40% of the actions move two or more |
| executed actions | only 30% of the random ones |
Variants (`--reward_mode`, `--reward_scale`):
* `progress`: potential-based shaping F = gamma phi(s') - phi(s) with phi = scale (sorted fraction - 1). Policy-invariant (proof by telescoping, unit-tested),
  gives credit for partial progress; the value range becomes [-100, scale] (handled by the target clipping).
* `distance`: the same with phi = -scale mean distance of the objects to their region / cabinet width: a denser signal for pushes, which rarely
  complete an object by themselves.
* `noop`: sparse + penalty (default 1) when no object moved by 5 mm. Since the measured no-ops are exactly the failed / unexecuted actions, this doubles
  the cost of a failed attempt: it may help the agent learn feasibility faster, but it also changes the optimal policy (not policy-invariant).
Default since the shaping was requested: `progress` (the paper's sparse reward is `--reward_mode sparse`). Ablation order: sparse (paper baseline) vs `progress` (safe, invariant); `distance` for the push-heavy settings; `noop` last. The sparse signal is
less sparse than the paper suggests (a third of the random actions change the count), so large gains from shaping are not expected; the 70% of
actions that are not executed are the bigger loss, and they are better attacked by feasibility masks (`--footprint_placement`) than by rewards.
Hindsight relabelling keeps the sparse goal reward for the relabelled transition (the shaping term of the relabelled world is not known).

## ACT-style sequence prediction: not added
Decision: no ACT ablation for now. ACT (action chunking with transformers) is an *imitation* method: it needs demonstrations, trains a CVAE to predict a chunk of
future actions and executes them open loop with temporal ensembling. Here:
* there are no demonstrations (the policy is learned with Q-learning); an ACT would need a scripted expert first;
* every primitive already is a multi-second action chunk (a whole push or pick-and-place trajectory), so the usual motivation of chunking (smooth
  control, fewer compounding errors at the joint level) is already served;
* a *sequence of primitives* planned from one observation goes stale: measured with random actions, **40% of the actions move two or more objects** and 1.3
  objects move per action on average, so the scene after the first primitive is rarely the one the plan assumed; the dense Q-map re-plans from the new observation at every step,
  which is the right behaviour here, and an open-loop chunk would give it up (executing only the first action of the chunk = receding horizon is then an
  auxiliary loss, not a policy);
* the problem is partially observable with a changing scene: the belief variants address memory more directly.
If sequence models are to be tried, the order I would take: (1) a scripted expert on the ground-truth state (sort the objects in an order that does not block
the others) and behaviour cloning with the existing dense network, as a demonstration baseline and as pre-training for Q-learning; (2) only if long-horizon
coupling remains the bottleneck, a chunk predictor over *primitives* (K = 3-4) trained on the expert data, executed with replanning after every action and
used as an auxiliary prediction task for the Q-network.

## Parallel-jaw gripper (`--end_effector gripper`)

A programmatic wide-stroke gripper (`l2r/sim/gripper.py`: palm + two prismatic pads, 20 cm opening, 80 N per finger) replaces the
suction cup. Same interface towards the primitives; the grasp is held by **friction only** (no weld constraint).

* **Action space.** Objects stand on the cabinet floor, so the grasp is a horizontal pinch approached from the front. The pick
  Q-map gets one channel per jaw yaw (`--jaw_yaws`, default 5, evenly over +-60 degrees): the pick action is `(yaw index, row, col)`,
  as a push is `(direction*length, row, col)`. A yaw also turns the approach direction and the UR5e reaches into the cabinet only
  within about +-60 degrees, and the jaws are symmetric under 180 degrees, so this range covers both ways of aligning with an
  object's axes (the oracle uses the better of the two).
* **Geometry facts measured on the scene.** Objects are up to ~17 cm across, so an 8.5 cm (Robotiq-sized) opening cannot grasp
  most of them (0/12 with it); the pads are centred on the object centre (not the fingertips) and at mid-height (pads on the rim
  of a can tip it); the jaws are 23 cm wide, so grasps with |y| > ~17 cm hit the side walls and fail.
* **Contact modelling.** PyBullet combines the coefficients of the two bodies multiplicatively: without torsional friction on the
  object the pinch is a pivot and the held can swings about the pad axis; the held object gets `spinningFriction` while grasped. A
  grasped object may hang tilted until placed, so `is_valid` skips it while it is held (it is checked after release).
* **Oracle result** (true centre, jaw yaw aligned with the object, front-most object with |y| < 17 cm, 12 scenes): 12/12 hold the
  object through the lift. This is an upper bound for the learned policy, not its performance.
* **Placement** uses `place_offsets(orientation)` from the held object's pose in the tool frame, as for the cup.
* Run: `python train.py --end_effector gripper --jaw_yaws 5` (combines with `--hierarchy`, `--backbone hybrid`, `--placement learned`).
