"""Central configuration: every tunable of the paper's method in one place."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    # --- Reward (paper Eq. 2) ---
    reward_step: float = -1.0       # action that does not reach a goal state
    reward_goal: float = 0.0        # action that solves the task
    reward_invalid: float = -10.0   # action that leads to an invalid state (object dropped, toppled)

    # --- Reward variants (ablation). 'sparse' is the paper's reward. 'progress' / 'distance' add potential-based shaping
    #     F = gamma * phi(s') - phi(s), which leaves the optimal policy unchanged: phi = scale * (sorted fraction - 1) or
    #     -scale * mean distance of the objects to their target region (relative to the cabinet width). 'noop' adds a
    #     penalty to actions that moved no object (>= 5 mm). ---
    reward_mode: str = 'sparse'  # 'sparse' | 'progress' | 'distance' | 'noop'
    reward_shaping_scale: float = 1.0
    noop_penalty: float = 1.0

    # --- Q-learning ---
    gamma: float = 0.9              # future discount
    learning_rate: float = 1e-4     # Adam
    weight_decay: float = 2e-5
    batch_size: int = 32
    buffer_size: int = 8000
    polyak_beta: float = 1e-3       # target network soft update rate
    grad_clip: float = 10.0
    clip_q: bool = True             # clamp TD targets to the reachable range [reward_invalid/(1-gamma), reward_goal]
    lr_decay: float = 1.0           # exponential LR decay per optimisation step (1.0 = fixed LR as in the paper)
    amp: bool = True                # mixed precision (only used on CUDA)

    # --- Backbone (ablation): 'conv' = residual conv encoder-decoder with pooled global context;
    #     'hybrid' = same, with a transformer encoder on the 1/8-resolution bottleneck ---
    backbone: str = 'conv'
    transformer_dim: int = 192
    transformer_layers: int = 3

    # --- Hierarchical agent (ablation): level 1 chooses the primitive from global features, level 2 its parameters:
    #     the pixel from the dense Q-map and, for pushes, a *continuous* length from an actor head (DDPG-style) ---
    hierarchy: bool = False
    push_length_range: tuple = (0.03, 0.15)  # m, continuous push length
    hier_prim_weight: float = 0.5            # weight of the level-1 (primitive value) loss
    hier_length_noise: float = 0.15          # exploration noise on the normalised length at epsilon = eps_start

    # --- End effector: 'suction' (the cup) or 'gripper' (parallel jaws: the pick Q-map gets one channel per jaw yaw) ---
    end_effector: str = 'suction'
    jaw_yaws: int = 5  # jaw orientations about the vertical, evenly spread over +-60 degrees: a yaw also turns the approach
    jaw_max_yaw_deg: float = 60.0  # direction, and the arm only reaches into the cabinet within about +-60 degrees

    # --- Placement (ablation): 'heuristic' = hand-written rule inside the pick-and-place action (paper);
    #     'learned' = pick and place are two decisions, the place pixel comes from a third Q-head
    #     (conditioned on the class of the held object, restricted to free cells) ---
    placement: str = 'heuristic'
    footprint_placement: bool = False  # keep the placement target clear of the back wall by the object's depth
    place_heuristic_mix: float = 0.5   # when exploring a placement: share of heuristic proposals (rest: uniform)

    # --- Belief state (ablation): 'none' | 'spatial' | 'recurrent' ---
    # spatial:   persistent per-episode memory map (last seen colour/height/id, age, last action) as extra input channels
    # recurrent: ConvGRU latent state at the network bottleneck, updated from the encoded observation and last action
    belief: str = 'none'
    belief_age_horizon: int = 10    # steps after which a remembered cell counts as fully stale
    recurrent_hidden: int = 64      # channels of the ConvGRU hidden state

    # --- Prioritized experience replay (proportional) ---
    prioritized: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4     # annealed to 1.0 over per_beta_iters optimisation steps
    per_beta_iters: int = 50000
    per_eps: float = 1e-3

    # --- Exploration (epsilon-greedy, annealed per curriculum stage) ---
    eps_start: float = 0.8
    eps_end: float = 0.01
    eps_decay: float = 0.9995

    # --- Curriculum ---
    start_objects: int = 1
    max_objects: int = 8
    success_window: int = 50        # episodes
    success_threshold: float = 0.9
    max_actions: int = 40           # actions per episode

    # --- Action primitives ---
    push_length: float = 0.08       # m (paper: 8 cm); used when push_lengths is left at its default
    push_lengths: tuple = (0.08,)   # learned push: the policy chooses among these lengths (m), e.g. (0.04, 0.08, 0.12)
    push_height: float = 0.05       # m above the cabinet floor (paper: 5 cm)

    # --- Input representation ---
    offset_px: int = 4              # border (px) cropped from the Q-maps
    pixel_size: float = 0.005       # m / px of the heightmap

    @classmethod
    def from_args(cls, args):
        """Build the configuration from the command-line flags of train / evaluate / profile (missing flags: defaults)."""
        get = lambda name, default: getattr(args, name, default)
        defaults = cls()
        return cls(start_objects=get('num_obj', defaults.start_objects), max_objects=get('max_obj', defaults.max_objects),
                   max_actions=get('max_actions', defaults.max_actions), batch_size=get('batch_size', defaults.batch_size), belief=get('belief', 'none'),
                   placement=get('placement', 'heuristic'), eps_decay=get('eps_decay', defaults.eps_decay),
                   push_lengths=tuple(get('push_lengths', defaults.push_lengths)),
                   footprint_placement=get('footprint_placement', False), backbone=get('backbone', 'conv'),
                   end_effector=get('end_effector', 'suction'), jaw_yaws=get('jaw_yaws', defaults.jaw_yaws),
                   hierarchy=get('hierarchy', False), reward_mode=get('reward_mode', 'sparse'),
                   reward_shaping_scale=get('reward_scale', defaults.reward_shaping_scale),
                   noop_penalty=get('noop_penalty', defaults.noop_penalty), push_length_range=tuple(get('push_length_range', defaults.push_length_range)))

    @staticmethod
    def add_arguments(parser):
        """Flags that define the model / agent (must be identical for training and evaluation)."""
        parser.add_argument('--batch_size', type=int, default=32, help='replay batch size')
        parser.add_argument('--push_lengths', type=float, nargs='+', default=[0.08],
                            help='selectable push lengths in m; one value = fixed (paper: 0.08), several = learned')
        parser.add_argument('--placement', choices=['heuristic', 'learned'], default='heuristic',
                            help="'learned': a Q-head chooses where to place the held object (default: rule from the paper)")
        parser.add_argument('--belief', choices=['none', 'spatial', 'recurrent'], default='none',
                            help='belief state ablation: memoryless / spatial memory map / ConvGRU latent')
        parser.add_argument('--backbone', choices=['conv', 'hybrid'], default='conv',
                            help="'hybrid': transformer encoder on the 1/8-resolution bottleneck of the conv network")
        parser.add_argument('--jaw_yaws', type=int, default=5,
                            help='jaw orientations of the gripper over +-60 degrees (--end_effector gripper)')
        parser.add_argument('--hierarchy', action='store_true',
                            help='hierarchical agent: a primitive-value head selects the primitive, then a continuous push '
                                 'length (actor-critic) and the pixel are chosen')
        parser.add_argument('--push_length_range', type=float, nargs=2, default=[0.03, 0.15], metavar=('MIN', 'MAX'),
                            help='continuous push length range in m (with --hierarchy)')
        parser.add_argument('--reward_mode', choices=['sparse', 'progress', 'distance', 'noop'], default='sparse',
                            help="reward: the paper's sparse one, potential-based shaping by sorted fraction ('progress') or by "
                                 "distance to the target region ('distance'), or sparse + a penalty for actions that move nothing")
        parser.add_argument('--reward_scale', type=float, default=1.0, help='scale of the shaping potential')
        parser.add_argument('--noop_penalty', type=float, default=1.0, help="penalty of 'noop' actions (reward mode 'noop')")
        parser.add_argument('--footprint_placement', action='store_true',
                            help="keep placement targets clear of the back wall by the held object's depth")

    def jaw_angle(self, index):
        """Yaw (rad) of the jaws for a pick action index (evenly spread over +-jaw_max_yaw_deg)."""
        if self.jaw_yaws == 1:
            return 0.0
        return float(np.deg2rad(-self.jaw_max_yaw_deg + 2 * self.jaw_max_yaw_deg * index / (self.jaw_yaws - 1)))

    @property
    def pick_channels(self):
        """Pick Q-map channels: one per jaw yaw for the gripper, one for the suction cup."""
        return self.jaw_yaws if self.end_effector == 'gripper' else 1

    @property
    def q_range(self):
        """(min, max) of the action values: rewards are bounded, and potential-based shaping shifts Q by -phi(s) <= scale."""
        shaped = self.reward_mode in ('progress', 'distance')
        low = (self.reward_invalid - (self.noop_penalty if self.reward_mode == 'noop' else 0.0)) / (1.0 - self.gamma)
        return low, self.reward_goal + (self.reward_shaping_scale if shaped else 0.0)

    def __post_init__(self):
        if self.hierarchy:  # the length is continuous: the feasibility mask uses the longest push
            self.push_lengths = (self.push_length_range[1],)

    @property
    def push_lengths_px(self) -> tuple:
        return tuple(int(round(length / self.pixel_size)) for length in self.push_lengths)

    @property
    def push_actions(self) -> int:
        """Number of push Q-map channels: two directions x the selectable lengths (index = direction * L + length)."""
        return 2 * len(self.push_lengths)


@dataclass
class SimParams:
    """Physics / scene parameters of the PyBullet simulation.

    'legacy'  reproduces the original code (0.5 x 0.35 m cabinet, very strong suction, heavy objects).
    'paper'   uses the cabinet of the paper (40 x 60 x 27 cm) and more physical suction force and masses.
              NOTE: not validated against the real robot; check the camera framing when you first use it.
    """
    name: str = 'legacy'
    cabinet_depth: float = 0.5      # m, x extent of the workspace
    cabinet_height: float = 0.35    # m, z extent of the workspace
    suction_max_force: float = 1000.0   # N, max force of the constraint simulating the suction cup
    suction_erp: float = 0.1
    cabinet_friction: float = 0.8
    block_mass: float = 1.0         # kg
    cylinder_mass: float = 0.7      # kg

    # motion / speed (defaults reproduce the original behaviour; larger step / smaller budget = faster, coarser)
    step_distance: float = 0.002            # m between IK waypoints of straight-line moves
    move_joints_max_steps: int = 1000       # simulation-step budget of one joint move

    # simulation quality options ('improved' preset). All default to the original behaviour.
    controller: str = 'legacy'              # 'legacy' | 'pd': rate-limited joint commands (smooth, velocity-limited)
    velocity_scale: float = 1.0             # 'pd': fraction of the URDF joint velocity limits used
    pd_motor_forces: str = 'default'        # 'pd': 'default' = PyBullet default motor force, 'urdf' = URDF effort limits
    solver_iterations: object = None        # PyBullet numSolverIterations (default 50); more = stiffer contacts, slower
    deterministic: bool = False             # deterministic overlapping pairs: identical runs for identical seeds
    strict_place: bool = False              # place(): a failed approach (after one retry) fails the action
    contact_force_limit: float = 300.0      # N: a motion is aborted when the joint reaction force exceeds this
    force_sense_x: object = None            # m: only check forces where the tool x < this (None = legacy: 0.5, or 0.2 holding)
    adaptive_step: bool = False             # coarse IK waypoints (5x) while the tool is outside the cabinet front
    ik_iterations: object = None            # IK solver iterations (None = PyBullet default; 100 made IK ~7x slower)
    camera_scale: float = 1.0               # scale of the rendered image (0.5 = 4x fewer pixels; check the heightmaps)
    ik_limits: bool = False                 # IK with joint limits, rest pose and tighter solver tolerances
    filtered_force: bool = False            # contact stop from tared, median-filtered, debounced F/T readings
    force_window: int = 5                   # samples of the median filter
    force_debounce: int = 2                 # consecutive samples above the limit before a contact is reported
    suction_min_alignment: float = 0.5      # min cos(angle) between cup axis and surface normal to seal (0.5 = 60 deg)
    suction_verify: bool = False            # after lifting, check that the object is still held at its grasp offset
    suction_hold_tolerance: float = 0.04    # m the object may drift from the grasp offset before the grasp is lost

    # perception: where the object masks come from
    perception: str = 'gt'                  # 'gt' = PyBullet ground truth, 'network' = trained segmentation network
    perception_weights: str = 'assets/perception/segnet.pth'

    # end effector: the suction cup (default) or a parallel-jaw gripper (`--end_effector gripper`)
    end_effector: str = 'suction'

    # domain randomisation (applied at every reset)
    randomize: bool = False
    friction_range: tuple = (0.3, 0.8)      # lateral friction of the objects
    mass_scale_range: tuple = (0.7, 1.3)    # multiplicative object mass noise
    cam_pos_jitter: float = 0.01            # m
    cam_rot_jitter_deg: float = 1.0
    sensor_noise: bool = False              # RGB-D noise of the simulated camera
    depth_noise_std: float = 0.0015         # m; the heightmap treats cells above 8 mm as objects, 3 mm noise (2.7 sigma)
                                            # produced ~0.4% false object cells

    @staticmethod
    def preset(name, randomize=False):
        if name == 'paper':
            return SimParams(name='paper', cabinet_depth=0.40, cabinet_height=0.27, suction_max_force=50.0,
                             block_mass=0.3, cylinder_mass=0.25, randomize=randomize, sensor_noise=randomize)
        return SimParams(randomize=randomize, sensor_noise=randomize)

    @classmethod
    def from_args(cls, args):
        """Build the parameters from the command-line flags shared by train / evaluate / profile."""
        params = cls.preset(args.physics, randomize=getattr(args, 'randomize', False))
        params.step_distance, params.move_joints_max_steps = args.step_distance, args.move_budget
        params.camera_scale = getattr(args, 'camera_scale', 1.0)
        params.contact_force_limit = getattr(args, 'force_limit', 300.0)
        params.force_sense_x = getattr(args, 'force_sense_x', None)
        params.adaptive_step = getattr(args, 'adaptive_step', False)
        quality = getattr(args, 'sim_quality', 'legacy')
        if quality in ('improved', 'modern'):
            params.ik_limits = params.filtered_force = params.suction_verify = True
            params.suction_min_alignment = 0.8
        if quality == 'modern':  # + rate-limited controller, stiffer solver, reproducible contacts, strict place errors
            params.controller, params.solver_iterations = 'pd', 100
            params.deterministic = params.strict_place = params.adaptive_step = True
            params.force_sense_x = 0.27  # do not check forces while the tool is still outside the cabinet
        params.velocity_scale = getattr(args, 'velocity_scale', 1.0)
        params.perception = getattr(args, 'perception', 'gt')
        params.end_effector = getattr(args, 'end_effector', 'suction')
        params.perception_weights = getattr(args, 'perception_weights', params.perception_weights)
        return params

    @staticmethod
    def add_arguments(parser):
        """Command-line flags that configure the simulation."""
        parser.add_argument('--physics', choices=['legacy', 'paper'], default='legacy',
                            help="'paper': 40x60x27 cm cabinet, weaker suction, lighter objects (unvalidated)")
        parser.add_argument('--sim_quality', choices=['legacy', 'improved', 'modern'], default='legacy',
                            help="'improved': IK with joint limits, filtered force contact detection, verified suction "
                                 "grasps. 'modern': improved + rate-limited joint controller, 100 solver iterations, "
                                 "deterministic contacts, strict place errors, adaptive waypoints, force check only near the "
                                 "cabinet (all opt-in, not yet validated against the legacy behaviour)")
        parser.add_argument('--perception', choices=['gt', 'network'], default='gt',
                            help="object masks: PyBullet ground truth, or a trained segmentation network (train_perception.py)")
        parser.add_argument('--perception_weights', default='assets/perception/segnet.pth', help='weights of the network')
        parser.add_argument('--end_effector', choices=['suction', 'gripper'], default='suction',
                            help='suction cup, or a parallel-jaw gripper (the pick Q-map gets one channel per jaw yaw, --jaw_yaws)')
        parser.add_argument('--velocity_scale', type=float, default=1.0,
                            help="'modern' controller: fraction of the URDF joint velocity limits")
        parser.add_argument('--force_limit', type=float, default=300.0,
                            help='N: abort a motion when the joint reaction force exceeds this (legacy 300; ~50%% of the '
                                 'actions hit it, see profile_sim.py)')
        parser.add_argument('--force_sense_x', type=float, default=None,
                            help='m: only check the force near/inside the cabinet (x below this); the cabinet front is 0.25')
        parser.add_argument('--adaptive_step', action='store_true',
                            help='5x coarser IK waypoints while the tool is outside the cabinet (faster, same contact physics)')
        parser.add_argument('--camera_scale', type=float, default=1.0,
                            help='scale of the simulated camera image; <1 renders faster but gives sparser heightmaps')
        parser.add_argument('--step_distance', type=float, default=0.002,
                            help='m between IK waypoints of straight moves (larger = faster, coarser; try 0.004)')
        parser.add_argument('--move_budget', type=int, default=1000, help='simulation steps allowed per joint move')
