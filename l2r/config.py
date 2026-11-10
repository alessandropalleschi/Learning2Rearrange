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
    reward_mode: str = 'progress'  # 'sparse' (paper) | 'progress' (default: potential-based shaping) | 'distance' | 'noop' |
                                   # 'combined' (progress + distance potential, plus the no-op penalty)
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

    # --- Agent (architecture comparison): 'dense' = pixel Q-maps (default); 'object' = object-centric Q (one token per visible object,
    #     Q per object-action slot); 'policy_critic' = dense Q critic + a policy heatmap trained on demonstrations that proposes cells ---
    agent: str = 'dense'
    policy_topk: int = 10  # policy_critic: cells proposed per primitive
    policy_bc_weight: float = 1.0  # policy_critic: weight of the behaviour-cloning loss on demonstration samples
    object_dim: int = 64
    object_layers: int = 2

    # --- Network architecture (ablation): U-Net skip connections; explicit task-geometry input channels (target-region bands per
    #     class + row / column coordinates) ---
    arch_skips: bool = False
    arch_stride4: bool = False  # bottleneck at 1/4 resolution (U-Net decoders): 4x more cells at the context / belief level
    arch_encoder: str = 'resnet'  # 'resnet' | 'convnext' (blocks of the encoder stages)
    arch_decoder: str = 'plain'  # 'plain' | 'unet' | 'fpn' | 'attn_unet' | 'unet_attn4'  (arch_skips is the same as 'unet')
    equiv_sigma: float = 0.0  # 0: equivalent cells all weigh 1 (hard); s > 0: weight exp(-d^2 / 2 s^2) by distance (cells) to the taken cell
    equiv_targets: bool = False  # TD target applied to all cells equivalent to the taken one (an object's front band, a 5x5 neighbourhood)
    region_channels: bool = False

    # --- Action masks on the dense Q-maps: 'basic' = grasp on any object cell, push near objects (original);
    #     'structured' = grasp only at the front face of an object, push only where the blade has something in its path ---
    mask_level: str = 'basic'
    grasp_edge_margin: float = 0.15  # structured masks / equivalence bands: skip this fraction of an object's width at each edge

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

    # --- Demonstrations (DQfD): a large-margin loss keeps the expert action on top, TD keeps the values meaningful ---
    demo_margin: float = 0.8          # margin l(aE, a) of the expert action over every other feasible action
    demo_margin_weight: float = 1.0
    demo_margin_radius: int = 0       # 0: the margin applies to every other pixel; r > 0: it grows linearly from 0 at the expert pixel to
                                      # the full margin at Chebyshev distance r in the expert's own map (neighbouring cells of one
                                      # object are equally good, a smooth network can not separate them)
    demo_fraction: float = 0.25       # share of each batch drawn from the demonstrations (when loaded)
    object_algo: str = 'dqn'          # object agent: 'dqn' (double DQN) | 'iql' (implicit Q-learning, advantage-weighted policy)
    iql_expectile: float = 0.8        # IQL: expectile of the value regression (0.5 = mean, -> 1 = max over the data's actions)
    iql_beta: float = 3.0             # IQL: inverse temperature of the advantage weights
    n_step: int = 1                   # > 1: n-step returns in the replay (see l2r/learning/nstep.py)
    twin_q: bool = False              # object agent: two value heads, TD targets from their minimum (TD3-style clipped double Q)
    outcome_weight: float = 0.0       # > 0: object agent outcome heads (executed / invalid / changed) trained with this weight
    outcome_penalty: float = 0.0      # > 0: action selection subtracts this x predicted P(invalid) x |reward_invalid| from Q
    demo_anneal_steps: int = 0        # > 0: the demonstration share (and with it the margin loss) fades linearly to 0 over this many
                                      #      gradient steps of RL, so the demonstrations bootstrap the agent without capping it

    # --- Placement (ablation): 'heuristic' = hand-written rule inside the pick-and-place action (paper);
    #     'learned' = pick and place are two decisions, the place pixel comes from a third Q-head
    #     (conditioned on the class of the held object, restricted to free cells) ---
    placement: str = 'heuristic'
    grasp_approach: str = 'straight'  # 'straight' (original) | 'clear': the env picks the approach line with the fewest obstacles
    placement_space: str = 'depth'  # with footprint_placement: 'depth' = keep the back wall clear (original); 'region' = fit the whole
                                     # footprint and offer only cells inside the object's own target region
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
                   success_threshold=get('success_threshold', defaults.success_threshold),
                   max_actions=get('max_actions', defaults.max_actions), batch_size=get('batch_size', defaults.batch_size), belief=get('belief', 'none'),
                   placement=get('placement', 'heuristic'), eps_decay=get('eps_decay', defaults.eps_decay),
                   push_lengths=tuple(get('push_lengths', defaults.push_lengths)),
                   footprint_placement=get('footprint_placement', False), placement_space=get('placement_space', 'depth'), grasp_approach=get('grasp_approach', 'straight'), backbone=get('backbone', 'conv'),
                   end_effector=get('end_effector', 'suction'), jaw_yaws=get('jaw_yaws', defaults.jaw_yaws),
                   hierarchy=get('hierarchy', False), agent=get('agent', 'dense'), arch_decoder=get('arch_decoder', 'plain'), arch_encoder=get('arch_encoder', 'resnet'), policy_topk=get('policy_topk', 10), arch_skips=get('arch_skips', False), arch_stride4=get('arch_stride4', False),
                   region_channels=get('region_channels', False), equiv_targets=get('equiv_targets', False),
                   equiv_sigma=get('equiv_sigma', 0.0), mask_level=get('mask_level', 'basic'),
                   grasp_edge_margin=get('grasp_edge_margin', 0.15), demo_fraction=get('demo_fraction', defaults.demo_fraction), demo_anneal_steps=get('demo_anneal_steps', defaults.demo_anneal_steps),
                   twin_q=get('twin_q', False), n_step=get('n_step', 1), object_algo=get('object_algo', 'dqn'),
                   iql_expectile=get('iql_expectile', 0.8), iql_beta=get('iql_beta', 3.0), outcome_weight=get('outcome_weight', 0.0), outcome_penalty=get('outcome_penalty', 0.0),
                   eps_end=get('eps_end', defaults.eps_end), reward_mode=get('reward_mode', defaults.reward_mode),
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
        parser.add_argument('--backbone', choices=['conv', 'hybrid', 'ppm', 'aspp', 'ppm_aspp'], default='conv',
                            help="'hybrid': transformer encoder on the 1/8-resolution bottleneck of the conv network")
        parser.add_argument('--jaw_yaws', type=int, default=5,
                            help='jaw orientations of the gripper over +-60 degrees (--end_effector gripper)')
        parser.add_argument('--hierarchy', action='store_true',
                            help='hierarchical agent: a primitive-value head selects the primitive, then a continuous push '
                                 'length (actor-critic) and the pixel are chosen')
        parser.add_argument('--push_length_range', type=float, nargs=2, default=[0.03, 0.15], metavar=('MIN', 'MAX'),
                            help='continuous push length range in m (with --hierarchy)')
        parser.add_argument('--reward_mode', choices=['sparse', 'progress', 'distance', 'noop', 'combined'], default='progress',
                            help="reward: the paper's sparse one, potential-based shaping by sorted fraction ('progress') or by "
                                 "distance to the target region ('distance'), or sparse + a penalty for actions that move nothing")
        parser.add_argument('--reward_scale', type=float, default=1.0, help='scale of the shaping potential')
        parser.add_argument('--noop_penalty', type=float, default=1.0, help="penalty of 'noop' actions (reward mode 'noop')")
        parser.add_argument('--agent', choices=['dense', 'object', 'policy_critic'], default='dense',
                            help="'object': object-centric Q (transformer over visible objects, Q per object-action slot; heuristic "
                                 "placement); 'policy_critic': dense critic + demonstration-trained policy heatmap proposing cells")
        parser.add_argument('--policy_topk', type=int, default=10, help='policy_critic: cells the policy proposes per primitive')
        parser.add_argument('--arch_encoder', choices=['resnet', 'convnext', 'resnet18_imagenet', 'resnet18_scratch'], default='resnet',
                            help='encoder: residual 3x3 blocks, ConvNeXt 7x7 blocks, or torchvision ResNet-18 stages (ImageNet-pretrained, '
                                 'or the same structure from scratch); the ResNet-18 encoders need a U-Net style decoder')
        parser.add_argument('--arch_decoder', choices=['plain', 'unet', 'fpn', 'attn_unet', 'unet_attn4', 'hrnet'], default='plain',
                            help='decoder: plain (original), unet skips, fpn pyramid, attn_unet (attention-gated skips), unet_attn4 '
                                 '(plus global attention at 1/4 resolution)')
        parser.add_argument('--arch_stride4', action='store_true', help='bottleneck at 1/4 resolution instead of 1/8 (needs a U-Net decoder)')
        parser.add_argument('--arch_skips', action='store_true', help='U-Net skip connections from the encoder to the decoder')
        parser.add_argument('--equiv_targets', action='store_true',
                            help='regress the TD target at every cell equivalent to the taken action (the whole front band of the '
                                 'grasped object; a 5x5 neighbourhood for pushes / places) instead of one pixel')
        parser.add_argument('--equiv_sigma', type=float, default=0.0,
                            help='with --equiv_targets: Gaussian weight (cells) around the taken cell instead of weight 1 on all '
                                 'equivalent cells (0 = hard)')
        parser.add_argument('--region_channels', action='store_true',
                            help='input channels with the target region of each class and row / column coordinates (needs observations '
                                 'that carry the regions: demonstrations must be re-recorded)')
        parser.add_argument('--grasp_edge_margin', type=float, default=0.15,
                            help="structured masks and equivalence bands skip this fraction of an object's width at each edge "
                                 '(suction on an edge fails: scripts/grasp_profile.py); 0 = whole front band')
        parser.add_argument('--mask_level', choices=['basic', 'structured'], default='basic',
                            help="'structured': mask the Q-maps to grasps at an object's front face and pushes with something in the "
                                 "blade's path")
        parser.add_argument('--demo_fraction', type=float, default=0.25, help='share of each batch from the demonstrations')
        parser.add_argument('--object_algo', choices=['dqn', 'iql'], default='dqn', help='object agent learning algorithm')
        parser.add_argument('--iql_expectile', type=float, default=0.8)
        parser.add_argument('--iql_beta', type=float, default=3.0)
        parser.add_argument('--n_step', type=int, default=1, help='n-step returns in the replay (1 = one-step TD, as before)')
        parser.add_argument('--twin_q', action='store_true', default=False,
                            help='object agent: clipped double Q (two value heads, targets from their minimum)')
        parser.add_argument('--outcome_weight', type=float, default=0.0,
                            help='object agent: weight of the outcome heads (executed / invalid / changed, free labels; 0 = off)')
        parser.add_argument('--outcome_penalty', type=float, default=0.0,
                            help='object agent: subtract this x P(invalid) x |reward_invalid| from Q when acting')
        parser.add_argument('--demo_anneal_steps', type=int, default=0,
                            help='fade the demonstration share linearly to 0 over this many RL gradient steps (0 = constant)')
        parser.add_argument('--eps_end', type=float, default=0.01, help='floor of the epsilon-greedy exploration')
        parser.add_argument('--grasp_approach', choices=['straight', 'clear'], default='straight',
                            help="grasp approach line: straight in (original) or the clearest of several angles in the height map")
        parser.add_argument('--placement_space', choices=['depth', 'region'], default='depth',
                            help="with --footprint_placement: 'region' fits the held object's whole footprint and offers only cells "
                                 "inside its target region (see SIMULATION_NOTES.md)")
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
        shift = {'progress': 1.0, 'distance': 1.0, 'combined': 2.0}.get(self.reward_mode, 0.0) * self.reward_shaping_scale
        noop = self.noop_penalty if self.reward_mode in ('noop', 'combined') else 0.0
        low = (self.reward_invalid - noop) / (1.0 - self.gamma)
        return low, self.reward_goal + shift

    @property
    def decoder(self):
        """The decoder kind: arch_decoder, or 'unet' if only arch_skips is set."""
        return 'unet' if (self.arch_skips and self.arch_decoder == 'plain') else self.arch_decoder

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
    joint_tolerance: float = 1e-3           # rad: a joint move is done when every joint is this close to its goal
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
    object_types: tuple = None              # YCB types the episodes draw from (>= 3; default: the whole catalogue)

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
        params.joint_tolerance = getattr(args, 'joint_tolerance', params.joint_tolerance)
        params.perception = getattr(args, 'perception', 'gt')
        params.end_effector = getattr(args, 'end_effector', 'suction')
        params.perception_weights = getattr(args, 'perception_weights', params.perception_weights)
        params.object_types = tuple(getattr(args, 'object_types', None) or ()) or None
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
        parser.add_argument('--perception', choices=['gt', 'network', 'agnostic'], default='gt',
                            help="object masks: PyBullet ground truth, or a trained segmentation network (train_perception.py)")
        parser.add_argument('--object_types', nargs='+', default=None, help='YCB types of the episodes (>= 3), e.g. to hold one out')
        parser.add_argument('--perception_weights', default='assets/perception/segnet.pth', help='weights of the network')
        parser.add_argument('--end_effector', choices=['suction', 'gripper'], default='suction',
                            help='suction cup, or a parallel-jaw gripper (the pick Q-map gets one channel per jaw yaw, --jaw_yaws)')
        parser.add_argument('--joint_tolerance', type=float, default=1e-3,
                            help='rad: a joint move is finished when all joints are this close (larger = fewer physics steps)')
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
