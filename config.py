"""Central configuration: every tunable of the paper's method in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- Reward (paper Eq. 2) ---
    reward_step: float = -1.0       # action that does not reach a goal state
    reward_goal: float = 0.0        # action that solves the task
    reward_invalid: float = -10.0   # action that leads to an invalid state (object dropped, toppled)

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

    # --- Placement (ablation): 'heuristic' = hand-written rule inside the pick-and-place action (paper);
    #     'learned' = pick and place are two decisions, the place pixel comes from a third Q-head
    #     (conditioned on the class of the held object, restricted to free cells) ---
    placement: str = 'heuristic'
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
    push_length: float = 0.08       # m (paper: 8 cm)
    push_height: float = 0.05       # m above the cabinet floor (paper: 5 cm)

    # --- Input representation ---
    max_dim: int = 160              # padded square size fed to the network
    offset_px: int = 4              # border (px) cropped from the Q-maps
    pixel_size: float = 0.005       # m / px of the heightmap

    @property
    def push_length_px(self) -> int:
        return int(round(self.push_length / self.pixel_size))


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

    # domain randomisation (applied at every reset)
    randomize: bool = False
    friction_range: tuple = (0.3, 0.8)      # lateral friction of the objects
    mass_scale_range: tuple = (0.7, 1.3)    # multiplicative object mass noise
    cam_pos_jitter: float = 0.01            # m
    cam_rot_jitter_deg: float = 1.0
    sensor_noise: bool = False              # RGB-D noise of the simulated camera

    @staticmethod
    def preset(name, randomize=False):
        if name == 'paper':
            return SimParams(name='paper', cabinet_depth=0.40, cabinet_height=0.27, suction_max_force=50.0,
                             block_mass=0.3, cylinder_mass=0.25, randomize=randomize, sensor_noise=randomize)
        return SimParams(randomize=randomize, sensor_noise=randomize)
