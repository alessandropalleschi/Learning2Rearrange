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
