"""MuJoCo backend (`--backend mujoco`), scaffold.

Stage 1 (this package): `MujocoClient` answers the PyBullet calls of the existing simulator code from a MuJoCo model, so the
same `Arm`, `SuctionCup`, `Scene`, `Manipulator`, observation pipeline, expert and planner run on both engines and can be
compared seed by seed (same scenes: the random draws happen in the same order).

Stage 2 (next): native MuJoCo controllers where the facade is slow or unfaithful: Cartesian streaming instead of
IK-per-waypoint, a suction model with a force limit (adhesion actuator or a breakable weld), the paddle and the gripper as
bodies declared at build time, batched worlds (MJX / MuJoCo Warp) once the primitives are vectorised.

Not supported yet: `--pusher paddle`, `--end_effector gripper`, the GUI.
"""
import os

os.environ.setdefault('MUJOCO_GL', 'egl')  # headless GPU rendering unless the caller chose (e.g. MUJOCO_GL=osmesa without a GPU)
