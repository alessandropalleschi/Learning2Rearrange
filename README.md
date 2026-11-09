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
| `--randomize` | domain randomisation: object friction/mass, camera pose, sensor noise |
| `--gui` | PyBullet GUI for the first simulator |

Logs go to `logs/<logging_directory>/` (`episodes.csv`, TensorBoard in `tb/`, snapshots in `models/`):

```bash
tensorboard --logdir logs
```

Evaluate a snapshot on fixed seeds (success rate and mean number of actions per object count):

```bash
python evaluate.py --snapshot logs/train/models/snapshot-0010000.pth --objects 3 5 8 --episodes 20 --num_envs 4
```

Tests (no PyBullet needed, uses a fake simulator): `python tests/test_training_loop.py`

## Without a container
`pip install torch` (CUDA build if you have a GPU) and `pip install -r requirements.txt`.

## Code layout
| file | purpose |
|---|---|
| `config.py` | all hyper-parameters, plus `SimParams` (physics presets, domain randomisation) |
| `env.py` | PyBullet scene: UR5, cabinet, objects, suction, camera, heightmaps (physics constants at the top) |
| `rearrange_env.py` | Gym-style `reset()/step()` wrapper: action execution, reward, termination, relabelling |
| `models.py` | `QNet`: 4-channel input, GroupNorm, global-context block, shared backbone + push/pick heads |
| `trainer.py` | batched double DQN: prioritized replay, Huber loss on the taken pixel, clipped targets, mixed precision, Polyak target update |
| `policy.py` | epsilon-greedy action selection shared by training and evaluation |
| `vec_env.py` | parallel simulators in worker processes |
| `evaluate.py` | fixed-seed evaluation per object count |
| `logger.py` | episode CSV + TensorBoard |
| `train.py` | training / testing loop, curriculum 1 -> 8 objects, episode CSV log, snapshots |
| `tests/test_training_loop.py` | smoke test of the learning stack with a fake simulator (no PyBullet) |

Snapshots saved by older versions (two separate BatchNorm networks, 6 input channels) are not compatible.
