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

Useful flags: `--num_obj` (start), `--max_obj`, `--max_actions`, `--gui`.

## Without a container
`pip install torch` (CUDA build if you have a GPU) and `pip install -r requirements.txt`.

## Code layout
| file | purpose |
|---|---|
| `config.py` | all hyper-parameters (reward, epsilon, curriculum, push length, ...) |
| `env.py` | PyBullet scene: UR5, cabinet, objects, suction, camera, heightmaps (physics constants at the top) |
| `rearrange_env.py` | Gym-style `reset()/step()` wrapper: action execution, reward, termination, relabelling |
| `models.py` | `QNet`: 4-channel input, GroupNorm, global-context block, shared backbone + push/pick heads |
| `trainer.py` | batched double DQN (Huber loss on the taken pixel, Polyak target update) |
| `train.py` | training / testing loop, curriculum 1 -> 8 objects, episode CSV log, snapshots |
| `tests/test_training_loop.py` | smoke test of the learning stack with a fake simulator (no PyBullet) |

Snapshots saved by older versions (two separate BatchNorm networks, 6 input channels) are not compatible.
