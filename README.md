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
python train.py --is_testing --load_snapshot --snapshot_push ... --snapshot_pick ... --snapshot_push_target ... --snapshot_pick_target ...
```

Useful flags: `--num_obj` (start), `--max_obj`, `--max_actions`, `--gui`.

## Without a container
`pip install torch` (CUDA build if you have a GPU) and `pip install -r requirements.txt`.
