"""Manual check of the simulator: spawn objects, execute the three primitives, show a Q-map.

    python demo_sim.py                # needs a display (PyBullet GUI + matplotlib window)
"""
import matplotlib.pyplot as plt
import numpy as np

from l2r.envs.rearrange import RearrangeEnv
from l2r.learning.trainer import Trainer


def main():
    env = RearrangeEnv(gui=True, num_obj=5)
    obs = env.reset()
    sim = env.sim
    sim.pick([0.0, 0.0])
    sim.place([0.0, 0.0])
    sim.push([0.0, 0.0], np.pi / 2)
    push_q, pick_q = Trainer().forward(obs['colormap'], obs['depthmap'])
    print(push_q.shape, pick_q.shape)
    plt.imshow(push_q[0])
    plt.show()


if __name__ == '__main__':
    main()
