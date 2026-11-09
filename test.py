"""Manual check of the simulator: spawn objects, execute the three primitives, show a Q-map.

    xvfb-run -a python test.py        # inside the dev container
"""
import numpy as np
import matplotlib.pyplot as plt
from rearrange_env import RearrangeEnv
from trainer import Trainer

env = RearrangeEnv(gui=True, num_obj=5)
obs = env.reset()
trainer = Trainer()
sim = env.sim
sim.pick([0, 0, 0.5])
sim.place([0, 0, 0.5])
sim.push([0, 0, 0.5], np.pi / 2)

push_q, pick_q = trainer.forward(obs["colormap"], obs["depthmap"])
print(push_q.shape, pick_q.shape)
plt.imshow(push_q[0])
plt.show()
