from env import *
import matplotlib.pyplot as plt
from trainer import *
env = Environment(gui=True, num_obj=[15])
env.reset()
obs = env.get_observation()
trainer = Trainer()
env.pick([0,0,0.5])
env.place([0,0,0.5])
env.push([0,0,0.5],np.pi/2)

image = obs["freespace"]
push_predictions, pick_predictions = trainer.forward(obs["colormap"],obs["depthmap"],is_volatile=True)
print(push_predictions.shape)
plt.imshow(push_predictions[0])
plt.show()