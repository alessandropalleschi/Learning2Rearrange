"""Action selection shared by training and evaluation."""
import numpy as np

from rearrange_env import check_push


def select_action(trainer, obs, push_q, pick_q, explore_prob, cfg):
    """Epsilon-greedy over the two primitives.

    push_q (2,h,w) and pick_q (1,h,w) are the raw Q-maps of the network for `obs`.
    Returns (primitive, (rotation, row, col), predicted_value). explore_prob=0 gives the greedy policy.
    """
    push_q = trainer.preprocessQmaps(push_q, obs['depthmap'], 'push')
    pick_q = trainer.preprocessQmaps(pick_q, obs['depthmap'], 'grasp')
    primitive = 'push' if np.max(push_q) > np.max(pick_q) else 'grasp'
    q = {'push': push_q, 'grasp': pick_q}

    if explore_prob > 0 and np.random.uniform() < explore_prob:
        if np.random.randint(0, 2) == 1:
            primitive = 'push' if np.random.randint(0, 2) == 0 else 'grasp'
        valid = np.flatnonzero(q[primitive].ravel() > -1000)
        if valid.size:
            pix = np.unravel_index(np.random.choice(valid), q[primitive].shape)
            if primitive == 'push':  # prefer pushes that actually touch something
                for _ in range(100):
                    angle = np.deg2rad(pix[0] * 180.0 - 90)
                    if check_push(obs['depthmap'], pix[2], pix[1], angle, cfg.push_length):
                        break
                    pix = np.unravel_index(np.random.choice(valid), q[primitive].shape)
        else:
            pix = np.unravel_index(np.argmax(q[primitive]), q[primitive].shape)
    else:
        pix = np.unravel_index(np.argmax(q[primitive]), q[primitive].shape)
    return primitive, tuple(int(i) for i in pix), float(q[primitive][pix])


# ---------------------------------------------------------------------------------------------------
# Recurrent belief bookkeeping. The hidden state and the last action travel with the observations:
#   obs_t['h_prev'], obs_t['prev_action']  = the inputs of the belief update at time t (h_{t-1}, a_{t-1}).
# After acting at time t the network returns h_t, which becomes obs_{t+1}['h_prev'] (with a_t).
# ---------------------------------------------------------------------------------------------------
def attach_initial(trainer, obs):
    """Episode start: empty belief, no previous action."""
    if trainer.belief == 'recurrent':
        obs['h_prev'] = trainer.zero_hidden()
        obs['prev_action'] = None
    return obs


def carry(trainer, next_obs, h_new, primitive, pix, place_pix):
    """Attach the updated belief and the executed action to the observation that follows the action."""
    if trainer.belief == 'recurrent':
        next_obs['h_prev'] = h_new
        next_obs['prev_action'] = {'primitive': primitive, 'pix': pix, 'place_pix': place_pix}
    return next_obs


def carry_over(trainer, src, dst):
    """Copy the belief inputs of `src` to the (relabelled) copy `dst`."""
    if trainer.belief == 'recurrent':
        dst['h_prev'], dst['prev_action'] = src['h_prev'], src['prev_action']
    return dst
