"""Policy heatmap + critic: heads, behaviour cloning of the policy, proposals ranked by the critic."""
import numpy as np
import torch

from l2r.config import Config
from l2r.learning.memory import Transition
from l2r.learning.models import QNet
from l2r.learning.policy_critic import PolicyCriticTrainer

H, W = 40, 36


def state():
    depth = np.zeros((H, W), np.float32)
    depth[14:26, 12:24] = 0.06  # one object: action rows 10..21, columns 8..19
    return {'colormap': np.random.randint(0, 255, (H, W, 3), dtype=np.uint8), 'depthmap': depth}


def make_trainer(**kw):
    return PolicyCriticTrainer(Config(agent='policy_critic', batch_size=4, amp=False, prioritized=False, learning_rate=2e-3, **kw))


def test_policy_heads_have_the_shapes_of_the_q_heads_and_follow_the_flip():
    net = QNet(4, 'none', 8, 2, False, policy_heads=True)
    x = torch.randn(2, 4, 40, 40)
    out = net.forward_ext(x)
    push, pick, place = out.policy
    assert push.shape == out.push.shape == (2, 4, 40, 40) and pick.shape == out.pick.shape and place is None
    assert QNet(4, 'none', 8, 2, False).forward_ext(x).policy is None  # off by default: the dense network is unchanged


def test_top_k_cells_keeps_the_best_valid_cells_only():
    scores = np.arange(12, dtype=np.float32).reshape(3, 4)
    valid = np.ones((3, 4), bool)
    valid[2, 3] = False  # the best score is not feasible
    keep = PolicyCriticTrainer.top_k_cells(scores, valid, 3)
    assert keep.sum() == 3 and keep[2, 2] and keep[2, 1] and keep[2, 0] and not keep[2, 3]
    assert PolicyCriticTrainer.top_k_cells(scores, np.zeros((3, 4), bool), 3).sum() == 0
    assert PolicyCriticTrainer.top_k_cells(scores, valid, 50).sum() == 11  # fewer valid cells than k


def test_behaviour_cloning_moves_the_policy_to_the_demonstrated_cell_and_proposals_contain_it():
    torch.manual_seed(0)
    np.random.seed(0)
    trainer, s = make_trainer(policy_topk=5), state()
    expert = (0, 14, 12)  # a pick on the object (action coordinates)
    demos = [Transition(s, expert, s, -1.0, 'grasp', True)] * 4
    first = None
    for step in range(120):
        loss = float(trainer.bc_loss(demos).detach())
        first = first if first is not None else loss
        trainer.train_step(demos, demo=np.ones(4, bool))
    assert float(trainer.bc_loss(demos).detach()) < first / 3  # the cross-entropy falls
    q = trainer.forward_batch([s])
    proposed = q.pick[0, 0] > -999
    assert 1 <= proposed.sum() <= 5 and proposed[expert[1], expert[2]]  # the policy proposes at most k cells, the expert's among them
    assert (q.push[0] > -999).sum() <= 5
    assert trainer.last_bc > 0  # the behaviour-cloning term is logged on demonstration batches


def test_no_policy_loss_without_demonstrations():
    trainer, s = make_trainer(), state()
    batch = [Transition(s, (0, 14, 12), s, -1.0, 'grasp', True)] * 4
    assert trainer.train_step(batch) is not None  # plain TD step: the extra loss hook returns nothing
    assert not hasattr(trainer, 'last_bc')
