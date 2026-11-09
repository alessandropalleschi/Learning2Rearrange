"""Hierarchical agent: level-1 primitive values, continuous push length (actor-critic), selection."""
import numpy as np
import torch

from l2r.config import Config
from l2r.learning.memory import Transition
from l2r.learning.policy import select_for, select_hierarchical
from l2r.learning.trainer import Trainer
from l2r.sim.workspace import Workspace

H, W = 40, 36  # a tiny map: 3 poolings need multiples of 8 (the padded 4 px border is cropped from the action maps)


def tiny_state():
    depth = np.zeros((H, W), np.float32)
    depth[14:26, 12:24] = 0.06  # one object
    return {'colormap': np.random.randint(0, 255, (H, W, 3), dtype=np.uint8), 'depthmap': depth}


def make_trainer(**kw):
    return Trainer(Config(hierarchy=True, batch_size=4, amp=False, prioritized=False, learning_rate=1e-3, **kw))


def test_actor_critic_learns_a_continuous_push_length():
    """Bandit: the reward only depends on the executed length (best at 0.8 of the range). The critic must learn the
    reward curve and the actor, which only sees the critic's gradient, must move to its maximum."""
    torch.manual_seed(0)
    np.random.seed(0)
    trainer, state = make_trainer(), tiny_state()
    row, col = 9, 10  # cropped action coordinates, next to the object
    for _ in range(220):
        lengths = np.random.uniform(0, 1, 4)
        batch = [Transition(state, (0, row, col, float(l)), state, -(float(l) - 0.8) ** 2, 'push', True) for l in lengths]
        trainer.train_step(batch)
    obs_q = trainer.forward_batch([state])
    actor_length = float(obs_q.lengths[0, 0, row, col])
    assert abs(actor_length - 0.8) < 0.25, actor_length
    x = trainer.prepare(state)[None]
    with torch.no_grad():
        f0, _, _ = trainer.model._features(x, None, None, None)
        critic = lambda l: trainer.model._push_critic(f0, torch.full((1, 1, *f0.shape[2:]), l))[0, 0, row + 4 + 0, col + 4 + 2].item()
    assert critic(0.8) > critic(0.05)  # the critic knows that a very short push is worse


def test_gradient_routing_actor_only_reaches_the_length_head():
    trainer, state = make_trainer(), tiny_state()
    batch = [Transition(state, (0, 9, 10, 0.5), state, -1.0, 'push', True)] * 4
    trainer.train_step(batch)
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    trainer.train_step(batch)
    changed = {n for n, p in trainer.model.named_parameters() if not torch.equal(p, before[n])}
    assert any(n.startswith('length_head') for n in changed) and any(n.startswith('push_head') for n in changed)
    assert any(n.startswith('prim_head') for n in changed)  # level 1 learns too


def test_primitive_head_regresses_the_best_value_of_each_primitive():
    torch.manual_seed(1)
    trainer, state = make_trainer(hier_prim_weight=5.0), tiny_state()
    for _ in range(80):  # pick pays 0, pushes pay -10: level 1 should rank pick above push
        batch = [Transition(state, (0, 9, 10, 0.5), state, -10.0, 'push', True),
                 Transition(state, (0, 20, 16), state, 0.0, 'grasp', True)] * 2
        trainer.train_step(batch)
    prim = trainer.forward_batch([state]).prim[0]
    assert prim[1] > prim[0], prim  # order: push, pick, place


def test_selection_follows_the_primitive_values_and_returns_a_continuous_length():
    cfg, ws = Config(hierarchy=True), Workspace()
    trainer, state = make_trainer(), tiny_state()
    q = trainer.forward_batch([state])
    greedy = lambda prim_q: select_hierarchical(trainer, state, q.push[0], q.pick[0], np.array(prim_q, np.float32),
                                                q.lengths[0], 0.0, cfg, ws)
    assert greedy([5.0, -5.0, 0.0])[0] == 'push' and greedy([-5.0, 5.0, 0.0])[0] == 'grasp'  # level 1 decides
    primitive, pix, _ = greedy([5.0, -5.0, 0.0])
    assert len(pix) == 4 and 0.0 <= pix[3] <= 1.0 and pix[0] in (0, 1)
    for _ in range(5):  # exploring: random feasible primitive; any length in the range
        primitive, pix, _ = select_for(trainer, state, q, 0, 1.0, cfg, ws)
        assert primitive in ('push', 'grasp') and (len(pix) == 4) == (primitive == 'push')
        if primitive == 'push':
            assert 0.0 <= pix[3] <= 1.0
