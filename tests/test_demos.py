"""Demonstrations: the DQfD margin loss, mixed batches, pre-training."""
import numpy as np
import torch

from l2r.config import Config
from l2r.learning.memory import Transition
from l2r.learning.models import DEVICE, QNet
from l2r.learning.trainer import Trainer

H, W = 40, 36


def state():
    depth = np.zeros((H, W), np.float32)
    depth[14:26, 12:24] = 0.06  # one object: padded rows 14..25, columns 12..23 -> action rows 10..21, columns 8..19
    return {'colormap': np.random.randint(0, 255, (H, W, 3), dtype=np.uint8), 'depthmap': depth}


def make_trainer(**kw):
    return Trainer(Config(batch_size=4, amp=False, prioritized=False, learning_rate=1e-3, **kw))


def test_margin_loss_puts_the_expert_action_on_top():
    torch.manual_seed(0)
    np.random.seed(0)
    trainer, s = make_trainer(), state()
    expert = (0, 14, 12)  # a pick on the object, not where the untrained network happens to peak
    demos = [Transition(s, expert, s, -1.0, 'grasp', True)] * 8
    batch = demos[:4]
    before = float(trainer._margin_loss(trainer.prepare(s)[None].repeat(4, 1, 1, 1).to(next(trainer.model.parameters()).device), [s] * 4,
                                        torch.tensor([QNet.PICK] * 4, device=DEVICE), torch.zeros(4, dtype=torch.long, device=DEVICE),
                                        torch.full((4,), 14 + 4 + trainer._pads(H, W)[0], device=DEVICE),
                                        torch.full((4,), 12 + 4 + trainer._pads(H, W)[1], device=DEVICE), np.ones(4, bool),
                                        (None, None, None), H, W))
    trainer.pretrain(demos, 200, log=lambda *a: None)
    pick = trainer.preprocessQmaps(trainer.forward_batch([s]).pick[0], s['depthmap'], 'grasp')
    assert np.unravel_index(np.argmax(pick), pick.shape)[1:] == expert[1:]  # the expert action is the greedy one
    after = trainer.train_step(batch, demo=np.ones(4, bool))
    assert before > 0.7 and after < before  # the margin term (initially ~the margin) shrinks; the gap itself grows slowly


def test_margin_loss_ignores_samples_that_are_not_demonstrations():
    torch.manual_seed(1)
    trainer, s = make_trainer(), state()
    batch = [Transition(s, (0, 14, 12), s, -1.0, 'grasp', True)] * 4
    plain = trainer.train_step(batch)
    trainer2 = make_trainer()
    trainer2.model.load_state_dict(trainer.model.state_dict())
    assert trainer2.train_step(batch, demo=np.zeros(4, bool)) is not None  # no demo flag: the margin term is skipped


def test_optimize_mixes_demonstrations_and_agent_transitions():
    np.random.seed(2)
    trainer, s = make_trainer(demo_fraction=0.5), state()
    for _ in range(8):
        trainer.buffer.push(s, (0, 15, 13), s, -1.0, 'grasp', True)
    trainer.demos = [Transition(s, (0, 14, 12), s, 0.0, 'grasp', True)] * 6
    seen = []
    original = trainer.train_step
    trainer.train_step = lambda t, w=None, demo=None: (seen.append(demo.copy()), original(t, w, demo))[1]
    assert trainer.optimize() is not None
    assert seen[0].tolist() == [True, True, False, False]  # half of the batch are demonstrations


def test_expert_aims_at_a_cell_of_the_object_even_when_its_front_is_notched():
    """Regression: the median row of the front columns can be a hole in the object; a grasp on an empty cell is ignored."""
    from l2r.demos import GRASP_DEPTH_CELLS, ScriptedExpert
    cells = {(r, c) for r in range(2, 16) for c in range(6, 27) if not (c >= 22 and 5 <= r <= 11)}
    rows, cols = np.array([r for r, c in cells]), np.array([c for r, c in cells])
    row, col = ScriptedExpert._aim_cell(rows, cols)
    assert (row, col) in cells and col <= 26 - GRASP_DEPTH_CELLS
    thin = {(r, c) for r in range(3, 5) for c in range(10, 13)}  # an object only 3 cells deep
    row, col = ScriptedExpert._aim_cell(np.array([r for r, c in thin]), np.array([c for r, c in thin]))
    assert (row, col) in thin


def test_expert_pushes_objects_that_are_one_push_from_their_region_and_checks_the_blade_strip():
    from types import SimpleNamespace
    from l2r.demos import ScriptedExpert
    scene = SimpleNamespace(regions={'blue': [0.10, 0.25]}, class_name=lambda i: 'blue',
                            poses=lambda: {7: ((0.0, 0.06, 0.4), None), 8: ((0.0, -0.20, 0.4), None), 9: ((0.0, 0.15, 0.4), None)})
    expert = ScriptedExpert(SimpleNamespace(sim=SimpleNamespace(scene=scene)))
    assert expert._near_region(7)  # 4 cm from the region: one push
    assert not expert._near_region(8)  # 30 cm away: pick and place
    assert not expert._near_region(9)  # already inside
    expert.pushes[7] = 1
    assert not expert._near_region(7)  # a push that did not finish it is followed by a grasp
    depth = np.zeros((128, 104), np.float32)
    depth[4 + 60:4 + 70, 4 + 40:4 + 60] = 0.05  # an object at rows 60..69, columns 40..59 (cropped coordinates)
    obs = {'depthmap': depth}
    assert ScriptedExpert._blade_touches(obs, 55, 50, 1, 0.08)  # in its column, pushing towards it
    assert not ScriptedExpert._blade_touches(obs, 55, 70, 1, 0.08)  # beside it: check_push would accept, the blade would miss
    assert not ScriptedExpert._blade_touches(obs, 75, 50, 1, 0.08)  # the object is behind the blade
