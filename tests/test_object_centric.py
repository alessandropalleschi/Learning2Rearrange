"""Object-centric agent: instances from the observation alone, candidate slots, and a bandit the trainer must solve."""
import numpy as np
import torch

from l2r.config import Config
from l2r.learning.memory import Transition
from l2r.learning.object_centric import CLASS_RGB, FEATURES, ObjectQNet, build_state, collate, rasterize, slot_of
from l2r.learning.models import DEVICE
from l2r.learning.object_trainer import ObjectTrainer

H, W = 128, 104


def observation(blue_rows=(35, 46), green_rows=(75, 86)):
    """Padded maps (4 px border) with a blue object that is NOT in its region and a green one that is."""
    color = np.full((H, W, 3), 255, np.uint8)
    depth = np.zeros((H, W), np.float32)
    color[4 + blue_rows[0]:4 + blue_rows[1], 4 + 30:4 + 50] = CLASS_RGB[0]
    depth[4 + blue_rows[0]:4 + blue_rows[1], 4 + 30:4 + 50] = 0.05
    color[4 + green_rows[0]:4 + green_rows[1], 4 + 60:4 + 80] = CLASS_RGB[1]
    depth[4 + green_rows[0]:4 + green_rows[1], 4 + 60:4 + 80] = 0.05
    color[4 + 35:4 + 46, 4:4 + 30] = 105  # grey cells behind the blue object
    return {'colormap': color, 'depthmap': depth, 'regions': np.array([[10, 30], [70, 90], [100, 115]])}


def test_instances_features_and_candidates_come_from_the_observation_only():
    state = build_state(observation(), [16])
    assert [(i.cls, i.rows.size) for i in state['instances']] == [(0, 220), (1, 220)]
    tokens = state['tokens']
    assert tokens.shape == (1 + 3 + 2, FEATURES)
    assert tokens[4, 12] == 0 and tokens[5, 12] == 1  # blue: misplaced, green: inside its region
    assert tokens[4, 15] > 0.5 and tokens[5, 15] == 0  # the blue object has grey (hidden) cells behind it
    pick = state['actions'][0, 0]
    assert tuple(pick[1:]) == (40, 47)  # front face is column 49: aimed 2 cells inside, in the middle of the width
    assert state['valid'].shape == (2, 3) and state['valid'].all()
    assert slot_of(state, 'grasp', (0, 41, 48)) == (0, 0) and slot_of(state, 'grasp', (0, 100, 5)) is None
    assert slot_of(state, 'place', (0, 40, 47)) is None  # placement is not a decision of this agent


def test_a_push_slot_needs_an_object_in_the_blades_path():
    obs = observation()
    state = build_state(obs, [4])  # a 4-cell push reaches nothing from 3 cells beside the object... except along its width
    assert state['valid'][:, 0].all()  # the pick is always valid
    far = build_state(obs, [1])
    assert not far['valid'][:, 1:].any()  # a 1-cell push from 3 cells away touches nothing


def test_network_and_rasterisation():
    state = build_state(observation(), [16])
    tokens, n_objects, valid = collate([state, state], 'cpu')
    q = ObjectQNet(n_slots=3)(tokens, n_objects)
    assert q.shape == (2, 2, 3) and valid.shape == (2, 2, 3)
    push, pick = rasterize(state, q[0].detach().numpy(), (120, 100), 2)
    assert push.shape == (2, 120, 100) and pick.shape == (1, 120, 100)
    assert (pick > -999).sum() == 2 and (push > -999).sum() == 4  # only the candidate pixels carry values


def test_trainer_learns_which_object_to_grasp_and_acts_on_it():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Config(placement='heuristic', batch_size=8, learning_rate=3e-3, prioritized=False, reward_mode='sparse')
    trainer = ObjectTrainer(cfg)
    obs = observation()
    state = trainer.info(obs)
    good, bad = tuple(state['actions'][0, 0]), tuple(state['actions'][1, 0])  # grasp the misplaced blue / the placed green
    batch = [Transition(obs, good, obs, 0.0, 'grasp', True)] * 4 + [Transition(obs, bad, obs, -5.0, 'grasp', True)] * 4
    for _ in range(150):
        trainer.train_step(batch)
    q = trainer.forward_batch([obs])
    pick = q.pick[0, 0]
    assert pick[good[1], good[2]] > pick[bad[1], bad[2]] + 2  # about 0 against about -5
    assert np.unravel_index(np.argmax(pick), pick.shape) == (good[1], good[2])
    assert trainer.last_td.shape == (8,)


def belief_for(obs, ghost_rows=(60, 70), ghost_cols=(30, 40), age=0.5):
    """The 7-channel compact belief of the observation (rgb, height, age, ...) plus a remembered BLUE object that is hidden now."""
    belief = np.zeros((7, H, W), np.uint8)
    belief[:3] = np.transpose(obs['colormap'], (2, 0, 1))
    belief[3] = np.clip(obs['depthmap'] / 0.4, 0, 1) * 255
    belief[4] = 0  # everything visible now: age 0
    r0, r1 = (4 + ghost_rows[0], 4 + ghost_rows[1])
    c0, c1 = (4 + ghost_cols[0], 4 + ghost_cols[1])
    belief[:3, r0:r1, c0:c1] = CLASS_RGB[0][:, None, None]  # remembered blue ...
    belief[3, r0:r1, c0:c1] = 50  # ... with a height
    belief[4, r0:r1, c0:c1] = int(age * 255)  # ... last seen a while ago
    return belief


def test_ghost_tokens_are_remembered_stale_objects_only_and_have_no_actions():
    obs = observation()
    plain = build_state(obs, [16])
    assert plain['ghosts'] == [] and plain['ghost_tokens'].shape == (0, FEATURES)  # no belief: no ghosts
    obs['belief'] = belief_for(obs)
    state = build_state(obs, [16])
    assert len(state['ghosts']) == 1 and state['ghosts'][0].cls == 0  # the remembered blue object; visible objects (age 0) are not ghosts
    token = state['ghost_tokens'][0]
    assert token[18] == 1.0 and abs(token[19] - 0.5) < 0.01 and token[0] == 1.0 and token[12] == 0.0  # flag, age, class, outside its region
    assert state['tokens'][0, 2] > 0  # the global token knows there is a hidden object
    assert state['valid'].shape == (2, 3)  # actions only for the 2 visible instances


def test_the_network_attends_to_ghosts_but_only_outputs_instance_values():
    torch.manual_seed(0)
    obs = observation()
    plain = build_state(obs, [16])
    obs2 = dict(obs, belief=belief_for(obs))
    ghosty = build_state(obs2, [16])
    net = ObjectQNet(n_slots=3)
    for p in net.parameters():
        p.requires_grad_(True)
    tokens, counts, _ = collate([plain, ghosty], 'cpu')
    from l2r.learning.object_centric import MAX_MEMORY
    assert tokens.shape[1] == 4 + 2 + MAX_MEMORY and counts.tolist() == [[2, 0], [2, 1]]
    q = net(tokens, counts)
    assert q.shape == (2, 2, 3)  # instance values only
    assert not torch.allclose(q[0], q[1], atol=1e-5)  # the ghost changes what the instances are worth
    zero_age = dict(obs, belief=belief_for(obs, age=0.0))
    assert build_state(zero_age, [16])['ghosts'] == []  # age 0 means seen now


def test_continuous_trainer_learns_the_best_push_length_per_object():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Config(placement='heuristic', batch_size=64, learning_rate=3e-3, prioritized=False, reward_mode="sparse", hierarchy=True)
    trainer = ObjectTrainer(cfg)
    obs = observation()
    state = trainer.info(obs)
    row, col = state['actions'][0, 1][1:]
    lengths = np.random.RandomState(0).uniform(0, 1, 64)
    batch = [Transition(obs, (0, row, col, float(x)), obs, -5.0 * abs(float(x) - 0.8), 'push', True) for x in lengths]
    for _ in range(400):
        trainer.train_step(batch)
    q = trainer.forward_batch([obs])
    assert q.lengths is not None and q.prim.shape == (1, 3)
    assert abs(q.lengths[0, 0, row, col] - 0.8) < 0.15  # the actor climbs the critic to the best length


def test_a_sliver_narrower_than_the_grasp_depth_still_gets_a_grasp_candidate():
    obs = observation()
    obs['colormap'][4 + 100:4 + 110, 4 + 90:4 + 91] = CLASS_RGB[2]  # one column wide (e.g. an object mostly hidden behind another)
    obs['depthmap'][4 + 100:4 + 110, 4 + 90:4 + 91] = 0.05
    state = build_state(obs, [16])
    sliver = [i for i, inst in enumerate(state['instances']) if inst.cls == 2]
    assert sliver and state['valid'][sliver[0], 0] and state['actions'][sliver[0], 0][2] == 90


def held_observation():
    """`observation()` while holding a blue object (class label 1), with free space on the right half of the map."""
    obs = observation()
    obs['held_class'] = 1
    free = np.zeros((H - 8, W - 4), np.uint8)
    free[:, 50:] = 1
    obs['freespace'] = free
    return obs


def test_place_candidates_cover_the_free_cells_and_know_the_held_class_region():
    from l2r.learning.object_centric import PLACE_FEATURES, place_candidates, place_index_of
    actions, tokens = place_candidates(held_observation())
    assert tokens.shape == (len(actions), PLACE_FEATURES) and len(actions) > 100
    assert (actions[:, 2] >= 50).all()  # only free cells
    assert tokens[:, 0].all() and not tokens[:, 1:3].any()  # blue held
    inside = (actions[:, 1] >= 10) & (actions[:, 1] <= 30)  # the blue region rows
    assert (tokens[:, 5] == inside).all()
    assert place_candidates(observation())[0].shape == (0, 3)  # nothing held: no candidates
    assert place_index_of({'place_actions': actions}, (0, 21, 61)) is not None


def test_learned_object_variant_imitates_placement_and_push_length():
    from l2r.arch_benchmark import ObjectVariant
    torch.manual_seed(0)
    variant = ObjectVariant('object+learned', 300, 0, lr=3e-3)
    t = variant.trainer
    assert t.continuous and t.learned_place
    obs, held = observation(), held_observation()
    state = t.info(obs)
    row, col = state['actions'][0, 1][1:]
    push = Transition(obs, (0, row, col, 0.7), obs, 0.0, 'push', False)  # push the blue object, 70% of the length range
    place = Transition(held, (0, 20, 70), held, 0.0, 'place', False)  # inside the blue region, at column 70
    for _ in range(300):
        variant.train_step([push, place] * 4)
    primitive, pix = variant.predict(obs)
    assert primitive == 'push' and (pix[1], pix[2]) == (row, col) and abs(pix[3] - 0.7) < 0.1
    primitive, pix = variant.predict(held)
    assert primitive == 'place' and abs(pix[1] - 20) <= 2 and abs(pix[2] - 70) <= 2
    q = t.forward_batch([held])
    assert q.place.max() > -1000 and q.prim.shape == (1, 3)


def test_rl_trainer_learns_placement_and_backs_the_place_value_up_to_the_grasp_undiscounted():
    torch.manual_seed(0)
    cfg = Config(placement='learned', batch_size=8, learning_rate=3e-3, prioritized=False, reward_mode='sparse', gamma=0.5)
    trainer = ObjectTrainer(cfg)
    obs, held = observation(), held_observation()
    candidates = trainer.info(held)['place_actions']
    nearest = lambda r, c: tuple(int(v) for v in candidates[np.argmin(np.abs(candidates[:, 1] - r) + np.abs(candidates[:, 2] - c))])
    good, bad = nearest(20, 70), nearest(100, 70)  # inside the blue region / far from it
    grasp = tuple(trainer.info(obs)['actions'][0, 0])
    # rewards inside the sparse mode's Q range [-20, 0]: the good placement is worth -2, the bad one -6
    batch = [Transition(held, good, obs, -2.0, 'place', True)] * 3 + [Transition(held, bad, obs, -6.0, 'place', True)] * 3 + \
        [Transition(obs, grasp, held, 0.0, 'grasp', False)] * 2
    for step in range(300):
        trainer.train_step(batch)
        if step % 20 == 19:  # the training loop's target update
            trainer.model_target.load_state_dict(trainer.model.state_dict())
    q = trainer.forward_batch([held, obs])
    place = q.place[0, 0]
    assert abs(place[good[1], good[2]] + 2) < 0.5 and abs(place[bad[1], bad[2]] + 6) < 0.5
    assert abs(q.pick[1, 0][grasp[1], grasp[2]] + 2) < 0.5  # the grasp is worth the best placement (-2), not gamma x it (-1)


def test_tokens_know_the_last_action_and_its_outcome():
    obs = observation()
    state = build_state(obs, [16])
    assert not state['tokens'][4:, 20:22].any() and not state['tokens'][0, 3:9].any()  # no record: no flags
    blue = state['instances'][0]
    failed = dict(obs, last_action=np.array([0, 1, 0, 0, 0, 0.0], np.float32),  # a grasp that did not execute and changed nothing
                  last_pix=np.array([int(blue.rows.mean()), int(blue.cols.max()) - 2], np.int16))
    tokens = build_state(failed, [16])['tokens']
    assert tokens[4, 20] == 1 and tokens[5, 20] == 0  # on the blue object, not on the green one
    assert tokens[0, 4] == 1 and tokens[0, 7] == 0 and tokens[0, 8] == 0  # global: last was a grasp, not executed, nothing changed
    push = dict(obs, last_action=np.array([1, 0, 0, 0, 1, 0.02], np.float32),
                last_pix=np.array([blue.rows.min() - 3, int(blue.cols.mean())], np.int16))
    tokens = build_state(push, [16])['tokens']
    assert tokens[4, 21] == 1 and tokens[4, 20] == 0 and abs(tokens[0, 8] - 0.2) < 1e-6


def test_the_agent_learns_not_to_repeat_a_grasp_that_just_failed():
    """Same scene twice: after a failed grasp on the blue object, grasping it again is bad; otherwise it is the best action."""
    torch.manual_seed(0)
    cfg = Config(placement='heuristic', batch_size=8, learning_rate=3e-3, prioritized=False, reward_mode='sparse')
    trainer = ObjectTrainer(cfg)
    obs = observation()
    state = trainer.info(obs)
    blue, green = tuple(state['actions'][0, 0]), tuple(state['actions'][1, 0])
    failed = dict(obs, last_action=np.array([0, 1, 0, 0, 0, 0.0], np.float32), last_pix=np.array(blue[1:], np.int16))
    batch = [Transition(obs, blue, obs, 0.0, 'grasp', True), Transition(obs, green, obs, -3.0, 'grasp', True),
             Transition(failed, blue, failed, -6.0, 'grasp', True), Transition(failed, green, failed, -3.0, 'grasp', True)] * 2
    for _ in range(300):
        trainer.train_step(batch)
    fresh, after = trainer.forward_batch([obs]).pick[0, 0], trainer.forward_batch([failed]).pick[0, 0]
    assert fresh[blue[1], blue[2]] > fresh[green[1], green[2]] + 1
    assert after[green[1], green[2]] > after[blue[1], blue[2]] + 1


def test_outcome_heads_learn_from_free_labels_and_the_risk_penalty_avoids_toppling_actions():
    torch.manual_seed(0)
    cfg = Config(placement='heuristic', batch_size=8, learning_rate=3e-3, prioritized=False, reward_mode='sparse', outcome_weight=1.0)
    trainer = ObjectTrainer(cfg)
    obs = observation()
    state = trainer.info(obs)
    blue, green = tuple(state['actions'][0, 0]), tuple(state['actions'][1, 0])
    worked = dict(obs, last_action=np.array([0, 1, 0, 0, 1, 0.05, 0], np.float32))  # executed, changed, valid
    toppled = dict(obs, last_action=np.array([0, 1, 0, 0, 0, 0.02, 1], np.float32))  # not executed, scene invalid
    batch = [Transition(obs, blue, worked, -1.0, 'grasp', False), Transition(obs, green, toppled, -1.0, 'grasp', False)] * 4
    for _ in range(200):
        trainer.train_step(batch)
    tokens, counts, _ = collate([state], 'cpu' if not torch.cuda.is_available() else 'cuda')
    with torch.no_grad():
        p = torch.sigmoid(trainer.model.outcomes(tokens, counts))[0].cpu().numpy()
    assert p[0, 0, 0] > 0.8 and p[1, 0, 0] < 0.2  # executed: blue yes, green no
    assert p[0, 0, 1] < 0.2 and p[1, 0, 1] > 0.8  # invalid: green
    plain = trainer.forward_batch([obs]).pick[0, 0]
    trainer.cfg.outcome_penalty = 0.5  # 0.5 x P(invalid) x 10
    careful = trainer.forward_batch([obs]).pick[0, 0]
    assert careful[green[1], green[2]] < plain[green[1], green[2]] - 3 and abs(careful[blue[1], blue[2]] - plain[blue[1], blue[2]]) < 1


def test_twin_heads_are_independent_and_the_target_takes_their_minimum():
    """Clipped double Q: two value heads (initialised differently); the TD target of a next state is the minimum of the two target
    heads at the action the policy (their mean) prefers. Without --twin_q there is one head and the target is its value."""
    torch.manual_seed(0)
    obs, held = observation(), held_observation()
    for placement, nxt in (('heuristic', obs), ('learned', held)):
        trainer = ObjectTrainer(Config(placement=placement, twin_q=True, reward_mode='sparse'))
        state = trainer.info(nxt)
        tokens, counts, _ = collate([state], DEVICE)
        with torch.no_grad():
            if placement == 'learned':
                from l2r.learning.object_centric import collate_place
                place, _ = collate_place([state], DEVICE)
                heads = trainer.model_target.place_values(tokens, counts, place, heads=True)[:, 0]
                choice = trainer.model.place_values(tokens, counts, place)[0].argmax()
                expected = heads[:, choice].min()
            else:
                heads = trainer.model_target.q_heads(tokens, counts)[:, 0]
                valid = torch.as_tensor(state['valid'], device=DEVICE)
                pol = trainer.model(tokens, counts)[0].masked_fill(~valid, -1e9).flatten()
                expected = heads.flatten(1)[:, pol.argmax()].min()
            assert heads.shape[0] == 2 and not torch.allclose(heads[0], heads[1])
            assert torch.isclose(trainer._future([state])[0], expected)
    single = ObjectTrainer(Config(placement='heuristic', reward_mode='sparse'))
    assert single.model.q_heads(*collate([single.info(obs)], DEVICE)[:2]).shape[0] == 1


def test_iql_policy_prefers_the_better_actions_of_the_data_for_grasps_placements_and_push_lengths():
    from l2r.learning.factory import make_trainer
    from l2r.learning.object_centric import collate_place
    torch.manual_seed(0)
    cfg = Config(agent='object', object_algo='iql', placement='learned', hierarchy=True, twin_q=True, batch_size=8,
                 learning_rate=3e-3, prioritized=False, reward_mode='sparse')
    trainer = make_trainer(cfg)
    obs, held = observation(), held_observation()
    state, held_state = trainer.info(obs), trainer.info(held)
    blue, green = tuple(state['actions'][0, 0]), tuple(state['actions'][1, 0])
    row, col = state['actions'][0, 1][1:]
    candidates = held_state['place_actions']
    good = tuple(int(v) for v in candidates[0])
    bad = tuple(int(v) for v in candidates[-1])
    batch = [Transition(obs, blue, obs, 0.0, 'grasp', True), Transition(obs, green, obs, -5.0, 'grasp', True),
             Transition(held, good, obs, 0.0, 'place', True), Transition(held, bad, obs, -5.0, 'place', True),
             Transition(obs, (0, row, col, 0.2), obs, 0.0, 'push', True), Transition(obs, (0, row, col, 0.8), obs, -5.0, 'push', True)]
    for step in range(400):
        trainer.train_step(batch)
        if step % 20 == 19:  # the training loop's target update (the advantages use the target critic)
            trainer.model_target.load_state_dict(trainer.model.state_dict())
    tokens, counts, _ = collate([state], DEVICE)
    with torch.no_grad():
        logits = trainer.model.policy_logits(tokens, counts)[0]
        place_logits = trainer.model.place_policy_logits(*collate([held_state], DEVICE)[:2], collate_place([held_state], DEVICE)[0])[0]
        value = float(trainer.model.value(tokens, counts)[0])
    assert logits[0, 0] > logits[1, 0] + 1  # grasp the blue object, not the green one
    assert place_logits[0] > place_logits[-1] + 1
    q = trainer.forward_batch([obs])
    assert abs(q.lengths[0, 0, row, col] - 0.2) < 0.15  # the actor follows the better executed length
    assert -5.0 < value <= 0.5  # an expectile of the data's action values, no max over unseen actions
    assert q.pick[0, 0][blue[1], blue[2]] > q.pick[0, 0][green[1], green[2]]  # acting follows the policy


def test_history_tokens_record_what_was_done_and_the_agent_learns_from_them():
    from l2r.learning.object_centric import HISTORY, make_history_entry
    obs = observation()
    state = build_state(obs, [16])
    blue = state['instances'][0]
    pix = (0, int(blue.rows.mean()), int(blue.cols.max()) - 2)
    after = dict(obs, last_action=np.array([0, 1, 0, 0, 0, 0.0, 0], np.float32))  # a grasp that did not execute
    entry = make_history_entry(obs, 'grasp', pix, after)
    assert entry[0] == 1 and entry[6] == 1 and entry[9] == 0 and entry[22] == 1  # blue object, grasp, not executed, history token
    remembered = dict(obs, history=np.stack([entry]))
    tokens = build_state(remembered, [16])['ghost_tokens']
    assert len(tokens) == 1 and np.isclose(tokens[0, 23], 1 / HISTORY)
    # without the last-action record, the history alone tells the agent that grasping the blue object just failed
    torch.manual_seed(0)
    trainer = ObjectTrainer(Config(placement='heuristic', batch_size=8, learning_rate=3e-3, prioritized=False, reward_mode='sparse'))
    s = trainer.info(obs)
    blue_grasp, green_grasp = tuple(s['actions'][0, 0]), tuple(s['actions'][1, 0])
    batch = [Transition(obs, blue_grasp, obs, 0.0, 'grasp', True), Transition(obs, green_grasp, obs, -3.0, 'grasp', True),
             Transition(remembered, blue_grasp, remembered, -6.0, 'grasp', True),
             Transition(remembered, green_grasp, remembered, -3.0, 'grasp', True)] * 2
    for _ in range(300):
        trainer.train_step(batch)
    fresh, again = trainer.forward_batch([obs]).pick[0, 0], trainer.forward_batch([remembered]).pick[0, 0]
    assert fresh[blue_grasp[1], blue_grasp[2]] > fresh[green_grasp[1], green_grasp[2]] + 1
    assert again[green_grasp[1], green_grasp[2]] > again[blue_grasp[1], blue_grasp[2]] + 1


def test_carry_builds_the_history_newest_first_and_only_when_enabled():
    from types import SimpleNamespace

    from l2r.learning.object_centric import HISTORY
    from l2r.learning.policy import carry
    obs = observation()
    on, off = SimpleNamespace(cfg=Config(history_tokens=True), belief='none'), SimpleNamespace(cfg=Config(), belief='none')
    current = obs
    for k in range(HISTORY + 2):
        nxt = carry(on, dict(obs), None, 'push' if k % 2 else 'grasp', (0, 40, 40), None, current)
        current = nxt
    assert current['history'].shape == (HISTORY, 24) and current['history'][0, 5] == 1  # newest (a push) first
    assert 'history' not in carry(off, dict(obs), None, 'grasp', (0, 40, 40), None, obs)


def test_outcome_mask_drops_candidates_predicted_to_fail_when_acting_and_in_targets():
    torch.manual_seed(0)
    cfg = Config(placement='heuristic', batch_size=8, learning_rate=3e-3, prioritized=False, reward_mode='sparse', outcome_weight=1.0)
    trainer = ObjectTrainer(cfg)
    obs = observation()
    state = trainer.info(obs)
    blue, green = tuple(state['actions'][0, 0]), tuple(state['actions'][1, 0])
    failed = dict(obs, last_action=np.array([0, 1, 0, 0, 0, 0.0, 0], np.float32))  # not executed, nothing changed
    worked = dict(obs, last_action=np.array([0, 1, 0, 0, 1, 0.2, 0], np.float32))
    # the grasp on blue is (slightly) better valued but never executes; the green one works
    batch = [Transition(obs, blue, failed, -1.0, 'grasp', True), Transition(obs, green, worked, -1.3, 'grasp', True)] * 4
    for _ in range(300):
        trainer.train_step(batch)
    pick = trainer.forward_batch([obs]).pick[0, 0]
    assert pick[blue[1], blue[2]] > pick[green[1], green[2]]  # without the mask: the failing grasp wins
    trainer.cfg.outcome_mask = 0.5
    masked = trainer.forward_batch([obs]).pick[0, 0]
    assert masked[blue[1], blue[2]] == -1000.0 and masked[green[1], green[2]] > -1000.0
    tokens, counts, valid = collate([state], DEVICE)
    with torch.no_grad():
        keep = trainer._feasible(torch.sigmoid(trainer.model.outcomes(tokens, counts)), valid)
    assert not keep[0, 0, 0] and keep[0, 1, 0]
    nothing = trainer._feasible(torch.zeros(1, 2, 3, 3, device=DEVICE), valid)  # everything predicted to fail: keep the valid ones
    assert torch.equal(nothing, valid)


def test_goal_masks_drive_the_goal_tokens_and_the_in_goal_features():
    from l2r.learning.object_centric import region_features
    obs = observation()  # blue object at rows 35..45, cols 30..49; green at rows 75..85, cols 60..79
    goal_map = np.zeros((120, 100), np.uint8)
    goal_map[30:50, 25:55] = 1  # blue goal: a slot around the blue object
    goal_map[0:20, 80:100] = 2  # green goal: a slot far from the green object
    obs['goal_map'] = goal_map
    state = build_state(obs, [16])
    tokens = state['tokens']
    assert tokens[4, 12] == 1 and tokens[5, 12] == 0  # blue inside its goal, green not
    assert tokens[5, 13] < 0 and tokens[5, 14] > 0  # the green goal is up (fewer rows) and to the front (more columns)
    goals = region_features(obs)
    assert np.isclose(goals[0, 5], 20 / 120 * 4) and np.isclose(goals[0, 6], 30 / 100 * 4)  # extent of the blue slot


def test_goal_map_channels_of_the_dense_agent():
    from l2r.learning.trainer import Trainer
    out = np.zeros((5, 128, 104), np.float32)
    goal_map = np.zeros((120, 100), np.uint8)
    goal_map[10:20, 30:40] = 1
    Trainer._geometry_channels(out, {'goal_map': goal_map, 'regions': np.zeros((3, 2))}, 0, 0, 128, 104)
    assert out[0, 4 + 10:4 + 20, 4 + 30:4 + 40].all() and out[0].sum() == 100  # 4-cell border: rows both sides, columns left
