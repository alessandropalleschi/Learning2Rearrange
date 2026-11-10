"""Supervised architecture benchmark on the expert demonstrations.

Reinforcement learning takes hours per run and is noisy. The recorded expert transitions are a cheap proxy to rank architectures: every
variant is trained on the SAME transitions to predict the expert's action (cross-entropy over the feasible actions) and evaluated on
held-out episodes. It measures how well an architecture can represent and learn the action structure, not how well RL works with it.

    python scripts/arch_benchmark.py --demos data/demos/expert.pkl --variants baseline skips skips+region --seeds 3 --steps 1500

Metrics on held-out transitions: `primitive` (push / pick / place), and for the cases where the primitive is right: `in-class`
(the chosen cell lies in the equivalence class of the expert's cell: for a grasp the central front band of the same object, for a push /
place a 5x5 neighbourhood), `near` (within 3 cells) and `exact` (the expert's own cell).
"""
import argparse
import json

import numpy as np
import torch
from scipy import ndimage

from l2r.config import Config
from l2r.demos import load_demos
from l2r.learning.models import DEVICE, QNet
from l2r.learning.trainer import Trainer
from l2r.sim.constants import CLASSES, COLORS

BASE = dict(placement='learned', footprint_placement=True)
VARIANTS = {
    'baseline': dict(),
    'skips': dict(arch_skips=True),
    'region': dict(region_channels=True),
    'skips+region': dict(arch_skips=True, region_channels=True),
    'skips+region+equiv': dict(arch_skips=True, region_channels=True, equiv_targets=True),
    'hybrid': dict(backbone='hybrid'),
    'policy+skips': dict(agent='policy_critic', arch_skips=True),
    'object': dict(agent='object', placement='heuristic'),
    'fpn': dict(arch_decoder='fpn'),
    'attn_unet': dict(arch_decoder='attn_unet'),
    'unet_attn4': dict(arch_decoder='unet_attn4'),
    'unet+hybrid': dict(arch_skips=True, backbone='hybrid'),
    'unet+ppm': dict(arch_skips=True, backbone='ppm'),
    'object+continuous': dict(agent='object', placement='heuristic', hierarchy=True),
    'object+place': dict(agent='object', placement='learned'),
    'object+learned': dict(agent='object', placement='learned', hierarchy=True),  # learned placement and continuous push length
    'object+place+belief': dict(agent='object', placement='learned', belief='spatial'),  # ghost tokens from the spatial belief
    'skips+belief': dict(arch_skips=True, belief='spatial'),  # belief channels as network input
    'unet+convnext+ppm': dict(arch_skips=True, arch_encoder='convnext', backbone='ppm'),
    'unet+convnext+aspp': dict(arch_skips=True, arch_encoder='convnext', backbone='aspp'),
    'unet+aspp': dict(arch_skips=True, backbone='aspp'),
    'unet+ppm+aspp': dict(arch_skips=True, backbone='ppm_aspp'),
    'unet+os4': dict(arch_skips=True, arch_stride4=True),
    'unet+os4+ppm': dict(arch_skips=True, arch_stride4=True, backbone='ppm'),
    'unet+os4+aspp': dict(arch_skips=True, arch_stride4=True, backbone='aspp'),
    'fpn+hybrid': dict(arch_decoder='fpn', backbone='hybrid'),
    'hrnet': dict(arch_decoder='hrnet'),
    'unet+convnext': dict(arch_skips=True, arch_encoder='convnext'),
    'unet+r18_imagenet': dict(arch_skips=True, arch_encoder='resnet18_imagenet'),
    'unet+r18_scratch': dict(arch_skips=True, arch_encoder='resnet18_scratch'),
    'skips+region+hybrid': dict(arch_skips=True, region_channels=True, backbone='hybrid'),
}


class DenseVariant:
    """A dense Q-map network trained with a cross-entropy over the flattened, masked action space."""

    def __init__(self, name, steps, seed, batch=8, lr=3e-4):
        torch.manual_seed(seed)
        self.cfg = Config(**BASE, **VARIANTS[name])
        self.trainer = Trainer(self.cfg)
        self.reference = Trainer(Config(**BASE, equiv_targets=True))  # defines the equivalence classes used by the metric
        self.optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, steps)
        self.batch = batch
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.trainer.amp)  # fp16 autocast needs loss scaling (it diverged without)

    # ------------------------------------------------------------------ layout of the flat action space
    def _flat_index(self, t, prim, rot, rows, cols, canvas, n_push):
        channel = torch.where(prim == QNet.PUSH, rot, torch.where(prim == QNet.PICK, torch.zeros_like(rot) + n_push,
                                                                     torch.zeros_like(rot) + n_push + 1))
        return channel * (canvas[0] * canvas[1]) + rows * canvas[1] + cols

    def _logits(self, states, grad):
        t = self.trainer
        h, w = states[0]['depthmap'].shape[:2]
        x = torch.stack([t.prepare(s) for s in states]).to(DEVICE)
        with torch.set_grad_enabled(grad), torch.autocast(device_type=DEVICE.type, enabled=t.amp):
            out = t.model(x, None, None, t._held(states))
        push, pick = out[0].float(), out[1].float()
        place = out[2].float() if out[2] is not None else torch.full_like(pick, -1e9)
        push_mask, pick_mask, place_mask = t._masks_padded(states, h, w)
        logits = torch.cat([push.flatten(1), pick.flatten(1), place.flatten(1)], dim=1)
        masks = torch.cat([push_mask.expand_as(push).flatten(1), pick_mask.flatten(1), place_mask.flatten(1)], dim=1)
        return logits, masks, push.shape[1], push.shape[-2:], (h, w)

    def _expert(self, transitions, h, w):
        t = self.trainer
        top, left = t._pads(h, w)
        prim = torch.tensor([t.PRIMITIVES[tr.primitive_action] for tr in transitions], device=DEVICE)
        rot = torch.tensor([int(tr.action[0]) for tr in transitions], device=DEVICE)
        rows = torch.tensor([int(tr.action[1]) + t.offset_px + top for tr in transitions], device=DEVICE)
        cols = torch.tensor([int(tr.action[2]) + t.offset_px + left for tr in transitions], device=DEVICE)
        return prim, rot, rows, cols, top, left

    # ------------------------------------------------------------------ training / prediction
    def train_step(self, transitions):
        states = [tr.state for tr in transitions]
        logits, masks, n_push, canvas, (h, w) = self._logits(states, grad=True)
        prim, rot, rows, cols, top, left = self._expert(transitions, h, w)
        expert = self._flat_index(self.trainer, prim, rot, rows, cols, canvas, n_push)
        target = torch.zeros_like(logits).scatter(1, expert[:, None], 1.0)
        if self.cfg.equiv_targets:  # uniform over the equivalence class of the expert's cell, in the expert's own channel
            cells = self.trainer._equivalent_cells(states, prim, rows, cols, h, w, top, left).flatten(1)
            channel = (expert // (canvas[0] * canvas[1]))[:, None]
            offset = channel * (canvas[0] * canvas[1])
            idx = torch.arange(cells.shape[1], device=DEVICE)[None] + offset
            target = torch.zeros_like(logits).scatter(1, idx, cells)
            target = target / target.sum(1, keepdim=True).clamp(min=1e-9)
        masks = masks | (target > 0)  # the expert's own cells are feasible by definition
        log_p = torch.log_softmax(logits.masked_fill(~masks, -1e9), dim=1)
        loss = -(target * log_p).sum(1).mean()
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.trainer.model.parameters(), 5.0)
        self.scaler.step(self.optimizer)  # skips the step if the gradients overflowed
        self.scaler.update()
        self.scheduler.step()
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, state):
        logits, masks, n_push, canvas, (h, w) = self._logits([state], grad=False)
        flat = logits.masked_fill(~masks, -1e9)[0].argmax().item()
        size = canvas[0] * canvas[1]
        channel, rest = divmod(flat, size)
        row, col = divmod(rest, canvas[1])
        top, left = self.trainer._pads(h, w)
        primitive = 'push' if channel < n_push else ('grasp' if channel == n_push else 'place')
        return primitive, (channel if primitive == 'push' else 0, row - top - self.trainer.offset_px, col - left - self.trainer.offset_px)


CLASS_RGB = np.array([np.multiply(COLORS[name][:3], 255) for name in CLASSES]).round().astype(int)


def sensible_action(state, primitive, pix):
    """Is the predicted action reasonable, judged from the observation alone (no expert label)?
    grasp: the cell lies on an object that is NOT yet inside its own target region (class colour + the region rows of the observation);
    place: the cell lies inside the target region of the held object's class. Pushes are not judged (returns None)."""
    regions = np.asarray(state['regions'])
    row, col = pix[1] + 4, pix[2] + 4  # padded map coordinates
    if primitive == 'grasp':
        colour = state['colormap'][row, col].astype(int)
        match = np.nonzero((np.abs(CLASS_RGB - colour).sum(1) <= 6))[0]
        if match.size == 0 or state['depthmap'][row, col] <= 0.008:
            return False
        k = int(match[0])
        same = (np.abs(state['colormap'].astype(int) - CLASS_RGB[k]).sum(-1) <= 6) & (state['depthmap'] > 0.008)
        labels, _ = ndimage.label(same, structure=np.ones((3, 3)))
        centre = np.nonzero(labels == labels[row, col])[0].mean() - 4  # un-padded row of the object's centre
        return bool(not regions[k][0] <= centre <= regions[k][1])
    if primitive == 'place':
        k = {1: 0, 3: 1, 5: 2}[int(state['held_class'])]  # class label -> index in CLASSES
        return bool(regions[k][0] <= pix[1] <= regions[k][1])
    return None


class PolicyVariant(DenseVariant):
    """The dense network with a policy head: the cross-entropy trains the POLICY logits (behaviour cloning), the Q heads are untouched."""

    def __init__(self, name, steps, seed, batch=8, lr=3e-4):
        from l2r.learning.policy_critic import PolicyCriticTrainer
        super().__init__(name, steps, seed, batch, lr)
        self.trainer = PolicyCriticTrainer(self.cfg)
        self.optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, steps)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.trainer.amp)

    def _logits(self, states, grad):
        logits, masks, (n_push, _), canvas, size = self.trainer.policy_logits(states, grad)
        return logits, masks, n_push, canvas, size


class ObjectVariant:
    """The object-centric Q-function trained with a cross-entropy over its object-action slots on the demonstrations.
    * heuristic placement (default): the demonstrated placements are neither trained on nor evaluated;
    * `placement='learned'`: also a cross-entropy over the place candidates (free cells the held object fits in);
    * `hierarchy=True`: continuous push length; the actor regresses the demonstrated length (fraction of `push_length_range`)."""

    def __init__(self, name, steps, seed, batch=8, lr=1e-3):
        from l2r.learning.object_trainer import ObjectTrainer
        torch.manual_seed(seed)
        self.cfg = Config(**{**BASE, **VARIANTS[name]})
        self.trainer = ObjectTrainer(self.cfg, imitation=True)
        self.skips_place = not self.trainer.learned_place
        self.reference = Trainer(Config(**BASE, equiv_targets=True))
        self.optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, steps)

    def _push_label(self, action):
        """Demonstrated push -> (direction, length fraction). A 4-tuple is already (direction, row, col, fraction); a 3-tuple's first
        entry is direction * L + length index over the expert's selectable lengths (`ScriptedExpert.CONTINUOUS_LENGTHS`)."""
        from l2r.demos import ScriptedExpert
        if len(action) == 4:
            return int(action[0]), float(action[3])
        lengths, (low, high) = ScriptedExpert.CONTINUOUS_LENGTHS, self.cfg.push_length_range
        if not self.trainer.continuous:
            return int(action[0]), None
        direction, index = divmod(int(action[0]), len(lengths))
        return direction, float(np.clip((lengths[index] - low) / (high - low), 0, 1))

    def train_step(self, transitions):
        from l2r.learning.object_centric import collate, collate_place, place_index_of, slot_of
        t = self.trainer
        losses = []
        acts, places = [], []
        for tr in transitions:
            state = t.info(tr.state)
            if tr.primitive_action == 'place':
                index = place_index_of(state, tr.action) if t.learned_place else None
                if index is not None:
                    places.append((state, index))
                continue
            action, fraction = tr.action, None
            if tr.primitive_action == 'push':
                direction, fraction = self._push_label(tr.action)
                action = (direction, tr.action[1], tr.action[2])
            slot = slot_of(state, tr.primitive_action, action)
            if slot is not None:
                acts.append((state, slot, fraction))
        if acts:
            tokens, n_objects, valid = collate([s for s, _, _ in acts], DEVICE)
            if t.continuous:
                q, actor, _ = t.model.forward_full(tokens, n_objects)
            else:
                q = t.model(tokens, n_objects)
            index = torch.tensor([slot for _, slot, _ in acts], device=DEVICE)
            arange = torch.arange(len(acts), device=DEVICE)
            flat = index[:, 0] * q.shape[2] + index[:, 1]
            valid = valid.clone()
            valid[arange, index[:, 0], index[:, 1]] = True
            losses.append(torch.nn.functional.cross_entropy(q.masked_fill(~valid, -1e9).flatten(1), flat))
            pushes = [k for k, (_, slot, fraction) in enumerate(acts) if slot[1] > 0 and fraction is not None]
            if t.continuous and pushes:
                k = torch.tensor(pushes, device=DEVICE)
                predicted = actor[k, index[k, 0], index[k, 1] - 1]
                target = torch.tensor([acts[j][2] for j in pushes], device=DEVICE)
                losses.append(10.0 * torch.nn.functional.mse_loss(predicted, target))
        if places:
            tokens, n_objects, _ = collate([s for s, _ in places], DEVICE)
            place_tokens, place_valid = collate_place([s for s, _ in places], DEVICE)
            values = t.model.place_values(tokens, n_objects, place_tokens).masked_fill(~place_valid, -1e9)
            losses.append(torch.nn.functional.cross_entropy(values, torch.tensor([i for _, i in places], device=DEVICE)))
        if not losses:
            return 0.0
        loss = sum(losses)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(t.model.parameters(), 5.0)
        self.optimizer.step()
        self.scheduler.step()
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, state):
        from l2r.learning.object_centric import collate, collate_place
        t = self.trainer
        info = t.info(state)
        if t.learned_place and state.get('held_class'):
            if len(info['place_actions']) == 0:
                col, row = state['heuristic_place']  # no free cell the object fits in: fall back to the rule
                return 'place', (0, int(row), int(col))
            tokens, n_objects, _ = collate([info], DEVICE)
            values = t.model.place_values(tokens, n_objects, collate_place([info], DEVICE)[0])[0]
            return 'place', tuple(int(v) for v in info['place_actions'][int(values.argmax())])
        if len(info['instances']) == 0:
            return 'grasp', (0, 0, 0)
        tokens, n_objects, valid = collate([info], DEVICE)
        if t.continuous:
            q, actor, _ = t.model.forward_full(tokens, n_objects)
            q, actor = q[0].masked_fill(~valid[0], -1e9), actor[0]
        else:
            q = t.model(tokens, n_objects).masked_fill(~valid, -1e9)[0]
        i, slot = divmod(int(q.flatten().argmax()), q.shape[1])
        rot, row, col = (int(v) for v in info['actions'][i, slot])
        if slot == 0:
            return 'grasp', (rot, row, col)
        if t.continuous:
            return 'push', (slot - 1, row, col, float(actor[i, slot - 1]))
        return 'push', (rot, row, col)


def make_env(cfg, num_obj, max_actions=20):
    """A simulator environment with the benchmark's (legacy, fast) physics."""
    from types import SimpleNamespace

    from l2r.config import SimParams
    from l2r.envs.rearrange import RearrangeEnv
    flags = SimpleNamespace(physics='legacy', randomize=False, step_distance=0.002, move_budget=1000, sim_quality='legacy',
                            camera_scale=1.0, force_limit=300.0, force_sense_x=None, adaptive_step=True, velocity_scale=1.0,
                            joint_tolerance=3e-3)
    cfg.max_actions = max_actions
    return RearrangeEnv(cfg, num_obj=num_obj, params=SimParams.from_args(flags))


def rollout(variant, episodes, num_obj, max_actions=20, base_seed=7000, env=None):
    """Closed-loop evaluation of a trained variant in the simulator (greedy): success rate and actions per episode. This is the real
    test: it does not care WHICH valid strategy the policy uses (other objects, push or grasp, other order), only whether it sorts."""
    env = env or make_env(variant.cfg, num_obj, max_actions)
    outcomes = []
    for episode in range(episodes):
        obs = env.reset(num_obj=num_obj, seed=base_seed + episode)
        done, info = False, {'sorted': False, 'actions': 0}
        while not done:
            primitive, pix = variant.predict(obs)
            obs, _, terminated, truncated, info = env.step({'primitive': primitive, 'pix': pix})
            done = terminated or truncated
        outcomes.append((bool(info['sorted']), info['actions']))
    return {'rollout success': float(np.mean([o[0] for o in outcomes])), 'rollout actions': float(np.mean([o[1] for o in outcomes]))}


def split(demos, fraction=0.2):
    cut = int(len(demos) * (1 - fraction))
    return demos[:cut], demos[cut:]


def evaluate(variant, test):
    ref, counts = variant.reference, dict(n=0, primitive=0, exact=0, near=0, inclass=0, pix_n=0, sensible=0, sensible_n=0)
    for tr in test:
        if getattr(variant, 'skips_place', False) and tr.primitive_action == 'place':
            continue  # this agent does not decide placements
        primitive, pix = variant.predict(tr.state)
        counts['n'] += 1
        judged = sensible_action(tr.state, primitive, pix) if 'regions' in tr.state else None
        if judged is not None:
            counts['sensible_n'] += 1
            counts['sensible'] += judged
        counts['primitive'] += primitive == tr.primitive_action
        if primitive != tr.primitive_action:
            continue
        counts['pix_n'] += 1
        a = tr.action
        same_channel = primitive != 'push' or pix[0] == a[0]
        counts['exact'] += same_channel and pix[1] == a[1] and pix[2] == a[2]
        counts['near'] += same_channel and abs(pix[1] - a[1]) <= 3 and abs(pix[2] - a[2]) <= 3
        h, w = tr.state['depthmap'].shape[:2]
        top, left = ref._pads(h, w)
        o = ref.offset_px
        prim = torch.tensor([ref.PRIMITIVES[tr.primitive_action]], device=DEVICE)
        cells = ref._equivalent_cells([tr.state], prim, torch.tensor([a[1] + top + o], device=DEVICE),
                                      torch.tensor([a[2] + left + o], device=DEVICE), h, w, top, left)[0]
        counts['inclass'] += same_channel and bool(cells[pix[1] + top + o, pix[2] + left + o] > 0)
    n, p = max(counts['n'], 1), max(counts['pix_n'], 1)
    return {'primitive': counts['primitive'] / n, 'in-class': counts['inclass'] / p, 'near': counts['near'] / p,
            'exact': counts['exact'] / p, 'sensible': counts['sensible'] / max(counts['sensible_n'], 1)}


def make_variant(name, steps, seed, batch):
    agent = VARIANTS[name].get('agent', 'dense')
    return {'object': ObjectVariant, 'policy_critic': PolicyVariant}.get(agent, DenseVariant)(name, steps, seed, batch)


def run(demos_path, names, seeds, steps, batch, fraction=1.0, rollouts=0, rollout_objects=3, log=print):
    demos, stats = load_demos(demos_path)
    train, test = split(demos)
    train_eval = train[::4]  # a fixed quarter of the training transitions: accuracy on what the network was trained on
    train = train[:max(int(len(train) * fraction), batch)]
    log('%d training / %d held-out transitions (expert %.2f success over %d episodes)'
        % (len(train), len(test), stats['success_rate'], stats['episodes']))
    results = {}
    for name in names:
        runs = []
        for seed in range(seeds):
            variant = make_variant(name, steps, seed, batch)
            rng = np.random.default_rng(seed)
            losses = []
            for step in range(steps):
                losses.append(variant.train_step([train[i] for i in rng.integers(len(train), size=batch)]))
            metrics = evaluate(variant, test)
            metrics['loss'] = float(np.mean(losses[-100:]))
            seen = evaluate(variant, train_eval)
            metrics['train in-class'], metrics['train sensible'] = seen['in-class'], seen['sensible']
            if rollouts:
                metrics.update(rollout(variant, rollouts, rollout_objects))
            runs.append(metrics)
            log('  %-22s seed %d: %s' % (name, seed, ' '.join('%s %.3f' % kv for kv in metrics.items())))
            del variant
            torch.cuda.empty_cache()
        results[name] = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in runs[0]}
    extra = ' rollout success | rollout actions |' if rollouts else ''
    log('\n| variant | primitive | in-class | sensible | near (3 cells) | exact | train loss | train in-class | train sensible |' + extra +
        '\n|---|---|---|---|---|---|---|---|---|' + ('---|---|' if rollouts else ''))
    for name, m in results.items():
        keys = ('primitive', 'in-class', 'sensible', 'near', 'exact', 'loss', 'train in-class', 'train sensible') + \
            (('rollout success', 'rollout actions') if rollouts else ())
        log('| %s | %s |' % (name, ' | '.join('%.3f +- %.3f' % m[k] for k in keys)))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description='Supervised architecture benchmark on expert demonstrations.')
    parser.add_argument('--demos', default='data/demos/expert.pkl')
    parser.add_argument('--variants', nargs='+', default=['baseline', 'skips', 'region', 'skips+region'], choices=sorted(VARIANTS))
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--rollouts', type=int, default=0, help='closed-loop greedy episodes per run in the simulator (0 = skip: slow)')
    parser.add_argument('--rollout_objects', type=int, default=3)
    parser.add_argument('--fraction', type=float, default=1.0, help='fraction of the training transitions to use (data scaling)')
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)
    results = run(args.demos, args.variants, args.seeds, args.steps, args.batch, args.fraction, args.rollouts, args.rollout_objects)
    if args.out:
        json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
