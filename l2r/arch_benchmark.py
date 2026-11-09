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
    'fpn': dict(arch_decoder='fpn'),
    'attn_unet': dict(arch_decoder='attn_unet'),
    'unet_attn4': dict(arch_decoder='unet_attn4'),
    'unet+hybrid': dict(arch_skips=True, backbone='hybrid'),
    'unet+ppm': dict(arch_skips=True, backbone='ppm'),
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


def split(demos, fraction=0.2):
    cut = int(len(demos) * (1 - fraction))
    return demos[:cut], demos[cut:]


def evaluate(variant, test):
    ref, counts = variant.reference, dict(n=0, primitive=0, exact=0, near=0, inclass=0, pix_n=0, sensible=0, sensible_n=0)
    for tr in test:
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


def run(demos_path, names, seeds, steps, batch, fraction=1.0, log=print):
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
            variant = DenseVariant(name, steps, seed, batch)
            rng = np.random.default_rng(seed)
            losses = []
            for step in range(steps):
                losses.append(variant.train_step([train[i] for i in rng.integers(len(train), size=batch)]))
            metrics = evaluate(variant, test)
            metrics['loss'] = float(np.mean(losses[-100:]))
            seen = evaluate(variant, train_eval)
            metrics['train in-class'], metrics['train sensible'] = seen['in-class'], seen['sensible']
            runs.append(metrics)
            log('  %-22s seed %d: %s' % (name, seed, ' '.join('%s %.3f' % kv for kv in metrics.items())))
            del variant
            torch.cuda.empty_cache()
        results[name] = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in runs[0]}
    log('\n| variant | primitive | in-class | sensible | near (3 cells) | exact | train loss | train in-class | train sensible |\n|---|---|---|---|---|---|---|---|---|')
    for name, m in results.items():
        keys = ('primitive', 'in-class', 'sensible', 'near', 'exact', 'loss', 'train in-class', 'train sensible')
        log('| %s | %s |' % (name, ' | '.join('%.3f +- %.3f' % m[k] for k in keys)))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description='Supervised architecture benchmark on expert demonstrations.')
    parser.add_argument('--demos', default='data/demos/expert.pkl')
    parser.add_argument('--variants', nargs='+', default=['baseline', 'skips', 'region', 'skips+region'], choices=sorted(VARIANTS))
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--fraction', type=float, default=1.0, help='fraction of the training transitions to use (data scaling)')
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)
    results = run(args.demos, args.variants, args.seeds, args.steps, args.batch, args.fraction)
    if args.out:
        json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
