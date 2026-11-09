"""Trainer of the object-centric Q-function (double DQN over object-action slots). Drop-in for `Trainer`: same interface, the dense Q-maps
it returns have values only at the candidate pixels of the visible objects (see `object_centric`)."""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from l2r.learning.models import DEVICE
from l2r.learning.object_centric import ObjectQNet, build_state, collate, rasterize, slot_of
from l2r.learning.trainer import QMaps, Trainer

CACHE_SIZE = 2048  # extracted states kept (the replay buffer re-uses observations many times)


class ObjectTrainer(Trainer):
    def __init__(self, cfg=None, is_testing=False, load_snapshot=False, snapshot=None, snapshot_target=None):
        super().__init__(cfg, is_testing, False, None, None)  # builds the dense network we replace below: cheap
        if self.cfg.placement != 'heuristic':
            raise ValueError("the object-centric agent uses the heuristic placement (--placement heuristic)")
        self.n_slots = 1 + 2 * len(self.cfg.push_lengths)
        make = lambda: ObjectQNet(self.n_slots, self.cfg.object_dim, self.cfg.object_layers).to(DEVICE)
        self.model, self.model_target = make(), make()
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot, map_location=DEVICE))
            self.model_target.load_state_dict(torch.load(snapshot_target or snapshot, map_location=DEVICE))
        else:
            self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval()
        for p in self.model_target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.lr_decay)
        self._cache = OrderedDict()

    # ------------------------------------------------------------------ state extraction
    def info(self, obs):
        """`build_state` of an observation, cached by object identity (the cache keeps the observation alive: ids are not re-used)."""
        key = id(obs)
        if key in self._cache:
            return self._cache[key][1]
        state = build_state(obs, self.push_lengths_px)
        self._cache[key] = (obs, state)
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return state

    # ------------------------------------------------------------------ acting
    def forward_batch(self, obs_list, network='policy'):
        """Q-maps in the dense interface: values at the candidate pixels of every object-action slot, -1000 elsewhere."""
        net = self.model if network == 'policy' else self.model_target
        states = [self.info(o) for o in obs_list]
        tokens, n_objects, _ = collate(states, DEVICE)
        with torch.no_grad():
            q = net(tokens, n_objects).float().cpu().numpy()
        h, w = obs_list[0]['depthmap'].shape[:2]
        shape = (h - 2 * self.offset_px, w - self.offset_px)
        push, pick = zip(*[rasterize(s, q[b], shape, 2 * len(self.cfg.push_lengths)) for b, s in enumerate(states)])
        push, pick = np.stack(push), np.stack(pick)
        return QMaps(push, pick, np.zeros_like(pick), None)

    # ------------------------------------------------------------------ learning
    def train_step(self, transitions, weights=None, demo=None):
        """Double DQN on the taken object-action slot; for demonstration samples also a large margin over the other valid slots.
        Samples whose action matches no candidate slot are skipped (they carry no gradient)."""
        B = len(transitions)
        states = [self.info(t.state) for t in transitions]
        slots = [slot_of(s, t.primitive_action, t.action) for s, t in zip(states, transitions)]
        keep = [i for i, slot in enumerate(slots) if slot is not None]
        self.last_td = np.zeros(B, np.float32)
        if not keep:
            return 0.0
        nexts = [self.info(transitions[i].next_state) for i in keep]
        reward = torch.tensor([float(transitions[i].reward) for i in keep], device=DEVICE)
        done = torch.tensor([float(transitions[i].done) for i in keep], device=DEVICE)

        with torch.no_grad():  # double DQN: the policy network picks the next slot, the target network values it
            tokens_n, n_n, valid_n = collate(nexts, DEVICE)
            neg = -1e9
            pol = self.model(tokens_n, n_n).masked_fill(~valid_n, neg).flatten(1)
            tgt = self.model_target(tokens_n, n_n).flatten(1)
            future = tgt.gather(1, pol.argmax(1, keepdim=True))[:, 0]
            future = torch.where(pol.max(1).values < neg / 2, torch.zeros_like(future), future)
            label = (reward + self.future_reward_discount * future * (1.0 - done)).clamp(self.q_min, self.q_max)

        tokens, n_objects, valid = collate([states[i] for i in keep], DEVICE)
        q = self.model(tokens, n_objects)
        index = torch.tensor([slots[i] for i in keep], device=DEVICE)
        arange = torch.arange(len(keep), device=DEVICE)
        pred = q[arange, index[:, 0], index[:, 1]]
        per_sample = F.smooth_l1_loss(pred, label, reduction='none')
        if weights is not None:
            per_sample = per_sample * torch.as_tensor(np.asarray(weights)[keep], device=DEVICE, dtype=per_sample.dtype)
        loss = per_sample.mean()
        if demo is not None and np.any(np.asarray(demo)[keep]):
            flagged = torch.as_tensor(np.asarray(demo)[keep], device=DEVICE)
            penalty = torch.full_like(q, self.cfg.demo_margin)
            penalty[arange, index[:, 0], index[:, 1]] = 0.0
            valid = valid.clone()
            valid[arange, index[:, 0], index[:, 1]] = True
            best = torch.where(valid, q + penalty, torch.full_like(q, neg)).flatten(1).max(1).values
            margin = (best - pred)[flagged].mean()
            self.last_margin = float(margin.detach())
            loss = loss + self.cfg.demo_margin_weight * margin
        self.last_td_loss = float(per_sample.detach().mean())
        self.last_td[keep] = (pred - label).detach().cpu().numpy()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.last_q = float(pred.detach().mean())
        return float(loss.detach())
