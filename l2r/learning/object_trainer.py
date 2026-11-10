"""Trainer of the object-centric Q-function (double DQN over object-action slots). Drop-in for `Trainer`: same interface, the dense Q-maps
it returns have values only at the candidate pixels of the visible objects (see `object_centric`)."""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from l2r.learning.models import DEVICE
from l2r.learning.object_centric import (ObjectQNet, ObjectQNetContinuous, build_state, collate, collate_place, place_index_of, rasterize,
                                         slot_of)
from l2r.learning.trainer import QMaps, Trainer

CACHE_SIZE = 2048  # extracted states kept (the replay buffer re-uses observations many times)


class ObjectTrainer(Trainer):
    def __init__(self, cfg=None, is_testing=False, load_snapshot=False, snapshot=None, snapshot_target=None, imitation=False):
        """imitation: the network is trained by a supervised loss (`arch_benchmark.ObjectVariant`, DAgger) instead of `train_step`
        (kept for the call sites; both support learned placement)."""
        super().__init__(cfg, is_testing, False, None, None)  # builds the dense network we replace below: cheap
        self.learned_place = self.cfg.placement == 'learned'
        self.outcome = self.cfg.outcome_weight > 0  # outcome heads (executed / invalid / changed), trained from free labels
        self.twin = bool(self.cfg.twin_q)  # clipped double Q: two value heads, targets from their minimum
        self.iql = self.cfg.object_algo == 'iql'
        self.n_slots = 1 + 2 * len(self.cfg.push_lengths)
        self.continuous = bool(self.cfg.hierarchy)  # --agent object --hierarchy: a continuous push length (actor + length-conditioned critic)
        if self.continuous:
            self.n_slots = 3
            make = lambda: ObjectQNetContinuous(self.cfg.object_dim, self.cfg.object_layers, placement=self.learned_place,
                                                     outcome=self.outcome, twin=self.twin, iql=self.iql).to(DEVICE)
        else:
            make = lambda: ObjectQNet(self.n_slots, self.cfg.object_dim, self.cfg.object_layers, placement=self.learned_place,
                                      outcome=self.outcome, twin=self.twin, iql=self.iql).to(DEVICE)
        self.model, self.model_target = make(), make()
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot, map_location=DEVICE))
            self.model_target.load_state_dict(torch.load(snapshot_target or snapshot, map_location=DEVICE))
        else:
            self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval()
        for p in self.model_target.parameters():
            p.requires_grad_(False)
        if self.continuous:  # the actor learns slower than the critic (it would otherwise saturate the sigmoid on a still-monotone critic)
            actor_ids = {id(p) for p in self.model.actor.parameters()}
            rest = [p for p in self.model.parameters() if id(p) not in actor_ids]
            groups = [{'params': rest}, {'params': list(self.model.actor.parameters()), 'lr': self.cfg.learning_rate * 0.1}]
            self.optimizer = torch.optim.Adam(groups, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.lr_decay)
        self._cache = OrderedDict()

    # ------------------------------------------------------------------ state extraction
    def info(self, obs):
        """`build_state` of an observation, cached by object identity (the cache keeps the observation alive: ids are not re-used)."""
        key = id(obs)
        if key in self._cache:
            return self._cache[key][1]
        state = build_state(obs, self.push_lengths_px[:1] if self.continuous else self.push_lengths_px)
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
            if self.continuous:
                q, actor, _ = net.forward_full(tokens, n_objects)
                q, actor = q.float().cpu().numpy(), actor.float().cpu().numpy()
            else:
                q = net(tokens, n_objects).float().cpu().numpy()
            q = self._acting_scores(net, tokens, n_objects, q)
            risk = self.cfg.outcome_penalty * abs(self.cfg.reward_invalid) if self.outcome and network == 'policy' else 0.0
            if risk:  # act cautiously: subtract the expected cost of making the scene invalid (the outcome model's P(invalid))
                q = q - risk * torch.sigmoid(net.outcomes(tokens, n_objects)[..., 1]).float().cpu().numpy()
        h, w = obs_list[0]['depthmap'].shape[:2]
        shape = (h - 2 * self.offset_px, w - self.offset_px)
        push, pick = zip(*[rasterize(s, q[b], shape, 2 * len(self.cfg.push_lengths)) for b, s in enumerate(states)])
        push, pick = np.stack(push), np.stack(pick)
        place = np.full_like(pick, -1000.0)
        if self.learned_place and any(len(s['place_tokens']) for s in states):
            with torch.no_grad():
                if risk:
                    values, outcome = net.place_values(tokens, n_objects, collate_place(states, DEVICE)[0], with_outcome=True)
                    values = (values - risk * torch.sigmoid(outcome[..., 1])).float().cpu().numpy()
                else:
                    values = self._acting_place_scores(net, tokens, n_objects, collate_place(states, DEVICE)[0]).float().cpu().numpy()
            for b, s in enumerate(states):
                for p, (_, row, col) in enumerate(s['place_actions']):
                    place[b, 0, row, col] = values[b, p]
        if not self.continuous:
            return QMaps(push, pick, place, None)
        lengths = np.full(push.shape, 0.5, np.float32)  # the actor's length at every candidate push pixel (0.5 elsewhere)
        for b, state in enumerate(states):
            for i in range(len(state['instances'])):
                for slot in (1, 2):
                    if state['valid'][i, slot]:
                        _, row, col = state['actions'][i, slot]
                        lengths[b, slot - 1, row, col] = actor[b, i, slot - 1]
        prim = np.stack([push.reshape(len(states), -1).max(1), pick.reshape(len(states), -1).max(1), np.full(len(states), -1000.0)], axis=1)
        prim[:, 2] = place.reshape(len(states), -1).max(1)
        return QMaps(push, pick, place, None, lengths, prim.astype(np.float32))

    def _acting_scores(self, net, tokens, counts, q):
        """What the greedy action maximises over the object-action slots (numpy (B, N, S)): the Q-values (subclasses: a policy)."""
        return q

    def _acting_place_scores(self, net, tokens, counts, place_tokens):
        return net.place_values(tokens, counts, place_tokens)

    # ------------------------------------------------------------------ learning
    def _future(self, nexts):
        """Double-DQN value of each next state (B,): the policy network picks the action, the target network values it. A next state
        that holds an object (learned placement) is valued over its place candidates, any other over its object-action slots."""
        neg = -1e9
        future = torch.zeros(len(nexts), device=DEVICE)
        holding = [k for k, s in enumerate(nexts) if self.learned_place and len(s['place_tokens'])]
        acting = [k for k in range(len(nexts)) if k not in set(holding)]
        if acting:
            group = [nexts[k] for k in acting]
            tokens_n, n_n, valid_n = collate(group, DEVICE)
            if self.continuous:  # the target critic values the policy actor's lengths
                pol_q, pol_actor, _ = self.model.forward_full(tokens_n, n_n)
                pol = pol_q.masked_fill(~valid_n, neg).flatten(1)
                tgt = self.model_target.forward_full(tokens_n, n_n, lengths=pol_actor, heads=True)[0]
            else:
                pol = self.model(tokens_n, n_n).masked_fill(~valid_n, neg).flatten(1)
                tgt = self.model_target.q_heads(tokens_n, n_n)
            tgt = tgt.min(0).values.flatten(1)  # clipped double Q (a single head without --twin_q)
            value = tgt.gather(1, pol.argmax(1, keepdim=True))[:, 0]
            future[acting] = torch.where(pol.max(1).values < neg / 2, torch.zeros_like(value), value)
        if holding:
            group = [nexts[k] for k in holding]
            tokens_n, n_n, _ = collate(group, DEVICE)
            place_n, place_valid = collate_place(group, DEVICE)
            pol = self.model.place_values(tokens_n, n_n, place_n).masked_fill(~place_valid, neg)
            tgt = self.model_target.place_values(tokens_n, n_n, place_n, heads=True).min(0).values
            future[holding] = tgt.gather(1, pol.argmax(1, keepdim=True))[:, 0]
        return future

    @staticmethod
    def outcome_labels(transition):
        """(executed, invalid, changed) of the action, from the outcome record in the next observation, or None (older data)."""
        record = transition.state.get('_outcome') if isinstance(transition.state, dict) and '_outcome' in transition.state else (
            transition.next_state.get('last_action') if isinstance(transition.next_state, dict) else None)
        if record is None or len(record) < 7:
            return None
        record = np.asarray(record, np.float32)
        return record[4], record[6], min(10.0 * float(record[5]), 1.0)

    def _outcome_loss(self, predictions, transitions):
        """Binary cross-entropy of the outcome logits of the taken actions against the free labels (rows without labels skipped)."""
        logits, targets = [], []
        for rows, pred in predictions:
            for k, row in enumerate(rows):
                labels = self.outcome_labels(transitions[row])
                if labels is not None:
                    logits.append(pred[k])
                    targets.append(labels)
        if not logits:
            return None
        return F.binary_cross_entropy_with_logits(torch.stack(logits).float(), torch.tensor(np.asarray(targets), device=DEVICE))

    def train_step(self, transitions, weights=None, demo=None):
        """Double DQN on the taken object-action slot or (learned placement) place candidate; for demonstration samples also a large
        margin over the other valid slots / candidates. Samples whose action matches no candidate are skipped (no gradient). As in the
        dense trainer, the grasp -> place step (the next state holds the object) is not discounted: pick and place are one action."""
        B = len(transitions)
        states = [self.info(t.state) for t in transitions]
        kinds = []
        for s, t in zip(states, transitions):
            if t.primitive_action == 'place':
                p = place_index_of(s, t.action) if self.learned_place else None
                kinds.append(None if p is None else ('place', p))
            else:
                slot = slot_of(s, t.primitive_action, t.action)
                kinds.append(None if slot is None else ('slot', slot))
        keep = [i for i, k in enumerate(kinds) if k is not None]
        self.last_td = np.zeros(B, np.float32)
        if not keep:
            return 0.0
        nexts = [self.info(transitions[i].next_state) for i in keep]
        reward = torch.tensor([float(transitions[i].reward) for i in keep], device=DEVICE)
        done = torch.tensor([float(transitions[i].done) for i in keep], device=DEVICE)
        gamma = torch.tensor([float(transitions[i].state['_nstep_discount']) if '_nstep_discount' in transitions[i].state
                              else 1.0 if (self.learned_place and len(n['place_tokens']) and kinds[i][0] == 'slot')
                              else self.future_reward_discount for i, n in zip(keep, nexts)], device=DEVICE)
        with torch.no_grad():
            label = (reward + gamma * self._future(nexts) * (1.0 - done)).clamp(self.q_min, self.q_max)

        n_heads = 2 if self.twin else 1
        pred = torch.zeros(len(keep), device=DEVICE)
        pred_heads = torch.zeros(n_heads, len(keep), device=DEVICE)  # every head regresses the same (clipped) target
        margins, outcome_pred = [], []
        flags = np.zeros(len(keep), bool) if demo is None else np.asarray(demo)[keep].astype(bool)
        neg = -1e9
        actor_loss = None
        slot_rows = [k for k, i in enumerate(keep) if kinds[i][0] == 'slot']
        place_rows = [k for k, i in enumerate(keep) if kinds[i][0] == 'place']
        if slot_rows:
            items = [keep[k] for k in slot_rows]
            tokens, n_objects, valid = collate([states[i] for i in items], DEVICE)
            index = torch.tensor([kinds[i][1] for i in items], device=DEVICE)
            arange = torch.arange(len(items), device=DEVICE)
            if self.continuous:
                exec_len = torch.full((len(items), valid.shape[1], 2), 0.5, device=DEVICE)
                for k, i in enumerate(items):
                    if transitions[i].primitive_action == 'push':
                        exec_len[k, kinds[i][1][0], kinds[i][1][1] - 1] = float(transitions[i].action[3])
                qh, actor, push_at_actor = self.model.forward_full(tokens, n_objects, lengths=exec_len, heads=True)
                push_valid = valid[..., 1:3]
                actor_loss = -(push_at_actor * push_valid).sum() / push_valid.sum().clamp(min=1)
            else:
                qh = self.model.q_heads(tokens, n_objects)
            q = qh.mean(0)
            taken = q[arange, index[:, 0], index[:, 1]]
            rows_t = torch.tensor(slot_rows, device=DEVICE)
            for h in range(n_heads):
                pred_heads[h] = pred_heads[h].index_put((rows_t,), qh[h, arange, index[:, 0], index[:, 1]])
            if self.outcome:
                outcome_pred.append((slot_rows, self.model.outcomes(tokens, n_objects)[arange, index[:, 0], index[:, 1]]))
            pred = pred.index_put((torch.tensor(slot_rows, device=DEVICE),), taken)
            flagged = torch.as_tensor(flags[slot_rows], device=DEVICE)
            if flagged.any():
                penalty = torch.full_like(q, self.cfg.demo_margin)
                penalty[arange, index[:, 0], index[:, 1]] = 0.0
                valid = valid.clone()
                valid[arange, index[:, 0], index[:, 1]] = True
                best = torch.where(valid, q + penalty, torch.full_like(q, neg)).flatten(1).max(1).values
                margins.append((best - taken)[flagged])
        if place_rows:
            items = [keep[k] for k in place_rows]
            group = [states[i] for i in items]
            tokens, n_objects, _ = collate(group, DEVICE)
            place_tokens, place_valid = collate_place(group, DEVICE)
            if self.outcome:
                values_h, place_outcome = self.model.place_values(tokens, n_objects, place_tokens, with_outcome=True, heads=True)
            else:
                values_h = self.model.place_values(tokens, n_objects, place_tokens, heads=True)
            values = values_h.mean(0)
            index = torch.tensor([kinds[i][1] for i in items], device=DEVICE)
            arange = torch.arange(len(items), device=DEVICE)
            rows_t = torch.tensor(place_rows, device=DEVICE)
            for h in range(n_heads):
                pred_heads[h] = pred_heads[h].index_put((rows_t,), values_h[h, arange, index])
            if self.outcome:
                outcome_pred.append((place_rows, place_outcome[arange, index]))
            taken = values[arange, index]
            pred = pred.index_put((torch.tensor(place_rows, device=DEVICE),), taken)
            flagged = torch.as_tensor(flags[place_rows], device=DEVICE)
            if flagged.any():
                penalty = torch.full_like(values, self.cfg.demo_margin)
                penalty[arange, index] = 0.0
                best = torch.where(place_valid, values + penalty, torch.full_like(values, neg)).max(1).values
                margins.append((best - taken)[flagged])
        per_sample = F.smooth_l1_loss(pred_heads, label[None].expand_as(pred_heads), reduction='none').sum(0)
        outcome_loss = self._outcome_loss(outcome_pred, [transitions[i] for i in keep]) if self.outcome else None
        if weights is not None:
            per_sample = per_sample * torch.as_tensor(np.asarray(weights)[keep], device=DEVICE, dtype=per_sample.dtype)
        loss = per_sample.mean()
        if outcome_loss is not None:
            loss = loss + self.cfg.outcome_weight * outcome_loss
            self.last_outcome_loss = float(outcome_loss.detach())
        if margins:
            margin = torch.cat(margins).mean()
            self.last_margin = float(margin.detach())
            loss = loss + self.cfg.demo_margin_weight * margin
        self.last_td_loss = float(per_sample.detach().mean())
        self.last_td[keep] = (pred - label).detach().cpu().numpy()
        self.optimizer.zero_grad()
        if actor_loss is not None:  # the actor ascends the critic: its own gradient only, the critic is not pushed by it
            actor_params = list(self.model.actor.parameters())
            actor_grads = torch.autograd.grad(actor_loss, actor_params, retain_graph=True, allow_unused=True)
        loss.backward()
        if actor_loss is not None:
            for p, g in zip(actor_params, actor_grads):
                p.grad = g
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.last_q = float(pred.detach().mean())
        return float(loss.detach())
