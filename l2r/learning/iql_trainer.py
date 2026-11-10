"""Implicit Q-learning (Kostrikov et al. 2022) for the object-centric agent (`--agent object --object_algo iql`).

Offline-to-online without the max over unseen actions that made double DQN overestimate (the re-grasp loops):
* V(s) regresses an upper EXPECTILE of the target Q-values of the actions actually taken (no argmax anywhere);
* Q(s, a) regresses r + g V(s') (g: the stored n-step discount, or the one-step rule; twin heads each, targets clipped to the Q range);
* the policy (logits over the object-action slots and the place candidates) is fitted by advantage-weighted cross-entropy on the
  taken actions, weights exp(beta (Q_target(s, a) - V(s))) capped at 100; the continuous push length by advantage-weighted
  regression of the actor towards the executed length.
Pre-training on the demonstrations (`Trainer.pretrain`) is the offline phase; the RL loop the online phase. Acting follows the
policy logits (epsilon-greedy exploration as for the other agents)."""
import numpy as np
import torch
import torch.nn.functional as F

from l2r.learning.models import DEVICE
from l2r.learning.object_centric import collate, collate_place, place_index_of, slot_of
from l2r.learning.object_trainer import ObjectTrainer

MAX_WEIGHT = 100.0


def expectile_loss(diff, tau):
    """Asymmetric squared loss: |tau - 1(diff < 0)| diff^2 (diff = target - prediction)."""
    weight = torch.where(diff < 0, torch.full_like(diff, 1.0 - tau), torch.full_like(diff, tau))
    return (weight * diff ** 2).mean()


class IQLObjectTrainer(ObjectTrainer):
    def _acting_scores(self, net, tokens, counts, q):
        return net.policy_logits(tokens, counts).float().cpu().numpy()

    def _acting_place_scores(self, net, tokens, counts, place_tokens):
        return net.place_policy_logits(tokens, counts, place_tokens)

    def train_step(self, transitions, weights=None, demo=None):
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
            tokens_n, counts_n, _ = collate(nexts, DEVICE)
            label = (reward + gamma * self.model.value(tokens_n, counts_n) * (1.0 - done)).clamp(self.q_min, self.q_max)

        heads = 2 if self.twin else 1
        q_pred = torch.zeros(heads, len(keep), device=DEVICE)
        q_target = torch.zeros(len(keep), device=DEVICE)
        log_prob = torch.zeros(len(keep), device=DEVICE)
        length_error = torch.zeros(len(keep), device=DEVICE)
        outcome_pred = []
        neg = -1e9
        slot_rows = [k for k, i in enumerate(keep) if kinds[i][0] == 'slot']
        place_rows = [k for k, i in enumerate(keep) if kinds[i][0] == 'place']
        if slot_rows:
            items = [keep[k] for k in slot_rows]
            tokens, counts, valid = collate([states[i] for i in items], DEVICE)
            index = torch.tensor([kinds[i][1] for i in items], device=DEVICE)
            arange = torch.arange(len(items), device=DEVICE)
            rows = torch.tensor(slot_rows, device=DEVICE)
            if self.continuous:
                exec_len = torch.full((len(items), valid.shape[1], 2), 0.5, device=DEVICE)
                for k, i in enumerate(items):
                    if transitions[i].primitive_action == 'push':
                        exec_len[k, kinds[i][1][0], kinds[i][1][1] - 1] = float(transitions[i].action[3])
                qh, actor, _ = self.model.forward_full(tokens, counts, lengths=exec_len, heads=True)
                with torch.no_grad():
                    qt = self.model_target.forward_full(tokens, counts, lengths=exec_len, heads=True)[0]
                pushes = [k for k, i in enumerate(items) if transitions[i].primitive_action == 'push']
                if pushes:
                    pk = torch.tensor(pushes, device=DEVICE)
                    executed = exec_len[pk, index[pk, 0], index[pk, 1] - 1]
                    length_error = length_error.index_put((rows[pk],), (actor[pk, index[pk, 0], index[pk, 1] - 1] - executed) ** 2)
            else:
                qh = self.model.q_heads(tokens, counts)
                with torch.no_grad():
                    qt = self.model_target.q_heads(tokens, counts)
            for h in range(heads):
                q_pred[h] = q_pred[h].index_put((rows,), qh[h, arange, index[:, 0], index[:, 1]])
            q_target = q_target.index_put((rows,), qt.min(0).values[arange, index[:, 0], index[:, 1]])
            logits = self.model.policy_logits(tokens, counts)
            valid = valid.clone()
            valid[arange, index[:, 0], index[:, 1]] = True
            flat = F.log_softmax(logits.masked_fill(~valid, neg).flatten(1), dim=1)
            log_prob = log_prob.index_put((rows,), flat[arange, index[:, 0] * logits.shape[2] + index[:, 1]])
            if self.outcome:
                outcome_pred.append((slot_rows, self.model.outcomes(tokens, counts)[arange, index[:, 0], index[:, 1]]))
        if place_rows:
            items = [keep[k] for k in place_rows]
            group = [states[i] for i in items]
            tokens, counts, _ = collate(group, DEVICE)
            place_tokens, place_valid = collate_place(group, DEVICE)
            index = torch.tensor([kinds[i][1] for i in items], device=DEVICE)
            arange = torch.arange(len(items), device=DEVICE)
            rows = torch.tensor(place_rows, device=DEVICE)
            if self.outcome:
                values_h, place_outcome = self.model.place_values(tokens, counts, place_tokens, with_outcome=True, heads=True)
                outcome_pred.append((place_rows, place_outcome[arange, index]))
            else:
                values_h = self.model.place_values(tokens, counts, place_tokens, heads=True)
            with torch.no_grad():
                values_t = self.model_target.place_values(tokens, counts, place_tokens, heads=True).min(0).values
            for h in range(heads):
                q_pred[h] = q_pred[h].index_put((rows,), values_h[h, arange, index])
            q_target = q_target.index_put((rows,), values_t[arange, index])
            logits = self.model.place_policy_logits(tokens, counts, place_tokens).masked_fill(~place_valid, neg)
            log_prob = log_prob.index_put((rows,), F.log_softmax(logits, dim=1)[arange, index])

        tokens_s, counts_s, _ = collate([states[i] for i in keep], DEVICE)
        value = self.model.value(tokens_s, counts_s)
        v_loss = expectile_loss(q_target - value, self.cfg.iql_expectile)
        advantage = (q_target - value).detach()
        weight = torch.exp(self.cfg.iql_beta * advantage).clamp(max=MAX_WEIGHT)
        pi_loss = -(weight * log_prob).mean() + 10.0 * (weight * length_error).mean()
        per_sample = F.smooth_l1_loss(q_pred, label[None].expand_as(q_pred), reduction='none').sum(0)
        if weights is not None:
            per_sample = per_sample * torch.as_tensor(np.asarray(weights)[keep], device=DEVICE, dtype=per_sample.dtype)
        loss = per_sample.mean() + v_loss + pi_loss
        outcome_loss = self._outcome_loss(outcome_pred, [transitions[i] for i in keep]) if self.outcome else None
        if outcome_loss is not None:
            loss = loss + self.cfg.outcome_weight * outcome_loss
        self.last_td_loss = float(per_sample.detach().mean())
        self.last_td[keep] = (q_pred.mean(0) - label).detach().cpu().numpy()
        self.last_v_loss, self.last_pi_loss = float(v_loss.detach()), float(pi_loss.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.last_q = float(q_pred.mean().detach())
        return float(loss.detach())
