"""Policy heatmap + Q critic.

The dense Q-network gets a second set of heads, a **policy** (logits per cell, per primitive), trained by cross-entropy on the
demonstrations: this is a behaviour-cloning objective, which suits a pixel map better than DQfD's margin loss (neighbouring equivalent cells
get soft, shared probability; with `--equiv_targets` the target is spread over the equivalence class of the expert's cell). The critic is
trained by TD as before. Acting: the policy proposes its top-K feasible cells per primitive and the critic ranks them (Transporter /
CLIPort style proposals with a value function on top); exploration draws among the proposals, so it stays near what the demonstrations
show. The output has the same dense interface as every other agent (cells that are not proposed carry -1000).
"""
import numpy as np
import torch

from l2r.learning.models import DEVICE, QNet
from l2r.learning.trainer import QMaps, Trainer

NEG = -1e9


class PolicyCriticTrainer(Trainer):
    def __init__(self, cfg=None, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if self.cfg.hierarchy or self.belief == 'recurrent':
            raise ValueError('the policy-critic agent supports neither --hierarchy nor --belief recurrent')

    # ------------------------------------------------------------------ policy logits and behaviour cloning
    def policy_logits(self, states, grad):
        """Flat logits over [push channels, pick channels, place], their feasibility mask, and the geometry needed to index them."""
        h, w = states[0]['depthmap'].shape[:2]
        x = torch.stack([self.prepare(s) for s in states]).to(DEVICE)
        with torch.set_grad_enabled(grad), torch.autocast(device_type=DEVICE.type, enabled=self.amp):
            out = self.model.forward_ext(x, None, None, self._held(states))
        push, pick, place = out.policy
        push, pick = push.float(), pick.float()
        place = place.float() if place is not None else torch.full_like(push[:, :1], NEG)
        push_mask, pick_mask, place_mask = self._masks_padded(states, h, w)
        logits = torch.cat([push.flatten(1), pick.flatten(1), place.flatten(1)], dim=1)
        masks = torch.cat([push_mask.expand_as(push).flatten(1), pick_mask.expand_as(pick).flatten(1),
                           place_mask.expand_as(place).flatten(1)], dim=1)
        return logits, masks, (push.shape[1], pick.shape[1]), push.shape[-2:], (h, w)

    def bc_loss(self, transitions):
        """Cross-entropy of the policy over the feasible actions against the demonstrated action (spread over its equivalence class
        with --equiv_targets)."""
        states = [t.state for t in transitions]
        logits, masks, (n_push, n_pick), canvas, (h, w) = self.policy_logits(states, grad=True)
        top, left = self._pads(h, w)
        o = self.offset_px
        prim = torch.tensor([self.PRIMITIVES[t.primitive_action] for t in transitions], device=DEVICE)
        rot = torch.tensor([int(t.action[0]) for t in transitions], device=DEVICE)
        rows = torch.tensor([int(t.action[1]) + o + top for t in transitions], device=DEVICE)
        cols = torch.tensor([int(t.action[2]) + o + left for t in transitions], device=DEVICE)
        size = canvas[0] * canvas[1]
        channel = torch.where(prim == QNet.PUSH, rot, torch.where(prim == QNet.PICK, rot.clamp(max=n_pick - 1) + n_push,
                                                                   torch.zeros_like(rot) + n_push + n_pick))
        expert = channel * size + rows * canvas[1] + cols
        target = torch.zeros_like(logits).scatter(1, expert[:, None], 1.0)
        if self.cfg.equiv_targets:
            cells = self._equivalent_cells(states, prim, rows, cols, h, w, top, left).flatten(1)
            target = torch.zeros_like(logits).scatter(1, torch.arange(size, device=DEVICE)[None].expand(len(states), -1) +
                                                      (channel * size)[:, None], cells)
            target = target / target.sum(1, keepdim=True).clamp(min=1e-9)
        masks = masks | (target > 0)
        log_p = torch.log_softmax(logits.masked_fill(~masks, NEG), dim=1)
        return -(target * log_p).sum(1).mean()

    def _extra_loss(self, x, states, transitions, demo, inputs, h, w):
        if demo is None or not np.any(demo):
            return 0.0
        chosen = [transitions[i] for i in np.flatnonzero(np.asarray(demo))]
        loss = self.cfg.policy_bc_weight * self.bc_loss(chosen)
        self.last_bc = float(loss.detach())
        return loss

    # ------------------------------------------------------------------ acting: proposals ranked by the critic
    @staticmethod
    def top_k_cells(scores, valid, k):
        """Boolean mask of the k highest-scoring valid cells (all valid cells if there are fewer)."""
        flat = np.where(valid, scores, -np.inf).ravel()
        keep = np.zeros(flat.shape, bool)
        count = int(valid.sum())
        if count:
            keep[np.argpartition(-flat, min(k, count) - 1)[:min(k, count)]] = True
        return keep.reshape(scores.shape) & valid

    def forward_batch(self, obs_list, network='policy'):
        net = self.model if network == 'policy' else self.model_target
        x = torch.stack([self.prepare(o) for o in obs_list]).to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=self.amp):
            out = net.forward_ext(x, None, None, self._held(obs_list))
        h, w = obs_list[0]['depthmap'].shape[:2]
        crop = lambda q: self._crop(q.float(), h, w).cpu().numpy()
        push, pick = crop(out.push), crop(out.pick)
        place = crop(out.place) if out.place is not None else np.zeros_like(pick)
        pi_push, pi_pick = crop(out.policy[0]), crop(out.policy[1])
        pi_place = crop(out.policy[2]) if out.policy[2] is not None else None
        k = self.cfg.policy_topk
        for b, obs in enumerate(obs_list):
            depth = obs['depthmap']
            push[b] = np.where(self.top_k_cells(pi_push[b], self.valid_mask(depth, 'push'), k), push[b], -1000.0)
            pick_valid = np.broadcast_to(self.valid_mask(depth, 'grasp')[None], pick[b].shape)
            pick[b] = np.where(self.top_k_cells(pi_pick[b], pick_valid, k), pick[b], -1000.0)
            if pi_place is not None and obs.get('held_class'):
                free = np.broadcast_to(self.place_valid(obs)[None], place[b].shape)
                place[b] = np.where(self.top_k_cells(pi_place[b], free, k), place[b], -1000.0)
        return QMaps(push, pick, place, None)
