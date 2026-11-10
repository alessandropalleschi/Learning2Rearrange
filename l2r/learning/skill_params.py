"""Object agent with PARAMETERISED skills and a flow-matching parameter policy (`--agent object --object_algo iql --skill_params`).

Per visible object two skills, each with continuous parameters x in [-1, 1]^2:
* grasp: x[0] = approach angle / MAX_APPROACH (the approach line rotated about the vertical, see `l2r.sim.primitives`), x[1] unused;
* push:  x[0] = heading u, phi = pi + u pi/2 in [pi/2, 3 pi/2] (never towards the robot), x[1] = 2 * length fraction - 1.
The push start cell follows from the heading: the object's face opposite the motion, PUSH_GAP cells away (`push_start`).

Networks (on the object-centric encoder): a parameter-conditioned critic Q(object, skill, x) (Fourier features, twin heads), a
conditional flow-matching velocity field v(x_t, t | object, skill) for the parameters, and IQL's value V(s) and skill policy
(logits over (object, skill)); placement keeps the discrete place candidates. Training: IQL (expectile V, Q -> r + g V(s'),
advantage-weighted skill policy) plus advantage-weighted conditional flow matching on the executed parameters. Acting: the skill
policy picks (object, skill); its parameters are the best of K flow samples under the critic (IDQL-style)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from l2r.learning.iql_trainer import MAX_WEIGHT, expectile_loss
from l2r.learning.models import DEVICE
from l2r.learning.object_centric import PUSH_GAP, ObjectQNet, collate, collate_place, crop, place_index_of, slot_of
from l2r.learning.trainer import QMaps

MAX_APPROACH_PARAM = math.radians(45)
PARAMS = 2
SKILLS = 2  # 0 grasp, 1 push
SAMPLES = 16
FLOW_STEPS = 8


# ------------------------------------------------------------------ parameter <-> action
def heading_of(u):
    return math.pi + float(u) * math.pi / 2


def heading_param(heading):
    """Heading (rad) -> u in [-1, 1] (pi/2 -> -1, pi -> 0, 3 pi/2 -> 1)."""
    h = (heading - math.pi / 2) % (2 * math.pi) + math.pi / 2  # in [pi/2, 5 pi/2)
    return float(np.clip((h - math.pi) / (math.pi / 2), -1.0, 1.0))


def push_start(inst, heading, shape):
    """Start cell (row, col) of a push of the instance along `heading` (world (cos, sin) = cells (d col, d row)), or None if it is
    outside the map."""
    d_row, d_col = math.sin(heading), math.cos(heading)
    proj = inst.rows * d_row + inst.cols * d_col
    c_row, c_col = inst.rows.mean(), inst.cols.mean()
    back = proj.min() - (c_row * d_row + c_col * d_col) - PUSH_GAP
    row, col = int(round(c_row + back * d_row)), int(round(c_col + back * d_col))
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return None
    return row, col


def action_params(primitive, pix):
    """Normalised parameters x (2,) of an executed action tuple (push (0, row, col, length, heading), grasp (0, row, col, approach))."""
    if primitive == 'push':
        return np.array([heading_param(float(pix[4])), 2.0 * float(pix[3]) - 1.0], np.float32)
    approach = float(pix[3]) if len(pix) > 3 else 0.0
    return np.array([np.clip(approach / MAX_APPROACH_PARAM, -1, 1), 0.0], np.float32)


def push_instance_of(state, pix, tolerance=3):
    """Index of the instance whose push start for the action's heading is the action's cell, or None."""
    heading = float(pix[4]) if len(pix) > 4 else (math.pi / 2 if int(pix[0]) % 2 == 1 else 3 * math.pi / 2)
    best, best_distance = None, tolerance + 1
    for i, inst in enumerate(state['instances']):
        start = push_start(inst, heading, (10 ** 6, 10 ** 6))
        if start is None:
            continue
        distance = max(abs(start[0] - int(pix[1])), abs(start[1] - int(pix[2])))
        if distance < best_distance:
            best, best_distance = i, distance
    return best


def adapt_demo_actions(transitions):
    """Demonstrations -> parameterised action tuples: a push (direction, row, col, fraction) gets heading pi/2 (+y, direction 1) or
    3 pi/2 (-y, direction 0); a grasp gets approach 0."""
    from l2r.learning.memory import Transition
    out = []
    for t in transitions:
        a = t.action
        if t.primitive_action == 'push' and len(a) == 4:
            heading = math.pi / 2 if int(a[0]) == 1 else 3 * math.pi / 2
            t = Transition(t.state, (0, int(a[1]), int(a[2]), float(a[3]), heading), *t[2:])
        elif t.primitive_action == 'grasp' and len(a) == 3:
            t = Transition(t.state, (0, int(a[1]), int(a[2]), 0.0), *t[2:])
        out.append(t)
    return out


# ------------------------------------------------------------------ network
def fourier(x, k=3):
    freq = torch.arange(1, k + 1, device=x.device, dtype=x.dtype) * math.pi
    return torch.cat([x, torch.sin(x[..., None] * freq).flatten(-2), torch.cos(x[..., None] * freq).flatten(-2)], dim=-1)


class ObjectSkillNet(ObjectQNet):
    def __init__(self, dim=64, layers=2, heads=4, placement=False, outcome=False, twin=False):
        super().__init__(n_slots=SKILLS, dim=dim, layers=layers, heads=heads, placement=placement, outcome=outcome, twin=twin, iql=True)
        del self.head
        if twin:
            del self.head_twin
        in_critic = dim + SKILLS + PARAMS * 7
        make_critic = lambda: nn.Sequential(nn.Linear(in_critic, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.critics = nn.ModuleList([make_critic() for _ in range(2 if twin else 1)])
        self.flow = nn.Sequential(nn.Linear(dim + SKILLS + PARAMS + 7, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(),
                                  nn.Linear(dim, PARAMS))

    @staticmethod
    def _onehot(skill, like):
        out = torch.zeros(*like.shape[:-1], SKILLS, device=like.device)
        out[..., skill] = 1.0
        return out

    def critic(self, emb, skill, x, heads=False):
        """emb (..., dim), skill int, x (..., PARAMS) -> Q (...) (mean of the heads; heads=True: (H, ...))."""
        inputs = torch.cat([emb, self._onehot(skill, emb), fourier(x)], dim=-1)
        q = torch.stack([c(inputs)[..., 0] for c in self.critics])
        return q if heads else q.mean(0)

    def velocity(self, emb, skill, x_t, t):
        return self.flow(torch.cat([emb, self._onehot(skill, emb), x_t, fourier(t[..., None])], dim=-1))

    def flow_loss(self, emb, skill, x1, weight):
        """Advantage-weighted conditional flow matching (rectified flow: x_t = (1 - t) x0 + t x1, target velocity x1 - x0)."""
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[:-1], device=x1.device)
        x_t = (1 - t[..., None]) * x0 + t[..., None] * x1
        error = ((self.velocity(emb, skill, x_t, t) - (x1 - x0)) ** 2).sum(-1)
        return (weight * error).mean()

    @torch.no_grad()
    def sample(self, emb, skill, k=SAMPLES, steps=FLOW_STEPS):
        """emb (N, dim) -> (N, k, PARAMS) parameter samples (Euler integration of the flow from Gaussian noise), clipped to [-1, 1]."""
        e = emb[:, None].expand(-1, k, -1)
        x = torch.randn(*e.shape[:-1], PARAMS, device=emb.device)
        for s in range(steps):
            t = torch.full(x.shape[:-1], s / steps, device=emb.device)
            x = x + self.velocity(e, skill, x, t) / steps
        return x.clamp(-1, 1)


# ------------------------------------------------------------------ trainer
def make_skill_trainer_class():
    from l2r.learning.object_trainer import ObjectTrainer

    class SkillObjectTrainer(ObjectTrainer):
        """IQL object agent with parameterised grasp / push skills and a flow-matching parameter policy."""

        def __init__(self, cfg=None, is_testing=False, load_snapshot=False, snapshot=None, snapshot_target=None, imitation=False):
            super().__init__(cfg, is_testing, False, None, None, imitation)  # builds networks we replace below
            make = lambda: ObjectSkillNet(self.cfg.object_dim, self.cfg.object_layers, placement=self.learned_place,
                                          outcome=False, twin=self.twin).to(DEVICE)
            self.model, self.model_target = make(), make()
            if load_snapshot:
                self.model.load_state_dict(torch.load(snapshot, map_location=DEVICE))
                self.model_target.load_state_dict(torch.load(snapshot_target or snapshot, map_location=DEVICE))
            else:
                self.model_target.load_state_dict(self.model.state_dict())
            for p in self.model_target.parameters():
                p.requires_grad_(False)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.lr_decay)
            self._chosen = {}  # (batch index, primitive, row, col) -> parameter tuple, from the last forward_batch

        # ---------------------------------------------------------- acting
        def forward_batch(self, obs_list, network='policy'):
            net = self.model if network == 'policy' else self.model_target
            states = [self.info(o) for o in obs_list]
            h, w = obs_list[0]['depthmap'].shape[:2]
            shape = (h - 2 * self.offset_px, w - self.offset_px)
            push = np.full((len(states), 2, *shape), -1000.0, np.float32)
            pick = np.full((len(states), 1, *shape), -1000.0, np.float32)
            place = np.full((len(states), 1, *shape), -1000.0, np.float32)
            self._chosen = {}
            tokens, counts, _ = collate(states, DEVICE)
            with torch.no_grad():
                emb = net.embed_tokens(tokens, counts)
                logits = net.policy_logits(tokens, counts).float().cpu().numpy()
                for b, state in enumerate(states):
                    n = len(state['instances'])
                    if n == 0:
                        continue
                    for skill in range(SKILLS):
                        samples = net.sample(emb[b, :n], skill)  # (n, K, 2)
                        scores = net.critic(emb[b, :n, None].expand(-1, SAMPLES, -1), skill, samples)
                        best = samples[torch.arange(n), scores.argmax(1)].cpu().numpy()
                        for i, inst in enumerate(state['instances']):
                            x = best[i]
                            if skill == 0:
                                if not state['valid'][i, 0]:
                                    continue
                                _, row, col = state['actions'][i, 0]
                                params = (float(x[0]) * MAX_APPROACH_PARAM,)
                                target = pick[b, 0]
                            else:
                                heading = heading_of(x[0])
                                start = push_start(inst, heading, shape)
                                if start is None or crop(obs_list[b]['depthmap'])[start] > 0.008:
                                    continue
                                row, col = start
                                params = ((float(x[1]) + 1) / 2, heading)
                                target = push[b, 0]
                            if logits[b, i, skill] > target[row, col]:
                                target[row, col] = logits[b, i, skill]
                                self._chosen[(b, 'grasp' if skill == 0 else 'push', int(row), int(col))] = params
                if self.learned_place and any(len(s['place_tokens']) for s in states):
                    values = net.place_policy_logits(tokens, counts, collate_place(states, DEVICE)[0]).float().cpu().numpy()
                    for b, s in enumerate(states):
                        for p, (_, row, col) in enumerate(s['place_actions']):
                            place[b, 0, row, col] = values[b, p]
            return QMaps(push, pick, place, None)

        def action_params(self, b, primitive, pix, explore_prob=0.0):
            """The parameters chosen for this candidate in the last forward_batch, appended to the action tuple (with exploration noise)."""
            if primitive not in ('grasp', 'push'):
                return pix
            params = self._chosen.get((b, primitive, int(pix[1]), int(pix[2])))
            if params is None:
                params = (0.0,) if primitive == 'grasp' else (0.5, math.pi / 2 if np.random.rand() < 0.5 else 3 * math.pi / 2)
            if explore_prob > 0:
                noise = 0.3 * explore_prob / max(self.cfg.eps_start, 1e-6)
                if primitive == 'grasp':
                    params = (float(np.clip(params[0] + noise * MAX_APPROACH_PARAM * np.random.randn(), -MAX_APPROACH_PARAM, MAX_APPROACH_PARAM)),)
                else:
                    params = (float(np.clip(params[0] + noise * np.random.randn(), 0, 1)),
                              heading_of(np.clip(heading_param(params[1]) + noise * np.random.randn(), -1, 1)))
            return (0, int(pix[1]), int(pix[2])) + tuple(params)

        # ---------------------------------------------------------- learning
        def train_step(self, transitions, weights=None, demo=None):
            B = len(transitions)
            states = [self.info(t.state) for t in transitions]
            kinds = []
            for s, t in zip(states, transitions):
                if t.primitive_action == 'place':
                    p = place_index_of(s, t.action) if self.learned_place else None
                    kinds.append(None if p is None else ('place', p))
                elif t.primitive_action == 'grasp':
                    slot = slot_of(s, 'grasp', tuple(t.action[:3]))
                    kinds.append(None if slot is None else ('skill', slot[0], 0))
                elif t.primitive_action == 'push' and len(t.action) >= 5:
                    i = push_instance_of(s, t.action)
                    kinds.append(None if i is None else ('skill', i, 1))
                else:
                    kinds.append(None)
            keep = [k for k, kind in enumerate(kinds) if kind is not None]
            self.last_td = np.zeros(B, np.float32)
            if not keep:
                return 0.0
            nexts = [self.info(transitions[k].next_state) for k in keep]
            reward = torch.tensor([float(transitions[k].reward) for k in keep], device=DEVICE)
            done = torch.tensor([float(transitions[k].done) for k in keep], device=DEVICE)
            gamma = torch.tensor([float(transitions[k].state['_nstep_discount']) if '_nstep_discount' in transitions[k].state
                                  else 1.0 if (self.learned_place and len(n['place_tokens']) and kinds[k][0] == 'skill')
                                  else self.future_reward_discount for k, n in zip(keep, nexts)], device=DEVICE)
            with torch.no_grad():
                label = (reward + gamma * self.model.value(*collate(nexts, DEVICE)[:2]) * (1.0 - done)).clamp(self.q_min, self.q_max)
            heads = 2 if self.twin else 1
            q_pred = torch.zeros(heads, len(keep), device=DEVICE)
            q_target = torch.zeros(len(keep), device=DEVICE)
            log_prob = torch.zeros(len(keep), device=DEVICE)
            flow_terms = []
            neg = -1e9
            skill_rows = [r for r, k in enumerate(keep) if kinds[k][0] == 'skill']
            place_rows = [r for r, k in enumerate(keep) if kinds[k][0] == 'place']
            if skill_rows:
                items = [keep[r] for r in skill_rows]
                tokens, counts, _ = collate([states[k] for k in items], DEVICE)
                emb = self.model.embed_tokens(tokens, counts)
                with torch.no_grad():
                    emb_t = self.model_target.embed_tokens(tokens, counts)
                obj = torch.tensor([kinds[k][1] for k in items], device=DEVICE)
                skill = [kinds[k][2] for k in items]
                x = torch.tensor(np.stack([action_params(transitions[k].primitive_action, transitions[k].action) for k in items]),
                                 device=DEVICE)
                arange = torch.arange(len(items), device=DEVICE)
                e, et = emb[arange, obj], emb_t[arange, obj]
                rows = torch.tensor(skill_rows, device=DEVICE)
                for sk in (0, 1):
                    sel = [r for r, s in enumerate(skill) if s == sk]
                    if not sel:
                        continue
                    sel_t = torch.tensor(sel, device=DEVICE)
                    qh = self.model.critic(e[sel_t], sk, x[sel_t], heads=True)
                    with torch.no_grad():
                        qt = self.model_target.critic(et[sel_t], sk, x[sel_t], heads=True).min(0).values
                    for h in range(heads):
                        q_pred[h] = q_pred[h].index_put((rows[sel_t],), qh[h])
                    q_target = q_target.index_put((rows[sel_t],), qt)
                    flow_terms.append((rows[sel_t], sk, e[sel_t].detach(), x[sel_t]))
                logits = self.model.policy_logits(tokens, counts)
                valid = torch.zeros_like(logits, dtype=torch.bool)
                for r, k in enumerate(items):
                    n = len(states[k]['instances'])
                    valid[r, :n, 0] = torch.as_tensor(states[k]['valid'][:, 0], device=DEVICE)
                    valid[r, :n, 1] = True
                valid[arange, obj, torch.tensor(skill, device=DEVICE)] = True
                flat = F.log_softmax(logits.masked_fill(~valid, neg).flatten(1), dim=1)
                log_prob = log_prob.index_put((rows,), flat[arange, obj * SKILLS + torch.tensor(skill, device=DEVICE)])
            if place_rows:
                items = [keep[r] for r in place_rows]
                group = [states[k] for k in items]
                tokens, counts, _ = collate(group, DEVICE)
                place_tokens, place_valid = collate_place(group, DEVICE)
                index = torch.tensor([kinds[k][1] for k in items], device=DEVICE)
                arange = torch.arange(len(items), device=DEVICE)
                rows = torch.tensor(place_rows, device=DEVICE)
                values_h = self.model.place_values(tokens, counts, place_tokens, heads=True)
                with torch.no_grad():
                    values_t = self.model_target.place_values(tokens, counts, place_tokens, heads=True).min(0).values
                for h in range(heads):
                    q_pred[h] = q_pred[h].index_put((rows,), values_h[h, arange, index])
                q_target = q_target.index_put((rows,), values_t[arange, index])
                logits = self.model.place_policy_logits(tokens, counts, place_tokens).masked_fill(~place_valid, neg)
                log_prob = log_prob.index_put((rows,), F.log_softmax(logits, dim=1)[arange, index])
            value = self.model.value(*collate([states[k] for k in keep], DEVICE)[:2])
            v_loss = expectile_loss(q_target - value, self.cfg.iql_expectile)
            weight = torch.exp(self.cfg.iql_beta * (q_target - value).detach()).clamp(max=MAX_WEIGHT)
            pi_loss = -(weight * log_prob).mean()
            flow_loss = sum(self.model.flow_loss(e, sk, xs, weight[r]) for r, sk, e, xs in flow_terms) if flow_terms else 0.0
            per_sample = F.smooth_l1_loss(q_pred, label[None].expand_as(q_pred), reduction='none').sum(0)
            if weights is not None:
                per_sample = per_sample * torch.as_tensor(np.asarray(weights)[keep], device=DEVICE, dtype=per_sample.dtype)
            loss = per_sample.mean() + v_loss + pi_loss + flow_loss
            self.last_td_loss = float(per_sample.detach().mean())
            self.last_td[keep] = (q_pred.mean(0) - label).detach().cpu().numpy()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.last_q = float(q_pred.mean().detach())
            return float(loss.detach())

    return SkillObjectTrainer
