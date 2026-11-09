"""Trainer: batched double-DQN on the shared Q-network (see models.py)."""
from collections import namedtuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage

from l2r.config import Config
from l2r.learning.memory import ReplayMemory, PrioritizedReplay
from l2r.learning.belief import BELIEF_CHANNELS
from l2r.learning.models import QNet, DEVICE, ACTION_CHANNELS, HELD_CLASSES

QMaps = namedtuple('QMaps', ['push', 'pick', 'place', 'hidden', 'lengths', 'prim'], defaults=(None, None))


class Trainer(object):
    def __init__(self, cfg=None, is_testing=False, load_snapshot=False, snapshot=None, snapshot_target=None):
        self.cfg = cfg or Config()
        if self.cfg.prioritized:
            self.buffer = PrioritizedReplay(self.cfg.buffer_size, self.cfg.per_alpha, self.cfg.per_eps)
        else:
            self.buffer = ReplayMemory(self.cfg.buffer_size)
        self.opt_steps = 0
        self.last_td = None
        self.amp = bool(self.cfg.amp and DEVICE.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        # Q values are bounded: rewards lie in [reward_invalid, reward_goal]
        self.q_min, self.q_max = self.cfg.q_range
        self.episode = ReplayMemory(50)
        self.double = True
        self.iteration = 0
        self.is_testing = is_testing
        self.offset_px = self.cfg.offset_px
        self.push_lengths_px = self.cfg.push_lengths_px
        self.future_reward_discount = self.cfg.gamma

        self.belief = self.cfg.belief
        self.learned_placement = self.cfg.placement == 'learned'
        in_channels = 4 + (BELIEF_CHANNELS if self.belief == 'spatial' else 0)
        n_len = len(self.cfg.push_lengths)
        if DEVICE.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # the input size is fixed: let cuDNN pick the fastest kernels
        net_args = (in_channels, self.belief, self.cfg.recurrent_hidden, n_len, self.learned_placement, self.cfg.backbone,
                    self.cfg.transformer_dim, self.cfg.transformer_layers, self.cfg.hierarchy, self.cfg.pick_channels)
        self.model, self.model_target = QNet(*net_args).to(DEVICE), QNet(*net_args).to(DEVICE)
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot, map_location=DEVICE))
            self.model_target.load_state_dict(torch.load(snapshot_target or snapshot, map_location=DEVICE))
            print('Pre-trained model snapshot')
        else:
            self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval()
        for p in self.model_target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate,
                                          weight_decay=self.cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.lr_decay)

    # ------------------------------------------------------------------ persistence
    def save(self, snapshot_path, snapshot_target_path=None):
        torch.save(self.model.state_dict(), snapshot_path)
        if snapshot_target_path:
            torch.save(self.model_target.state_dict(), snapshot_target_path)

    # ------------------------------------------------------------------ input / output geometry
    def _pads(self, h, w):
        height, width = self.canvas(h, w)
        top, left = (height - h) // 2, (width - w) // 2
        return top, left

    def prepare(self, obs):
        """Observation dict -> (C,H,W) float tensor on the network canvas (the map size rounded up to a multiple of 8).
        Channels: RGB, height (/0.4 m) and, for the spatial belief, the 7 belief channels."""
        color, depth = obs['colormap'], obs['depthmap']
        h, w = depth.shape[:2]
        c = 4 + (BELIEF_CHANNELS if self.belief == 'spatial' else 0)
        x = np.zeros((c, *self.canvas(h, w)), np.float32)
        top, left = self._pads(h, w)
        x[:3, top:top + h, left:left + w] = np.asarray(color, np.float32).transpose(2, 0, 1) / 255.0
        x[3, top:top + h, left:left + w] = np.asarray(depth, np.float32) / 0.4
        if self.belief == 'spatial':
            x[4:, top:top + h, left:left + w] = np.asarray(obs['belief'], np.float32) / 255.0
        return torch.from_numpy(x)

    # ------------------------------------------------------------------ recurrent belief inputs
    @staticmethod
    def canvas(h, w):
        """Network input size for an h x w map: rounded up to a multiple of 8 (three 2x poolings). The maps of this
        task (128 x 104) already are, so nothing is padded and no compute is spent on empty borders."""
        return -(-h // 8) * 8, -(-w // 8) * 8

    def _cells(self, h, w):
        """Size of the bottleneck grid (where the recurrent belief lives)."""
        height, width = self.canvas(h, w)
        return height // 8, width // 8

    def zero_hidden(self, h, w):
        return np.zeros((self.cfg.recurrent_hidden, *self._cells(h, w)), np.float16)

    def encode_action(self, action, h, w):
        """Last action -> (ACTION_CHANNELS, cells, cells): push 0deg, push 180deg, pick, place (one-hot cells)."""
        rows, cols = self._cells(h, w)
        a = np.zeros((ACTION_CHANNELS, rows, cols), np.float32)
        if action is None:
            return a
        top, left = self._pads(h, w)
        o = self.offset_px

        def cell(row, col):
            return (min((row + o + top) // 8, rows - 1), min((col + o + left) // 8, cols - 1))

        rot, row, col = action['pix'][:3]
        channel = {'push': rot // len(self.cfg.push_lengths), 'grasp': 2, 'place': 3}[action['primitive']]
        a[channel][cell(row, col)] = 1.0
        if action.get('place_pix') is not None:
            a[3][cell(action['place_pix'][1], action['place_pix'][0])] = 1.0
        return a

    def _recurrent_inputs(self, obs_list):
        """Stacked (h_prev, a_prev) tensors of a list of observations (None unless the belief is recurrent)."""
        if self.belief != 'recurrent':
            return None, None
        h, w = obs_list[0]['depthmap'].shape[:2]
        hp = torch.from_numpy(np.stack([o['h_prev'] for o in obs_list]).astype(np.float32)).to(DEVICE)
        ap = torch.from_numpy(np.stack([self.encode_action(o.get('prev_action'), h, w) for o in obs_list])).to(DEVICE)
        return hp, ap

    def _crop(self, q, h, w):
        """(...,P,P) padded Q-map -> (...,h-2*off,w-off) in the coordinates used by the action indices."""
        top, left = self._pads(h, w)
        q = q[..., top:top + h, left:left + w]
        return q[..., self.offset_px:h - self.offset_px, self.offset_px:w]

    def valid_mask(self, depth, primitive):
        """Boolean mask (cropped coordinates) of pixels where the primitive is feasible: (h,w) for 'grasp',
        (2L,h,w) for 'push' (index = direction*L + length; a longer push can start farther from an object)."""
        o = self.offset_px
        crop = depth[o:depth.shape[0] - o, o:depth.shape[1]]
        object_pixels = (crop > 0.008).astype(np.uint8)
        if primitive == 'grasp':
            return object_pixels.astype(bool)
        per_length = []
        for k in self.push_lengths_px:
            contactable = cv2.dilate(object_pixels, np.ones((k, k), np.uint8), iterations=1)
            contactable[np.nonzero(object_pixels)] = 0
            per_length.append(contactable.astype(bool))
        return np.tile(np.stack(per_length), (2, 1, 1))  # same feasibility for both directions

    def _held(self, obs_list):
        """(B,3) one-hot class of the held object (learned placement only; zeros when nothing is held)."""
        if not self.learned_placement:
            return None
        held = np.zeros((len(obs_list), len(HELD_CLASSES)), np.float32)
        for i, o in enumerate(obs_list):
            if o.get('held_class'):
                held[i, HELD_CLASSES[int(o['held_class'])]] = 1.0
        return torch.from_numpy(held).to(DEVICE)

    @staticmethod
    def place_valid(obs):
        """Free cells (cropped coordinates) where the held object may be placed."""
        return np.asarray(obs['freespace']) > 0

    def preprocessQmaps(self, map, depth, primitve):
        """Set infeasible pixels to -1000 (numpy Q-maps in cropped coordinates)."""
        mask = self.valid_mask(depth, primitve)
        if mask.ndim == 2:
            mask = mask[None]
        return np.where(mask, map, -1000.0)

    # ------------------------------------------------------------------ acting
    def forward(self, color_input, depth_input, network='policy'):
        """Q-maps of one memoryless observation: push (2,h,w), pick (1,h,w) as numpy (belief 'none' only)."""
        q = self.forward_batch([{'colormap': color_input, 'depthmap': depth_input}], network)
        return q.push[0], q.pick[0]

    def forward_batch(self, obs_list, network='policy'):
        """Q-maps of several observations at once: push (B,2,h,w), pick (B,1,h,w) as numpy, cropped to the action
        coordinates, the place Q-map (B,1,h,w) (used while an object is held) and (recurrent belief) the new
        hidden states (B,hid,cells,cells) float16, else None -- returned as QMaps(push, pick, place, hidden).
        With the recurrent belief every observation must carry 'h_prev' and 'prev_action'."""
        net = self.model if network == 'policy' else self.model_target
        x = torch.stack([self.prepare(o) for o in obs_list]).to(DEVICE)
        hp, ap = self._recurrent_inputs(obs_list)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=self.amp):
            out = net.forward_ext(x, hp, ap, self._held(obs_list))
        h, w = obs_list[0]['depthmap'].shape[:2]
        crop = lambda q: self._crop(q.float(), h, w).cpu().numpy()
        return QMaps(crop(out.push), crop(out.pick),
                     crop(out.place) if out.place is not None else np.zeros_like(crop(out.pick)),
                     None if out.hidden is None else out.hidden.float().cpu().numpy().astype(np.float16),
                     None if out.lengths is None else crop(out.lengths),
                     None if out.prim is None else out.prim.float().cpu().numpy())

    # ------------------------------------------------------------------ learning
    def _soft_update(self, alpha):
        with torch.no_grad():
            for pt, p in zip(self.model_target.parameters(), self.model.parameters()):
                pt.data.copy_(alpha * p.data + (1 - alpha) * pt.data)

    def _masks_padded(self, obs_list, h, w):
        """Feasible-action masks of a batch of observations in padded coordinates, each (B,1,P,P) bool:
        push, pick and place (place only while an object is held, push/pick only otherwise)."""
        top, left = self._pads(h, w)
        o = self.offset_px
        B = len(obs_list)
        canvas = self.canvas(h, w)
        masks = {'push': np.zeros((B, self.cfg.push_actions, *canvas), bool),
                 'grasp': np.zeros((B, self.cfg.pick_channels, *canvas), bool),
                 'place': np.zeros((B, 1, *canvas), bool)}
        for i, obs in enumerate(obs_list):
            window = (slice(top + o, top + h - o), slice(left + o, left + w))
            if obs.get('held_class'):
                masks['place'][(i, 0) + window] = self.place_valid(obs)
            else:
                masks['push'][(i, slice(None)) + window] = self.valid_mask(obs['depthmap'], 'push')
                masks['grasp'][(i, slice(None)) + window] = self.valid_mask(obs['depthmap'], 'grasp')
        return [torch.from_numpy(masks[k]).to(DEVICE) for k in ('push', 'grasp', 'place')]

    PRIMITIVES = {'grasp': QNet.PICK, 'push': QNet.PUSH, 'place': QNet.PLACE}

    def _hierarchical_pass(self, x, states, transitions, prim, rot, rows, cols, inputs, h, w):
        """Hierarchical agent: (Q of the taken actions, level-1 loss, actor loss).

        * critic: the taken push is evaluated at its *executed* continuous length;
        * level 1: the value of every primitive is regressed to the best Q of that primitive at the actor's length
          (over the feasible pixels), for the primitives that are available in the state;
        * actor: maximise the critic at the taken pixel / direction (only for pushes)."""
        hp, ap, held = inputs
        B, arange = len(transitions), torch.arange(len(transitions), device=DEVICE)
        length = torch.tensor([float(t.action[3]) if t.primitive_action == 'push' else 0.0 for t in transitions],
                              device=DEVICE)
        out = self.model.forward_train(x, length, hp, ap, held)
        push_map = out['push_exec'].gather(1, (rot * (prim == QNet.PUSH))[:, None, None, None].expand(-1, 1, *out['pick'].shape[2:]))[:, 0]
        pick_rot = (rot * (prim == QNet.PICK)).clamp(max=out['pick'].shape[1] - 1)
        pick_map = out['pick'].gather(1, pick_rot[:, None, None, None].expand(-1, 1, *out['pick'].shape[2:]))[:, 0]
        place_map = out['place'][:, 0] if out['place'] is not None else pick_map
        taken = torch.where((prim == QNet.PUSH)[:, None, None], push_map,
                            torch.where((prim == QNet.PLACE)[:, None, None], place_map, pick_map))
        pred = taken.float()[arange, rows, cols]

        neg = -1e9
        push_mask, pick_mask, place_mask = self._masks_padded(states, h, w)
        with torch.no_grad():
            def best(q, mask):
                return torch.where(mask.expand_as(q), q.float(), torch.full_like(q.float(), neg)).flatten(1).max(1).values

            values = torch.stack([best(out['push_actor'].detach(), push_mask), best(out['pick'].detach(), pick_mask),
                                  best(out['place'].detach(), place_mask) if out['place'] is not None
                                  else torch.full((B,), neg, device=DEVICE)], dim=1)  # order of `prim`: push, pick, place
            available = values > neg / 2
            targets = values.clamp(self.q_min, self.q_max)
        level1 = (F.smooth_l1_loss(out['prim'].float(), targets, reduction='none') * available).sum() / available.sum().clamp(min=1)
        actor_loss = None
        pushed = prim == QNet.PUSH
        if pushed.any():
            q_actor = out['push_actor'].float()[arange, rot * pushed, rows, cols]
            actor_loss = -q_actor[pushed].mean()
        return pred, self.cfg.hier_prim_weight * level1, actor_loss

    def train_step(self, transitions, weights=None):
        """One double-DQN gradient step on a list of Transition. Gradients flow only through the taken pixel."""
        B = len(transitions)
        states = [t.state for t in transitions]
        nexts = [t.next_state for t in transitions]
        h, w = states[0]['depthmap'].shape[:2]
        top, left = self._pads(h, w)

        x = torch.stack([self.prepare(s) for s in states]).to(DEVICE)
        xn = torch.stack([self.prepare(s) for s in nexts]).to(DEVICE)
        hp, ap = self._recurrent_inputs(states)
        hpn, apn = self._recurrent_inputs(nexts)
        held, held_n = self._held(states), self._held(nexts)
        reward = torch.tensor([float(t.reward) for t in transitions], device=DEVICE)
        done = torch.tensor([float(t.done) for t in transitions], device=DEVICE)
        prim = torch.tensor([self.PRIMITIVES[t.primitive_action] for t in transitions], device=DEVICE)
        rot = torch.tensor([int(t.action[0]) for t in transitions], device=DEVICE)
        rows = torch.tensor([int(t.action[1]) + self.offset_px + top for t in transitions], device=DEVICE)
        cols = torch.tensor([int(t.action[2]) + self.offset_px + left for t in transitions], device=DEVICE)
        # pick -> place is one pick-and-place action: no discount between its two decisions
        gamma = torch.tensor([1.0 if (n.get('held_class') and not s.get('held_class')) else self.future_reward_discount
                              for s, n in zip(states, nexts)], device=DEVICE)

        # ---- TD target (double DQN: policy net selects, target net evaluates, over all feasible actions)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=self.amp):
            push_mask, pick_mask, place_mask = self._masks_padded(nexts, h, w)
            neg = -1e9

            def flat(q):
                pp, pk = q[0].float(), q[1].float()
                pl = q[2].float() if q[2] is not None else torch.full_like(pk, neg)
                return torch.cat([torch.where(push_mask.expand_as(pp), pp, torch.full_like(pp, neg)).flatten(1),
                                  torch.where(pick_mask, pk, torch.full_like(pk, neg)).flatten(1),
                                  torch.where(place_mask, pl, torch.full_like(pl, neg)).flatten(1)], dim=1)

            pol = flat(self.model(xn, hpn, apn, held_n))
            if self.double:
                tgt = flat(self.model_target(xn, hpn, apn, held_n))
                future = tgt.gather(1, pol.argmax(1, keepdim=True)).squeeze(1)
            else:
                future = pol.max(1).values
            # no valid action at all (e.g. empty scene): no bootstrap
            future = torch.where(future < neg / 2, torch.zeros_like(future), future)
            label = reward + gamma * future * (1.0 - done)
            if self.cfg.clip_q:
                label = label.clamp(self.q_min, self.q_max)

        # ---- prediction at the taken pixel only
        arange = torch.arange(B, device=DEVICE)
        extra_loss, actor_loss = 0.0, None
        with torch.autocast(device_type=DEVICE.type, enabled=self.amp):
            if self.cfg.hierarchy:
                pred, extra_loss, actor_loss = self._hierarchical_pass(x, states, transitions, prim, rot, rows, cols,
                                                                       (hp, ap, held), h, w)
            else:
                pred = self.model.forward_selected(x, prim, rot, hp, ap, held).float()[arange, 0, rows, cols]
        per_sample = F.smooth_l1_loss(pred, label, reduction='none')  # Huber loss (paper Eq. 3)
        if weights is not None:  # importance-sampling correction of prioritized replay
            per_sample = per_sample * torch.as_tensor(weights, device=DEVICE)
        loss = per_sample.mean() + extra_loss
        self.last_td = (pred - label).detach().cpu().numpy()

        self.optimizer.zero_grad()
        length_params = list(self.model.length_head.parameters()) if self.cfg.hierarchy else []
        actor_grads = None
        if actor_loss is not None:  # DDPG: the actor follows the gradient of the critic; no other parameter is touched
            actor_grads = torch.autograd.grad(self.scaler.scale(actor_loss), length_params, allow_unused=True)
        self.scaler.scale(loss).backward()
        if actor_grads is not None:
            for parameter, grad in zip(length_params, actor_grads):
                parameter.grad = grad
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.last_q = float(pred.detach().mean().cpu())
        return float(loss.detach().cpu())

    def optimize(self):
        """Sample a batch from the replay buffer, take a gradient step and Polyak-update the target."""
        if self.is_testing or len(self.buffer) < self.cfg.batch_size:
            return None
        if self.cfg.prioritized:
            beta = min(1.0, self.cfg.per_beta_start +
                       (1.0 - self.cfg.per_beta_start) * self.opt_steps / self.cfg.per_beta_iters)
            transitions, idx, weights = self.buffer.sample(self.cfg.batch_size, beta)
            loss = self.train_step(transitions, weights)
            self.buffer.update(idx, self.last_td)
        else:
            loss = self.train_step(self.buffer.sample(self.cfg.batch_size))
        self.scheduler.step()
        self.opt_steps += 1
        self._soft_update(self.cfg.polyak_beta)
        return loss

    def get_prediction_vis(self, predictions, color_input, best_pix_ind=[], final_pix=[]):

        canvas = None
        color_heightmap = color_input.copy()
        color_heightmap= color_heightmap[self.offset_px:color_heightmap.shape[0]-self.offset_px,self.offset_px:color_heightmap.shape[1]]
        num_rotations = predictions.shape[0]
        final_pix = np.array(final_pix)
        best_pix_ind = np.array(best_pix_ind)
        size_final = final_pix.shape
        size_init = best_pix_ind.shape
        if num_rotations > 1 and len(best_pix_ind)>0:
            rot_selected = best_pix_ind[0]
        else:
            rot_selected = 0
        for canvas_row in range(int(num_rotations/num_rotations)):
            tmp_row_canvas = None
            for canvas_col in range(predictions.shape[0]):
                rotate_idx = canvas_row+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()

                prediction_vis = prediction_vis+1
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == rot_selected and len(best_pix_ind)>0:
                    # prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[size_init[0]-1]), int(best_pix_ind[size_init[0]-2])), 7, (0,0,255), 2)
                    # prediction_vis = cv2.circle(prediction_vis, (int(final_pix[size_final[0]-1]), int(final_pix[size_final[0]-2])), 7, (255,0,0), 2)
                    prediction_vis = cv2.arrowedLine(prediction_vis, (int(best_pix_ind[size_init[0]-1]), int(best_pix_ind[size_init[0]-2])),(int(final_pix[size_final[0]-1]), int(final_pix[size_final[0]-2])), (0,255,0), 2)
                # if(num_rotations == 1):
                #     prediction_vis = cv2.rectangle(prediction_vis, (int(best_pix_ind[size_init[0]-1])-self.model_pick_and_place.crop_size//2, int(best_pix_ind[size_init[0]-2]-self.model_pick_and_place.crop_size//2)) , (int(best_pix_ind[size_init[0]-1])+self.model_pick_and_place.crop_size//2, int(best_pix_ind[size_init[0]-2]+self.model_pick_and_place.crop_size//2)), (0,0,255), 2)

                prediction_vis = ndimage.zoom(prediction_vis, zoom=[2,2,1], order=0)
                color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0) 
                size = prediction_vis.shape
       
                top_pad = int((320-size[0])/2)
                left_pad = int((320-size[1])/2)

                prediction_vis=cv2.copyMakeBorder(prediction_vis.copy(),top_pad,top_pad,left_pad,left_pad,cv2.BORDER_CONSTANT,value=[0,0,0])
                rot = cv2.getRotationMatrix2D((int(320 // 2),int(320 // 2)), np.radians(rotate_idx*(360/(num_rotations))-90) * 180 / np.pi, 1.0)
 
 
                prediction_vis = cv2.warpAffine(prediction_vis, rot, (320, 320))
                # prediction_vis = ndimage.rotate(prediction_vis,(rotate_idx*(180/num_rotations)), reshape=False, order=0)
                size = color_heightmap_2x.shape
                
                top_pad = int((320-size[0])/2)
                left_pad = int((320-size[1])/2)

                color_heightmap_2x =cv2.copyMakeBorder(color_heightmap_2x.copy(),top_pad,top_pad,left_pad,left_pad,cv2.BORDER_CONSTANT,value=[0,0,0])

                background_image = cv2.warpAffine(color_heightmap_2x, rot, (320, 320))
                   
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if(rotate_idx == 1):
                    prediction_vis = prediction_vis[::-1,:,:]

                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas
