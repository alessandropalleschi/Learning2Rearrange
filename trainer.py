"""Trainer: batched double-DQN on the shared Q-network (see models.py)."""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage

from config import Config
from memory import ReplayMemory, PrioritizedReplay
from models import QNet, DEVICE


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
        self.q_min = self.cfg.reward_invalid / (1.0 - self.cfg.gamma)
        self.q_max = self.cfg.reward_goal
        self.episode = ReplayMemory(50)
        self.double = True
        self.iteration = 0
        self.is_testing = is_testing
        self.max_dim = self.cfg.max_dim
        self.offset_px = self.cfg.offset_px
        self.push_length_px = self.cfg.push_length_px
        self.future_reward_discount = self.cfg.gamma

        self.model = QNet().to(DEVICE)
        self.model_target = QNet().to(DEVICE)
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
        top, left = (self.max_dim - h) // 2, (self.max_dim - w) // 2
        return top, left

    def prepare(self, color, depth):
        """RGB (H,W,3 uint8) + height (H,W, m) -> (4,max_dim,max_dim) float tensor, zero padded."""
        h, w = depth.shape[:2]
        x = np.zeros((4, self.max_dim, self.max_dim), np.float32)
        top, left = self._pads(h, w)
        x[:3, top:top + h, left:left + w] = np.asarray(color, np.float32).transpose(2, 0, 1) / 255.0
        x[3, top:top + h, left:left + w] = np.asarray(depth, np.float32) / 0.4
        return torch.from_numpy(x)

    def _crop(self, q, h, w):
        """(...,P,P) padded Q-map -> (...,h-2*off,w-off) in the coordinates used by the action indices."""
        top, left = self._pads(h, w)
        q = q[..., top:top + h, left:left + w]
        return q[..., self.offset_px:h - self.offset_px, self.offset_px:w]

    def valid_mask(self, depth, primitive):
        """Boolean mask (cropped coordinates) of pixels where the primitive is feasible."""
        o = self.offset_px
        crop = depth[o:depth.shape[0] - o, o:depth.shape[1]]
        object_pixels = (crop > 0.008).astype(np.uint8)
        if primitive == 'grasp':
            return object_pixels.astype(bool)
        kernel = np.ones((self.push_length_px, self.push_length_px), np.uint8)
        contactable = cv2.dilate(object_pixels, kernel, iterations=1)
        contactable[np.nonzero(object_pixels)] = 0
        return contactable.astype(bool)

    def preprocessQmaps(self, map, depth, primitve):
        """Set infeasible pixels to -1000 (numpy Q-maps in cropped coordinates)."""
        mask = self.valid_mask(depth, primitve)
        return np.where(mask[None], map, -1000.0)

    # ------------------------------------------------------------------ acting
    def forward(self, color_input, depth_input, is_volatile=True, specific_rotation=-1, network='policy'):
        """Q-maps of one observation: push (2,h,w), pick (1,h,w) as numpy, cropped to action coordinates."""
        push, pick = self.forward_batch([color_input], [depth_input], network)
        return push[0], pick[0]

    def forward_batch(self, colors, depths, network='policy'):
        """Q-maps of several observations at once: push (B,2,h,w), pick (B,1,h,w) as numpy."""
        net = self.model if network == 'policy' else self.model_target
        x = torch.stack([self.prepare(c, d) for c, d in zip(colors, depths)]).to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=self.amp):
            push, pick = net(x)
        h, w = depths[0].shape[:2]
        return (self._crop(push.float(), h, w).cpu().numpy(), self._crop(pick.float(), h, w).cpu().numpy())

    # ------------------------------------------------------------------ learning
    def _soft_update(self, alpha):
        with torch.no_grad():
            for pt, p in zip(self.model_target.parameters(), self.model.parameters()):
                pt.data.copy_(alpha * p.data + (1 - alpha) * pt.data)

    @staticmethod
    def _soft_update_q_network_parameters(q_network_1, q_network_2, alpha):
        """In-place soft update of q_network_1 with the parameters of q_network_2."""
        for p1, p2 in zip(q_network_1.parameters(), q_network_2.parameters()):
            p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)

    def _masks_padded(self, depths, h, w):
        """Feasibility masks of a batch of depth maps in padded coordinates: (B,1,P,P) push, pick."""
        top, left = self._pads(h, w)
        o = self.offset_px
        B = len(depths)
        out = {}
        for prim in ('push', 'grasp'):
            m = np.zeros((B, 1, self.max_dim, self.max_dim), bool)
            for i, d in enumerate(depths):
                m[i, 0, top + o:top + h - o, left + o:left + w] = self.valid_mask(d, prim)
            out[prim] = torch.from_numpy(m).to(DEVICE)
        return out['push'], out['grasp']

    def train_step(self, transitions, weights=None):
        """One double-DQN gradient step on a list of Transition. Gradients flow only through the taken pixel."""
        B = len(transitions)
        states = [t.state for t in transitions]
        nexts = [t.next_state for t in transitions]
        h, w = states[0]['depthmap'].shape[:2]
        top, left = self._pads(h, w)

        x = torch.stack([self.prepare(s['colormap'], s['depthmap']) for s in states]).to(DEVICE)
        xn = torch.stack([self.prepare(s['colormap'], s['depthmap']) for s in nexts]).to(DEVICE)
        reward = torch.tensor([float(t.reward) for t in transitions], device=DEVICE)
        done = torch.tensor([float(t.done) for t in transitions], device=DEVICE)
        is_push = torch.tensor([t.primitive_action == 'push' for t in transitions], device=DEVICE)
        rot = torch.tensor([int(t.action[0]) for t in transitions], device=DEVICE)
        rows = torch.tensor([int(t.action[1]) + self.offset_px + top for t in transitions], device=DEVICE)
        cols = torch.tensor([int(t.action[2]) + self.offset_px + left for t in transitions], device=DEVICE)

        # ---- TD target (double DQN: policy net selects, target net evaluates, over both primitives)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=self.amp):
            push_mask, pick_mask = self._masks_padded([s['depthmap'] for s in nexts], h, w)
            neg = -1e9
            pp, pk = self.model(xn)
            pp, pk = pp.float(), pk.float()
            pol = torch.cat([torch.where(push_mask.expand_as(pp), pp, torch.full_like(pp, neg)).flatten(1),
                             torch.where(pick_mask, pk, torch.full_like(pk, neg)).flatten(1)], dim=1)
            if self.double:
                tp, tk = self.model_target(xn)
                tp, tk = tp.float(), tk.float()
                tgt = torch.cat([torch.where(push_mask.expand_as(tp), tp, torch.full_like(tp, neg)).flatten(1),
                                 torch.where(pick_mask, tk, torch.full_like(tk, neg)).flatten(1)], dim=1)
                future = tgt.gather(1, pol.argmax(1, keepdim=True)).squeeze(1)
            else:
                future = pol.max(1).values
            # no valid action at all (e.g. empty scene): no bootstrap
            future = torch.where(future < neg / 2, torch.zeros_like(future), future)
            label = reward + self.future_reward_discount * future * (1.0 - done)
            if self.cfg.clip_q:
                label = label.clamp(self.q_min, self.q_max)

        # ---- prediction at the taken pixel only
        with torch.autocast(device_type=DEVICE.type, enabled=self.amp):
            out = self.model.forward_selected(x, is_push, rot)
        pred = out.float()[torch.arange(B, device=DEVICE), 0, rows, cols]
        per_sample = F.smooth_l1_loss(pred, label, reduction='none')  # Huber loss (paper Eq. 3)
        if weights is not None:  # importance-sampling correction of prioritized replay
            per_sample = per_sample * torch.as_tensor(weights, device=DEVICE)
        loss = per_sample.mean()
        self.last_td = (pred - label).detach().cpu().numpy()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
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
