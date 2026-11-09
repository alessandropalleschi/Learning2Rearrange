"""Object-centric Q-function: one token per visible object instead of one value per pixel.

The dense agents score ~13,000 pixels, of which a handful matter. Here the observation (colour map + height map + the target regions)
is turned into **instances** (connected pieces of one class colour), each described by a small feature vector (class, position, extent,
whether it already stands in its region and how far it is, how hidden it is). A transformer attends over the instance tokens, the three
region tokens and a global token, and outputs a Q-value per **object-action slot**: pick, or push left / right at each push length.
The number of actions per state is 5 objects x (1 + 2L) instead of 13,000 pixels, the network generalises over the number of objects
(a set), and the primitive / object choice is explicit (a natural place for a manager).

Every slot corresponds to a fixed pixel action computed from the instance's mask (the same geometry the scripted expert uses), so the agent
plugs into the unchanged environment: `rasterize` writes the slot values onto the dense Q-map interface (everything else -1000).
Only observation information is used (no ground-truth ids). Placement is the heuristic one (`--placement heuristic`).
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from l2r.learning.masks import BLADE_HALF_WIDTH_CELLS, GRASP_EDGE_MARGIN, PUSH_OBJECT_HEIGHT, push_path_cells
from l2r.sim.constants import CLASSES, COLORS

MAX_OBJECTS = 16
MAX_GHOSTS = 8  # remembered-but-hidden objects kept as tokens
FEATURES = 20  # 18 object features + ghost flag + age (see instance_features)
COLOR_TOLERANCE = 6  # summed absolute RGB difference that still counts as the class colour
MIN_AREA = 6  # cells (5 mm): smaller pieces are noise
OCCLUDED_RGB = 105
GRASP_DEPTH_CELLS = 2  # aim this many cells behind the front-most cell (see demos.GRASP_DEPTH_CELLS)
PUSH_GAP = 3  # cells between the blade start and the object
TYPE_INSTANCE, TYPE_REGION, TYPE_GLOBAL, TYPE_GHOST = 0, 1, 2, 3
CLASS_RGB = np.array([np.multiply(COLORS[name][:3], 255) for name in CLASSES]).round().astype(int)


@dataclass
class Instance:
    cls: int  # index in CLASSES
    rows: np.ndarray  # cells in cropped (action) coordinates
    cols: np.ndarray


def crop(array, offset=4):
    """Observation map (padded by `offset` rows on both sides and columns on the left) -> action coordinates."""
    return array[offset:array.shape[0] - offset, offset:array.shape[1]]


def extract_instances(obs):
    """Instances (largest first) from the colour and height map: connected pieces of one class colour on cells above the floor."""
    color, depth = crop(obs['colormap']).astype(int), crop(obs['depthmap'])
    raised = depth > 0.008
    found = []
    for k in range(len(CLASSES)):
        same = raised & (np.abs(color - CLASS_RGB[k]).sum(-1) <= COLOR_TOLERANCE)
        labels, count = ndimage.label(same, structure=np.ones((3, 3)))
        for label in range(1, count + 1):
            rows, cols = np.nonzero(labels == label)
            if rows.size >= MIN_AREA:
                found.append(Instance(k, rows, cols))
    found.sort(key=lambda i: -i.rows.size)
    return found[:MAX_OBJECTS]


def extract_ghosts(obs):
    """Remembered objects that are NOT visible now, from the spatial belief (`--belief spatial`): connected pieces of one class colour in the
    remembered colour map whose memory is stale (age > 0), largest first. Returns (ghosts, remembered height map in metres, mean age per
    ghost in 0..1). Empty without a belief."""
    if 'belief' not in obs:
        return [], None, []
    belief = np.asarray(obs['belief'])
    color = crop(np.transpose(belief[:3], (1, 2, 0))).astype(int)
    height, age = crop(belief[3]).astype(np.float32) / 255.0 * 0.4, crop(belief[4]).astype(np.float32) / 255.0
    stale = (height > 0.008) & (age > 0)
    found = []
    for k in range(len(CLASSES)):
        same = stale & (np.abs(color - CLASS_RGB[k]).sum(-1) <= COLOR_TOLERANCE)
        labels, count = ndimage.label(same, structure=np.ones((3, 3)))
        for label in range(1, count + 1):
            rows, cols = np.nonzero(labels == label)
            if rows.size >= MIN_AREA:
                found.append((Instance(k, rows, cols), float(age[rows, cols].mean())))
    found.sort(key=lambda g: -g[0].rows.size)
    found = found[:MAX_GHOSTS]
    return [g[0] for g in found], height, [g[1] for g in found]


def instance_features(instances, obs, height_map=None, ages=None):
    """(N, FEATURES) float32: class one-hot, position, extent, in-region flag and distance, height, occlusion; for ghosts (height_map and
    ages given) the remembered heights, the ghost flag and the age of the memory."""
    color, depth = crop(obs['colormap']).astype(int), (crop(obs['depthmap']) if height_map is None else height_map)
    h, w = depth.shape
    regions = np.asarray(obs['regions'], np.float32)  # (3, 2) rows, un-padded = cropped coordinates
    grey = np.abs(color - OCCLUDED_RGB).sum(-1) <= 6
    raised = depth > 0.008
    out = np.zeros((len(instances), FEATURES), np.float32)
    for i, inst in enumerate(instances):
        r, c = inst.rows, inst.cols
        centre = r.mean()
        low, high = regions[inst.cls]
        distance = max(low - centre, centre - high, 0.0)
        back = grey[np.unique(r)][:, max(int(c.min()) - 4, 0):int(c.min())]  # cells behind the object, on its rows
        ahead = raised[np.unique(r)][:, int(c.max()) + 1:int(c.max()) + 9]  # cells in front of it, on its rows
        out[i, inst.cls] = 1.0
        out[i, 3:5] = (2 * centre / (h - 1) - 1, 2 * c.mean() / (w - 1) - 1)
        out[i, 5:7] = ((r.max() - r.min() + 1) / h * 4, (c.max() - c.min() + 1) / w * 4)
        out[i, 7:9] = (2 * c.max() / (w - 1) - 1, 2 * c.min() / (w - 1) - 1)
        out[i, 9] = r.size / 400.0
        out[i, 10:12] = (depth[r, c].mean() / 0.2, depth[r, c].max() / 0.2)
        out[i, 12] = float(low <= centre <= high)
        out[i, 13:15] = ((centre - (low + high) / 2) / h * 4, distance / h * 4)
        out[i, 15] = back.mean() if back.size else 0.0
        out[i, 16] = ahead.mean() if ahead.size else 0.0
        out[i, 17] = 1.0  # a real token (padding rows are all zero)
        if ages is not None:
            out[i, 18], out[i, 19] = 1.0, ages[i]  # a ghost: remembered, not visible now
    return out


def region_features(obs):
    """(3, FEATURES) float32 tokens of the target regions: class one-hot, rows (normalised)."""
    h = crop(obs['depthmap']).shape[0]
    out = np.zeros((len(CLASSES), FEATURES), np.float32)
    for k, (low, high) in enumerate(np.asarray(obs['regions'], np.float32)):
        out[k, k] = 1.0
        out[k, 3:6] = (2 * low / (h - 1) - 1, 2 * high / (h - 1) - 1, (high - low) / h * 4)
        out[k, 17] = 1.0
    return out


def global_features(instances, obs, ghosts=()):
    out = np.zeros(FEATURES, np.float32)
    features = instance_features(instances, obs) if instances else np.zeros((0, FEATURES), np.float32)
    out[0] = len(instances) / 8.0
    out[2] = len(ghosts) / 8.0
    out[1] = float((features[:, 12] == 0).mean()) if len(features) else 0.0  # fraction still misplaced
    out[17] = 1.0
    return out


def candidate_actions(instances, obs, push_lengths_px, edge_margin=GRASP_EDGE_MARGIN):
    """(N, 1 + 2L, 3) int pixel actions (rot, row, col) and (N, 1 + 2L) bool validity of every object-action slot.

    slot 0: pick at the instance's front face (aimed GRASP_DEPTH_CELLS inside it, in the central part of its width);
    slot 1 + d * L + l: push in direction d (0: towards decreasing rows, 1: towards increasing rows) with length index l, the blade
    starting PUSH_GAP cells beside the object on the opposite side. A push is valid only if the blade's own strip has object cells in
    its path."""
    depth = crop(obs['depthmap'])
    h, w = depth.shape
    near = depth > PUSH_OBJECT_HEIGHT
    n_len = len(push_lengths_px)
    actions = np.zeros((len(instances), 1 + 2 * n_len, 3), np.int64)
    valid = np.zeros((len(instances), 1 + 2 * n_len), bool)
    paths = {(d, l): push_path_cells(near, push_lengths_px[l], d == 1, BLADE_HALF_WIDTH_CELLS)
             for d in (0, 1) for l in range(n_len)}
    for i, inst in enumerate(instances):
        rows, cols = inst.rows, inst.cols
        col = int(cols.max()) - GRASP_DEPTH_CELLS
        while not (cols == col).any() and col > cols.min():
            col -= 1
        at = rows[cols == col]
        span = rows.max() - rows.min()
        central = at[(at >= rows.min() + edge_margin * span) & (at <= rows.max() - edge_margin * span)]
        pool = central if central.size else at
        row = int(pool[np.argmin(np.abs(pool - np.median(rows)))])
        actions[i, 0] = (0, row, col)
        valid[i, 0] = True
        push_col = int(np.median(cols))
        for direction in (0, 1):
            start = int(rows.min()) - PUSH_GAP if direction == 1 else int(rows.max()) + PUSH_GAP
            for l in range(n_len):
                slot = 1 + direction * n_len + l
                actions[i, slot] = (direction * n_len + l, start, push_col)
                valid[i, slot] = 0 <= start < h and not depth[start, push_col] > 0.008 and bool(paths[(direction, l)][start, push_col])
    return actions, valid


def build_state(obs, push_lengths_px):
    """Everything the network and the action mapping need for one observation."""
    instances = extract_instances(obs)
    actions, valid = candidate_actions(instances, obs, push_lengths_px)
    ghosts, remembered, ages = extract_ghosts(obs)
    ghost_tokens = instance_features(ghosts, obs, remembered, ages) if ghosts else np.zeros((0, FEATURES), np.float32)
    return {'instances': instances, 'actions': actions, 'valid': valid, 'ghosts': ghosts, 'ghost_tokens': ghost_tokens,
            'tokens': np.concatenate([global_features(instances, obs, ghosts)[None], region_features(obs),
                                      instance_features(instances, obs)], axis=0)}


def slot_of(state, primitive, action, tolerance=3):
    """(object index, slot) of the taken pixel action, or None if no candidate lies within `tolerance` cells (same slot kind)."""
    if len(state['instances']) == 0:
        return None
    best, best_distance = None, tolerance + 1
    rot = int(action[0])
    slots = [0] if primitive == 'grasp' else ([1 + rot] if primitive == 'push' else [])
    for slot in slots:
        for i in range(len(state['instances'])):
            if not state['valid'][i, slot]:
                continue
            _, row, col = state['actions'][i, slot]
            distance = max(abs(int(row) - int(action[1])), abs(int(col) - int(action[2])))
            if distance < best_distance:
                best, best_distance = (i, slot), distance
    return best


class ObjectQNet(nn.Module):
    """Transformer over [global, 3 region, N instance] tokens -> Q for every (instance, slot)."""

    def __init__(self, n_slots, dim=64, layers=2, heads=4):
        super().__init__()
        self.n_slots = n_slots
        self.embed = nn.Linear(FEATURES, dim)
        self.type_embed = nn.Embedding(4, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_slots))

    def embed_tokens(self, tokens, counts):
        """tokens (B, 4 + Nmax + MAX_GHOSTS, FEATURES), counts (B, 2) long (instances, ghosts) -> contextualised tokens of the
        instances (B, Nmax, dim): global + 3 region + Nmax instance + MAX_GHOSTS ghost tokens through the transformer."""
        batch, length, _ = tokens.shape
        n_max = length - 4 - MAX_GHOSTS
        position = torch.arange(length, device=tokens.device)
        types = torch.full((length,), TYPE_INSTANCE, dtype=torch.long, device=tokens.device)
        types[0], types[1:4], types[4 + n_max:] = TYPE_GLOBAL, TYPE_REGION, TYPE_GHOST
        pad = torch.where(position[None] < 4, torch.zeros_like(position[None], dtype=torch.bool),
                          torch.where(position[None] < 4 + n_max, (position[None] - 4) >= counts[:, :1],
                                      (position[None] - 4 - n_max) >= counts[:, 1:2]))
        x = self.embed(tokens) + self.type_embed(types)[None]
        return self.norm(self.encoder(x, src_key_padding_mask=pad))[:, 4:4 + n_max]

    def forward(self, tokens, counts):
        """-> Q (B, Nmax, n_slots) for every instance and slot."""
        return self.head(self.embed_tokens(tokens, counts))


def collate(states, device):
    """List of `build_state` results -> (tokens (B, 4 + Nmax + MAX_GHOSTS, F), counts (B, 2) = (instances, ghosts), valid (B, Nmax, S))."""
    n_max = max(max(len(s['instances']) for s in states), 1)
    n_slots = states[0]['valid'].shape[1] if len(states[0]['valid']) else 1
    tokens = np.zeros((len(states), 4 + n_max + MAX_GHOSTS, FEATURES), np.float32)
    valid = np.zeros((len(states), n_max, n_slots), bool)
    for b, s in enumerate(states):
        n, g = len(s['instances']), len(s['ghost_tokens'])
        tokens[b, :4 + n] = s['tokens']
        tokens[b, 4 + n_max:4 + n_max + g] = s['ghost_tokens']
        valid[b, :n] = s['valid']
    counts = torch.tensor([[len(s['instances']), len(s['ghost_tokens'])] for s in states], device=device)
    return torch.from_numpy(tokens).to(device), counts, torch.from_numpy(valid).to(device)


def rasterize(state, q, shape, n_push, fill=-1000.0):
    """Slot values (N, S) -> the dense Q-map interface: push (2L, h, w), pick (1, h, w) with the value at each valid slot's pixel and
    `fill` everywhere else, so the unchanged action selection (masked argmax / random among valid cells) only sees real candidates."""
    push = np.full((n_push, *shape), fill, np.float32)
    pick = np.full((1, *shape), fill, np.float32)
    for i in range(len(state['instances'])):
        for slot in range(state['valid'].shape[1]):
            if not state['valid'][i, slot]:
                continue
            rot, row, col = state['actions'][i, slot]
            target = pick[0] if slot == 0 else push[int(rot)]
            target[row, col] = max(target[row, col], q[i, slot])
    return push, pick


class ObjectQNetContinuous(ObjectQNet):
    """Object-level agent with a CONTINUOUS push length (hierarchical: the manager's values rank object-skills, the worker's parameter is
    the length). Per instance token: a pick value, an actor that proposes a push length per direction (0..1 of the length range), and a
    push critic Q(token, direction, length). Slots: 0 pick, 1 / 2 push in direction 0 / 1."""

    def __init__(self, dim=64, layers=2, heads=4):
        super().__init__(n_slots=3, dim=dim, layers=layers, heads=heads)
        del self.head
        self.pick_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.actor = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
        self.push_critic = nn.Sequential(nn.Linear(dim + 9, dim), nn.GELU(), nn.Linear(dim, 1))

    def _push_values(self, emb, lengths):
        """emb (B,N,dim), lengths (B,N,2) -> (B,N,2) push values for the two directions at those lengths."""
        outs = []
        for direction in range(2):
            onehot = torch.zeros(*emb.shape[:2], 2, device=emb.device)
            onehot[..., direction] = 1.0
            length = lengths[..., direction:direction + 1]
            freq = torch.arange(1, 4, device=emb.device) * 3.14159265  # Fourier features: a peaked value over the length is hard for a plain MLP
            feats = torch.cat([length, torch.sin(length * freq), torch.cos(length * freq)], dim=-1)
            outs.append(self.push_critic(torch.cat([emb, feats, onehot], dim=-1))[..., 0])
        return torch.stack(outs, dim=-1)

    def forward_full(self, tokens, counts, lengths=None):
        """-> (Q (B,N,3), actor lengths (B,N,2), push values at the actor's lengths computed from DETACHED features (B,N,2)).
        `lengths` (B,N,2) overrides the lengths the critic is evaluated at (training: the executed ones)."""
        emb = self.embed_tokens(tokens, counts)
        actor = torch.sigmoid(self.actor(emb.detach()))
        used = actor.detach() if lengths is None else lengths
        q = torch.cat([self.pick_head(emb), self._push_values(emb, used)], dim=-1)
        return q, actor, self._push_values(emb.detach(), actor)

    def forward(self, tokens, counts):
        return self.forward_full(tokens, counts)[0]
