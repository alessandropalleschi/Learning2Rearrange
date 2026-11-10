"""Return decomposition (RUDDER, Arjona-Medina et al. 2019) for long-horizon credit assignment (`--rudder LAMBDA`).

A causal transformer reads the episode step by step (each step: a summary of the state and what the agent did and what came of it)
and predicts the episode's return after every step. The change of that prediction caused by a step is credited to it:

    r'_t = g(tau_0..t) - g(tau_0..t-1),     g(tau_0..-1) = 0,

and the last step absorbs the residual so that the redistributed rewards sum to the true return. A clearing push that makes a later
grasp possible raises the predicted return when it happens, so it is credited immediately instead of through several bootstraps.
The redistributed reward replaces a share LAMBDA of the environment reward in the TD targets (`training.py`).

Step tokens: the object agent's global token of the state (object counts, misplaced share, last action, held class) and a history
entry of the action (object class at the action cell, cell, primitive, executed, changed, parameters; see
`object_centric.make_history_entry`); both come from the observations alone."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from l2r.learning.models import DEVICE
from l2r.learning.object_centric import FEATURES, build_state, make_history_entry

STEP_FEATURES = 2 * FEATURES
MAX_STEPS = 64


def step_token(obs, primitive, pix, next_obs):
    """(STEP_FEATURES,) float32 summary of one step from the observations."""
    state = build_state(obs, [16])
    return np.concatenate([state['tokens'][0], make_history_entry(obs, primitive, pix, next_obs)]).astype(np.float32)


class ReturnPredictor(nn.Module):
    def __init__(self, dim=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(STEP_FEATURES, dim)
        self.position = nn.Embedding(MAX_STEPS, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, steps, lengths):
        """steps (B, T, STEP_FEATURES), lengths (B,) -> (B, T) predicted return after each step (causal)."""
        T = steps.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=steps.device), diagonal=1)
        pad = torch.arange(T, device=steps.device)[None] >= lengths[:, None]
        x = self.embed(steps) + self.position(torch.arange(T, device=steps.device).clamp(max=MAX_STEPS - 1))[None]
        return self.head(self.encoder(x, mask=causal, src_key_padding_mask=pad))[..., 0]


class Rudder:
    """Episode memory, return predictor and redistribution."""

    def __init__(self, capacity=2000, lr=1e-3, batch=16, device=DEVICE):
        self.model = ReturnPredictor().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.episodes, self.capacity, self.batch, self.device = [], capacity, batch, device
        self.mean_return = 0.0

    def add(self, steps, ret):
        """Store a finished episode: steps (T, STEP_FEATURES), its (discounted or plain) return."""
        steps = np.asarray(steps, np.float32)[:MAX_STEPS]
        self.episodes.append((steps, float(ret)))
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
        self.mean_return = float(np.mean([r for _, r in self.episodes]))

    def _collate(self, episodes):
        T = max(len(s) for s, _ in episodes)
        steps = np.zeros((len(episodes), T, STEP_FEATURES), np.float32)
        for b, (s, _) in enumerate(episodes):
            steps[b, :len(s)] = s
        lengths = torch.tensor([len(s) for s, _ in episodes], device=self.device)
        returns = torch.tensor([r for _, r in episodes], device=self.device)
        return torch.from_numpy(steps).to(self.device), lengths, returns

    def train(self, updates=1, rng=np.random):
        """Regress the final return from every prefix (later prefixes weigh more: they know more)."""
        if len(self.episodes) < 2:
            return None
        self.model.train()
        loss = None
        for _ in range(updates):
            batch = [self.episodes[i] for i in rng.randint(len(self.episodes), size=min(self.batch, len(self.episodes)))]
            steps, lengths, returns = self._collate(batch)
            pred = self.model(steps, lengths)
            T = steps.shape[1]
            valid = torch.arange(T, device=self.device)[None] < lengths[:, None]
            weight = valid.float() * (torch.arange(T, device=self.device)[None] + 1).float() / lengths[:, None].float()
            last = pred.gather(1, (lengths - 1)[:, None])[:, 0]
            loss = (weight * (pred - returns[:, None]) ** 2).sum() / weight.sum() + F.mse_loss(last, returns)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def redistribute(self, steps, ret):
        self.model.eval()
        """(T,) redistributed rewards of an episode: differences of the predicted return, summing to `ret`."""
        steps = np.asarray(steps, np.float32)[:MAX_STEPS]
        s, lengths, _ = self._collate([(steps, ret)])
        pred = self.model(s, lengths)[0].cpu().numpy()
        out = np.diff(np.concatenate([[0.0], pred]))  # the first step carries the expected return, later steps the changes
        out[-1] += ret - out.sum()  # the redistribution sums to the true return
        self.model.train()
        return out.astype(np.float32)
