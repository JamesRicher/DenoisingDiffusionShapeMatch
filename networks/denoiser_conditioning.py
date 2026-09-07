"""Timestep conditioning for the denoiser: c = MLP_t(sinusoid(t)), which drives the
AdaLN-Zero modulation in every block.
"""
import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# conditioning spine: t -> c
# --------------------------------------------------------------------------- #
def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal embedding of a scalar per sample. t: (B,) -> (B, dim).

    Frequencies are geometric in (1/max_period, 1]; scale t into a range that spans
    them before calling (see ConditioningSpine.time_scale) or the embedding barely
    varies over t in [0, 1].
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad to dim when odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ConditioningSpine(nn.Module):
    """Timestep pathway c = MLP_t(sinusoid(t)) driving AdaLN-Zero in every block.

    forward returns a SUM of contributions (currently just the t pathway); the cascade's
    s and level embeddings add into that same sum later, zero-init, with no surgery here.

    Args:
        dim: conditioning width (matches the token dim the AdaLN blocks expect).
        embed_dim: sinusoidal embedding width before the MLP (defaults to dim).
        time_scale: t is multiplied by this before embedding so continuous t in [0, 1]
            spans the sinusoidal frequencies (1000 mirrors 1000-step diffusion).
        max_period: sinusoidal max period.
    """
    def __init__(self, dim: int, embed_dim: int | None = None,
                 time_scale: float = 1000.0, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim or dim
        self.time_scale = time_scale
        self.max_period = max_period
        self.t_mlp = nn.Sequential(nn.Linear(self.embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=self.t_mlp[0].weight.device)
        if t.dim() == 0:
            t = t[None]
        emb = sinusoidal_embedding(t * self.time_scale, self.embed_dim, self.max_period)
        c = self.t_mlp(emb)
        # cascade extras add into this sum later: c = c + self.s_mlp(sinusoid(s)) + level_emb
        return c
