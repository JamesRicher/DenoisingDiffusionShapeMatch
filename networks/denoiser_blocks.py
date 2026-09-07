"""DiT-style AdaLN-Zero modulation and the shared feed-forward block.

AdaLN-Zero is driven by a conditioning vector c (the timestep pathway, from
networks/denoiser_conditioning.py).

Gates are zero-init, so a block using this is exactly the identity at init and the
timestep conditioning fades in during training. Consumed by the MPNN intra and cross
stages, which supply their own sublayers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN affine modulation: x * (1 + scale) + shift. shift/scale are (B, 1, D),
    broadcast over tokens."""
    return x * (1 + scale) + shift


class AdaLNModulation(nn.Module):
    """Maps conditioning c -> (shift, scale, gate) for each of a block's sublayers.

    Zero-init (AdaLN-Zero): at init shift=scale=0 (norm passes through) and gate=0 (the
    sublayer's residual contributes nothing), so the block starts as the identity.
    """
    def __init__(self, dim: int, n_sublayers: int = 2):
        super().__init__()
        self.n_sublayers = n_sublayers
        self.proj = nn.Linear(dim, 3 * n_sublayers * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: torch.Tensor):
        # SiLU then Linear (DiT convention). (B, 3*n_sublayers*D) -> tokens broadcast.
        params = self.proj(F.silu(c)).unsqueeze(1)          # (B, 1, 3*n_sublayers*D)
        return params.chunk(3 * self.n_sublayers, dim=-1)   # 3*n_sublayers x (B, 1, D)


class FeedForward(nn.Module):
    """Two linear layers with a GELU between them, the sublayer FFN used by the MPNN
    intra and cross stages.

    Args:
        d_model: token width.
        mlp_ratio: hidden expansion factor.
        dropout: dropout applied after each linear.
    """
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
