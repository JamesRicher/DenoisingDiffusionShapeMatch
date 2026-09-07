"""State track for the MPNN denoiser: the assignment lives inside the trunk.

The denoiser maintains an internal log-assignment state u through the blocks:

    u^0 = log P_t                      (gauge-fixed read-in)
    per block:  u <- u + gate_t * pair_mlp(affinity, rbf(u))       (StateWrite)
                u <- soft_project_sym(u)                           (re-gauge)
    output:     u0_hat = final u       (unconstrained logits; Pi_S external)

Every write is a zero-init gated residual, so at init u0_hat = log P_t and
Pi_S(u0_hat) = P_t exactly. The network therefore estimates the correction to the
current assignment.

Orientation: u, P are (B, n_y, n_x), rows = Y. Y tokens pull back X features
through rows of P; X tokens through rows of P^T.
"""
import torch
import torch.nn as nn

from utils.sinkhorn import log_sinkhorn, safe_log
from networks.mpnn.geometry import RBFEmbed

U_LO = -18.0   # ~log(1e-8): the state RBF grid / clamp range, matches safe_log's floor


def warp_stats(P_dir: torch.Tensor, F_src: torch.Tensor,
               eps: float = 1e-8) -> torch.Tensor:
    """Assignment-aware context for one shape's tokens.

    P_dir: (B, n_dst, n_src) receiver-major soft assignment (rows ~sum to 1).
    F_src: (B, n_src, d) partner-shape features to pull back.
    Returns (B, n_dst, d + 2): [warped features, row max, row entropy].

    The warp alone averages confidence away, hence the explicit row max/entropy
    scalars alongside it.
    """
    warped = P_dir @ F_src                                       # (B, n_dst, d)
    row_max = P_dir.max(dim=-1, keepdim=True).values             # (B, n_dst, 1)
    row_ent = -(P_dir * safe_log(P_dir, eps)).sum(-1, keepdim=True)
    return torch.cat([warped, row_max, row_ent], dim=-1)


class InputEmbed(nn.Module):
    """Initial tokens: [own features, warped partner features, state stats] -> dim.

    One projection shared by both shapes (pair-swap symmetry). No anchor
    coordinates: the intra MPNN supplies all geometric structure through its
    edges, which buys full permutation equivariance.
    """

    def __init__(self, feat_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(2 * feat_dim + 2, dim)

    def forward(self, F_own: torch.Tensor, P_dir: torch.Tensor,
                F_partner: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([F_own, warp_stats(P_dir, F_partner)], dim=-1))


class Rewarp(nn.Module):
    """Per-block belief bridge: re-inject warp+stats from the freshly written state.

    Zero-init projection -> contributes nothing at init; the trunk learns to use
    the refreshed belief. One instance per block, shared across shapes.
    """

    def __init__(self, feat_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim + 2, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, P_dir: torch.Tensor,
                F_partner: torch.Tensor) -> torch.Tensor:
        return h + self.proj(warp_stats(P_dir, F_partner))


class StateWrite(nn.Module):
    """The only writer of u: gated residual from feature affinity + state.

    Pair features are scalars, never concatenated token pairs, so the dense
    (n_y, n_x) application is one matmul plus a tiny MLP:

        affinity = (W h_y) (W h_x)^T / sqrt(D)     [shared W -> swap-symmetric]
        du = pair_mlp([affinity, rbf(u), 0])
        u <- u + gate(c) * du                       [gate zero-init, time-driven]

    The pair-MLP's third input channel is a legacy slot, always fed zeros. It is kept so
    the trained weight layout (and hence the released checkpoints) stays loadable.
    """

    def __init__(self, dim: int, n_rbf: int = 16, hidden: int = 32):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.u_rbf = RBFEmbed(n_rbf, U_LO, 0.0)
        self.pair_mlp = nn.Sequential(nn.Linear(1 + n_rbf + 1, hidden), nn.SiLU(),
                                      nn.Linear(hidden, 1))
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, hx: torch.Tensor, hy: torch.Tensor, u: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        """hx: (B, n_x, D); hy: (B, n_y, D); u: (B, n_y, n_x); c: (B, D)."""
        affinity = (self.W(hy) @ self.W(hx).transpose(-1, -2)) * self.scale
        feats = torch.cat([affinity.unsqueeze(-1),
                           self.u_rbf(u.clamp(U_LO, 0.0)),
                           torch.zeros_like(u).unsqueeze(-1)], dim=-1)  # (B, n_y, n_x, 2+n_rbf)
        du = self.pair_mlp(feats).squeeze(-1)                    # (B, n_y, n_x)
        gate = self.gate(c).unsqueeze(-1)                        # (B, 1, 1)
        return u + gate * du


def soft_project_sym(u: torch.Tensor, n_iters: int = 3) -> torch.Tensor:
    """Cheap in-block re-gauging toward the Birkhoff polytope, transpose-symmetric.

    Truncated Sinkhorn is order-dependent (rows-then-cols != cols-then-rows), which
    would break the pair-swap symmetry f(u^T) = f(u)^T; averaging the two orderings
    restores it exactly. Full-strength projection stays external (loss / sampler).
    """
    a = log_sinkhorn(u, n_iters=n_iters)
    b = log_sinkhorn(u.transpose(-1, -2), n_iters=n_iters).transpose(-1, -2)
    return 0.5 * (a + b)
