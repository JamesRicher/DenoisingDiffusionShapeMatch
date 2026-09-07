"""MPNNMatrixDenoiser: geodesic-MPNN + state-track denoiser.

External contract, as consumed by MPNNDiffusionModel:

    forward(P_t, F_x, F_y, D_x, D_y, t) -> u0_hat   (B, n_y, n_x) logits

Per call (stateless across diffusion steps - only P_t carries the trajectory):

    c = spine(t);  precompute geodesic kNN graphs, distance RBFs and feature-kNN
    candidates;  u = log P_t;  tokens from [features, warp, stats].

    L x block:
        intra(X), intra(Y)      geodesic MPNN, shared instance      (intra_mpnn)
        cross(X<->Y)            candidate-graph bipartite GNN       (cross_stage)
        u <- StateWrite(...)    feature/affinity write              (state_track)
        u <- soft_project_sym   cheap re-gauge, transpose-symmetric
        rewarp(X), rewarp(Y)    refreshed belief back into tokens

    return u  (identity at init: all writes gated to zero -> u0_hat = log P_t).

Properties by construction: full permutation equivariance (no anchor frame), pair-swap
symmetry, size agnosticism, intrinsic inputs only, identity at init.
"""
import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY
from utils.sinkhorn import safe_log
from networks.denoiser_conditioning import ConditioningSpine
from networks.mpnn.geometry import (RBFEmbed, feature_knn, knn_from_dist,
                                    normalise_knn_dist)
from networks.mpnn.intra_mpnn import IntraMPNNLayer
from networks.mpnn.cross_stage import build_cross_stage
from networks.mpnn.state_track import (InputEmbed, Rewarp, StateWrite,
                                       soft_project_sym)


@NETWORK_REGISTRY.register()
class MPNNMatrixDenoiser(nn.Module):
    """Denoise P_t -> clean logit matrix via interleaved intra-MPNN / cross / state writes.

    Args:
        feat_dim: per-point input feature dimension.
        dim: token width.
        depth: number of [intra|intra|cross|write] blocks.
        k_intra: geodesic-kNN neighbours for the intra graphs.
        k_feat: static feature-kNN candidates per receiver (mpnn cross only).
        k_state: dynamic state-top-k candidates per receiver (mpnn cross only).
        n_rbf: RBF basis size (distance and state embeddings).
        rbf_max: upper end of the normalised-distance RBF grid (distances are
            divided by the mean k-th-NN radius, so ~[0, 3] covers the graph).
        inner_iters: soft-Sinkhorn iterations per in-block re-gauge.
        mlp_ratio, dropout: FFN settings.
        time_scale: t scaling into the sinusoidal spine.
    """

    def __init__(self, feat_dim: int, dim: int, depth: int = 5,
                 k_intra: int = 12, k_feat: int = 10,
                 k_state: int = 10, n_rbf: int = 16, rbf_max: float = 3.0,
                 inner_iters: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 time_scale: float = 1000.0):
        super().__init__()
        self.k_intra = k_intra
        self.k_feat = k_feat
        self.inner_iters = inner_iters

        self.spine = ConditioningSpine(dim, time_scale=time_scale)
        self.embed = InputEmbed(feat_dim, dim)
        self.dist_rbf = RBFEmbed(n_rbf, 0.0, rbf_max)

        self.intra = nn.ModuleList(
            IntraMPNNLayer(dim, n_rbf, mlp_ratio, dropout) for _ in range(depth))
        self.cross = nn.ModuleList(
            build_cross_stage(dim, k_state, n_rbf, mlp_ratio, dropout)
            for _ in range(depth))
        self.write = nn.ModuleList(StateWrite(dim, n_rbf) for _ in range(depth))
        # no rewarp after the final block: its output tokens are never read again
        self.rewarp = nn.ModuleList(Rewarp(feat_dim, dim) for _ in range(depth - 1))


    def _geo(self, D: torch.Tensor):
        """Static per-call intra-graph tensors: kNN indices + RBF-embedded distances."""
        idx, dist = knn_from_dist(D, min(self.k_intra, D.shape[-1] - 1))
        return idx, self.dist_rbf(normalise_knn_dist(dist))

    def _regauge(self, u: torch.Tensor) -> torch.Tensor:
        """Re-gauge the state after a write: transpose-symmetric truncated Sinkhorn."""
        return soft_project_sym(u, n_iters=self.inner_iters)

    def _warps(self, u: torch.Tensor):
        """The two warp matrices read off the current state, (P_y, P_x): P_y (B, n_y, n_x)
        rows=Y for the Y-token warp, P_x (B, n_x, n_y) rows=X for the X-token warp. Both are
        views of the single (near) doubly-stochastic P = u.exp()."""
        P = u.exp()
        return P, P.transpose(-1, -2)

    def forward(self, P_t: torch.Tensor, F_x: torch.Tensor, F_y: torch.Tensor,
                D_x: torch.Tensor, D_y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """P_t: (B, n_y, n_x) projected read-in Pi_S(u_t); F_*: (B, n, feat_dim);
        D_*: (B, n, n) geodesics; t: (B,) or scalar. Returns u0_hat logits."""
        c = self.spine(t)

        # ---- static per-call precompute ---- #
        idx_x, rbf_x = self._geo(D_x)
        idx_y, rbf_y = self._geo(D_y)
        k_f = min(self.k_feat, F_y.shape[1])
        cand_x = feature_knn(F_x, F_y, k_f)              # X receivers <- Y senders
        cand_y = feature_knn(F_y, F_x, k_f)              # Y receivers <- X senders

        # ---- state + tokens ---- #
        u = safe_log(P_t)                                # gauge-fixed read-in
        # Y warps through rows of P_t; X through P_t^T
        P_y = P_t
        P_x = P_t.transpose(-1, -2)
        hy = self.embed(F_y, P_y, F_x)                   # rows of P: Y pulls back X
        hx = self.embed(F_x, P_x, F_y)

        # ---- blocks ---- #
        for i, (intra, cross, write) in enumerate(zip(self.intra, self.cross, self.write)):
            hx = intra(hx, idx_x, rbf_x, c)
            hy = intra(hy, idx_y, rbf_y, c)
            hx, hy = cross(hx, hy, u, c, cand_x=cand_x, cand_y=cand_y)

            u = write(hx, hy, u, c)
            u = self._regauge(u)

            if i < len(self.rewarp):                     # refreshed belief bridge
                P_y, P_x = self._warps(u)
                hy = self.rewarp[i](hy, P_y, F_x)
                hx = self.rewarp[i](hx, P_x, F_y)

        return u
