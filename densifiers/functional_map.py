"""Functional-map densifier: solve the dense map in the LBO spectral domain.

Two signals set the map, both as spectral descriptor-preservation constraints on a functional
map C (k_y x k_x) that transfers functions from X to Y:

  * global  -- the trained extractor's dense per-vertex descriptors (ctx.feat_x/feat_y, filled
    by the model before densify; feat_source=diffnet). Carries the mid-band structure.
  * landmark -- each sparse correspondence enters as wave-kernel "bumps": the Gaussian
    band-passed response of a delta at the landmark, in the standard WKS sense. In the LBO basis
    the bump for landmark p at energy e has a closed-form spectral coefficient
    so no dense function is ever built.
"""
import torch

from utils.registry import DENSIFIER_REGISTRY
from utils.spectral_features import shared_wks_grid, wks_coefs
from .base_densifier import BaseDensifier, DensifyContext


@DENSIFIER_REGISTRY.register()
class FunctionalMapDensifier(BaseDensifier):
    """Extractor-feature + wave-kernel-landmark functional-map densifier with
    Laplacian-commutativity regularisation. densify() returns the hard nearest-neighbour p2p
    (Y vertex -> X vertex).

    Hyperparameters: k_fm and mu must move together, since a larger basis lets C fit errors
    in the sparse map and the ridge has to grow with it. n_e has no independent effect (it
    reaches the solve only through sigma), so set band width with variance instead.
    lm_weight is a weak knob, as it equalises the two blocks' trace while their effective
    ranks differ.
    """

    def __init__(self, opt=None):
        super().__init__(opt)
        o = self.opt
        self.k_fm = o.get('k_fm', 120)            # spectral basis size for C and the p2p readout
        self.n_e = o.get('n_e', 100)              # landmark energy grid resolution (sets sigma)
        self.variance = o.get('variance', 7.0)    # wave-kernel band-width factor
        self.lm_bands = o.get('lm_bands', 20)     # wave-kernel bands per landmark
        self.lm_weight = o.get('lm_weight', 5.0)  # landmark vs feature block trace weight
        self.mu = o.get('mu', 1e-1)               # Laplacian-commutativity weight (scale with k_fm)
        self.chunk = o.get('chunk', 512)          # Y rows per NN block (memory bound)
        self.eps = o.get('eps', 1e-8)
        self._landmarks = None                    # (m_x, l_y) set per densify() call

    @staticmethod
    def _normalise_pair(Ax, Ay, eps):
        """Scale each corresponding column pair by its X-norm so descriptors are comparable
        across bands; the same factor divides the Y column so the correspondence is preserved."""
        s = Ax.norm(dim=0, keepdim=True).clamp_min(eps)
        return Ax / s, Ay / s

    def _descriptors(self, ctx, dev):
        """Build the stacked spectral descriptor matrices A (k x D) on X and B (k x D) on Y from
        the dense extractor features and the wave-kernel landmark bumps, plus the truncated
        basis/spectra used by the solve and readout."""
        ex, ey = ctx.evecs_x.to(dev).float(), ctx.evecs_y.to(dev).float()   # (Vx,K),(Vy,K)
        vx, vy = ctx.evals_x.to(dev).float(), ctx.evals_y.to(dev).float()   # (K,),(K,)
        mx, my = ctx.mass_x.to(dev).float(), ctx.mass_y.to(dev).float()     # (Vx,),(Vy,)
        k = min(self.k_fm, ex.shape[1], ey.shape[1])

        # shared energy grid for the landmark bumps: one grid spanning BOTH spectra so band b
        # means the same energy on each shape. Only `sigma` and the subsampled `lm_e` below are
        # used -- n_e itself acts solely by setting sigma (see class docstring).
        energies, sigma = shared_wks_grid(vx, vy, self.n_e, self.variance, self.eps)

        exk, eyk = ex[:, :k], ey[:, :k]
        etx = (exk * mx[:, None]).T                                         # (k, Vx) = Phi^T M
        ety = (eyk * my[:, None]).T

        # global descriptor block, projected into the truncated basis (etx @ field -> (k, D)).
        # The model runs its extractor densely and fills ctx.feat_x/feat_y before densify().
        assert ctx.feat_x is not None and ctx.feat_y is not None, \
            ("FunctionalMapDensifier needs dense extractor features on ctx.feat_x/feat_y -- set "
             "densifier.feat_source: diffnet and give the model a DiffusionNet extractor")
        Gx = ctx.feat_x.to(dev).float()                                    # (Vx, d)
        Gy = ctx.feat_y.to(dev).float()
        Ax_g, Ay_g = etx @ Gx, ety @ Gy                                    # (k, D) each
        Ax_g, Ay_g = self._normalise_pair(Ax_g, Ay_g, self.eps)
        # divide each block by sqrt(#columns) so its trace contribution is 1 regardless of how
        # many constraints it holds; landmark strength is then set solely by lm_weight, decoupled
        # from lm_bands and n_sparse (which otherwise silently reweight the landmark block).
        Ax_g, Ay_g = Ax_g / Ax_g.shape[1] ** 0.5, Ay_g / Ay_g.shape[1] ** 0.5

        A_blocks, B_blocks = [Ax_g], [Ay_g]

        # wave-kernel landmark bumps: matched (X vertex, Y vertex) pairs from the sparse map
        if self._landmarks is not None:
            m_x, l_y = self._landmarks                                      # (n,),(n,) full-mesh
            lm_e = energies[torch.linspace(0, self.n_e - 1, min(self.lm_bands, self.n_e),
                                           device=dev).long()]
            cx = wks_coefs(vx[:k], lm_e, sigma, self.eps)                   # (k, b)
            cy = wks_coefs(vy[:k], lm_e, sigma, self.eps)
            # bump coeffs (k, n, b): coefs[k,b] * phi_k(landmark); flatten landmark x band -> cols
            Ax_l = (cx[:, None, :] * exk[m_x].T[:, :, None]).reshape(k, -1)
            Ay_l = (cy[:, None, :] * eyk[l_y].T[:, :, None]).reshape(k, -1)
            Ax_l, Ay_l = self._normalise_pair(Ax_l, Ay_l, self.eps)
            # count-normalise then apply lm_weight, so the landmark block contributes lm_weight
            # to the trace balance no matter how many landmarks x bands it holds.
            w = (self.lm_weight / Ax_l.shape[1]) ** 0.5
            A_blocks.append(w * Ax_l)
            B_blocks.append(w * Ay_l)

        A = torch.cat(A_blocks, dim=1)                                      # (k, D)
        B = torch.cat(B_blocks, dim=1)
        return A, B, exk, eyk, vx[:k], vy[:k]

    def _solve_fm(self, A, B, vx, vy):
        """Closed-form functional map C (k_y x k_x): descriptor preservation C A = B with the
        diagonal Laplacian-commutativity mask, solved row by row as ridge regression."""
        k = A.shape[0]
        G = A @ A.T                                                        # (k, k)
        R = B @ A.T                                                        # (k_y, k_x) row RHS
        scale = torch.maximum(vx.max(), vy.max()).clamp_min(self.eps)
        mask = ((vx[None, :] - vy[:, None]) / scale) ** 2                  # (k_y, k_x)
        reg = self.mu * G.diagonal().mean()
        Gi = G[None] + reg * torch.diag_embed(mask)                       # (k_y, k_x, k_x)
        C = torch.linalg.solve(Gi, R.unsqueeze(-1)).squeeze(-1)           # (k_y, k_x)
        return C

    def _nn_p2p(self, emb_x, emb_y):
        """Nearest neighbour Y -> X in the shared spectral embedding, chunked over Y."""
        N_y = emb_y.shape[0]
        p2p = torch.empty(N_y, dtype=torch.long, device=emb_x.device)
        for lo in range(0, N_y, self.chunk):
            sl = slice(lo, min(lo + self.chunk, N_y))
            p2p[sl] = torch.cdist(emb_y[sl], emb_x).argmin(dim=1)
        return p2p

    def densify(self, sparse_p2p: torch.Tensor, ctx: DensifyContext) -> torch.Tensor:
        assert all(t is not None for t in (ctx.evecs_x, ctx.evals_x, ctx.mass_x,
                                           ctx.evecs_y, ctx.evals_y, ctx.mass_y)), \
            "FunctionalMapDensifier needs ctx.evecs/evals/mass -- enable ret_evecs in the dataset"
        dev = ctx.evecs_x.device
        idx_x, idx_y = ctx.idx_x.to(dev).long(), ctx.idx_y.to(dev).long()
        m_x = idx_x[sparse_p2p.to(dev).long()]                            # X vertex per Y anchor
        self._landmarks = (m_x, idx_y)

        A, B, exk, eyk, vx, vy = self._descriptors(ctx, dev)
        C = self._solve_fm(A, B, vx, vy)
        emb_x = exk @ C.T                                                 # (Vx, k_y) X in Y basis
        return self._nn_p2p(emb_x, eyk)                                   # (Vy,) Y -> X one-shot NN
