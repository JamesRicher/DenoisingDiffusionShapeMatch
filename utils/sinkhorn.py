"""Sinkhorn + Birkhoff-polytope utilities for the sparse matrix diffusion matcher.

Soft assignment matrices P have shape (..., R, C) with R = n_y rows, C = n_x cols.

Noising is DDPM in the logit domain: the diffusion variable is an
unconstrained logit matrix u; a doubly-stochastic P = Pi_S(u) = log_sinkhorn(u).exp() is
only ever the projected view. logit_target embeds the GT permutation as a finite logit u0,
q_sample runs the variance-preserving forward marginal on u0, and cosine_alpha_bar is the
schedule. Time convention: t=0 is clean data, t=1 is noise.
"""
import math
from typing import Optional

import torch


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """log with a floor so a zero entry gives a large finite value, not -inf.

    Used wherever a log of an assignment matrix feeds a bias/logit (a -inf bias is a
    dead, zero-gradient edge)."""
    return torch.log(x.clamp_min(eps))


def log_sinkhorn(
    log_alpha: torch.Tensor,
    n_iters: int = 10,
    tau: float = 1.0,
    log_mu: Optional[torch.Tensor] = None,
    log_nu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project a score matrix onto the Birkhoff polytope in the log domain.

    Alternating row/col log-normalisation on log_alpha / tau, differentiable by unrolling
    and stable at small tau.

    Args:
        log_alpha: (..., R, C) logits (log of the unnormalised assignment).
        n_iters: Sinkhorn iterations (each = one row + one col pass).
        tau: temperature; smaller sharpens toward a hard permutation.
        log_mu, log_nu: target log row/col marginals; default 0 (unit sums).

    Returns log P (..., R, C); the last pass normalises columns, so col marginals are
    exact and row marginals hold to tolerance.
    """
    log_p = log_alpha / tau
    for _ in range(n_iters):
        # rows -> mu
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        if log_mu is not None:
            log_p = log_p + log_mu.unsqueeze(-1)
        # cols -> nu
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
        if log_nu is not None:
            log_p = log_p + log_nu.unsqueeze(-2)
    return log_p


def row_logprob(u: torch.Tensor) -> torch.Tensor:
    """Row-stochastic read of a logit matrix: log-softmax over the last axis.

    The single-pass, one-sided analogue of log_sinkhorn for the row-stochastic (non
    doubly-stochastic) variant: each row j becomes log P(x | y_j), a proper distribution
    over columns, with no column constraint. Take .exp() for the assignment P.
    """
    return u - torch.logsumexp(u, dim=-1, keepdim=True)


def sample_gumbel(shape, device=None, dtype=torch.float32, generator=None) -> torch.Tensor:
    """i.i.d. standard Gumbel noise -log(-log(U)), U ~ Uniform(0, 1)."""
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    # clamp guards log(0) at the tails of the uniform draw
    u = u.clamp_(min=torch.finfo(dtype).tiny)
    return -torch.log(-torch.log(u))


def sample_doubly_stochastic(
    n_rows: int,
    n_cols: int,
    tau: float = 1.0,
    n_iters: int = 20,
    batch_shape=(),
    device=None,
    dtype=torch.float32,
    generator=None,
) -> torch.Tensor:
    """Random doubly-stochastic matrix via Sinkhorn(Gumbel noise).

    Maximum-entropy sample on the polytope; as tau -> 0 it concentrates on a uniformly
    random permutation. A DS prior / test fixture, not the forward noise (see q_sample).

    Returns P_noise (*batch_shape, n_rows, n_cols) in the probability domain.
    """
    g = sample_gumbel((*batch_shape, n_rows, n_cols), device=device, dtype=dtype, generator=generator)
    return log_sinkhorn(g, n_iters=n_iters, tau=tau).exp()


def logit_target(P0: torch.Tensor, eta: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    """Clean logit target u0 = log((1 - eta)*P0 + eta/m*1) for the logit-space diffusion.

    log(P0) has -inf on the zeros of a (near-)permutation, so smoothing toward the
    row-barycenter gives a bounded target whose magnitude sets the near-t=0 difficulty.

    Args:
        P0: (..., R, C) ground-truth assignment, rows sum to 1.
        eta: smoothing weight in [0.05, 0.2]; smaller = sharper target, larger logits.
    Returns u0 (..., R, C), an unconstrained logit matrix.
    """
    m = P0.shape[-1]
    P_tilde = (1.0 - eta) * P0 + eta / m
    return safe_log(P_tilde, eps)


def _normalise_kernel(
    K: torch.Tensor,
    doubly_stochastic: bool = False,
    ds_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalise an affinity kernel K into a soft-assignment target.

    doubly_stochastic False (default): row-normalise, so each query is a distribution over
    anchors. True: Sinkhorn to a doubly-stochastic entropic-OT coupling, which puts the
    target in the same Birkhoff polytope as the read-in Pi_S. Sinkhorn ends on a column
    pass so rows still sum to ~1, leaving row-CE unchanged.
    """
    if not doubly_stochastic:
        return K / K.sum(-1, keepdim=True).clamp_min(eps)
    return log_sinkhorn(safe_log(K, eps), n_iters=ds_iters).exp()


def gaussian_target(
    P0: torch.Tensor,
    D_x: torch.Tensor,
    sigma: float,
    cutoff: Optional[float] = None,
    floor: float = 2e-4,
    eps: float = 1e-8,
    doubly_stochastic: bool = False,
    ds_iters: int = 20,
) -> torch.Tensor:
    """Geodesic soft assignment target: row j proportional to exp(-d(g(j), .)^2 / 2*sigma^2),
    floored past cutoff.

    Puts a metric on the label space that plain row-CE against a permutation lacks: a
    geodesically close near-miss keeps target mass, where a far miss costs the same as any
    other. The GT match's kernel row is selected by P0 itself (T = P0 K).

    Args:
        P0: (..., R, C) ground-truth assignment, rows sum to 1.
        D_x: (..., C, C) geodesic distances between X sparse points (sqrt-area units).
        sigma: kernel width in D_x units, roughly the anchor NN spacing.
        cutoff: distance beyond which the kernel drops to floor; default 3*sigma.
        floor: tail value past the cutoff, i.e. the uniform label-smoothing mass.
        doubly_stochastic: Sinkhorn-normalise to an entropic-OT target instead.
        ds_iters: Sinkhorn iterations when doubly_stochastic.
    Returns T (..., R, C) soft target; safe_log(T) embeds it as logits.
    """
    if cutoff is None:
        cutoff = 3.0 * sigma
    K = torch.exp(-D_x.pow(2) / (2.0 * sigma ** 2))
    K = torch.where(D_x > cutoff, K.new_tensor(floor), K)
    return _normalise_kernel(P0 @ K, doubly_stochastic, ds_iters, eps)


def gaussian_target_from_dist(
    D_cross: torch.Tensor,
    sigma: float,
    cutoff: Optional[float] = None,
    floor: float = 2e-4,
    eps: float = 1e-8,
    doubly_stochastic: bool = False,
    ds_iters: int = 20,
) -> torch.Tensor:
    """Geodesic soft target for INDEPENDENT (non-corresponding) sparse points: row j is
    proportional to exp(-D_cross[j,.]^2 / 2*sigma^2), floored past cutoff.

    The independent-FPS analogue of gaussian_target. There is no P0 to select a kernel row,
    since the query's GT image need not coincide with any source anchor; the precomputed
    D_cross[j, i] = geodesic(GT image of query j, source anchor i) is used directly. Same
    kernel semantics as gaussian_target, so losses stay comparable.

    Args:
        D_cross: (..., R, C) query-image -> source-anchor geodesic distances (sqrt-area units).
        sigma, cutoff, floor: as in gaussian_target.
        doubly_stochastic, ds_iters: as in gaussian_target (default False = row-stochastic).
    Returns T (..., R, C) soft target (row-stochastic by default; doubly-stochastic if requested).
    """
    if cutoff is None:
        cutoff = 3.0 * sigma
    K = torch.exp(-D_cross.pow(2) / (2.0 * sigma ** 2))
    K = torch.where(D_cross > cutoff, K.new_tensor(floor), K)
    return _normalise_kernel(K, doubly_stochastic, ds_iters, eps)


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008, logsnr_shift: float = 0.0) -> torch.Tensor:
    """Cosine VP schedule alpha_bar(t) (Nichol & Dhariwal). t in [0, 1]: alpha_bar(0)=1
    (clean), alpha_bar(1)=0.

    alpha_bar(t) = f(t)/f(0), f(u) = cos((u + s)/(1 + s)*pi/2)^2. The small offset s keeps
    alpha_bar from
    dropping too fast near t=0. Accepts a scalar or a batch tensor of times.

    logsnr_shift b uniformly shifts the schedule's log-SNR by b nats (Hoogeboom et al.
    2023): b < 0 lowers SNR at every interior t, sliding the informative transition toward
    mid-trajectory; b = 0 is the unshifted cosine. The endpoints are preserved for any b,
    so it stays a valid VP schedule. This is the knob for matching the schedule to a
    high-effective-SNR target rather than inheriting the image-domain profile.
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    f = lambda u: torch.cos((u + s) / (1.0 + s) * math.pi / 2.0) ** 2
    ab = f(t) / f(torch.zeros_like(t))
    if logsnr_shift != 0.0:
        e = math.exp(-logsnr_shift)
        ab = ab / (ab + e * (1.0 - ab)).clamp_min(1e-12)
    return ab


def q_sample(
    u0: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    s: float = 0.008,
    logsnr_shift: float = 0.0,
) -> torch.Tensor:
    """Forward marginal in logit space: u_t = sqrt(alpha_bar(t))*u0 +
    sqrt(1-alpha_bar(t))*eps, with eps ~ N(0, I).

    Variance-preserving DDPM noising applied entirely in the unconstrained logit chart,
    which is what makes the DDPM/DDIM scaffold legitimate over the polytope. Project with
    Pi_S to get the doubly-stochastic P_t the denoiser reads.

    Args:
        u0: (..., R, C) clean logit target (from logit_target).
        t: scalar, or a batch-shaped tensor reshaped to broadcast over R, C.
        noise: optional eps (same shape as u0); sampled standard normal if None.
    Returns u_t (..., R, C), an unconstrained logit matrix.
    """
    if noise is None:
        noise = torch.randn_like(u0)
    ab = cosine_alpha_bar(t, s, logsnr_shift)
    if ab.dim() > 0 and ab.dim() == u0.dim() - 2:
        ab = ab.reshape(*ab.shape, 1, 1)
    return ab.sqrt() * u0 + (1.0 - ab).clamp_min(0.0).sqrt() * noise


def tau_schedule(
    t: torch.Tensor,
    tau_min: float,
    tau_max: float,
    mode: str = "geometric",
) -> torch.Tensor:
    """Sampler-time temperature vs diffusion time t (sampler only; training stays temperate).

    Anneals from tau_max at t=1 to tau_min at t=0, so Pi_S sharpens toward a permutation
    as sampling approaches the data. mode is "geometric" (log-linear) or "linear".
    """
    if mode == "geometric":
        return tau_min ** (1.0 - t) * tau_max ** t
    if mode == "linear":
        return (1.0 - t) * tau_min + t * tau_max
    raise ValueError(f"unknown tau_schedule mode: {mode!r}")


# --------------------------------------------------------------------------- #
# unit tests (Step 1 "done when"): marginals in tolerance, tau->0 recovers a
# permutation, gradients flow.  Run: python -m utils.sinkhorn
# --------------------------------------------------------------------------- #
def _run_tests() -> None:
    torch.manual_seed(0)
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f'[{"PASS" if ok else "FAIL"}] {name:42s}' + (f'  {detail}' if detail else ""))

    B, n = 4, 32

    # --- marginals within tolerance (square, unit marginals) --------------- #
    logits = torch.randn(B, n, n)
    P = log_sinkhorn(logits, n_iters=30, tau=1.0).exp()
    row_err = (P.sum(-1) - 1).abs().max().item()
    col_err = (P.sum(-2) - 1).abs().max().item()
    check("square marginals -> 1", max(row_err, col_err) < 1e-4,
          f"row_err={row_err:.1e} col_err={col_err:.1e}")

    # --- rectangular unit-row / scaled-col marginals ----------------------- #
    R, C = 24, 32
    lr = torch.randn(B, R, C)
    log_nu = torch.full((B, C), torch.log(torch.tensor(R / C)).item())
    Pr = log_sinkhorn(lr, n_iters=50, tau=1.0, log_nu=log_nu).exp()
    r_err = (Pr.sum(-1) - 1).abs().max().item()          # rows -> 1
    c_err = (Pr.sum(-2) - R / C).abs().max().item()      # cols -> R/C
    check("rect marginals (rows->1, cols->R/C)", max(r_err, c_err) < 1e-3,
          f"row_err={r_err:.1e} col_err={c_err:.1e}")

    # --- tau -> 0 recovers a permutation ----------------------------------- #
    # Plant a permutation with a separated cost; entropic Sinkhorn must converge to
    # it (a near one-hot matrix). Random logits are avoided here: a row with two
    # near-equal-cost columns keeps ~0.5/0.5 mass at any finite tau even though its
    # argmax is a valid permutation, so per-entry sharpness would test the cost, not
    # the solver.
    perm = torch.stack([torch.randperm(n) for _ in range(B)])
    planted = torch.zeros(B, n, n)
    planted.scatter_(-1, perm.unsqueeze(-1), 1.0)
    planted = 5.0 * planted + 0.1 * torch.randn(B, n, n)
    Phard = log_sinkhorn(planted, n_iters=200, tau=0.02).exp()
    row_max = Phard.max(-1).values.min().item()          # every row nearly one-hot
    recovered = torch.equal(Phard.argmax(-1), perm)
    check("tau->0 recovers a permutation", row_max > 0.99 and recovered,
          f"min row-max={row_max:.3f} recovered={recovered}")

    # --- random doubly stochastic sampler ---------------------------------- #
    Pn = sample_doubly_stochastic(n, n, tau=1.0, n_iters=30, batch_shape=(B,))
    ds_err = max((Pn.sum(-1) - 1).abs().max().item(), (Pn.sum(-2) - 1).abs().max().item())
    check("sample_doubly_stochastic is DS", ds_err < 1e-4, f"marg_err={ds_err:.1e}")

    # --- row/col logprob: proper distributions along their axis ------------ #
    u = torch.randn(B, n, n)
    Pr = row_logprob(u).exp()
    Pc = col_logprob(u).exp()
    r_err = (Pr.sum(-1) - 1).abs().max().item()          # rows -> 1
    c_err = (Pc.sum(-2) - 1).abs().max().item()          # cols -> 1
    check("row_logprob/col_logprob normalise their axis", max(r_err, c_err) < 1e-5,
          f"row_err={r_err:.1e} col_err={c_err:.1e}")

    # --- invariant: on a ROW-normalised u, col_softmax == col-normalise(P) -- #
    # (what makes the cheap X-side col read equal the uniform-prior inversion of P)
    ur = row_logprob(u)                                  # enforce the maintained invariant Z_j = 1
    P = ur.exp()
    Q_via_col = col_logprob(ur).exp()                    # col-softmax of the row-normalised logits
    Q_via_norm = P / P.sum(-2, keepdim=True)             # explicit column-normalise of P
    inv_err = (Q_via_col - Q_via_norm).abs().max().item()
    check("col_logprob(row-normalised u) == col-normalise(P)", inv_err < 1e-5,
          f"max_diff={inv_err:.1e}")

    # --- row/col logprob gradients flow ------------------------------------ #
    x = torch.randn(B, n, n, requires_grad=True)
    (row_logprob(x).exp().sum() + col_logprob(x).exp().sum()).backward()
    rc_grad = x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    check("row/col logprob gradients flow", bool(rc_grad))

    # --- logit_target: row-stochastic, log finite, recovers the permutation - #
    perm2 = torch.stack([torch.randperm(n) for _ in range(B)])
    P0 = torch.zeros(B, n, n).scatter_(-1, perm2.unsqueeze(-1), 1.0)
    u0 = logit_target(P0, eta=0.1)
    P_tilde = u0.exp()
    lt_row = (P_tilde.sum(-1) - 1).abs().max().item()               # rows sum to 1
    lt_finite = torch.isfinite(u0).all().item()
    lt_recover = torch.equal(u0.argmax(-1), perm2)                  # argmax = GT match
    check("logit_target row-stochastic + finite + recovers perm",
          lt_row < 1e-5 and bool(lt_finite) and lt_recover, f"row_err={lt_row:.1e}")

    # --- cosine_alpha_bar endpoints + monotone decreasing ------------------ #
    ts = torch.linspace(0, 1, 11)
    ab = cosine_alpha_bar(ts)
    ab_ends = abs(ab[0].item() - 1.0) < 1e-6 and ab[-1].item() < 1e-6
    ab_mono = bool((ab[1:] <= ab[:-1]).all())
    check("cosine_alpha_bar endpoints + monotone", ab_ends and ab_mono,
          f"abar(0)={ab[0]:.3f} abar(1)={ab[-1]:.1e}")

    # --- q_sample: VP forward marginal; endpoints + projection is DS -------- #
    t = torch.rand(B)
    eps = torch.randn(B, n, n)
    u_t = q_sample(u0, t, noise=eps)
    q0_err = (q_sample(u0, torch.zeros(B), noise=eps) - u0).abs().max().item()  # t=0 -> u0
    q1_err = (q_sample(u0, torch.ones(B), noise=eps) - eps).abs().max().item()  # t=1 -> noise
    Pt = log_sinkhorn(u_t, n_iters=30).exp()                        # Pi_S(u_t) is DS
    q_ds = max((Pt.sum(-1) - 1).abs().max().item(), (Pt.sum(-2) - 1).abs().max().item())
    check("q_sample endpoints + Pi_S doubly stochastic",
          q0_err < 1e-6 and q1_err < 1e-6 and q_ds < 1e-4,
          f"t0={q0_err:.1e} t1={q1_err:.1e} marg={q_ds:.1e}")

    # --- tau_schedule endpoints and monotonicity --------------------------- #
    ts = torch.linspace(0, 1, 11)
    tg = tau_schedule(ts, 0.05, 1.0, "geometric")
    tl = tau_schedule(ts, 0.05, 1.0, "linear")
    mono = bool((tg[1:] >= tg[:-1]).all() and (tl[1:] >= tl[:-1]).all())
    ends = abs(tg[0] - 0.05) < 1e-6 and abs(tg[-1] - 1.0) < 1e-6
    check("tau_schedule endpoints + monotone", mono and ends,
          f"tau(0)={tg[0]:.3f} tau(1)={tg[-1]:.3f}")

    # --- gradients flow through the unrolled Sinkhorn ---------------------- #
    x = torch.randn(B, n, n, requires_grad=True)
    loss = log_sinkhorn(x, n_iters=10, tau=0.5).exp().sum()
    loss.backward()
    g_ok = x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    check("gradients flow", g_ok, f"grad_norm={x.grad.norm().item():.3e}")

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_tests()
