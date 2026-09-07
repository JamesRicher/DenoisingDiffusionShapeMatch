"""Loopy sum-product BP over a pairwise MRF, refining assignment logits as a
post-process (evaluate.py, via diagnostics.bp_postprocess_eval.attach_bp). Nothing here
is learned; all scalars are passed in. Log-domain and dense-batched throughout.

Orientation: logits is (B, n_src, n_tgt), with variables on n_src and labels on n_tgt.
D_x is the source metric (message graph and edge lengths), D_y the label metric. The
slack label is internal to BP and is dropped at reintegration.
"""
import torch

from networks.mpnn.geometry import knn_from_dist

NEG_INF = float("-inf")


def _dedupe_mask(cand: torch.Tensor) -> torch.Tensor:
    """(B, n, Kc) candidate indices -> bool mask, False on duplicate slots.

    Keeps the first occurrence per row; later repeats become inert -inf entries.
    """
    Kc = cand.shape[-1]
    eq = cand.unsqueeze(-1) == cand.unsqueeze(-2)                 # (B, n, Kc, Kc)
    earlier = torch.tril(torch.ones(Kc, Kc, dtype=torch.bool, device=cand.device), -1)
    is_dup = (eq & earlier).any(-1)
    return ~is_dup


def build_candidate_sets(logits, feats_x, feats_y, k_logit, k_feat, feat_cand=None):
    """Per-vertex candidate labels = top-k logits and feature-kNN, deduped.

    Returns cand_idx, cand_mask (B, n, Kc) with Kc = k_logit + k_feat. Slack is not
    included here; it is a separate column in the potentials. feat_cand supplies
    precomputed feature-kNN indices, recomputed here when None.
    """
    top_l = logits.topk(min(k_logit, logits.shape[-1]), dim=-1).indices
    if feat_cand is None:
        d2 = torch.cdist(feats_x, feats_y)
        feat_cand = d2.topk(min(k_feat, feats_y.shape[1]), dim=-1, largest=False).indices
    cand = torch.cat([top_l, feat_cand], dim=-1)                 # (B, n, Kc)
    return cand, _dedupe_mask(cand)


def _unary(theta, cand_idx, cand_mask, s):
    """Unary potentials phi (B, n, Kc+1), last column = slack. Fixed across sweeps.

    theta is the already-scaled field (beta*logits); folding beta in at the caller lets
    it be a per-sample tensor.
    """
    phi_real = torch.gather(theta, -1, cand_idx)                 # (B, n, Kc)
    phi_real = phi_real.masked_fill(~cand_mask, NEG_INF)
    phi_slack = torch.full(phi_real.shape[:-1] + (1,), float(s), device=theta.device)
    phi = torch.cat([phi_real, phi_slack], dim=-1)
    return phi - torch.logsumexp(phi, dim=-1, keepdim=True)


def _gather_rows(cand_idx, nbr):
    """cand_idx (B, n, Kc), nbr (B, n, k) -> (B, n, k, Kc): each neighbour's candidates."""
    B, n, Kc = cand_idx.shape
    k = nbr.shape[-1]
    flat = nbr.reshape(B, n * k, 1).expand(-1, -1, Kc)
    return torch.gather(cand_idx, 1, flat).reshape(B, n, k, Kc)


def edge_distortion(cand_idx, nbr, D_src, D_tgt):
    """Isometric distortion of every candidate pair on every message edge.

    For directed edge (i, slot) with source s = nbr[i,slot], returns
    d_tgt(cand(s)_a, cand(i)_c) - d_src(i, s) as (B, n, k, Kc_a, Kc_c), the signed
    quantity the pairwise potential squares. Exposed so sigma can be calibrated against
    the exact distortion the potential sees.
    """
    B, n, Kc = cand_idx.shape
    k = nbr.shape[-1]
    d_x_edge = torch.gather(D_src, -1, nbr)                      # (B, n, k)
    cand_s = _gather_rows(cand_idx, nbr)                        # (B, n, k, Kc) source cands
    cand_d = cand_idx.unsqueeze(2).expand(-1, -1, k, -1)        # (B, n, k, Kc) dest cands

    # d_Y(source_a, dest_c): gather rows of D_y by source cands, then cols by dest cands
    r_flat = cand_s.reshape(B, n * k * Kc)
    Dr = torch.gather(D_tgt, 1, r_flat.unsqueeze(-1).expand(-1, -1, D_tgt.shape[-1]))
    Dr = Dr.reshape(B, n, k, Kc, D_tgt.shape[-1])               # (B,n,k,Kc_a, m)
    c_exp = cand_d.unsqueeze(3).expand(-1, -1, -1, Kc, -1)      # (B,n,k,Kc_a,Kc_c)
    d_y = torch.gather(Dr, -1, c_exp)                          # (B,n,k,Kc_a,Kc_c)
    return d_y - d_x_edge[..., None, None]


def _pairwise(cand_idx, cand_mask, nbr, D_src, D_tgt, sigma, delta):
    """Pairwise log-potentials (B, n, k, Kc+1, Kc+1) per directed edge (source a-axis,
    dest c-axis).

    log psi = -min((d_Y(a,c) - d_X_edge)^2 / 2*sigma^2, delta). Slack row/col = 0, i.e.
    compatible with everything; rows of invalid source candidates and cols of invalid
    dest candidates = -inf.
    """
    B, n, Kc = cand_idx.shape
    k = nbr.shape[-1]
    Kp1 = Kc + 1

    diff = edge_distortion(cand_idx, nbr, D_src, D_tgt)
    log_psi_real = -torch.clamp(diff.pow(2) / (2.0 * sigma ** 2), max=float(delta))

    log_psi = torch.zeros(B, n, k, Kp1, Kp1, device=D_src.device)
    log_psi[..., :Kc, :Kc] = log_psi_real
    # Source rows only. Masking dest columns too would make all-(-inf) columns, so the
    # logsumexp below would see an empty set and produce nan gradients; invalid dest
    # candidates are instead dropped at reintegration.
    mask_s = _gather_rows(cand_mask.long(), nbr).bool()
    log_psi[..., :Kc, :] = log_psi[..., :Kc, :].masked_fill(~mask_s[..., None], NEG_INF)
    return log_psi


def _reverse_index(nbr):
    """For each directed edge (i, slot) with source s=nbr[i,slot], find slot' with
    nbr[s,slot']=i. Returns rev_slot (B,n,k) and valid (B,n,k) (reverse edge exists)."""
    B, n, k = nbr.shape
    nbr_s = _gather_rows(nbr, nbr)                              # (B,n,k,k)
    i_idx = torch.arange(n, device=nbr.device).view(1, n, 1, 1)
    match = nbr_s == i_idx
    return match.float().argmax(-1), match.any(-1)


def _gather_msg(msg, nbr, rev_slot, valid):
    """Reverse messages: out[b,i,slot] = msg[b, nbr[i,slot], rev_slot[i,slot]] (uniform
    where no reverse edge). msg (B,n,k,Kp1) -> (B,n,k,Kp1)."""
    B, n, k, Kp1 = msg.shape
    node = nbr.reshape(B, n * k)
    m1 = torch.gather(msg, 1, node[:, :, None, None].expand(-1, -1, k, Kp1))
    m1 = m1.reshape(B, n, k, k, Kp1)
    out = torch.gather(m1, 3, rev_slot[..., None, None].expand(-1, -1, -1, 1, Kp1)).squeeze(3)
    return out * valid[..., None]                              # 0 = uniform in log


def _sweep(phi, log_psi, nbr, rev_slot, valid, src_mask, n_sweeps, tau, alpha):
    """T damped sum-product sweeps.

    Returns (msg, msg_delta): msg (B,n,k,Kp1) where msg[i,slot] is the message FROM
    nbr[i,slot] TO i over i's label domain, and msg_delta (B,n) the per-variable max
    message change on the final sweep, a per-vertex convergence monitor. Zero when
    n_sweeps == 0."""
    B, n, k, Kp1 = phi.shape[0], phi.shape[1], nbr.shape[-1], phi.shape[-1]
    msg = torch.zeros(B, n, k, Kp1, device=phi.device)
    msg_delta = torch.zeros(B, n, device=phi.device)
    for _ in range(n_sweeps):
        H = phi + msg.sum(2)                                   # (B,n,Kp1)
        H_src = _gather_rows(H, nbr)                           # (B,n,k,Kp1)
        rev = _gather_msg(msg, nbr, rev_slot, valid)
        # Sanitise before subtracting: padded entries are -inf, and -inf - -inf = nan
        # (torch.where still differentiates the unselected branch).
        z = torch.zeros_like(H_src)
        safe = torch.where(src_mask, H_src, z) - torch.where(src_mask, rev, z)
        h_cav = torch.where(src_mask, safe, torch.full_like(H_src, NEG_INF))
        new = tau * torch.logsumexp((h_cav.unsqueeze(-1) + log_psi) / tau, dim=-2)
        new = new - torch.logsumexp(new, dim=-1, keepdim=True)
        upd = alpha * msg + (1.0 - alpha) * new
        msg_delta = (upd - msg).abs().amax(dim=(-1, -2)).detach()
        msg = upd
    return msg, msg_delta


def bp_delta(theta, cand_idx, cand_mask, nbr, D_src, D_tgt, *,
             sigma=0.05, delta=4.0, tau=1.0, alpha=0.5, s=-4.0, n_sweeps=3,
             return_info=False):
    """The BP core: potentials, sweeps and reintegration on given candidates and graph,
    returning the ungated update delta.

    Wrapped by bp_refine, which chooses the candidates and the unary field; the
    propagation itself lives here.

    Args:
        theta: (B, n_src, n_tgt) already-scaled unary field (see `_unary`).
        cand_idx, cand_mask: (B, n_src, Kc) candidate labels and validity.
        nbr: (B, n_src, k) message graph on the variable shape.
        D_src: (B, n_src, n_src) variable-shape metric (edge lengths).
        D_tgt: (B, n_tgt, n_tgt) label-shape metric.
    Returns:
        delta (B, n_src, n_tgt), the mean-centred belief residual scattered onto candidates
        (slack dropped, zero off-candidate). With return_info, also a dict with the
        normalised belief (B, n_src, Kc+1), slack_mass, belief_entropy and msg_delta.
    """
    Kc = cand_idx.shape[-1]
    phi = _unary(theta, cand_idx, cand_mask, s)                 # (B,n,Kp1)
    log_psi = _pairwise(cand_idx, cand_mask, nbr, D_src, D_tgt, sigma, delta)
    rev_slot, valid = _reverse_index(nbr)

    # slack column is always valid
    src_ok = torch.cat([_gather_rows(cand_mask.long(), nbr).bool(),
                        torch.ones(*nbr.shape, 1, dtype=torch.bool, device=nbr.device)], dim=-1)

    msg, msg_delta = _sweep(phi, log_psi, nbr, rev_slot, valid, src_ok, n_sweeps, tau, alpha)

    # Reintegrate the residual (incoming messages only, unary excluded): adding the full
    # belief would double-count the unary. Messages are normalised log-probs, so the
    # result is then mean-centred over each vertex's valid candidates -- without that it
    # would down-shift every candidate relative to the untouched non-candidate logits and
    # the argmax would flee off-candidate. Centring makes BP a zero-sum re-ranking.
    resid = msg.sum(2)                                          # (B,n,Kp1)
    r_real = resid[..., :Kc].masked_fill(~cand_mask, 0.0)
    cnt = cand_mask.sum(-1, keepdim=True).clamp_min(1)
    mean = (r_real.sum(-1, keepdim=True) / cnt)
    r_real = (r_real - mean).masked_fill(~cand_mask, 0.0)
    delta_mat = torch.zeros(theta.shape, device=theta.device, dtype=theta.dtype)
    delta_mat.scatter_add_(-1, cand_idx, r_real)

    if not return_info:
        return delta_mat
    belief = phi + resid
    belief = belief - torch.logsumexp(belief, dim=-1, keepdim=True)
    p = belief.exp().clamp_min(1e-12)
    return delta_mat, {"belief": belief, "slack_mass": p[..., Kc],
                       "belief_entropy": -(p * p.log()).sum(-1),
                       "msg_delta": msg_delta}


def bp_refine(logits, feats_x, feats_y, D_x, D_y, *,
              k_logit=8, k_feat=8, k_graph=8,
              beta=1.0, sigma=0.05, delta=4.0, tau=1.0, alpha=0.5, s=-4.0, g=1.0,
              n_sweeps=3, return_info=False):
    """Single-sided BP refinement of assignment logits.

    Thin wrapper over bp_delta: build candidates from the logits, run BP with the unary
    beta*logits, add g*delta. logits (B, n_src, n_tgt); feats_* (B, n, d); D_x the variable
    shape's metric, D_y the label shape's. Returns refined logits; with return_info, also
    the diagnostics dict.
    """
    cand_idx, cand_mask = build_candidate_sets(logits, feats_x, feats_y, k_logit, k_feat)
    nbr, _ = knn_from_dist(D_x, min(k_graph, D_x.shape[-1] - 1))
    out = bp_delta(beta * logits, cand_idx, cand_mask, nbr, D_x, D_y, sigma=sigma,
                   delta=delta, tau=tau, alpha=alpha, s=s, n_sweeps=n_sweeps,
                   return_info=return_info)
    if not return_info:
        return logits + g * out
    delta_mat, info = out
    info["beliefs"] = info["belief"]                            # back-compat key
    return logits + g * delta_mat, info
