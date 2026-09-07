"""Spectral point descriptors from the Laplace-Beltrami eigen-decomposition.

These are intrinsic (isometry-invariant) and directly comparable across shapes, computed from
the cached LBO spectrum (evals, evecs) that the datasets already load under ret_evecs. Used as a
non-learned, network-free feature source for the densifier data terms.
"""
import torch


def shared_wks_grid(evals_x: torch.Tensor, evals_y: torch.Tensor, n_e: int = 100,
                    variance: float = 7.0, eps: float = 1e-8):
    """One energy grid covering BOTH spectra, so band i means the same energy on each shape (the
    prerequisite for functional-map descriptor correspondence). Range is the overlap of the two
    log-eigenvalue spans -- [max of the mins, min of the maxes] -- so both shapes actually
    resolve every band. Returns (energies (n_e,), sigma)."""
    lx = torch.log(evals_x.abs().clamp_min(eps))
    ly = torch.log(evals_y.abs().clamp_min(eps))
    e_min = torch.maximum(lx[1], ly[1])
    e_max = torch.minimum(lx[-1], ly[-1])
    sigma = variance * (e_max - e_min) / n_e
    energies = torch.linspace((e_min + 2 * sigma).item(), (e_max - 2 * sigma).item(),
                              n_e, device=evals_x.device)
    return energies, sigma


def wks_coefs(evals: torch.Tensor, energies: torch.Tensor, sigma: torch.Tensor,
              eps: float = 1e-8) -> torch.Tensor:
    """Per-eigenvalue Gaussian band weights on the given energy grid: coefs[k, b] =
    exp(-(e_b - log lambda_k)^2 / 2 sigma^2). (K, n_e). The building block of both the WKS
    descriptor and the wave-kernel landmark bumps."""
    log_ev = torch.log(evals.abs().clamp_min(eps))
    return torch.exp(-(energies[None, :] - log_ev[:, None]) ** 2 / (2 * sigma ** 2 + eps))
