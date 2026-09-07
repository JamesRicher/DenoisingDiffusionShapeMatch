"""Geodesic-MPNN denoiser package.

Modules: geometry (graphs/RBF), intra_mpnn (geodesic GNN layer), cross_stage
(candidate-graph bipartite GNN), state_track (in-trunk assignment state), belief_prop
(the BP refiner, applied as a post-process by evaluate.py), denoiser (assembly,
registered as MPNNMatrixDenoiser).
"""
from networks.mpnn.denoiser import MPNNMatrixDenoiser

__all__ = ["MPNNMatrixDenoiser"]
