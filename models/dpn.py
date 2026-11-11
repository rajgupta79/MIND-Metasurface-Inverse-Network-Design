"""DPN -- Design-Predicting Network (Track A inverse model, Paper 2).

Maps a target (R, T) spectrum back to a 4-parameter open-cylinder design.
Two input branches (reflection, transmission) each pass through several dense
layers, then merge and compress to the 4 design outputs.

Trained two ways:
  * IDPN  -- directly on (spectrum, design) pairs  (suffers many-to-one).
  * PDPN  -- inside a PAD with a frozen SPN         (the fix; see pad.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from physics.spectra import Spectrum


class DPN(nn.Module):
    def __init__(self, n_lambda: int | None = None, hidden_base: int | None = None):
        super().__init__()
        self.n_lambda = n_lambda or CFG.grid.n_points
        N = hidden_base or CFG.train.spn_hidden_base

        def branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.n_lambda, N), nn.ReLU(),
                nn.Linear(N, N), nn.ReLU(),
                nn.Linear(N, N), nn.ReLU(),
            )

        self.r_branch = branch()
        self.t_branch = branch()
        self.merge = nn.Sequential(
            nn.Linear(2 * N, 2 * N), nn.ReLU(),
            nn.Linear(2 * N, N), nn.ReLU(),
            nn.Linear(N, N // 2), nn.ReLU(),
            nn.Linear(N // 2, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.Sigmoid(),      # normalized (r_o, r_i, h, L)
        )

    def forward(self, spectrum_vec: torch.Tensor) -> torch.Tensor:
        """spectrum_vec: (B, 2*N_lambda) [R|T] -> (B, 4) normalized design."""
        R = spectrum_vec[:, : self.n_lambda]
        T = spectrum_vec[:, self.n_lambda:]
        feats = torch.cat([self.r_branch(R), self.t_branch(T)], dim=1)
        return self.merge(feats)

    # ---- InverseModel interface ------------------------------------------ #
    @torch.no_grad()
    def propose_design(self, target: Spectrum) -> np.ndarray:
        """target spectrum -> un-normalized (4,) geometry."""
        self.eval()
        x = torch.from_numpy(target.as_vector()).float().unsqueeze(0)
        p_norm = self.forward(x).squeeze(0).cpu().numpy()
        return CFG.parametric.denormalize(p_norm)
