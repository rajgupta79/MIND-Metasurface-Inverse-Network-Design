"""SPN -- Spectra-Predicting Network (Track A forward surrogate, Paper 2).

Maps a 4-parameter open-cylinder design to its (R, T) spectra. Architecture
follows Eq. (1) of the paper: a shared trunk that forks into independent
reflection and transmission heads.

    4 -> 8N -> 6N -> 4N -> 2N -> { N -> N_lambda (R),  N -> N_lambda (T) }

ReLU on hidden layers, sigmoid on outputs (spectra live in [0, 1]).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from physics.spectra import Spectrum


class SPN(nn.Module):
    def __init__(self, n_lambda: int | None = None, hidden_base: int | None = None):
        super().__init__()
        self.n_lambda = n_lambda or CFG.grid.n_points
        N = hidden_base or CFG.train.spn_hidden_base

        self.trunk = nn.Sequential(
            nn.Linear(4, 8 * N), nn.ReLU(),
            nn.Linear(8 * N, 6 * N), nn.ReLU(),
            nn.Linear(6 * N, 4 * N), nn.ReLU(),
            nn.Linear(4 * N, 2 * N), nn.ReLU(),
        )
        self.r_head = nn.Sequential(
            nn.Linear(2 * N, N), nn.ReLU(),
            nn.Linear(N, self.n_lambda), nn.Sigmoid(),
        )
        self.t_head = nn.Sequential(
            nn.Linear(2 * N, N), nn.ReLU(),
            nn.Linear(N, self.n_lambda), nn.Sigmoid(),
        )

    def forward(self, p_norm: torch.Tensor) -> torch.Tensor:
        """p_norm: (B, 4) normalized design -> (B, 2*N_lambda) spectrum [R|T]."""
        z = self.trunk(p_norm)
        R = self.r_head(z)
        T = self.t_head(z)
        return torch.cat([R, T], dim=1)

    # ---- ForwardSurrogate interface -------------------------------------- #
    @torch.no_grad()
    def predict_spectrum(self, design: np.ndarray) -> Spectrum:
        """design: un-normalized (4,) geometry -> Spectrum."""
        self.eval()
        p_norm = CFG.parametric.normalize(np.asarray(design, dtype=np.float32))
        x = torch.from_numpy(p_norm).float().unsqueeze(0)
        vec = self.forward(x).squeeze(0).cpu().numpy()
        return Spectrum.from_vector(vec, CFG.grid.wavelengths)
