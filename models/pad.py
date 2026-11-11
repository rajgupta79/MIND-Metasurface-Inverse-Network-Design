"""PAD -- Pseudo-Autodecoder (Track A, Paper 2's key contribution).

Stitches a DPN (inverse) to a *frozen* pretrained SPN (forward):

    s_target --> DPN --> p_hat --> (frozen) SPN --> s_hat
    loss = MSE(s_hat, s_target)

Because the loss compares *spectra* rather than *geometry*, the DPN is free to
choose any geometry whose spectrum matches the target. This sidesteps the
many-to-one problem that cripples a directly-trained IDPN, and is what lifts the
reported average R^2 from ~0.84 (IDPN) to ~0.96 (PDPN).

The PAD also doubles as the inverse-design pipeline: once trained, `propose_design`
returns the geometry and `predict_spectrum` (via the SPN) gives its spectrum.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from physics.spectra import Spectrum
from .dpn import DPN
from .spn import SPN


class PAD(nn.Module):
    def __init__(self, spn: SPN, dpn: DPN | None = None):
        super().__init__()
        self.spn = spn
        self.dpn = dpn or DPN(n_lambda=spn.n_lambda)

        # Freeze the SPN: only the DPN learns inside the PAD.
        for param in self.spn.parameters():
            param.requires_grad_(False)
        self.spn.eval()

    def forward(self, s_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """s_target: (B, 2*N_lambda). Returns (predicted design, reconstructed spectrum)."""
        p_hat = self.dpn(s_target)
        s_hat = self.spn(p_hat)
        return p_hat, s_hat

    def train(self, mode: bool = True):  # keep SPN frozen in eval even in train()
        super().train(mode)
        self.spn.eval()
        return self

    # ---- InverseModel interface ------------------------------------------ #
    @torch.no_grad()
    def propose_design(self, target: Spectrum) -> np.ndarray:
        self.dpn.eval()
        x = torch.from_numpy(target.as_vector()).float().unsqueeze(0)
        p_norm = self.dpn(x).squeeze(0).cpu().numpy()
        return CFG.parametric.denormalize(p_norm)

    # ---- ForwardSurrogate interface (delegates to the SPN) --------------- #
    def predict_spectrum(self, design: np.ndarray) -> Spectrum:
        return self.spn.predict_spectrum(design)
