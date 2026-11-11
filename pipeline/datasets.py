"""Torch dataset wrapping the parametric surrogate physics for Track A."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG
from physics import parametric_sim


class ParametricDataset(Dataset):
    """(normalized design, spectrum) pairs for Track A (SPN / IDPN / PAD)."""

    def __init__(self, n: int, seed: int = 0):
        designs, spectra = parametric_sim.sample_dataset(n, seed=seed)
        self.designs_norm = CFG.parametric.normalize(designs).astype(np.float32)
        self.spectra = spectra.astype(np.float32)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.designs_norm[idx]),
            torch.from_numpy(self.spectra[idx]),
        )
