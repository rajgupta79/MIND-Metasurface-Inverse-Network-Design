"""Central configuration: physical constants, grids, and hyperparameters.

Everything tunable lives here so experiments stay reproducible and the rest of
the code imports from a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# --------------------------------------------------------------------------- #
# Spectral grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpectralGrid:
    """Wavelength sampling grid for reflection/transmission spectra."""
    lambda_min_nm: float = 1300.0   # Paper 2 targets the telecom band
    lambda_max_nm: float = 1800.0
    n_points: int = 64              # N_lambda

    @property
    def wavelengths(self) -> np.ndarray:
        return np.linspace(self.lambda_min_nm, self.lambda_max_nm, self.n_points)


# --------------------------------------------------------------------------- #
# Track A -- parametric open-cylinder design space (Paper 2, Table I)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParametricSpace:
    # (min, max) in nm for outer radius, inner radius, height, lattice spacing
    r_o: tuple[float, float] = (120.0, 2000.0)
    r_i: tuple[float, float] = (80.0, 1900.0)
    h:   tuple[float, float] = (200.0, 1000.0)
    L:   tuple[float, float] = (300.0, 2200.0)

    # physical feasibility constraints (Paper 2):
    #   r_o <= 0.4 L,  r_i <= 0.8 r_o,  r_i < r_o
    ro_frac_of_L: float = 0.4
    ri_frac_of_ro: float = 0.8

    @property
    def bounds(self) -> np.ndarray:
        """(4, 2) array of [min, max] per parameter, order (r_o, r_i, h, L)."""
        return np.array([self.r_o, self.r_i, self.h, self.L], dtype=np.float32)

    def normalize(self, p: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (p - lo) / (hi - lo)

    def denormalize(self, p_norm: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return p_norm * (hi - lo) + lo


# --------------------------------------------------------------------------- #
# Training hyperparameters (Track A)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cpu"             # set to "cuda" if available

    spn_hidden_base: int = 64       # "N" in the SPN width schedule
    spn_epochs: int = 200
    spn_lr: float = 1e-3
    pad_epochs: int = 200
    pad_lr: float = 1e-3
    parametric_batch: int = 256
    parametric_dataset_size: int = 9000   # Paper 2 used ~9000 designs


@dataclass
class Config:
    grid: SpectralGrid = field(default_factory=SpectralGrid)
    parametric: ParametricSpace = field(default_factory=ParametricSpace)
    train: TrainConfig = field(default_factory=TrainConfig)


# a ready-to-import default instance
CFG = Config()
