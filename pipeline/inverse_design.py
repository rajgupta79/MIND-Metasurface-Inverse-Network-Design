"""Inverse-design query for Track A: target spectrum -> 4-param design.

The PAD (or plain DPN) proposes a geometry; the parametric surrogate then
verifies it. This mirrors Paper 2's final step: propose with the network,
then verify with the physics simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import parametric_sim
from physics.spectra import Spectrum
from pipeline.metrics import average_accuracy, minimum_accuracy, spectrum_mse


@dataclass
class InverseResult:
    design: np.ndarray            # un-normalized params (4,): (r_o, r_i, h, L)
    verified_spectrum: Spectrum   # from the ground-truth surrogate
    avg_accuracy: float
    min_accuracy: float
    mse: float


def design_parametric(pad_or_dpn, target: Spectrum) -> InverseResult:
    """Track A: propose a 4-param geometry and verify with surrogate RCWA."""
    design = pad_or_dpn.propose_design(target)
    verified = parametric_sim.simulate(design)
    return InverseResult(
        design=design,
        verified_spectrum=verified,
        avg_accuracy=average_accuracy(target, verified),
        min_accuracy=minimum_accuracy(target, verified),
        mse=spectrum_mse(target, verified),
    )


def make_broadband_reflector_target() -> Spectrum:
    """Paper 2's showcase query: a broad high-reflection band at 1550 nm."""
    from config import CFG
    from physics.spectra import broadband_plateau
    wl = CFG.grid.wavelengths
    R = broadband_plateau(wl, center_nm=1550.0, bandwidth_nm=450.0, amplitude=0.99)
    T = np.clip(1.0 - R, 0.0, 1.0)
    return Spectrum(R=R, T=T, wavelengths=wl)
