"""Surrogate RCWA simulator for the Track-A open-cylinder metasurface.

Replaces the real `S4` RCWA call used in Paper 2. Given the geometry
``(r_o, r_i, h, L)`` it returns a physically plausible, resonance-shaped
reflection/transmission pair. The mapping is smooth, nonlinear and
deliberately *many-to-one* (different geometries can give similar spectra),
so the networks have the same hard inverse problem the paper describes.

Swap `simulate()` for a real RCWA call to do actual physics; the signature
and return type are what the rest of the code depends on.
"""
from __future__ import annotations

import numpy as np

from config import CFG, ParametricSpace, SpectralGrid
from .spectra import Spectrum, broadband_plateau, gaussian_resonance


def simulate(
    p: np.ndarray,
    grid: SpectralGrid | None = None,
    space: ParametricSpace | None = None,
) -> Spectrum:
    """Forward 'physics': geometry -> spectrum.

    Parameters
    ----------
    p : array-like, shape (4,)
        Un-normalized design (r_o, r_i, h, L) in nm.
    """
    grid = grid or CFG.grid
    space = space or CFG.parametric
    r_o, r_i, h, L = [float(v) for v in p]
    wl = grid.wavelengths

    # --- Map geometry to resonance descriptors -------------------------------
    # Physics motivation: a Mie-type dielectric resonator's optical response is
    # governed by a few *effective* quantities, not by each raw dimension in
    # isolation. We therefore make the spectrum depend on three physically
    # meaningful COMBINATIONS of the four parameters:
    #
    #   V_eff  (effective optical volume)  -> resonance CENTER wavelength
    #   fill   (material fill fraction)    -> resonance PEAK reflectivity
    #   AR     (aspect ratio)              -> resonance BANDWIDTH
    #
    # This creates a *genuine, controlled many-to-one* relationship, exactly the
    # difficulty Paper 2 describes: different geometries (r_o, r_i, h, L) that
    # share these effective quantities produce the SAME spectrum. The derived
    # quantities are recoverable from (R, T), but the individual parameters are
    # NOT -- which is why a naive inverse (IDPN) struggles and the PAD wins.
    annulus_area = max(r_o ** 2 - r_i ** 2, 1.0)        # high-index cross-section
    v_eff = annulus_area * h                            # effective optical volume
    fill = annulus_area / (L ** 2 + 1.0)                # fraction of cell filled
    aspect = h / (2.0 * r_o + 1.0)                      # tall-vs-wide ratio

    center = 1150.0 + 110.0 * np.tanh((v_eff - 6e7) / 6e7)      # from V_eff
    peak_R = 0.50 + 0.48 * np.tanh(6.0 * (fill - 0.15))        # from fill
    bandwidth = 90.0 + 160.0 * np.tanh(aspect)                 # from aspect ratio

    # A single broadband resonance fully determined by the three combinations.
    R = broadband_plateau(wl, center, bandwidth, amplitude=peak_R)
    R = np.clip(R, 0.0, 0.999)

    # Transmission is the lossless complement for an all-dielectric surface.
    T = np.clip(1.0 - R, 0.0, 1.0)
    return Spectrum(R=R, T=T, wavelengths=wl)


def sample_valid_design(
    rng: np.random.Generator,
    space: ParametricSpace | None = None,
) -> np.ndarray:
    """Draw one geometry obeying the physical feasibility constraints."""
    space = space or CFG.parametric
    L = rng.uniform(*space.L)
    ro_hi = min(space.r_o[1], space.ro_frac_of_L * L)
    ro_lo = min(space.r_o[0], ro_hi)
    r_o = rng.uniform(ro_lo, ro_hi)
    ri_hi = min(space.r_i[1], space.ri_frac_of_ro * r_o)
    ri_lo = min(space.r_i[0], ri_hi)
    r_i = rng.uniform(ri_lo, ri_hi)
    h = rng.uniform(*space.h)
    return np.array([r_o, r_i, h, L], dtype=np.float32)


def sample_dataset(
    n: int,
    seed: int = 0,
    grid: SpectralGrid | None = None,
    space: ParametricSpace | None = None,
):
    """Generate ``n`` (design, spectrum) pairs for training SPN/DPN.

    Returns
    -------
    designs : np.ndarray, shape (n, 4)   -- un-normalized geometries
    spectra : np.ndarray, shape (n, 2*N_lambda) -- concatenated [R, T]
    """
    grid = grid or CFG.grid
    space = space or CFG.parametric
    rng = np.random.default_rng(seed)
    designs = np.zeros((n, 4), dtype=np.float32)
    spectra = np.zeros((n, 2 * grid.n_points), dtype=np.float32)
    for i in range(n):
        p = sample_valid_design(rng, space)
        designs[i] = p
        spectra[i] = simulate(p, grid, space).as_vector()
    return designs, spectra
