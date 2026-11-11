"""Spectrum data type and resonance helpers shared by both tracks.

A `Spectrum` is a (reflection, transmission) pair sampled on a common
wavelength grid. Values are clamped to [0, 1] as in both source papers.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Spectrum:
    """Reflection / transmission spectra on a shared wavelength grid.

    Attributes
    ----------
    R, T : np.ndarray
        Reflection and transmission, shape (N_lambda,), values in [0, 1].
    wavelengths : np.ndarray
        The wavelength grid in nm, shape (N_lambda,).
    """
    R: np.ndarray
    T: np.ndarray
    wavelengths: np.ndarray

    def __post_init__(self) -> None:
        self.R = np.clip(np.asarray(self.R, dtype=np.float32), 0.0, 1.0)
        self.T = np.clip(np.asarray(self.T, dtype=np.float32), 0.0, 1.0)
        self.wavelengths = np.asarray(self.wavelengths, dtype=np.float32)
        if not (self.R.shape == self.T.shape == self.wavelengths.shape):
            raise ValueError("R, T and wavelengths must share the same shape")

    def as_vector(self) -> np.ndarray:
        """Concatenate R and T into a single (2*N_lambda,) feature vector."""
        return np.concatenate([self.R, self.T]).astype(np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray, wavelengths: np.ndarray) -> "Spectrum":
        n = len(wavelengths)
        return cls(R=vec[:n], T=vec[n:], wavelengths=wavelengths)


def gaussian_resonance(
    wavelengths: np.ndarray,
    center_nm: float,
    width_nm: float,
    amplitude: float,
) -> np.ndarray:
    """A single Gaussian resonance lineshape, used to compose spectra.

    Paper 1 builds target spectra as Gaussian-like resonances; Paper 2's
    broadband reflector is effectively a wide plateau (a broad Gaussian).
    """
    width_nm = max(width_nm, 1e-6)
    return amplitude * np.exp(-0.5 * ((wavelengths - center_nm) / width_nm) ** 2)


def broadband_plateau(
    wavelengths: np.ndarray,
    center_nm: float,
    bandwidth_nm: float,
    amplitude: float = 1.0,
    edge_sharpness: float = 6.0,
) -> np.ndarray:
    """A flat-topped reflection band (super-Gaussian).

    Used to express Paper 2's "R >= 99% over 450 nm bandwidth" target.
    """
    bandwidth_nm = max(bandwidth_nm, 1e-6)
    z = (wavelengths - center_nm) / (0.5 * bandwidth_nm)
    return amplitude * np.exp(-(np.abs(z) ** edge_sharpness))
