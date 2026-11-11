"""Ground-truth physics layer: spectra types + parametric RCWA surrogate."""
from .spectra import Spectrum, gaussian_resonance
from . import parametric_sim

__all__ = ["Spectrum", "gaussian_resonance", "parametric_sim"]
