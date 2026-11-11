"""Shared interfaces that unify both tracks.

The inverse-design pipeline depends only on these two protocols, so a GAN
Generator (Track B) and a PAD/DPN (Track A) are interchangeable from the
caller's point of view.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from physics.spectra import Spectrum


@runtime_checkable
class ForwardSurrogate(Protocol):
    """A fast, differentiable stand-in for the physics solver.

    Implemented by Track A's ``SPN`` and Track B's ``Simulator``.
    """

    def predict_spectrum(self, design) -> Spectrum:
        """Map a design (params or image) to a predicted spectrum."""
        ...


@runtime_checkable
class InverseModel(Protocol):
    """Proposes a design that should reproduce a target spectrum.

    Implemented by Track A's ``DPN``/``PAD`` and Track B's ``Generator``.
    """

    def propose_design(self, target: Spectrum) -> np.ndarray:
        """Return a candidate design for the requested spectrum."""
        ...
