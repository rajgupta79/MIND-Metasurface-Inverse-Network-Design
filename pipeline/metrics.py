"""Evaluation metrics drawn from both papers.

Paper 2 uses the coefficient of determination R^2 on predicted design params.
Paper 1 defines three spectrum accuracies: average and minimum (worst-point)
accuracy, plus a geometric accuracy (class-match of the generated shape).
"""
from __future__ import annotations

import numpy as np

from physics.spectra import Spectrum


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (Paper 2's regression metric)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def average_accuracy(target: Spectrum, achieved: Spectrum) -> float:
    """1 - mean absolute transmittance error per frequency point (Paper 1)."""
    err = np.concatenate([
        np.abs(target.R - achieved.R),
        np.abs(target.T - achieved.T),
    ])
    return float(1.0 - err.mean())


def minimum_accuracy(target: Spectrum, achieved: Spectrum) -> float:
    """1 - max absolute error over all frequencies (Paper 1)."""
    err = np.concatenate([
        np.abs(target.R - achieved.R),
        np.abs(target.T - achieved.T),
    ])
    return float(1.0 - err.max())


def spectrum_mse(target: Spectrum, achieved: Spectrum) -> float:
    err = np.concatenate([target.R - achieved.R, target.T - achieved.T])
    return float(np.mean(err ** 2))


def per_parameter_r2(designs_true: np.ndarray, designs_pred: np.ndarray) -> dict:
    """R^2 for each of (r_o, r_i, h, L) plus the overall value."""
    names = ["r_o", "r_i", "h", "L"]
    out = {names[i]: r2_score(designs_true[:, i], designs_pred[:, i]) for i in range(4)}
    out["overall"] = r2_score(designs_true, designs_pred)
    return out
