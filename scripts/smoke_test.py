"""End-to-end smoke test for Track A (SPN / IDPN / PAD) pipeline.

Not a real training run -- just enough iterations to confirm every module
imports, every tensor shape lines up, and the inverse-design query returns a
sensible result. Run from the project root:

    python -m scripts.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from physics import parametric_sim
from pipeline.datasets import ParametricDataset
from pipeline import train_parametric as tp
from pipeline import inverse_design as idz


def check_physics():
    print("[1/3] physics surrogate ...")
    p = np.array([280.0, 80.0, 600.0, 760.0], dtype=np.float32)
    s = parametric_sim.simulate(p)
    assert s.R.shape == (CFG.grid.n_points,)
    assert s.T.shape == (CFG.grid.n_points,)
    print("      ok  (R/T shapes correct)")


def check_track_a():
    print("[2/3] Track A: SPN + PAD tiny train ...")
    device = torch.device("cpu")
    ds = ParametricDataset(256, seed=1)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    spn = tp.train_spn(loader, device, epochs=3, lr=1e-3, verbose=False)
    pad = tp.train_pad(spn, loader, device, epochs=3, lr=1e-3, verbose=False)
    test = ParametricDataset(128, seed=7)
    r2 = tp.evaluate(pad, test, device)
    print(f"      ok  (PAD overall R^2 placeholder={r2['overall']:.3f})")
    return pad


def check_track_a_inverse(pad):
    print("[3/3] Track A inverse-design query ...")
    target = idz.make_broadband_reflector_target()
    res = idz.design_parametric(pad, target)
    assert res.design.shape == (4,)
    print(f"      ok  (design={np.round(res.design, 1)}, avg_acc={res.avg_accuracy:.3f})")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    check_physics()
    pad = check_track_a()
    check_track_a_inverse(pad)
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
