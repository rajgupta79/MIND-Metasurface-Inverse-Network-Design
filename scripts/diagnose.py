"""Diagnose why the PAD underperforms: is the SPN accurate enough, and is the
PAD exploiting SPN error? Compares, for PAD-proposed designs, the spectrum the
SPN *predicts* vs. the spectrum the ground-truth simulator *actually* produces.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from physics import parametric_sim
from physics.spectra import Spectrum
from pipeline.datasets import ParametricDataset
from pipeline import train_parametric as tp
from pipeline.metrics import average_accuracy

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cpu")

ds = ParametricDataset(6000, seed=1)
loader = DataLoader(ds, batch_size=256, shuffle=True)

print("Training SPN...")
spn = tp.train_spn(loader, device, epochs=150, lr=1e-3, verbose=False)

# 1) SPN accuracy on held-out data: SPN prediction vs ground truth
test = ParametricDataset(500, seed=42)
spn.eval()
errs = []
with torch.no_grad():
    for pn, spec in DataLoader(test, batch_size=256):
        pred = spn(pn).numpy()
        errs.append(np.abs(pred - spec.numpy()))
spn_abs_err = np.concatenate(errs).mean()
print(f"SPN mean abs error per point: {spn_abs_err:.4f}  (paper target < 0.01)")

print("Training PAD...")
pad = tp.train_pad(spn, loader, device, epochs=150, lr=1e-3, verbose=False)

# 2) For PAD designs: SPN-predicted spectrum vs TRUE simulator spectrum
designs, spectra = parametric_sim.sample_dataset(300, seed=7)
spn_says, truth_says = [], []
for i in range(300):
    target = Spectrum.from_vector(spectra[i], CFG.grid.wavelengths)
    p = pad.propose_design(target)
    spn_pred = pad.spn.predict_spectrum(p)         # what SPN thinks
    true_sim = parametric_sim.simulate(p)          # what really happens
    spn_says.append(average_accuracy(target, spn_pred))
    truth_says.append(average_accuracy(target, true_sim))

print(f"PAD design -- SPN believes accuracy: {np.mean(spn_says):.3f}")
print(f"PAD design -- TRUE accuracy:         {np.mean(truth_says):.3f}")
print(f"Gap (surrogate exploitation):        {np.mean(spn_says)-np.mean(truth_says):.3f}")
