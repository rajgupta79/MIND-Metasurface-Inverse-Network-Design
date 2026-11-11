"""Track A training (Paper 2): train the SPN, then the PAD; compare vs. IDPN.

Run:
    python -m pipeline.train_parametric --epochs 50

Produces trained SPN + PAD and prints per-parameter R^2 for the directly
trained inverse network (IDPN) versus the PAD-trained one (PDPN), reproducing
the paper's central finding that the PAD wins on the many-to-one problem.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from models.spn import SPN
from models.dpn import DPN
from models.pad import PAD
from pipeline.datasets import ParametricDataset
from pipeline.metrics import per_parameter_r2


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_spn(train_loader, device, epochs, lr, verbose=True) -> SPN:
    spn = SPN().to(device)
    opt = torch.optim.Adam(spn.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    spn.train()
    for ep in range(epochs):
        total = 0.0
        for p_norm, spec in train_loader:
            p_norm, spec = p_norm.to(device), spec.to(device)
            opt.zero_grad()
            loss = loss_fn(spn(p_norm), spec)
            loss.backward()
            opt.step()
            total += loss.item() * len(p_norm)
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"[SPN] epoch {ep:4d}  mse={total / len(train_loader.dataset):.6f}")
    return spn


def train_idpn(train_loader, device, epochs, lr, verbose=True) -> DPN:
    """Independent DPN: spectra -> design, trained directly (the baseline)."""
    dpn = DPN().to(device)
    opt = torch.optim.Adam(dpn.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    dpn.train()
    for ep in range(epochs):
        total = 0.0
        for p_norm, spec in train_loader:
            p_norm, spec = p_norm.to(device), spec.to(device)
            opt.zero_grad()
            loss = loss_fn(dpn(spec), p_norm)       # compares geometry directly
            loss.backward()
            opt.step()
            total += loss.item() * len(p_norm)
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"[IDPN] epoch {ep:4d}  mse={total / len(train_loader.dataset):.6f}")
    return dpn


def train_pad(spn, train_loader, device, epochs, lr, verbose=True) -> PAD:
    """PAD: freeze SPN, train DPN so SPN(DPN(s)) reconstructs s."""
    pad = PAD(spn).to(device)
    opt = torch.optim.Adam(pad.dpn.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    pad.train()
    for ep in range(epochs):
        total = 0.0
        for _p_norm, spec in train_loader:
            spec = spec.to(device)
            opt.zero_grad()
            _p_hat, s_hat = pad(spec)
            loss = loss_fn(s_hat, spec)             # compares spectra, not geometry
            loss.backward()
            opt.step()
            total += loss.item() * len(spec)
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"[PAD] epoch {ep:4d}  spec-mse={total / len(train_loader.dataset):.6f}")
    return pad


def train_pad_active(
    spn, train_loader, device, epochs, lr,
    rounds=3, samples_per_round=2000, verbose=True,
):
    """PAD training with active learning to prevent *surrogate exploitation*.

    Problem: the PAD can learn designs that fool the SPN (low SPN loss) but do
    not reproduce the target under the *real* simulator. Fix: alternate between
      (a) training the DPN inside the PAD, and
      (b) re-labelling the designs the DPN currently proposes with the real
          simulator and fine-tuning the SPN on them.
    This keeps the SPN accurate exactly in the region the DPN explores, closing
    the gap between "SPN believes" and "actually happens".
    """
    from physics import parametric_sim
    from physics.spectra import Spectrum

    pad = PAD(spn).to(device)
    dpn_opt = torch.optim.Adam(pad.dpn.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for rnd in range(rounds):
        # (a) train the DPN against the (frozen) current SPN
        pad.train()
        for ep in range(epochs):
            for _p, spec in train_loader:
                spec = spec.to(device)
                dpn_opt.zero_grad()
                _p_hat, s_hat = pad(spec)
                loss_fn(s_hat, spec).backward()
                dpn_opt.step()

        # (b) sample the designs the DPN now proposes, label them with the REAL
        #     simulator, and fine-tune the SPN on those (unfreeze -> train -> refreeze)
        _d, spectra = parametric_sim.sample_dataset(samples_per_round, seed=100 + rnd)
        proposed = np.stack([
            CFG.parametric.normalize(
                pad.propose_design(Spectrum.from_vector(spectra[i], CFG.grid.wavelengths)))
            for i in range(samples_per_round)
        ]).astype(np.float32)
        true_spec = np.stack([
            parametric_sim.simulate(CFG.parametric.denormalize(proposed[i])).as_vector()
            for i in range(samples_per_round)
        ]).astype(np.float32)

        for p in pad.spn.parameters():
            p.requires_grad_(True)
        pad.spn.train()
        spn_opt = torch.optim.Adam(pad.spn.parameters(), lr=lr)
        xb = torch.from_numpy(proposed).to(device)
        yb = torch.from_numpy(true_spec).to(device)
        for _ in range(epochs):
            spn_opt.zero_grad()
            loss_fn(pad.spn(xb), yb).backward()
            spn_opt.step()
        for p in pad.spn.parameters():
            p.requires_grad_(False)
        pad.spn.eval()

        if verbose:
            gap = loss_fn(pad.spn(xb), yb).item()
            print(f"[PAD-active] round {rnd + 1}/{rounds}  SPN-refit-mse={gap:.6f}")

    return pad


@torch.no_grad()
def evaluate(model, dataset, device) -> dict:
    """Per-parameter R^2 of predicted (normalized) designs on a held-out set."""
    loader = DataLoader(dataset, batch_size=512)
    true, pred = [], []
    for p_norm, spec in loader:
        spec = spec.to(device)
        dpn = model.dpn if isinstance(model, PAD) else model
        dpn.eval()
        pred.append(dpn(spec).cpu().numpy())
        true.append(p_norm.numpy())
    return per_parameter_r2(np.concatenate(true), np.concatenate(pred))


@torch.no_grad()
def evaluate_spectrum_match(model, n_samples=500, seed=999) -> dict:
    """Fair, physics-grounded metric: does the proposed design actually
    reproduce the requested spectrum when checked by the ground-truth solver?

    This is the metric the PAD is really optimizing (Paper 1's average/minimum
    accuracy). Unlike geometry-R^2 it is not fooled by the many-to-one problem.
    """
    from physics import parametric_sim
    from physics.spectra import Spectrum
    from pipeline.metrics import average_accuracy, minimum_accuracy

    designs, spectra = parametric_sim.sample_dataset(n_samples, seed=seed)
    avg, mn = [], []
    for i in range(n_samples):
        target = Spectrum.from_vector(spectra[i], CFG.grid.wavelengths)
        proposed = model.propose_design(target)          # InverseModel interface
        achieved = parametric_sim.simulate(proposed)     # ground-truth verify
        avg.append(average_accuracy(target, achieved))
        mn.append(minimum_accuracy(target, achieved))
    return {"avg_accuracy": float(np.mean(avg)),
            "min_accuracy": float(np.mean(mn))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CFG.train.spn_epochs)
    ap.add_argument("--n", type=int, default=CFG.train.parametric_dataset_size)
    ap.add_argument("--batch", type=int, default=CFG.train.parametric_batch)
    args = ap.parse_args()

    _seed(CFG.train.seed)
    device = torch.device(CFG.train.device)

    train_ds = ParametricDataset(args.n, seed=1)
    test_ds = ParametricDataset(max(1000, args.n // 9), seed=999)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    print("== Training SPN (forward surrogate) ==")
    spn = train_spn(train_loader, device, args.epochs, CFG.train.spn_lr)

    print("\n== Training IDPN (direct inverse baseline) ==")
    idpn = train_idpn(train_loader, device, args.epochs, CFG.train.spn_lr)

    print("\n== Training PAD (frozen SPN + DPN, active-learning refinement) ==")
    import copy
    pad = train_pad_active(copy.deepcopy(spn), train_loader, device,
                           max(20, args.epochs // 4), CFG.train.pad_lr)

    print("\n== Geometry R^2 (how well predicted params match the true params) ==")
    print("IDPN:", {k: round(v, 3) for k, v in evaluate(idpn, test_ds, device).items()})
    print("PDPN:", {k: round(v, 3) for k, v in evaluate(pad, test_ds, device).items()})

    print("\n== Spectrum-match accuracy (does the design reproduce the target?) ==")
    print("   This is the fair metric: the PAD is trained to match spectra, not")
    print("   to copy one specific geometry (the many-to-one problem).")
    print("IDPN:", {k: round(v, 3) for k, v in evaluate_spectrum_match(idpn).items()})
    print("PDPN:", {k: round(v, 3) for k, v in evaluate_spectrum_match(pad).items()})
    return spn, idpn, pad


if __name__ == "__main__":
    main()
