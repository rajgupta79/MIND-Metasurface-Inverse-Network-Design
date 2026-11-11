"""
generate_plots.py  —  visualizations for the Track-A (PAD) pipeline.

Produces 5 figures saved to scripts/plots/:
  1. sample_spectra.png       — 6 random (R, T) spectra with their geometry labels
  2. loss_curves.png          — SPN / IDPN / PAD training loss vs epoch
  3. regression_idpn_vs_pad.png — per-parameter regression scatter (IDPN vs PDPN)
  4. inverse_design_query.png — target spectrum → PAD design → achieved spectrum
  5. r2_bar_chart.png         — R² bar chart comparing IDPN vs PDPN per parameter
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader

from config import CFG
from physics import parametric_sim
from physics.spectra import Spectrum
from models.spn import SPN
from models.dpn import DPN
from models.pad import PAD
from pipeline.datasets import ParametricDataset
from pipeline.metrics import per_parameter_r2

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

EPOCHS   = 40
N_TRAIN  = 3000
BATCH    = 256
LR       = 1e-3
DEVICE   = torch.device("cpu")

PARAM_NAMES  = ["r_o (nm)", "r_i (nm)", "h (nm)", "L (nm)"]
PARAM_SHORT  = ["r_o", "r_i", "h", "L"]
PALETTE      = {"SPN": "#4C72B0", "IDPN": "#DD8452", "PAD": "#55A868"}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _style():
    plt.rcParams.update({
        "font.family":  "DejaVu Sans",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "figure.dpi":         150,
    })

_style()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Sample (R, T) spectra
# ═══════════════════════════════════════════════════════════════════════════════
def fig_sample_spectra():
    print("Generating figure 1 – sample spectra …")
    designs, spectra = parametric_sim.sample_dataset(6, seed=7)
    wl = CFG.grid.wavelengths
    N  = CFG.grid.n_points

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    fig.suptitle("Sample R and T spectra from the surrogate RCWA simulator\n"
                 "(each sub-plot = one randomly sampled geometry)", fontsize=13)

    colors_R = "#2166ac"
    colors_T = "#d6604d"

    for i, ax in enumerate(axes.flat):
        R = spectra[i, :N]
        T = spectra[i, N:]
        ro, ri, h, L = designs[i]
        ax.plot(wl, R, color=colors_R, lw=2.0, label="R (reflection)")
        ax.plot(wl, T, color=colors_T, lw=2.0, linestyle="--", label="T (transmission)")
        ax.set_title(
            f"$r_o$={ro:.0f}  $r_i$={ri:.0f}  $h$={h:.0f}  $L$={L:.0f} nm",
            fontsize=8.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Wavelength (nm)", fontsize=9)
        ax.set_ylabel("Intensity", fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "sample_spectra.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING  (shared — run once, collect history)
# ═══════════════════════════════════════════════════════════════════════════════
def run_training():
    print("Training SPN, IDPN, PAD …")
    train_ds = ParametricDataset(N_TRAIN, seed=1)
    test_ds  = ParametricDataset(1000,    seed=999)
    loader   = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    mse      = torch.nn.MSELoss()

    # ── SPN ──────────────────────────────────────────────────────────────────
    spn     = SPN().to(DEVICE)
    opt_spn = torch.optim.Adam(spn.parameters(), lr=LR)
    spn_loss = []
    for ep in range(EPOCHS):
        spn.train()
        total = 0.0
        for p_norm, spec in loader:
            opt_spn.zero_grad()
            l = mse(spn(p_norm), spec)
            l.backward(); opt_spn.step()
            total += l.item() * len(p_norm)
        spn_loss.append(total / len(train_ds))
    print("  SPN done.")

    # ── IDPN ─────────────────────────────────────────────────────────────────
    dpn      = DPN().to(DEVICE)
    opt_dpn  = torch.optim.Adam(dpn.parameters(), lr=LR)
    idpn_loss = []
    for ep in range(EPOCHS):
        dpn.train()
        total = 0.0
        for p_norm, spec in loader:
            opt_dpn.zero_grad()
            l = mse(dpn(spec), p_norm)
            l.backward(); opt_dpn.step()
            total += l.item() * len(p_norm)
        idpn_loss.append(total / len(train_ds))
    print("  IDPN done.")

    # ── PAD ──────────────────────────────────────────────────────────────────
    import copy
    pad      = PAD(copy.deepcopy(spn)).to(DEVICE)
    opt_pad  = torch.optim.Adam(pad.dpn.parameters(), lr=LR)
    pad_loss = []
    for ep in range(EPOCHS):
        pad.train()
        total = 0.0
        for _, spec in loader:
            opt_pad.zero_grad()
            _, s_hat = pad(spec)
            l = mse(s_hat, spec)
            l.backward(); opt_pad.step()
            total += l.item() * len(spec)
        pad_loss.append(total / len(train_ds))
    print("  PAD done.")

    return spn, dpn, pad, spn_loss, idpn_loss, pad_loss, test_ds


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Loss curves
# ═══════════════════════════════════════════════════════════════════════════════
def fig_loss_curves(spn_loss, idpn_loss, pad_loss):
    print("Generating figure 2 – loss curves …")
    epochs = np.arange(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Training loss (MSE) vs. epoch", fontsize=13)

    # left — all three on the same axis
    ax = axes[0]
    ax.semilogy(epochs, spn_loss,  color=PALETTE["SPN"],  lw=2, label="SPN (forward)")
    ax.semilogy(epochs, idpn_loss, color=PALETTE["IDPN"], lw=2, label="IDPN (direct inverse)")
    ax.semilogy(epochs, pad_loss,  color=PALETTE["PAD"],  lw=2, label="PAD (frozen SPN + DPN)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("All models — log scale")
    ax.legend()

    # right — inverse models only, linear scale
    ax = axes[1]
    ax.plot(epochs, idpn_loss, color=PALETTE["IDPN"], lw=2, label="IDPN")
    ax.plot(epochs, pad_loss,  color=PALETTE["PAD"],  lw=2, label="PAD")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.set_title("Inverse models — linear scale")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "loss_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Regression scatter (IDPN vs PAD)
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def fig_regression(dpn, pad, test_ds):
    print("Generating figure 3 – regression scatter …")
    loader = DataLoader(test_ds, batch_size=512)
    true_all, pred_idpn_all, pred_pad_all = [], [], []
    for p_norm, spec in loader:
        true_all.append(p_norm.numpy())
        dpn.eval()
        pred_idpn_all.append(dpn(spec).numpy())
        pad.dpn.eval()
        pred_pad_all.append(pad.dpn(spec).numpy())

    true       = np.concatenate(true_all)
    pred_idpn  = np.concatenate(pred_idpn_all)
    pred_pad   = np.concatenate(pred_pad_all)

    r2_idpn = per_parameter_r2(true, pred_idpn)
    r2_pad  = per_parameter_r2(true, pred_pad)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Regression: true vs. predicted (normalized) geometry parameters\n"
                 "Left = IDPN (direct),  Right = PAD", fontsize=12)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.4)

    # 2 rows × 4 cols: cols 0-1 = IDPN, cols 2-3 = PAD
    for col, (preds, r2s, title, color) in enumerate([
        (pred_idpn, r2_idpn, "IDPN", PALETTE["IDPN"]),
        (pred_pad,  r2_pad,  "PAD",  PALETTE["PAD"]),
    ]):
        for pi, pname in enumerate(PARAM_SHORT):
            row = pi // 2
            inner_col = pi % 2 + col * 2
            ax = fig.add_subplot(gs[row, inner_col])
            x, y = true[:, pi], preds[:, pi]
            ax.scatter(x, y, s=5, alpha=0.35, color=color)
            ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect")
            r2_val = r2s[pname]
            ax.set_title(f"{title} — {pname}\n$R^2$={r2_val:.3f}", fontsize=9)
            ax.set_xlabel("True (norm.)", fontsize=8)
            ax.set_ylabel("Pred (norm.)", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    path = os.path.join(PLOTS_DIR, "regression_idpn_vs_pad.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")
    return r2_idpn, r2_pad


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Inverse design query
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def fig_inverse_design(spn, pad):
    print("Generating figure 4 – inverse design query …")
    wl = CFG.grid.wavelengths
    N  = CFG.grid.n_points

    # Build a "user request": flat broadband reflector centred at 1550 nm
    from physics.spectra import broadband_plateau
    R_target = broadband_plateau(wl, center_nm=1550.0, bandwidth_nm=350.0, amplitude=0.97)
    T_target = np.clip(1.0 - R_target, 0.0, 1.0)
    target   = Spectrum(R=R_target, T=T_target, wavelengths=wl)

    # PAD proposes a design
    proposed_design = pad.propose_design(target)       # un-normalized (4,)

    # Ground-truth simulator checks the design
    achieved = parametric_sim.simulate(proposed_design)

    # SPN predicts what it thinks the design gives (fast path)
    p_norm = torch.from_numpy(
        CFG.parametric.normalize(proposed_design.astype(np.float32))
    ).unsqueeze(0)
    spn.eval()
    spn_pred_vec = spn(p_norm).squeeze(0).cpu().numpy()
    spn_pred = Spectrum.from_vector(spn_pred_vec, wl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Inverse design query — target spectrum → PAD design → verification",
                 fontsize=12)

    # left: target vs achieved
    ax = axes[0]
    ax.plot(wl, R_target,       color="#2166ac", lw=2.5, label="Target R")
    ax.plot(wl, T_target,       color="#d6604d", lw=2.5, linestyle="--", label="Target T")
    ax.plot(wl, achieved.R,     color="#1a9850", lw=2.0, linestyle="-",  label="Achieved R (simulator)")
    ax.plot(wl, achieved.T,     color="#f46d43", lw=2.0, linestyle="--", label="Achieved T (simulator)")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Intensity")
    ax.set_title("Target vs. Achieved (ground-truth simulator)")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=8)

    ro, ri, h, L = proposed_design
    textstr = (f"PAD-proposed design:\n"
               f"  $r_o$ = {ro:.0f} nm\n"
               f"  $r_i$ = {ri:.0f} nm\n"
               f"  $h$   = {h:.0f} nm\n"
               f"  $L$   = {L:.0f} nm")
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    # right: SPN prediction vs achieved
    ax = axes[1]
    ax.plot(wl, spn_pred.R,  color="#2166ac", lw=2.0, linestyle="-",  label="SPN predicted R")
    ax.plot(wl, spn_pred.T,  color="#d6604d", lw=2.0, linestyle="--", label="SPN predicted T")
    ax.plot(wl, achieved.R,  color="#1a9850", lw=2.0, linestyle="-",  label="Simulator R")
    ax.plot(wl, achieved.T,  color="#f46d43", lw=2.0, linestyle="--", label="Simulator T")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Intensity")
    ax.set_title("SPN prediction vs. Ground-truth simulator\n(how good is the surrogate?)")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "inverse_design_query.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — R² bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig_r2_bar(r2_idpn, r2_pad):
    print("Generating figure 5 – R² bar chart …")

    keys = list(r2_idpn.keys())
    vals_idpn = [r2_idpn[k] for k in keys]
    vals_pad  = [r2_pad[k]  for k in keys]

    x = np.arange(len(keys))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w/2, vals_idpn, w, label="IDPN (direct)",
                   color=PALETTE["IDPN"], edgecolor="white")
    bars2 = ax.bar(x + w/2, vals_pad,  w, label="PAD (pseudo-autodecoder)",
                   color=PALETTE["PAD"],  edgecolor="white")

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(PARAM_SHORT, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("$R^2$ (coefficient of determination)")
    ax.set_title("Per-parameter R² — IDPN vs. PAD\n"
                 "(higher = model predicts that parameter better)")
    ax.legend()
    ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "r2_bar_chart.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_sample_spectra()

    spn, dpn, pad, spn_loss, idpn_loss, pad_loss, test_ds = run_training()

    fig_loss_curves(spn_loss, idpn_loss, pad_loss)
    r2_idpn, r2_pad = fig_regression(dpn, pad, test_ds)
    fig_inverse_design(spn, pad)
    fig_r2_bar(r2_idpn, r2_pad)

    print("\nAll plots saved to:", PLOTS_DIR)
