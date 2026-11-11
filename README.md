# MIND — Metasurface Inverse Network Design

Deep-learning inverse design of optical metasurfaces using the **Pseudo-Autodecoder (PAD)** approach, reproducing results from:

> Harper, Coyle, Vernon, Mills — *Inverse Design of Broadband Highly Reflective Metasurfaces Using Neural Networks*, **Physical Review B 101, 195104 (2020)**

---

## What this project does

Designing a metasurface means finding a geometry that produces a target optical spectrum. Running physics *forward* (geometry → spectrum) is slow but tractable; running it *backward* (spectrum → geometry) has no closed-form solution and is non-unique — many geometries produce nearly identical spectra.

This project trains two neural networks to solve the problem:

1. **SPN** (Spectra-Predicting Network) — a fast, differentiable surrogate for the RCWA physics solver. Maps a 4-parameter geometry to its (R, T) spectra.
2. **DPN** (Design-Predicting Network) — the inverse model. Maps a target (R, T) spectrum to a proposed geometry.

The key trick is the **PAD (Pseudo-Autodecoder)**: instead of training the DPN directly on geometry labels (which causes regression-to-the-mean due to the many-to-one problem), the DPN is stitched to a frozen SPN and trained end-to-end on spectrum reconstruction loss. This lifts inverse-design R² from ~83% (direct IDPN) to ~96% (PAD).

---

## Architecture overview

```
target spectrum (R/T vs. λ)
        │
        ▼
  ┌───────────┐
  │    DPN    │  ──▶  candidate design (r_o, r_i, h, L)
  └─────┬─────┘
        │  design
        ▼
  ┌───────────┐
  │    SPN    │  ──▶  predicted spectrum   (PAD training loss)
  └─────┬─────┘
        │
        ▼
  RCWA / surrogate physics  (ground-truth verification)
```

The **SPN** architecture (after Eq. 1 of the paper):
```
input(4) → 8N → 6N → 4N → 2N → { N → N_λ (R-head, sigmoid)
                                 { N → N_λ (T-head, sigmoid)
```

The **DPN** merges two parallel branches (one per spectrum channel) and compresses to 4 outputs.

For full module-level math and contracts see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Design parameters

The metasurface unit cell is a **Si open-cylinder on SiO₂** in the telecom band (1300–1800 nm):

| Parameter | Symbol | Range |
|---|---|---|
| Outer radius | r_o | 120 – 2000 nm |
| Inner radius | r_i | 80 – 1900 nm |
| Height | h | 200 – 1000 nm |
| Lattice spacing | L | 300 – 2200 nm |

Physical constraints: `r_o ≤ 0.4 L`, `r_i ≤ 0.8 r_o`.

---

## Repository layout

```
sem_prj/
├── config.py                   ← all hyperparameters and physical constants
├── requirements.txt
├── ARCHITECTURE.md             ← detailed module contracts and math
├── Papers/                     ← source paper PDF + extracted text
│
├── physics/
│   ├── spectra.py              ← Spectrum dataclass, Gaussian-resonance helpers
│   └── parametric_sim.py       ← analytic surrogate RCWA data generator
│
├── models/
│   ├── base.py                 ← ForwardSurrogate / InverseModel interfaces
│   ├── spn.py                  ← SPN (forward surrogate)
│   ├── dpn.py                  ← DPN (inverse model)
│   └── pad.py                  ← PAD (SPN + DPN stitched end-to-end)
│
├── pipeline/
│   ├── datasets.py             ← PyTorch Dataset for parametric data
│   ├── train_parametric.py     ← trains SPN then PAD
│   ├── inverse_design.py       ← query a target spectrum, get a design back
│   └── metrics.py              ← R² and spectrum-error metrics
│
└── scripts/
    ├── smoke_test.py           ← end-to-end wiring check
    ├── generate_plots.py       ← reproduces paper figures
    └── plots/                  ← generated figures
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> The repo uses a lightweight analytic surrogate for physics so the full pipeline runs on a laptop without any special solvers. To use real RCWA, install [S4](https://github.com/victorliu/S4) separately and swap it into `physics/parametric_sim.py`.

### 2. Run the smoke test

```bash
python scripts/smoke_test.py
```

Verifies the full data → SPN → PAD → inverse-design chain in a few seconds.

### 3. Train

```bash
python pipeline/train_parametric.py
```

Trains the SPN first, then the PAD. All hyperparameters are in `config.py`.

### 4. Inverse design

```python
from pipeline.inverse_design import InverseDesignPipeline

pipeline = InverseDesignPipeline.from_checkpoint("checkpoints/")
design = pipeline.design_for_target(target_spectrum)
```

### 5. Generate plots

```bash
python scripts/generate_plots.py
```

Reproduces loss curves, regression scatter plots, sample spectra, and inverse-design query figures.

---

## Key results

| Model | Avg. R² (geometry prediction) |
|---|---|
| Direct IDPN | ~83.6% |
| PAD (this work) | ~96.4% |

The PAD improvement is most pronounced for the inner radius parameter, which has the highest many-to-one degeneracy.

---

## Configuration

All tunable values live in `config.py`:

```python
CFG.grid.n_points          # spectral resolution (default 64)
CFG.train.spn_epochs       # SPN training epochs (default 200)
CFG.train.pad_epochs       # PAD training epochs (default 200)
CFG.train.spn_hidden_base  # network width multiplier N (default 64)
CFG.train.parametric_dataset_size  # ~9000 designs (matches paper)
```

---

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7

---

## Reference

```bibtex
@article{harper2020inverse,
  title   = {Inverse Design of Broadband Highly Reflective Metasurfaces Using Neural Networks},
  author  = {Harper, E. S. and Coyle, E. J. and Vernon, J. P. and Mills, M. S.},
  journal = {Physical Review B},
  volume  = {101},
  pages   = {195104},
  year    = {2020}
}
```
