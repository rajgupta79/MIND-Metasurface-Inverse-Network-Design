# MetaInverse — Deep Learning for Inverse Design of Optical Metasurfaces

A semester project on neural-network-based **inverse design of metasurfaces**,
reproducing the Pseudo-Autodecoder (PAD) approach from:

> Harper, Coyle, Vernon, Mills — *Inverse Design of Broadband Highly Reflective
> Metasurfaces Using Neural Networks*, **Physical Review B 101, 195104 (2020)**.

---

### The one-paragraph summary
Designing a metasurface means choosing a geometry that produces a desired optical
spectrum. Running the physics *forward* (geometry → spectrum) is possible but
slow; running it *backward* (spectrum → geometry) has no formula. This project
trains a small neural network to imitate the forward physics (the **SPN**), then
uses it to train an **inverse** network (the **DPN**) that proposes a geometry
for any requested spectrum. The trick that makes it work — the **PAD** — is
explained below and in `VIVA.md`.

---

## 1. The unifying abstraction

```
                         ┌─────────────────────────────┐
   target spectrum  ───▶ │      INVERSE  MODEL         │ ───▶  candidate design
   (R/T vs. λ)           │           DPN               │       4 params (r_o,r_i,h,L)
                         └──────────────┬──────────────┘
                                        │ design
                                        ▼
                         ┌─────────────────────────────┐
                         │     FORWARD SURROGATE        │ ───▶  predicted spectrum
                         │            SPN               │
                         └──────────────┬──────────────┘
                                        │ (used as a differentiable loss)
                                        ▼
                         ┌─────────────────────────────┐
                         │   GROUND-TRUTH PHYSICS       │  RCWA
                         │  (data generation + verify)  │
                         └─────────────────────────────┘
```

The key insight:

1. Train a **forward surrogate** (SPN) that imitates the RCWA physics solver —
   fast and differentiable.
2. Train the **inverse model** (DPN) whose output design is scored by passing it
   *through the frozen SPN* and comparing the resulting spectrum to the target.
   This is the **PAD** (pseudo-autodecoder).
3. Confirm the final design with the **real RCWA physics**.

---

## 2. The approach — PAD (Pseudo-Autodecoder)

- Design = `(r_o, r_i, h, L)` for a Si open-cylinder on SiO₂.
- **SPN**: `design → (R, T)` spectra.  Forward surrogate.
- **DPN**: `(R, T) → design`.  Inverse model.
- **PAD**: freeze a trained SPN, stitch `DPN → SPN`, train end-to-end so the
  predicted design reproduces the *input* spectrum. Solves the many-to-one
  problem that a raw inverse-DPN (IDPN) suffers from.

---

## 3. Directory layout

```
sem_prj/
├── README.md                  ← this file
├── ARCHITECTURE.md            ← deeper diagrams + math, per-module contracts
├── requirements.txt
├── config.py                  ← all hyperparameters / physical constants
├── Papers/                    ← source paper PDF + extracted text
│
├── physics/
│   ├── __init__.py
│   ├── spectra.py             ← spectrum dataclass, Gaussian-resonance helpers
│   └── parametric_sim.py      ← surrogate RCWA-like data generator
│
├── models/
│   ├── __init__.py
│   ├── base.py                ← ForwardSurrogate / InverseModel interfaces
│   ├── spn.py                 ← forward surrogate (SPN)
│   ├── dpn.py                 ← inverse model (DPN)
│   └── pad.py                 ← SPN + DPN pseudo-autodecoder (PAD)
│
├── pipeline/
│   ├── __init__.py
│   ├── datasets.py            ← PyTorch Dataset for parametric data
│   ├── train_parametric.py    ← train SPN, then PAD
│   ├── inverse_design.py      ← "given a target spectrum, get a design"
│   └── metrics.py             ← R² and spectrum-error metrics
│
└── scripts/
    └── smoke_test.py          ← tiny end-to-end run to validate wiring
```

---

## 4. Suggested semester milestones

1. **Weeks 1–2** Literature + reproduce the problem statement. Build
   `physics/parametric_sim.py` surrogate data generator.
   Deliverable: labelled dataset of (design, spectrum) pairs.
2. **Weeks 3–5** Train SPN, then DPN (IDPN), then PAD. Reproduce the
   many-to-one finding (PAD beats IDPN).
   Deliverable: R² regression plots comparing IDPN vs. PDPN.
3. **Weeks 6–7** Unified inverse-design pipeline + evaluation metrics.
   Deliverable: query a target spectrum, get a design back, verify with
   the surrogate RCWA.
4. **Weeks 8–9** Analysis of the broadband reflector result, comparison
   of IDPN vs. PAD performance, write-up and presentation.

See `ARCHITECTURE.md` for the detailed per-module contracts and the math.

> **Note on physics.** The real RCWA solver (`S4`) is heavy to install. This
> repo ships a lightweight *analytic surrogate* simulator in `physics/` so the
> full ML pipeline runs end-to-end on a laptop. It produces physically-plausible,
> resonance-shaped spectra — enough to exercise and validate every network. To do
> real science, swap the surrogate call sites for `S4`/Meep; the interfaces are
> designed for exactly that substitution.
