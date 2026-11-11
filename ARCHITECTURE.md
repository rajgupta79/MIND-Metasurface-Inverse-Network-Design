# ARCHITECTURE — detailed design

This document specifies the math and the per-module contracts. It is the
engineering companion to `README.md`.

## 0. Notation

- `λ` wavelength grid, `N_λ` points.
- Spectrum `s = (R, T)` where `R, T ∈ [0, 1]^{N_λ}` are reflection / transmission.
- Parametric design `p = (r_o, r_i, h, L) ∈ R^4` (normalized to `[0,1]`).

---

## 1. Spectra-Predicting Network (SPN) — forward surrogate

Learns `f: p → s`. Architecture (following Eq. 1 of the paper):

```
input 4
  → Dense(8·N) → Dense(6·N) → Dense(4·N) → Dense(2·N)
       ├── R-branch: Dense(N) → Dense(N_λ)   [sigmoid]
       └── T-branch: Dense(N) → Dense(N_λ)   [sigmoid]
```

Loss: MSE on `(R, T)`. ReLU hidden activations, sigmoid output (spectra live in
`[0,1]`). Once trained the SPN is ~1 million times faster than the real RCWA
solver and is differentiable, making it usable as a loss component.

---

## 2. Design-Predicting Network (DPN) — inverse model

Learns `g: s → p`. Two input branches (R, T), merge, compress to 4 outputs:

```
R → Dense×5    ┐
                merge → Dense(2N) → Dense(N) → Dense(N/2) → Dense(32) → Dense(4) [sigmoid]
T → Dense×5    ┘
```

---

## 3. The many-to-one problem

`s = f(p)` is one-to-one, but `p = f⁻¹(s)` is **not**: different geometries can
yield near-identical spectra. A DPN trained directly on `(s, p)` pairs (the
**IDPN**) regresses toward the *mean* of the valid geometries and underfits
(paper reports inner-radius R² ≈ 83.6%).

---

## 4. Pseudo-Autodecoder (PAD) — the fix

Freeze a trained SPN and stitch `DPN → SPN`:

```
s_target → DPN → p̂ → (frozen) SPN → ŝ
loss = MSE(ŝ, s_target)       # compare spectra, NOT geometry
```

Now the DPN is free to pick *any* geometry whose spectrum matches, instead of
being forced to a single "correct" geometry. The paper reports this lifts the
average R² to ≈ 96.4%. The number of quantities to minimize over drops from
`N_R + N_T` to effectively the 4 design params.

---

## 5. Inverse-design pipeline (Paper Fig. 4)

"Flip and stitch": `s_target → PDPN → design → SPN → predicted spectrum`, then
verify `design` with the real RCWA solver.

```
user-defined target spectrum
         |
         v
    DPN (PDPN)       ← inverse model
         |
         v
   proposed design (r_o, r_i, h, L)
         |
         v
    SPN (frozen)     ← fast forward surrogate
         |
         v
   predicted spectrum  (shown to user; should match target)
         |
         v
   RCWA verification  (ground-truth physics check)
```

---

## 6. Data contracts (`physics/`)

| function | returns | used by |
|---|---|---|
| `parametric_sim.simulate(p, λ)` | `Spectrum(R, T)` | SPN training, verify |
| `parametric_sim.sample_dataset(n)` | `[(p, Spectrum)]` | SPN/DPN training |

The surrogate physics is intentionally simple but resonance-shaped: geometry
parameters map to resonance centers/widths/amplitudes via smooth functions, so
the networks have a genuine (nonlinear, many-to-one) relationship to learn.

---

## 7. Shared interfaces (`models/base.py`)

```python
class ForwardSurrogate(Protocol):
    def predict_spectrum(self, design) -> Spectrum: ...   # SPN

class InverseModel(Protocol):
    def propose_design(self, target: Spectrum) -> Design: ...  # DPN / PAD
```

`pipeline/inverse_design.py` depends only on these two interfaces.
