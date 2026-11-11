# VIVA Guide — PAD Method (Paper 2)

Everything here is in plain language so you can explain and defend the whole
project without memorizing jargon. Read the story first, then the Q&A.

---

## 1. The story (what the project does, in plain English)

### The problem
A **metasurface** is a flat surface covered with tiny repeating structures,
smaller than the wavelength of light. By changing the shape of those structures,
you control how light reflects or passes through. In this project the tiny
structure is a **cylinder with a hole in it** (an "open cylinder") made of
silicon on a glass base.

The shape is described by just **4 numbers**:
- outer radius of the cylinder
- inner radius (the hole)
- height
- spacing between neighbouring cylinders

There are two directions you can work in:

- **Forward (possible but slow):** "I have these 4 numbers — what does the light
  do?" You can compute this with RCWA simulation. It works, but it is slow:
  finding a good design means trying millions of combinations.
- **Backward / inverse (hard and valuable):** "I want light to behave *this
  specific way* — what 4 numbers give me that?" There is **no formula** for
  this. Normally you would guess-and-check with the slow simulator forever.

**Goal of the project:** skip the guess-and-check. You describe the light
behaviour you want, and the system hands you a design.

### The solution, in three steps

**Step 1 — Teach a network to imitate the physics (the SPN).**
Run the slow simulator on ~9,000 random designs to get many
"design → light-response" examples. Train a neural network on them until it can
predict the light response from the 4 numbers *instantly*. It is not doing real
physics — it learned the pattern. The payoff: about **a million times faster**
than the real simulator, and it is differentiable (important for step 3).

**Step 2 — Build the inverse network (the DPN).**
This network does the backward job: feed it the light response you want, it
outputs 4 design numbers. Simple idea, but training it the obvious way hits a
trap (next step).

**Step 3 — The clever fix (the PAD). This is the heart of the paper.**
The trap is the **"many-to-one" problem**: different designs can produce almost
the *same* light response. So if you train the inverse network by telling it
"for this response the correct answer is *exactly these* 4 numbers," you are
effectively lying — other designs are also correct. The network gets confused,
**averages** between valid answers, and outputs a mushy, wrong design. That naive
version is called the **IDPN**, and it performs poorly.

The fix: **stop grading the network on which design it picked; grade it on the
result.** Flow:

```
   you ask for a light response
            |
            v
      inverse network (DPN)  -->  proposes a design (4 numbers)
                                        |
                                        v
                       the fast imitate-physics network (SPN, frozen)
                                        |
                                        v
                            predicted light response
                                        |
      compare THIS to what you asked for  <-- train on this gap
```

The proposed design is fed into the (now frozen) fast-physics network, which
says what light that design would produce. If that matches the request, good —
it does not matter *which* valid design was chosen. This stitched setup is the
**PAD (pseudo-autodecoder)**.

**Result:** accuracy jumps. The fit-quality score R² (where 1.0 is perfect) goes
from about **0.84 with the naive IDPN to about 0.96 with the PAD**.

### The payoff demo
They asked for a hard target: a surface that **reflects 99%+ of light across a
very wide band** around 1550 nm (the telecom wavelength). The network designed
one; checked against the real slow simulator it not only worked but beat the
best previously published result, covering **more than twice the bandwidth**
(450 nm vs. 200 nm).

### One-sentence summary (memorize this)
> "Running metasurface physics backward has no formula, so I trained a fast
> neural network to imitate the physics forward, then used it to train an
> inverse network — and the key trick was grading the inverse network on whether
> its design *produces the right light*, not on whether it matched one specific
> 'correct' design. That solves the many-to-one problem and pushes accuracy from
> about 0.84 to about 0.96."

---

## 2. The three networks at a glance

| Name | Plain meaning | Direction | In the code |
|------|---------------|-----------|-------------|
| **SPN** | "fake physics" — predicts light from a design, fast | design → spectrum | `models/spn.py` |
| **DPN** | the inverse designer — proposes a design for a wanted spectrum | spectrum → design | `models/dpn.py` |
| **PAD** | DPN + frozen SPN stitched together; the training trick | spectrum → design → spectrum | `models/pad.py` |
| **IDPN** | the *naive* DPN trained the wrong way (baseline to beat) | spectrum → design | trained in `pipeline/train_parametric.py` |

"Spectrum" just means the pair of curves: **R** (how much light reflects) and
**T** (how much transmits), across a range of wavelengths.

---

## 3. Likely examiner questions + short answers

**Q: What is a metasurface and why is designing one hard?**
A flat sheet of sub-wavelength structures that shape light. Hard because the
inverse direction (wanted light → geometry) has no formula and the search space
is enormous, so brute force with a physics simulator is too slow.

**Q: What does "inverse design" mean here?**
Starting from the *desired optical response* and producing the *geometry* that
achieves it — the reverse of normal simulation.

**Q: Why not just use the physics simulator directly?**
It only runs forward and is slow. To invert it you would try millions of designs.
We replace it with a neural network that is ~a million times faster and can be
used inside gradient-based training.

**Q: What is the SPN and why train it first?**
The "forward surrogate" — it imitates the simulator (design → spectrum) quickly
and differentiably. We train it first because the inverse trick (PAD) needs a
trusted, fast forward model to grade proposed designs against.

**Q: What is the many-to-one problem? (most important question)**
Several different geometries can give nearly the same spectrum. If you train the
inverse network to output one "correct" geometry per spectrum, it averages the
valid options and produces a bad design. The relationship spectrum → geometry is
not a clean one-to-one function.

**Q: How does the PAD fix it?**
Freeze the trained SPN and connect DPN → SPN. Train so that
SPN(DPN(spectrum)) ≈ the requested spectrum. You grade on the *resulting
spectrum*, not on the exact geometry, so any valid geometry is accepted. This
removes the averaging problem.

**Q: What does "frozen" mean and why freeze the SPN?**
Its weights are held fixed during PAD training — only the DPN learns. If the SPN
also changed, it could "cheat" by distorting the physics to make the loss look
small. Freezing keeps it an honest physics stand-in.

**Q: How do you measure success?**
R² (coefficient of determination) on predicted design parameters — 1.0 is
perfect. The paper reports ~0.84 for the naive IDPN and ~0.96 for the PAD. We
also check the achieved spectrum against the requested one.

**Q: Why is R² better for the PAD than the IDPN?**
Because the PAD is not forced to reproduce a single arbitrary "correct" geometry;
it only has to produce *a* geometry whose spectrum matches, which is an easier,
well-posed target.

**Q: Why "pseudo-autodecoder"?**
It resembles the decoder half of an autoencoder: a compact input (the spectrum)
is decoded into a design and then mapped back to a spectrum to check it — but it
is not a full autoencoder, hence "pseudo."

**Q: How did you get your training dataset?**
The dataset is generated by running a physics simulator (RCWA) on randomly
sampled open-cylinder designs. Specifically:
- The 4 design parameters `(r_o, r_i, h, L)` are randomly sampled within the
  ranges from the paper (e.g. r_o: 120–2000 nm, h: 200–1000 nm, L: 300–2200 nm),
  with physical validity constraints enforced (e.g. `r_i < r_o`, `r_o ≤ 0.4L`).
- For each sampled design, RCWA computes the reflection (R) and transmission (T)
  spectra across the wavelength range of interest (1300–1800 nm, 512 points).
- This gives one labelled example: `(r_o, r_i, h, L) → (R, T)`.
- The paper uses ~9,000 such examples. This is enough for the networks to
  converge — no one has derived the exact minimum needed for this problem size.
- In this implementation, `physics/parametric_sim.py` acts as the data generator
  (a lightweight analytic surrogate standing in for the heavy RCWA solver, so
  the pipeline runs on a laptop without installing S4).

The SPN is trained on these `(design → spectrum)` pairs. The DPN/PAD is trained
on the same data but uses the spectra as inputs and designs as targets.

**Q: What are the limits / weaknesses?**
The design space is restricted to 4 parameters and one shape family (open
cylinders); results are only as good as the surrogate and the training data; the
network stays within the ranges it was trained on. It will return the closest
achievable design, not necessarily an exact match if none exists.

**Q: What would you do to extend it?**
Richer geometries, more parameters, or a real RCWA solver (e.g. the `S4`
package) to replace the surrogate simulator.

**Q: (If they ask about your code) What runs the physics here?**
A lightweight analytic *surrogate* simulator stands in for the heavy RCWA solver
so the whole pipeline runs on a laptop. The interfaces are built so a real solver
(e.g. the `S4` package) can be dropped in without changing the networks.

---

## 4. Quick demo commands (to show it live)

```bash
# Train the forward model (SPN), the naive inverse (IDPN), and the PAD,
# then print the R^2 comparison that is the headline result:
python -m pipeline.train_parametric --epochs 200

# Sanity check that everything is wired correctly (fast):
python -m scripts.smoke_test
```

Expect the printed PDPN R² to be clearly higher than the IDPN R² — that gap is
the result you are defending.

---

## 5. Cheat-sheet of terms

- **Metasurface** — flat sheet of tiny light-shaping structures.
- **Spectrum (R, T)** — reflection and transmission curves vs. wavelength.
- **Forward model** — geometry → spectrum (easy).
- **Inverse design** — spectrum → geometry (hard; the point of the project).
- **SPN** — fast neural imitation of the physics (forward).
- **DPN** — the inverse network (spectrum → geometry).
- **IDPN** — DPN trained naively; the weak baseline.
- **PAD / PDPN** — DPN + frozen SPN; the fix; the strong result.
- **Many-to-one** — many geometries share one spectrum; the core difficulty.
- **R²** — fit-quality score, 1.0 is perfect.
- **RCWA** — the real (slow) physics simulation method the SPN replaces.
