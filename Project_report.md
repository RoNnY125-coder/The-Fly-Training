# Makkhi: A Connectome-Constrained Model of the Fly Head-Direction Circuit
### Project report — Phase 0 through Phase 3

## What this project is

This project extracts a real biological neural circuit from FlyWire (a
complete synaptic wiring diagram of an adult fruit fly brain, ~139,000
neurons) and simulates it computationally, to test what real neural
wiring can and cannot do when turned into a spiking model.

**Important framing, stated upfront:** a connectome gives us *structure*
— which neurons connect to which, and how many synapses. It does not
give us *physiology* — time constants, synaptic strengths, thresholds.
Every dynamic parameter in this project is an assumption we chose, not
a measurement FlyWire provided. This is a connectome-**constrained**
model, not a simulated fly brain.

---

## Phase 0 — Data and circuit selection

- Downloaded and hash-verified FlyWire FAFB v783 (connections,
  cell-type, neurotransmitter, and coordinate data).
- Selected a **1,370-neuron circuit**: the central complex's
  head-direction system — the protocerebral bridge (PB, the "compass"),
  ellipsoid body ring neurons (EB, visual/landmark input), fan-shaped
  body local and columnar neurons (steering computation and output),
  and the noduli (velocity input).
- Every inclusion/exclusion was a documented judgment call, not an
  arbitrary cut — e.g. we excluded a 391-neuron interneuron family
  (vDelta) because its role in steering specifically isn't
  well-established, even though including it would have hit a rounder
  1,500-neuron target.
- Built a signed connectivity matrix (29,552 connections) from real
  synapse counts and predicted neurotransmitter types.
- **Validated the wiring is real, structured biology**, not noise:
  visual inspection showed anatomically sensible patterns (diagonal
  banding in the ring-neuron population, inhibitory ring→compass
  projections matching the literature), and a quantitative test against
  200 randomly shuffled controls showed the real wiring keeps 56% of
  its connection weight within functional regions vs. ~21% for random
  wiring (z = 124.5 — a very strong, non-random effect).

## Phase 1 — Spiking dynamics

- Built a leaky integrate-and-fire (LIF) simulator from scratch.
- Found that naive parameter choices produced either total silence or
  a stuck oscillator (every neuron firing the instant it was
  physically able to, regardless of any input).
- Diagnosed this using inter-spike interval analysis, not just
  population firing rate, which had been silently masking the problem.
- Found a real working regime via grid search across excitation
  strength, synaptic decay time, and refractory period, judged by
  spike-timing irregularity — the correct criterion, once we discovered
  the earlier metrics were blind to synchronized-but-flat activity.
- Discovered separating excitatory and inhibitory synaptic scaling
  (rather than one global multiplier) was necessary — otherwise
  weakening the network always weakened its own restraint by the same
  amount.

## Phase 2 — Closed-loop directional input

- Gave the circuit a simulated "landmark" input and tested whether it
  could localize direction and hold that representation over time.
- Found our first angle-assignment method (soma-position PCA) worked
  for EB (a real anatomical ring) but failed for PB (anatomically more
  linear) — confirmed by a null result even for a *static* heading.
- Switched to decoding from EB directly. Found the circuit could
  localize a driven heading moderately well (~42° error vs. ~90° for
  random guessing) but the localization **collapsed immediately** once
  input was removed — no sustained memory.

## Phase 3 — Real wiring vs. random wiring

- The core experiment: does the real connectome do better than randomly
  shuffled wiring (same neuron count, same per-neuron output strength)
  on the localization task?
- **Surprising first result: real wiring did *worse*** (42.1° error vs.
  24.4° for shuffled controls).
- Diagnosed why: real wiring's strong, structured local inhibition
  (independently confirmed in Phase 0) suppressed ~80% of the ring
  neuron population under our simplified drive, leaving too few active
  neurons for a good decode.
- **Proved this causally**: weakening inhibition on the *same real
  matrix* recovered performance, and at sufficiently low inhibition the
  real wiring actually **beat** shuffled controls (12.3° vs. 24.4°) —
  the real structure is an advantage once compensated for the missing
  excitatory context the rest of the (excluded) brain would normally
  provide.
- Tested whether this same fix restored memory/persistence: it did not.
  Weakening inhibition improved instantaneous localization further
  (down to 7.8°) but persistence stayed broken, and giving the PB
  region more activity to participate made persistence actively worse
  (131° error), ruling out "PB was just too quiet" as the explanation.
- **Conclusion:** localization quality is governed by an
  excitation/inhibition balance we can tune. Sustained memory appears
  to depend on a correctly-wired EB↔PB feedback loop that requires real
  anatomical angle-to-neuron mapping data we don't have (FlyWire's
  annotations don't include it, and it isn't derivable from soma
  position for PB's non-ring geometry).

## Status: Phase 3 core comparison complete; full statistical version
(idealized ring-attractor baseline, leaky-integrator baseline, larger
seed counts) not yet built.

## Data & citation
FlyWire FAFB v783, CC BY-NC 4.0. Dorkenwald et al., *Nature* 634,
124–138 (2024); Schlegel et al., *Nature* 634, 139–152 (2024);
Eckstein et al., *Cell* 187(10), 2574–2594 (2024).