"""
Phase 2, Step 2: decode from EB directly (trustworthy anatomical angle),
and test persistence -- does the bump survive after input is removed?

Why this changes from decoding PB: Phase 2 Step 1 showed PB's PCA-based
angle doesn't correspond to anything meaningful (error ~= chance even
for a STATIC heading). EB's angle assignment is more defensible since
EB is a real anatomical ring, so we decode from EB's own activity here.

This also tests something more meaningful than pure input-following:
we drive EB toward a fixed heading, then REMOVE the input, and watch
whether decoded heading stays roughly where it was (activity is doing
something ring-attractor-like, i.e. sustaining a representation) or
collapses immediately back to noise (the circuit has no real memory
and was just reflecting its input).

Run: python scripts/experiments/phase2_eb_persistence_test.py
Output: figures/eb_persistence.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lif_simulator import LIFNetwork, LIFParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

SCALE_EXC = 0.003
SCALE_INH = 0.002         # UPDATED: Phase 3 found default (0.02) suppresses
                          # ~80% of EB, and reducing inhibition let real
                          # wiring beat shuffled controls on localization.
                          # Retesting persistence under this corrected regime.
TAU_SYN = 5.0
REFRACTORY = 2.0

N_STEPS = 400
INPUT_ON_UNTIL = 200        # input is removed at this step
TRUE_HEADING = np.pi / 2
INPUT_STRENGTH = 1.5
EB_BASELINE_DRIVE = 0.15
PB_BASELINE_DRIVE = 0.15  # NEW: give PB the same baseline support, so it can
                          # actually participate in feedback to EB, instead
                          # of sitting silent with no drive at all -- tests
                          # whether the sustaining loop needs PB active   # same fix used for PB in Phase 1 -- a stand-in for
                           # excitatory drive EB would get in vivo from parts
                           # of the brain we excluded. Tests whether EB's
                           # earlier near-total silence after input removal
                           # was due to lacking this, same as PB's was.
DECODE_WINDOW = 20


def pca_angle(positions: np.ndarray) -> np.ndarray:
    centered = positions - positions.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    return np.mod(np.arctan2(proj[:, 1], proj[:, 0]), 2 * np.pi)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    circuit_data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = circuit_data["W"]
    regions = circuit_data["regions"]
    root_ids = circuit_data["root_ids"]
    n = W.shape[0]

    coords = pd.read_csv(DATA_DIR / "coordinates.csv.gz", compression="gzip")
    coords["xyz"] = coords["position"].str.strip("[]").str.split().apply(
        lambda parts: [float(p) for p in parts]
    )
    coord_lookup = dict(zip(coords["root_id"], coords["xyz"]))

    eb_mask = regions == "EB (ring/visual input)"
    pb_mask = regions == "PB (heading)"
    eb_indices = np.nonzero(eb_mask)[0]
    pb_indices = np.nonzero(pb_mask)[0]

    eb_pts, eb_valid_idx = [], []
    for idx in eb_indices:
        rid = root_ids[idx]
        if rid in coord_lookup:
            eb_pts.append(coord_lookup[rid])
            eb_valid_idx.append(idx)
    eb_pts = np.array(eb_pts)
    eb_valid_idx = np.array(eb_valid_idx)
    eb_angle_full = np.full(n, np.nan)
    eb_angle_full[eb_valid_idx] = pca_angle(eb_pts)

    print(f"EB neurons with known coordinates: {len(eb_valid_idx)}/{len(eb_indices)}")

    params = LIFParams(scale_exc=SCALE_EXC, scale_inh=SCALE_INH,
                        tau_syn_ms=TAU_SYN, refractory_ms=REFRACTORY)
    net = LIFNetwork(W, params)

    decoded = np.full(N_STEPS, np.nan)
    input_on = np.zeros(N_STEPS, dtype=bool)
    spike_history = np.zeros((DECODE_WINDOW, n), dtype=bool)

    for t in range(N_STEPS):
        ext = np.zeros(n, dtype=np.float32)
        ext[eb_valid_idx] = EB_BASELINE_DRIVE  # constant baseline, always on
        ext[pb_indices] = PB_BASELINE_DRIVE    # NEW: PB baseline, always on too
        if t < INPUT_ON_UNTIL:
            tuning = np.cos(TRUE_HEADING - eb_angle_full[eb_valid_idx])
            tuning = np.clip(tuning, 0, None)
            ext[eb_valid_idx] += INPUT_STRENGTH * tuning
            input_on[t] = True

        spiked = net.step(ext)
        spike_history[t % DECODE_WINDOW] = spiked

        if t >= DECODE_WINDOW:
            recent = spike_history[:, eb_valid_idx].sum(axis=0)
            if recent.sum() > 0:
                x = np.sum(recent * np.cos(eb_angle_full[eb_valid_idx]))
                y = np.sum(recent * np.sin(eb_angle_full[eb_valid_idx]))
                decoded[t] = np.mod(np.arctan2(y, x), 2 * np.pi)

    # Error while input is on vs. after it's removed -- the key comparison
    valid = ~np.isnan(decoded)
    err = np.degrees(np.abs(np.angle(np.exp(1j * (TRUE_HEADING - decoded)))))

    on_mask = valid & input_on
    off_mask = valid & ~input_on

    print(f"\nMean error WHILE input on:  {err[on_mask].mean():.1f} deg "
          f"(n={on_mask.sum()} steps)")
    print(f"Mean error AFTER input off: {err[off_mask].mean():.1f} deg "
          f"(n={off_mask.sum()} steps)")

    if err[off_mask].mean() < 45:
        verdict = "Bump PERSISTS after input removed -- attractor-like memory."
    elif err[off_mask].mean() < err[on_mask].mean() + 15:
        verdict = "Bump degrades gradually -- partial/weak persistence."
    else:
        verdict = "Bump COLLAPSES immediately -- no real memory, pure input-following."
    print(f"\nVerdict: {verdict}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(np.degrees(TRUE_HEADING), color="red", linestyle="--", label="true heading (while driven)")
    ax.plot(np.degrees(decoded), '.', markersize=3, color="blue", label="decoded from EB")
    ax.axvline(INPUT_ON_UNTIL, color="black", linestyle=":", label="input removed here")
    ax.set_xlabel("timestep")
    ax.set_ylabel("heading (degrees)")
    ax.set_title("EB decoded heading: driven, then input removed")
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "eb_persistence.png"
    plt.savefig(out_path, dpi=140)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()