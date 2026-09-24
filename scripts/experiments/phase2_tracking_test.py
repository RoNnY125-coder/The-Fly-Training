"""
Phase 2, Step 1: does PB's decoded activity TRACK a moving heading signal?

Fixes the Phase 1 issue: EB and PB angles were previously assigned by
arbitrary index order, unrelated to each other. Here we derive angle
from each neuron's real soma position (coordinates.csv.gz), via PCA
projection onto the region's two dominant spatial axes. This is a real
anatomical signal, though for PB (more line-like than ring-like) it's
an approximation of compass angle, not a verified one -- stated
honestly, not hidden.

Test: rotate a simulated "true heading" slowly over time (like a fly
turning), drive EB with input tuned to bearing-to-landmark, give PB a
constant baseline excitatory tone (see Phase 1 notes on why), and
decode PB's population-vector angle at each timestep. Compare decoded
vs. true heading -- this is the actual scientific claim from our
original project design: "can we decode heading, and how does the
error behave over time."

Run: python scripts/phase2_tracking_test.py
Output: figures/heading_tracking.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

from lif_simulator import LIFNetwork, LIFParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

SCALE_EXC = 0.003
SCALE_INH = 0.02
TAU_SYN = 5.0
REFRACTORY = 2.0

N_STEPS = 600
INPUT_STRENGTH = 1.5
PB_BASELINE_DRIVE = 0.15
HEADING_PERIOD_STEPS = 400  # one full 360-degree rotation takes this many steps
STATIC_HEADING_TEST = True  # if True, heading never moves -- isolates whether
                            # the problem is PB's angle assignment (bad even
                            # when static) or tracking a MOVING signal (fine
                            # when static, fails when rotating)
STATIC_HEADING_VALUE = np.pi / 2


def pca_angle(positions: np.ndarray) -> np.ndarray:
    """Project 3D positions onto their 2 dominant axes and return angle
    of each point around the centroid, in radians [0, 2pi)."""
    centered = positions - positions.mean(axis=0)
    # SVD gives us the dominant axes directly
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T  # (n, 2) -- coordinates in the dominant plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    return np.mod(angles, 2 * np.pi)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    circuit_data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = circuit_data["W"]
    regions = circuit_data["regions"]
    root_ids = circuit_data["root_ids"]
    n = W.shape[0]

    coords = pd.read_csv(DATA_DIR / "coordinates.csv.gz", compression="gzip")
    # position column is a string like "[352484 175164 229040]" -- parse it
    coords["xyz"] = coords["position"].str.strip("[]").str.split().apply(
        lambda parts: [float(p) for p in parts]
    )
    coord_lookup = dict(zip(coords["root_id"], coords["xyz"]))

    eb_mask = regions == "EB (ring/visual input)"
    pb_mask = regions == "PB (heading)"
    eb_indices = np.nonzero(eb_mask)[0]
    pb_indices = np.nonzero(pb_mask)[0]

    def get_positions(indices):
        pts = []
        valid = []
        for idx in indices:
            rid = root_ids[idx]
            if rid in coord_lookup:
                pts.append(coord_lookup[rid])
                valid.append(idx)
        return np.array(pts), np.array(valid)

    eb_pts, eb_valid_idx = get_positions(eb_indices)
    pb_pts, pb_valid_idx = get_positions(pb_indices)

    print(f"EB neurons with known coordinates: {len(eb_valid_idx)}/{len(eb_indices)}")
    print(f"PB neurons with known coordinates: {len(pb_valid_idx)}/{len(pb_indices)}")

    eb_angles_valid = pca_angle(eb_pts)
    pb_angles_valid = pca_angle(pb_pts)

    # Map back to full arrays (neurons missing coordinates get angle=nan,
    # excluded from decoding/driving)
    eb_angle_full = np.full(n, np.nan)
    eb_angle_full[eb_valid_idx] = eb_angles_valid
    pb_angle_full = np.full(n, np.nan)
    pb_angle_full[pb_valid_idx] = pb_angles_valid

    params = LIFParams(scale_exc=SCALE_EXC, scale_inh=SCALE_INH,
                        tau_syn_ms=TAU_SYN, refractory_ms=REFRACTORY)
    net = LIFNetwork(W, params)

    true_headings = np.zeros(N_STEPS)
    decoded_headings = np.full(N_STEPS, np.nan)

    # Sliding window population vector decode: average recent PB spikes
    # weighted by each neuron's angle (circular mean), rather than
    # decoding from a single instantaneous timestep (too noisy).
    window = 20
    spike_history = np.zeros((window, n), dtype=bool)

    for t in range(N_STEPS):
        if STATIC_HEADING_TEST:
            true_heading = STATIC_HEADING_VALUE
        else:
            true_heading = (2 * np.pi * t / HEADING_PERIOD_STEPS) % (2 * np.pi)
        true_headings[t] = true_heading

        ext = np.zeros(n, dtype=np.float32)
        ext[pb_valid_idx] = PB_BASELINE_DRIVE
        if t >= 20:
            tuning = np.cos(true_heading - eb_angle_full[eb_valid_idx])
            tuning = np.clip(tuning, 0, None)
            ext[eb_valid_idx] = INPUT_STRENGTH * tuning

        spiked = net.step(ext)
        spike_history[t % window] = spiked

        if t >= window:
            recent_pb_spikes = spike_history[:, pb_valid_idx].sum(axis=0)
            if recent_pb_spikes.sum() > 0:
                # circular mean, weighted by spike count
                x = np.sum(recent_pb_spikes * np.cos(pb_angle_full[pb_valid_idx]))
                y = np.sum(recent_pb_spikes * np.sin(pb_angle_full[pb_valid_idx]))
                decoded_headings[t] = np.mod(np.arctan2(y, x), 2 * np.pi)

    valid = ~np.isnan(decoded_headings)
    print(f"\nDecoded heading available for {valid.sum()}/{N_STEPS} steps "
          f"(rest had zero PB spikes in the window)")

    if valid.sum() > 0:
        err = np.abs(np.angle(np.exp(1j * (true_headings[valid] - decoded_headings[valid]))))
        print(f"Mean circular error: {np.degrees(err.mean()):.1f} degrees "
              f"(chance level for a uniform random guess is ~90 deg)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.degrees(true_headings), label="true heading", color="red", linewidth=1.5)
    ax.plot(np.degrees(decoded_headings), label="decoded from PB", color="blue",
            linewidth=0.8, alpha=0.7, marker=".", markersize=2, linestyle="none")
    ax.set_xlabel("timestep")
    ax.set_ylabel("heading (degrees)")
    ax.set_title("True vs. decoded heading over time")
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "heading_tracking.png"
    plt.savefig(out_path, dpi=140)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()