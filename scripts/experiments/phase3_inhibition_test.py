"""
Phase 3, Step 2: WHY does real wiring localize worse than shuffled?

Hypothesis: EB's real wiring has strong, structured local inhibition
(confirmed back in Phase 0's connectivity matrix). This may suppress
so many neurons that the population-vector decode has too few active,
spatially-tuned neurons to work with -- vs. shuffled networks, where
that structured inhibition is destroyed and more neurons stay active
(even if their tuning is now meaningless).

Two things this script does:
  1. Diagnostic: compare how many EB neurons are actually active (and
     their total spike counts) in the real network vs a shuffle, for
     the same driven-phase test as Phase 3 Step 1.
  2. Causal test: rerun the REAL matrix with inhibition deliberately
     weakened (scale_inh reduced), and see if localization improves
     toward the shuffled-network level. If it does, that's direct
     evidence inhibition strength is the mechanism, not just a
     correlation.

Run: python scripts/experiments/phase3_inhibition_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import numpy as np
import pandas as pd

from lif_simulator import LIFNetwork, LIFParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SCALE_EXC = 0.003
SCALE_INH_DEFAULT = 0.02
TAU_SYN = 5.0
REFRACTORY = 2.0

N_STEPS = 220
TRUE_HEADING = np.pi / 2
INPUT_STRENGTH = 1.5
EB_BASELINE_DRIVE = 0.15
DECODE_WINDOW = 20
RNG_SEED = 42


def pca_angle(positions):
    centered = positions - positions.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    return np.mod(np.arctan2(proj[:, 1], proj[:, 0]), 2 * np.pi)


def shuffle_matrix(W, rng):
    post_idx, pre_idx = np.nonzero(W)
    weights = W[post_idx, pre_idx]
    shuffled_post = rng.permutation(post_idx)
    W_shuffled = np.zeros_like(W)
    np.add.at(W_shuffled, (shuffled_post, pre_idx), weights)
    return W_shuffled


def run_driven_test(W, eb_valid_idx, eb_angle_full, n, scale_inh):
    params = LIFParams(scale_exc=SCALE_EXC, scale_inh=scale_inh,
                        tau_syn_ms=TAU_SYN, refractory_ms=REFRACTORY)
    net = LIFNetwork(W, params)

    errors = []
    spike_history = np.zeros((DECODE_WINDOW, n), dtype=bool)
    all_eb_spikes = np.zeros(len(eb_valid_idx), dtype=int)

    for t in range(N_STEPS):
        ext = np.zeros(n, dtype=np.float32)
        ext[eb_valid_idx] = EB_BASELINE_DRIVE
        tuning = np.cos(TRUE_HEADING - eb_angle_full[eb_valid_idx])
        tuning = np.clip(tuning, 0, None)
        ext[eb_valid_idx] += INPUT_STRENGTH * tuning

        spiked = net.step(ext)
        spike_history[t % DECODE_WINDOW] = spiked
        all_eb_spikes += spiked[eb_valid_idx].astype(int)

        if t >= DECODE_WINDOW:
            recent = spike_history[:, eb_valid_idx].sum(axis=0)
            if recent.sum() > 0:
                x = np.sum(recent * np.cos(eb_angle_full[eb_valid_idx]))
                y = np.sum(recent * np.sin(eb_angle_full[eb_valid_idx]))
                decoded = np.mod(np.arctan2(y, x), 2 * np.pi)
                err = np.degrees(abs(np.angle(np.exp(1j * (TRUE_HEADING - decoded)))))
                errors.append(err)

    n_active = (all_eb_spikes > 0).sum()
    total_spikes = all_eb_spikes.sum()
    mean_err = np.mean(errors) if errors else np.nan
    return mean_err, n_active, total_spikes


def main():
    circuit_data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W_real = circuit_data["W"]
    regions = circuit_data["regions"]
    root_ids = circuit_data["root_ids"]
    n = W_real.shape[0]

    coords = pd.read_csv(DATA_DIR / "coordinates.csv.gz", compression="gzip")
    coords["xyz"] = coords["position"].str.strip("[]").str.split().apply(
        lambda parts: [float(p) for p in parts]
    )
    coord_lookup = dict(zip(coords["root_id"], coords["xyz"]))

    eb_mask = regions == "EB (ring/visual input)"
    eb_indices = np.nonzero(eb_mask)[0]
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
    n_eb = len(eb_valid_idx)

    # --- Part 1: diagnostic, real vs one shuffle, default inhibition ---
    rng = np.random.default_rng(RNG_SEED)
    W_shuf = shuffle_matrix(W_real, rng)

    print("=== Part 1: activity diagnostic (default inhibition) ===\n")
    err_real, active_real, spikes_real = run_driven_test(
        W_real, eb_valid_idx, eb_angle_full, n, SCALE_INH_DEFAULT)
    print(f"REAL wiring:     error={err_real:.1f} deg, "
          f"active neurons={active_real}/{n_eb} ({100*active_real/n_eb:.0f}%), "
          f"total spikes={spikes_real}")

    err_shuf, active_shuf, spikes_shuf = run_driven_test(
        W_shuf, eb_valid_idx, eb_angle_full, n, SCALE_INH_DEFAULT)
    print(f"SHUFFLED wiring: error={err_shuf:.1f} deg, "
          f"active neurons={active_shuf}/{n_eb} ({100*active_shuf/n_eb:.0f}%), "
          f"total spikes={spikes_shuf}")

    # --- Part 2: causal test, weaken inhibition on REAL matrix only ---
    print("\n=== Part 2: does weakening inhibition on REAL wiring help? ===\n")
    for scale_inh in [0.02, 0.01, 0.005, 0.002]:
        err, active, spikes = run_driven_test(
            W_real, eb_valid_idx, eb_angle_full, n, scale_inh)
        print(f"scale_inh={scale_inh:.3f}: error={err:.1f} deg, "
              f"active={active}/{n_eb} ({100*active/n_eb:.0f}%), spikes={spikes}")


if __name__ == "__main__":
    main()