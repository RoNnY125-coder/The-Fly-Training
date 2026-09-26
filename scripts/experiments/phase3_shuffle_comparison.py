"""
Phase 3, Step 1: real wiring vs. shuffled controls.

The actual experiment this whole project was built around (Experiment A
vs B from the original design). We already know from Phase 2 that:
  - the real circuit localizes moderately well while directly driven
    (~42 deg error, vs ~90 deg chance)
  - that localization collapses once input is removed (no persistence)

Question for Phase 3: is even that DRIVEN localization actually due to
the SPECIFIC real wiring, or would any network with the same basic
statistics (same neuron count, same degree distribution, same E/I
balance) do just as well, since a lot of the localization might just
come from EB's ring-shaped geometry and our cosine-tuned INPUT, not
from anything specific about the real synapses?

Shuffle method (same as Phase 0's quantitative_check.py): each edge's
weight stays with its presynaptic neuron, but its postsynaptic target
is randomly reassigned across the whole circuit. This preserves each
neuron's total outgoing signed weight, approximately preserves the
incoming distribution in aggregate, and completely destroys any
specific real wiring pattern (e.g. EB's local ring structure).

We run the SAME drive-then-release test on the real matrix and on
N_SHUFFLES independent shuffles, and compare driven-phase error (the
part we know is meaningfully above chance) between them.

Run: python scripts/experiments/phase3_shuffle_comparison.py
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
SCALE_INH = 0.02
TAU_SYN = 5.0
REFRACTORY = 2.0

N_STEPS = 220             # driven phase only -- we established there's no
                          # persistence to measure, so focus the comparison
                          # on the part that showed a real signal
TRUE_HEADING = np.pi / 2
INPUT_STRENGTH = 1.5
EB_BASELINE_DRIVE = 0.15
DECODE_WINDOW = 20
N_SHUFFLES = 20
RNG_SEED = 42


def pca_angle(positions: np.ndarray) -> np.ndarray:
    centered = positions - positions.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    return np.mod(np.arctan2(proj[:, 1], proj[:, 0]), 2 * np.pi)


def shuffle_matrix(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same method as Phase 0's quantitative check: keep each edge's
    weight with its presynaptic neuron, randomly reassign postsynaptic
    target. See docstring above for exactly what this does/doesn't
    preserve."""
    post_idx, pre_idx = np.nonzero(W)
    weights = W[post_idx, pre_idx]
    shuffled_post = rng.permutation(post_idx)

    W_shuffled = np.zeros_like(W)
    # accumulate in case of collisions (multiple edges landing on same cell)
    np.add.at(W_shuffled, (shuffled_post, pre_idx), weights)
    return W_shuffled


def run_driven_test(W, eb_valid_idx, eb_angle_full, n):
    params = LIFParams(scale_exc=SCALE_EXC, scale_inh=SCALE_INH,
                        tau_syn_ms=TAU_SYN, refractory_ms=REFRACTORY)
    net = LIFNetwork(W, params)

    errors = []
    spike_history = np.zeros((DECODE_WINDOW, n), dtype=bool)

    for t in range(N_STEPS):
        ext = np.zeros(n, dtype=np.float32)
        ext[eb_valid_idx] = EB_BASELINE_DRIVE
        tuning = np.cos(TRUE_HEADING - eb_angle_full[eb_valid_idx])
        tuning = np.clip(tuning, 0, None)
        ext[eb_valid_idx] += INPUT_STRENGTH * tuning

        spiked = net.step(ext)
        spike_history[t % DECODE_WINDOW] = spiked

        if t >= DECODE_WINDOW:
            recent = spike_history[:, eb_valid_idx].sum(axis=0)
            if recent.sum() > 0:
                x = np.sum(recent * np.cos(eb_angle_full[eb_valid_idx]))
                y = np.sum(recent * np.sin(eb_angle_full[eb_valid_idx]))
                decoded = np.mod(np.arctan2(y, x), 2 * np.pi)
                err = np.degrees(abs(np.angle(np.exp(1j * (TRUE_HEADING - decoded)))))
                errors.append(err)

    return np.mean(errors) if errors else np.nan, len(errors)


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

    print("Running on REAL connectome wiring...")
    real_err, real_n = run_driven_test(W_real, eb_valid_idx, eb_angle_full, n)
    print(f"  real wiring: mean error = {real_err:.1f} deg (n={real_n} decodable steps)\n")

    print(f"Running on {N_SHUFFLES} shuffled controls...")
    rng = np.random.default_rng(RNG_SEED)
    shuffle_errors = []
    for i in range(N_SHUFFLES):
        W_shuf = shuffle_matrix(W_real, rng)
        err, n_dec = run_driven_test(W_shuf, eb_valid_idx, eb_angle_full, n)
        shuffle_errors.append(err)
        print(f"  shuffle {i+1}/{N_SHUFFLES}: mean error = {err:.1f} deg (n={n_dec})")

    shuffle_errors = np.array(shuffle_errors)
    valid_shuffles = shuffle_errors[~np.isnan(shuffle_errors)]

    print(f"\n{'=' * 60}")
    print(f"Real wiring error:      {real_err:.1f} deg")
    print(f"Shuffled controls:      mean={valid_shuffles.mean():.1f} deg, "
          f"std={valid_shuffles.std():.1f} deg, n={len(valid_shuffles)}/{N_SHUFFLES} valid")

    if len(valid_shuffles) > 1 and valid_shuffles.std() > 0:
        z = (real_err - valid_shuffles.mean()) / valid_shuffles.std()
        percentile = (valid_shuffles > real_err).mean() * 100  # lower error = better, so flip direction
        print(f"z-score (negative = real wiring is BETTER than shuffled): {z:.2f}")
        print(f"Real wiring beats {percentile:.0f}% of shuffled controls")

        if z < -1.5:
            verdict = "Real wiring localizes MEANINGFULLY BETTER than random wiring."
        elif z < -0.5:
            verdict = "Real wiring shows a modest advantage over random wiring."
        else:
            verdict = "No clear advantage over random wiring -- localization may be driven mostly by input geometry, not specific real connectivity."
        print(f"\nVerdict: {verdict}")
    else:
        print("Not enough valid shuffle results to compute a comparison.")


if __name__ == "__main__":
    main()