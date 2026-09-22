"""
Phase 1, Step 8 (exit test): directional input -> localized bump?

Assumption stated upfront: FlyWire doesn't annotate which EB/PB neuron
corresponds to which compass angle (that tiling isn't in our columns).
We assign each EB ring neuron and each PB heading neuron an arbitrary
angle by index order within its region -- NOT a real anatomical
mapping. This is a stand-in, and any "bump" we see is evidence the
CIRCUIT can produce localized activity given cosine-tuned input, not
evidence it encodes the fly's real compass geometry.

Method: drive EB neurons with input ~ cos(true_heading - assigned_angle)
(a simple von-Mises-like tuning), i.e. simulating "a landmark sits at
true_heading". Hold this constant for the whole run. Then look at PB
neurons' firing rate as a function of THEIR assigned angle: a peaked
curve = localized bump. A flat curve = no spatial structure emerged.

Run: python scripts/bump_test.py
Output: figures/bump_test.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lif_simulator import LIFNetwork, LIFParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

# locked-in parameters from the grid search
SCALE_EXC = 0.003
SCALE_INH = 0.02
TAU_SYN = 5.0
REFRACTORY = 2.0

N_STEPS = 400
TRUE_HEADING = np.pi / 2   # arbitrary fixed "landmark" direction, 90 degrees
INPUT_STRENGTH = 1.5       # increased substantially -- EB's strong self-inhibition
                           # may have absorbed the previous, weaker input entirely
PB_BASELINE_DRIVE = 0.15   # constant excitatory tone to PB, standing in for the
                           # real excitatory input PB gets in vivo from EPG's own
                           # recurrent loop and other brain regions we excluded


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = data["W"]
    regions = data["regions"]
    n = W.shape[0]

    eb_mask = regions == "EB (ring/visual input)"
    pb_mask = regions == "PB (heading)"
    n_eb = eb_mask.sum()
    n_pb = pb_mask.sum()

    # Assign angles by index order within each region (the stated assumption)
    eb_angles = np.linspace(0, 2 * np.pi, n_eb, endpoint=False)
    pb_angles = np.linspace(0, 2 * np.pi, n_pb, endpoint=False)

    eb_indices = np.nonzero(eb_mask)[0]
    pb_indices = np.nonzero(pb_mask)[0]

    def input_fn(t):
        ext = np.zeros(n, dtype=np.float32)
        # Baseline excitatory tone to PB -- in the real fly, PB receives
        # excitatory drive from EPG's own recurrent loop and other brain
        # regions we excluded from this circuit. Without SOME baseline
        # drive, PB can never cross threshold at all, since EB's
        # projection onto it is inhibitory (shown in Phase 0's matrix)
        # and can only shape an existing bump, not create one from zero.
        ext[pb_indices] = PB_BASELINE_DRIVE
        if t >= 20:  # let the network settle briefly before driving it
            tuning = np.cos(TRUE_HEADING - eb_angles)  # peaked at true_heading
            tuning = np.clip(tuning, 0, None)  # only positive drive, like real input
            ext[eb_indices] = INPUT_STRENGTH * tuning
        return ext

    params = LIFParams(scale_exc=SCALE_EXC, scale_inh=SCALE_INH,
                        tau_syn_ms=TAU_SYN, refractory_ms=REFRACTORY)
    net = LIFNetwork(W, params)
    raster = net.run(N_STEPS, input_fn)

    steady = raster[-150:]  # last 150 steps, well after input turned on
    pb_rates = steady[:, pb_indices].mean(axis=0)
    eb_rates = steady[:, eb_indices].mean(axis=0)

    print(f"EB firing rate stats: mean={eb_rates.mean():.1%}, max={eb_rates.max():.1%}  "
          f"(the driven region -- if this is ~0%, the input isn't reaching threshold at all)")
    print(f"PB firing rate stats: mean={pb_rates.mean():.1%}, max={pb_rates.max():.1%}, "
          f"min={pb_rates.min():.1%}")

    # Is there a peak near true_heading, or is it flat?
    # Bin PB neurons by angle and average rate per bin.
    n_bins = 16
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_rates = np.zeros(n_bins)
    for b in range(n_bins):
        in_bin = (pb_angles >= bin_edges[b]) & (pb_angles < bin_edges[b + 1])
        bin_rates[b] = pb_rates[in_bin].mean() if in_bin.any() else np.nan

    peak_bin = np.nanargmax(bin_rates)
    peak_angle = bin_centers[peak_bin]
    print(f"\nPeak PB activity at assigned angle {np.degrees(peak_angle):.0f} deg "
          f"(true heading was {np.degrees(TRUE_HEADING):.0f} deg)")

    flatness = bin_rates[~np.isnan(bin_rates)].std() / (bin_rates[~np.isnan(bin_rates)].mean() + 1e-9)
    print(f"Peakedness (std/mean across angle bins): {flatness:.2f}  "
          f"({'flat -- no localized bump' if flatness < 0.3 else 'PEAKED -- localized activity found'})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(np.degrees(bin_centers), bin_rates, width=20, color="steelblue")
    axes[0].axvline(np.degrees(TRUE_HEADING), color="red", linestyle="--", label="true heading (input)")
    axes[0].set_xlabel("assigned angle (degrees)")
    axes[0].set_ylabel("PB firing rate")
    axes[0].set_title("PB activity vs assigned angle")
    axes[0].legend()

    spike_times, spike_neurons = np.nonzero(raster[:, pb_indices])
    axes[1].scatter(spike_times, spike_neurons, s=1, c="black", marker="|")
    axes[1].axvline(20, color="blue", linestyle="--", alpha=0.5, label="input turns on")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("PB neuron")
    axes[1].set_title("PB raster over time")
    axes[1].legend()

    plt.tight_layout()
    out_path = FIGURES_DIR / "bump_test.png"
    plt.savefig(out_path, dpi=140)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()