"""
Phase 1, Step 3: is the activity structured, or just uniform noise?

A nonzero firing rate alone doesn't tell us much -- random noise can
also produce "20% of neurons active". This script runs the network at
our chosen parameter setting and checks:

  1. A spike raster plot (time x neuron), ordered by region, so we can
     SEE whether activity clusters in specific regions/types or is
     spread uniformly everywhere.
  2. Per-region firing rates -- are some regions much more/less active
     than others, or is it flat across the whole circuit?

If activity is roughly uniform across every region regardless of type,
that's a sign the dynamics aren't yet reflecting the real structure we
found in Phase 0 (the strong EB self-connectivity, NO as a relay, etc.)
-- it would suggest we're in a regime driven by raw connection density
rather than the specific, meaningful wiring pattern.

Run: python scripts/inspect_activity.py
Output: figures/spike_raster.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lif_simulator import LIFNetwork, LIFParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

SCALE = 0.02          # the lowest setting that showed sustained activity
N_STEPS = 500
KICKSTART_STEPS = 20
KICKSTART_STRENGTH = 0.5


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = data["W"]
    regions = data["regions"]
    types = data["primary_types"]
    n = W.shape[0]

    rng = np.random.default_rng(0)
    kickstart_neurons = rng.choice(n, size=max(1, n // 20), replace=False)

    params = LIFParams(synapse_scale=SCALE)
    net = LIFNetwork(W, params)

    def input_fn(t, neurons=kickstart_neurons):
        if t < KICKSTART_STEPS:
            ext = np.zeros(n, dtype=np.float32)
            ext[neurons] = KICKSTART_STRENGTH
            return ext
        return None

    raster = net.run(N_STEPS, input_fn)  # (N_STEPS, n) boolean

    # --- Per-region firing rate ---
    print("Per-region firing rate (fraction of steps each neuron in the region spiked):")
    unique_regions = np.unique(regions)
    for r in unique_regions:
        mask = regions == r
        region_rate = raster[:, mask].mean()
        print(f"  {r:35s} {region_rate:.1%}   ({mask.sum()} neurons)")

    print("\nPer-type firing rate (top 10 most active types):")
    unique_types = np.unique(types)
    type_rates = []
    for t in unique_types:
        mask = types == t
        if mask.sum() == 0:
            continue
        type_rates.append((t, raster[:, mask].mean(), mask.sum()))
    type_rates.sort(key=lambda x: -x[1])
    for t, rate, count in type_rates[:10]:
        print(f"  {t:15s} {rate:.1%}   ({count} neurons)")

    print("\nPer-type firing rate (10 LEAST active types):")
    for t, rate, count in type_rates[-10:]:
        print(f"  {t:15s} {rate:.1%}   ({count} neurons)")

    # --- Raster plot, ordered by region (neurons are already sorted this way) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    spike_times, spike_neurons = np.nonzero(raster)
    ax.scatter(spike_times, spike_neurons, s=0.5, c="black", marker="|")

    # Draw region boundary lines, same style as the connectivity plot
    region_bounds = []
    current_region = regions[0]
    start = 0
    for i in range(1, n + 1):
        if i == n or regions[i] != current_region:
            region_bounds.append((current_region, start, i))
            if i < n:
                current_region = regions[i]
                start = i
    for region, s, e in region_bounds:
        ax.axhline(s, color="red", linewidth=0.5, alpha=0.4)
    tick_positions = [(s + e) / 2 for _, s, e in region_bounds]
    tick_labels = [r for r, _, _ in region_bounds]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.axvline(KICKSTART_STEPS, color="blue", linestyle="--", alpha=0.5, label="kickstart ends")
    ax.set_xlabel("timestep")
    ax.set_ylabel("neuron (grouped by region)")
    ax.set_title(f"Spike raster, synapse_scale={SCALE}")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    out_path = FIGURES_DIR / "spike_raster.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved raster plot to {out_path}")


if __name__ == "__main__":
    main()