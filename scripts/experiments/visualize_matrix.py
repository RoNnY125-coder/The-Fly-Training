"""
Phase 0, Step 4: visualize the circuit weight matrix.

The neurons in circuit_matrix.npz are already ordered by region (PB,
EB, FB local, FB in, FB out, NO), courtesy of build_circuit.py's sort.
This script just plots that matrix as a heatmap with region boundaries
drawn on top, and prints per-region/per-type summary stats.

What we're looking for: real central-complex circuits are known to have
block/banded structure (e.g. PEN neurons projecting to EPG neurons in a
shifted pattern -- the mechanism that lets the compass bump rotate). If
this plot is uniform noise with no visible structure, that's a signal
something upstream is wrong. If it shows visible blocks, that's a good
sign the wiring reflects real, meaningful structure.

Run: python scripts/visualize_matrix.py
Output: figures/circuit_matrix.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = data["W"]
    regions = data["regions"]
    types = data["primary_types"]

    n = W.shape[0]
    print(f"Matrix: {W.shape}, {np.count_nonzero(W):,} nonzero entries\n")

    # Find where each region starts/ends, for drawing boundary lines.
    # Neurons are already sorted by region (done in build_circuit.py).
    region_bounds = []
    current_region = regions[0]
    start = 0
    for i in range(1, n + 1):
        if i == n or regions[i] != current_region:
            region_bounds.append((current_region, start, i))
            if i < n:
                current_region = regions[i]
                start = i

    print("Region blocks (in matrix order):")
    for region, start, end in region_bounds:
        print(f"  {region:35s} rows/cols {start:4d}-{end:4d}  ({end - start} neurons)")

    # Plot: use a diverging colormap so excitatory (+) and inhibitory (-)
    # are visually distinguishable, centered at zero.
    vmax = np.percentile(np.abs(W[W != 0]), 99)  # robust scale, ignore outlier synapse counts

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    # Draw region boundary lines
    for region, start, end in region_bounds:
        ax.axhline(start - 0.5, color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(start - 0.5, color="black", linewidth=0.5, alpha=0.5)

    # Label region blocks on the axes
    tick_positions = [(start + end) / 2 for _, start, end in region_bounds]
    tick_labels = [region for region, _, _ in region_bounds]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("presynaptic neuron (source)")
    ax.set_ylabel("postsynaptic neuron (target)")
    ax.set_title(f"Makkhi circuit connectivity matrix (N={n}, signed by predicted neurotransmitter)")

    plt.colorbar(im, ax=ax, label="signed synapse count", shrink=0.7)
    plt.tight_layout()

    out_path = FIGURES_DIR / "circuit_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # Also print a quick per-type connection summary as a sanity check
    df = pd.DataFrame({"type": types, "region": regions})
    print("\nNeurons per type (top 10 by count):")
    print(df["type"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()