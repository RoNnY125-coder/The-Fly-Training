"""
Phase 0, Step 5: quantitative structure check.

Question: does the real wiring concentrate synaptic weight WITHIN
regions (PB-PB, EB-EB, etc.) more than chance? This is the first real
instance of the "shuffled control" comparison from our original
experimental design (Experiment A vs B in the project plan).

Shuffle method (stated honestly): each edge's weight stays attached to
its original presynaptic neuron, but the postsynaptic target is
randomly reassigned across the whole 1,370-neuron circuit. This exactly
preserves each neuron's total OUTGOING signed weight (out-strength),
and approximately preserves the incoming distribution in aggregate --
but NOT per-neuron in-strength. This is a standard, simple configuration-
model-style null, not a perfect degree-preserving rewiring. Good enough
to ask "is there region structure at all", not good enough to claim
precise statistical rigor -- we'll note that limitation in the output.

Metric: fraction of total |signed weight| that connects two neurons in
the SAME region, vs different regions. Computed for the real matrix,
then for N_SHUFFLES random shuffles, to build a null distribution.

Run: python scripts/quantitative_check.py
"""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

N_SHUFFLES = 200
RNG_SEED = 42


def within_region_fraction(pre_idx, post_idx, weight, region_of):
    """Fraction of total |weight| where pre and post are in the same region."""
    same_region = region_of[pre_idx] == region_of[post_idx]
    total = np.abs(weight).sum()
    within = np.abs(weight[same_region]).sum()
    return within / total


def main():
    data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = data["W"]
    regions = data["regions"]
    n = W.shape[0]

    # Extract the real edge list from the matrix: (pre_idx, post_idx, weight)
    # Recall W[post, pre] convention from build_circuit.py.
    post_idx, pre_idx = np.nonzero(W)
    weight = W[post_idx, pre_idx]
    print(f"Real edges (matrix entries): {len(weight):,}")

    real_score = within_region_fraction(pre_idx, post_idx, weight, regions)
    print(f"\nReal circuit: {real_score:.1%} of total |signed weight| is within-region")

    # Build the null distribution by shuffling post_idx (see docstring for
    # exactly what this does and doesn't preserve).
    rng = np.random.default_rng(RNG_SEED)
    null_scores = np.empty(N_SHUFFLES)
    for i in range(N_SHUFFLES):
        shuffled_post = rng.permutation(post_idx)
        null_scores[i] = within_region_fraction(pre_idx, shuffled_post, weight, regions)

    print(f"\nShuffled controls (n={N_SHUFFLES}):")
    print(f"  mean:   {null_scores.mean():.1%}")
    print(f"  std:    {null_scores.std():.1%}")
    print(f"  range:  {null_scores.min():.1%} - {null_scores.max():.1%}")

    z = (real_score - null_scores.mean()) / null_scores.std()
    percentile = (null_scores < real_score).mean() * 100
    print(f"\nReal circuit vs null distribution:")
    print(f"  z-score:    {z:.2f}")
    print(f"  percentile: {percentile:.1f}  "
          f"(real score exceeds {percentile:.1f}% of shuffled controls)")

    if z > 3:
        verdict = ("Real wiring shows substantially more within-region structure "
                   "than shuffled controls. Consistent with genuine anatomical "
                   "modularity, not a plotting artifact.")
    elif z > 1:
        verdict = ("Real wiring shows somewhat more within-region structure than "
                   "shuffled controls, but the effect is modest -- worth reporting "
                   "with appropriate caution rather than as a strong claim.")
    else:
        verdict = ("Real wiring is not clearly more within-region than shuffled "
                   "controls by this metric. The visual block structure may be "
                   "driven by a few specific connections rather than a general "
                   "pattern -- worth investigating which ones before concluding "
                   "anything.")
    print(f"\nVerdict: {verdict}")

    # Per-region breakdown: which regions specifically drive the effect?
    print("\nPer-region within-region weight fraction (real circuit only):")
    unique_regions = np.unique(regions)
    for r in unique_regions:
        mask = (regions[pre_idx] == r) & (regions[post_idx] == r)
        region_total_mask = (regions[pre_idx] == r) | (regions[post_idx] == r)
        if region_total_mask.sum() == 0:
            continue
        frac = np.abs(weight[mask]).sum() / np.abs(weight[region_total_mask]).sum()
        print(f"  {r:35s} {frac:.1%} of this region's edge weight stays within-region")


if __name__ == "__main__":
    main()