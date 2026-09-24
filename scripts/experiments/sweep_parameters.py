"""
Phase 1, Step 2: parameter sweep.

We don't know tau, threshold, or synapse_scale for real -- these are
invented, not measured. This script runs the network under a range of
synapse_scale values (the parameter most likely to swing between
silence and saturation) with a small constant "kickstart" input, and
reports the population firing rate for each. We're looking for a
regime where firing rate is nonzero but well below saturation (e.g.
roughly 5-30% of neurons active per timestep, not 0% or ~100%).

Run: python scripts/sweep_parameters.py
"""

import numpy as np
from pathlib import Path

from lif_simulator import LIFNetwork, LIFParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

N_STEPS = 500
KICKSTART_STEPS = 20        # inject input only for the first N steps, then let it run free
KICKSTART_STRENGTH = 0.5
SCALES_TO_TRY = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def main():
    data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = data["W"]
    n = W.shape[0]

    rng = np.random.default_rng(0)
    # A fixed random subset of neurons gets kickstarted, simulating "some
    # sensory drive happened" -- arbitrary for now, just to see if
    # activity can propagate and persist at all.
    kickstart_neurons = rng.choice(n, size=max(1, n // 20), replace=False)

    print(f"{'scale':>8s}  {'mean rate':>10s}  {'final rate':>11s}  verdict")
    print("-" * 55)

    for scale in SCALES_TO_TRY:
        params = LIFParams(synapse_scale=scale)
        net = LIFNetwork(W, params)

        def input_fn(t, neurons=kickstart_neurons):
            if t < KICKSTART_STEPS:
                ext = np.zeros(n, dtype=np.float32)
                ext[neurons] = KICKSTART_STRENGTH
                return ext
            return None

        raster = net.run(N_STEPS, input_fn)

        mean_rate = raster.mean()               # fraction of (neuron, timestep) pairs that spiked
        final_rate = raster[-50:].mean()         # rate in the last 50 steps, i.e. after kickstart ended

        if final_rate < 0.001:
            verdict = "silent (died out after kickstart)"
        elif final_rate > 0.5:
            verdict = "saturated (near-constant firing)"
        else:
            verdict = "ACTIVE -- candidate regime"

        print(f"{scale:8.3f}  {mean_rate:10.1%}  {final_rate:11.1%}  {verdict}")


if __name__ == "__main__":
    main()