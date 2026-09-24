"""
Phase 1, Step 4: finer sweep + zoomed raster to diagnose saturation.

Two things in one script:
  1. A finer sweep between 0.010 (silent) and 0.020 (looked saturated),
     to see if there's a narrower band with more graded activity.
  2. A zoomed-in raster (first 60 steps only) at a couple of candidate
     scales, so we can actually SEE whether "active" neurons are firing
     every single timestep (= stuck at ceiling, bad) or firing at some
     more moderate, varying rate (= real dynamics, good).

We also try increasing the refractory period, since a neuron that can
only recover for 2ms and receives strong constant drive will basically
always be exactly at the ceiling rate 1/refractory_period -- that's a
plausible cause of the "solid bar" pattern we saw.

Run: python scripts/tune_dynamics.py
Output: figures/zoomed_raster_<scale>_<refractory>.png for each combo tried
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lif_simulator import LIFNetwork, LIFParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

N_STEPS = 300
KICKSTART_STEPS = 20
KICKSTART_STRENGTH = 0.5
ZOOM_STEPS = 60

# (synapse_scale, refractory_ms) combos to try
COMBOS = [
    (0.012, 2.0),
    (0.015, 2.0),
    (0.017, 2.0),
    (0.015, 5.0),
    (0.015, 10.0),
]


def run_one(W, n, kickstart_neurons, scale, refractory_ms):
    params = LIFParams(synapse_scale=scale, refractory_ms=refractory_ms)
    net = LIFNetwork(W, params)

    def input_fn(t, neurons=kickstart_neurons):
        if t < KICKSTART_STEPS:
            ext = np.zeros(n, dtype=np.float32)
            ext[neurons] = KICKSTART_STRENGTH
            return ext
        return None

    raster = net.run(N_STEPS, input_fn)
    return raster


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = data["W"]
    regions = data["regions"]
    n = W.shape[0]

    rng = np.random.default_rng(0)
    kickstart_neurons = rng.choice(n, size=max(1, n // 20), replace=False)

    print(f"{'scale':>8s}  {'refrac_ms':>9s}  {'final rate':>11s}  {'max single-neuron rate':>23s}  verdict")
    print("-" * 80)

    for scale, refractory_ms in COMBOS:
        raster = run_one(W, n, kickstart_neurons, scale, refractory_ms)
        final_rate = raster[-100:].mean()

        # The key new diagnostic: what's the firing rate of the SINGLE
        # most active neuron, in the steady-state window? If it's near
        # 1/refractory_steps, that neuron is at its physical ceiling.
        per_neuron_rate = raster[-100:].mean(axis=0)
        max_rate = per_neuron_rate.max()
        theoretical_ceiling = 1.0 / max(1, round(refractory_ms))  # dt=1ms, so refractory_ms ~= refractory_steps

        if final_rate < 0.001:
            verdict = "silent"
        elif max_rate > 0.9 * theoretical_ceiling:
            verdict = "SATURATED (neurons pinned at refractory ceiling)"
        else:
            verdict = "graded -- candidate"

        print(f"{scale:8.3f}  {refractory_ms:9.1f}  {final_rate:11.1%}  {max_rate:23.1%}  {verdict}")

        # Save a zoomed raster for this combo so we can eyeball it
        fig, ax = plt.subplots(figsize=(8, 6))
        zoomed = raster[:ZOOM_STEPS]
        spike_times, spike_neurons = np.nonzero(zoomed)
        ax.scatter(spike_times, spike_neurons, s=1.0, c="black", marker="|")
        ax.axvline(KICKSTART_STEPS, color="blue", linestyle="--", alpha=0.5)
        ax.set_xlabel("timestep (zoomed)")
        ax.set_ylabel("neuron")
        ax.set_title(f"scale={scale}, refractory={refractory_ms}ms, final_rate={final_rate:.1%}")
        plt.tight_layout()
        fname = FIGURES_DIR / f"zoomed_raster_scale{scale}_refrac{refractory_ms}.png"
        plt.savefig(fname, dpi=120)
        plt.close(fig)

    print(f"\nZoomed rasters saved to {FIGURES_DIR}/zoomed_raster_*.png")


if __name__ == "__main__":
    main()