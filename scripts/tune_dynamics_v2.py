"""
Phase 1, Step 5: retest dynamics with synaptic decay added.

lif_simulator.py now filters synaptic input through a decaying current
(tau_syn_ms) instead of applying it instantaneously. This should break
the network-wide "synchronized clock tick" pattern we saw before, since
a spike's influence is now smeared over several timesteps rather than
landing entirely on the next one.

We sweep synapse_scale and tau_syn_ms together, since they trade off:
a longer tau_syn means each spike's effect lasts longer, so a smaller
scale may be needed to avoid saturation.

Run: python scripts/tune_dynamics_v2.py
Output: figures/zoomed_raster_v2_*.png
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
ZOOM_STEPS = 100

# (synapse_scale, tau_syn_ms) combos
COMBOS = [
    (0.015, 5.0),
    (0.015, 10.0),
    (0.010, 10.0),
    (0.010, 20.0),
    (0.020, 5.0),
    (0.008, 15.0),
]


def run_one(W, n, kickstart_neurons, scale, tau_syn_ms):
    params = LIFParams(synapse_scale=scale, tau_syn_ms=tau_syn_ms)
    net = LIFNetwork(W, params)

    def input_fn(t, neurons=kickstart_neurons):
        if t < KICKSTART_STEPS:
            ext = np.zeros(n, dtype=np.float32)
            ext[neurons] = KICKSTART_STRENGTH
            return ext
        return None

    raster = net.run(N_STEPS, input_fn)
    return raster


def synchrony_index(raster: np.ndarray) -> float:
    """
    Crude synchrony measure: correlation between the population's
    total spike count per timestep and itself -- specifically, the
    coefficient of variation of the population spike count over time.
    High CV = bursty/synchronized (all-or-nothing per timestep).
    Low CV = smoother, more temporally distributed firing.
    """
    pop_count = raster.sum(axis=1).astype(np.float64)
    if pop_count.mean() == 0:
        return float("nan")
    return pop_count.std() / pop_count.mean()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = data["W"]
    n = W.shape[0]

    rng = np.random.default_rng(0)
    kickstart_neurons = rng.choice(n, size=max(1, n // 20), replace=False)

    print(f"{'scale':>8s}  {'tau_syn':>8s}  {'final rate':>11s}  {'max neuron rate':>16s}  {'sync (CV)':>10s}  verdict")
    print("-" * 90)

    for scale, tau_syn in COMBOS:
        raster = run_one(W, n, kickstart_neurons, scale, tau_syn)
        final_rate = raster[-100:].mean()
        per_neuron_rate = raster[-100:].mean(axis=0)
        max_rate = per_neuron_rate.max() if final_rate > 0 else 0.0
        sync = synchrony_index(raster[-100:])

        if final_rate < 0.001:
            verdict = "silent"
        elif final_rate > 0.5:
            verdict = "saturated"
        elif sync > 1.5:
            verdict = "still synchronized/bursty"
        else:
            verdict = "GRADED, non-synchronized -- good candidate"

        print(f"{scale:8.3f}  {tau_syn:8.1f}  {final_rate:11.1%}  {max_rate:16.1%}  {sync:10.2f}  {verdict}")

        fig, ax = plt.subplots(figsize=(8, 6))
        zoomed = raster[:ZOOM_STEPS]
        spike_times, spike_neurons = np.nonzero(zoomed)
        ax.scatter(spike_times, spike_neurons, s=1.0, c="black", marker="|")
        ax.axvline(KICKSTART_STEPS, color="blue", linestyle="--", alpha=0.5)
        ax.set_xlabel("timestep (zoomed)")
        ax.set_ylabel("neuron")
        ax.set_title(f"scale={scale}, tau_syn={tau_syn}ms, rate={final_rate:.1%}, sync_CV={sync:.2f}")
        plt.tight_layout()
        fname = FIGURES_DIR / f"zoomed_raster_v2_scale{scale}_tausyn{tau_syn}.png"
        plt.savefig(fname, dpi=120)
        plt.close(fig)

    print(f"\nZoomed rasters saved to {FIGURES_DIR}/zoomed_raster_v2_*.png")


if __name__ == "__main__":
    main()