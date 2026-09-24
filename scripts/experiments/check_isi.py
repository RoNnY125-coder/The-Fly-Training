"""
Phase 1, Step 6: inter-spike interval (ISI) check.

The previous "synchrony CV" metric measured population count over time,
which can't distinguish "healthy steady asynchronous firing" from
"everyone locked into firing every single timestep" -- both look flat.

This script instead looks at INDIVIDUAL neurons: for each of the most
active neurons, what are the gaps between its consecutive spikes?

  - If a neuron fires every 1-2 steps, every single time, with almost
    no variation (ISI histogram is a single sharp spike) -- it's a
    stuck oscillator at its refractory ceiling, not doing computation.
  - If ISIs vary meaningfully (a spread-out histogram) -- that's a sign
    of real, input-driven dynamics.

Run: python scripts/check_isi.py
Output: figures/isi_histogram.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lif_simulator import LIFNetwork, LIFParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

N_STEPS = 500
KICKSTART_STEPS = 20
KICKSTART_STRENGTH = 0.5

# use E/I-separated scaling now -- push excitation down specifically,
# since the ISI check proved neurons were pinned at their refractory
# ceiling (CV_ISI = 0.00), meaning excitatory drive was far past
# threshold, not delicately balanced near it.
SCALE_EXC = 0.006
SCALE_INH = 0.02
TAU_SYN = 10.0


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = data["W"]
    n = W.shape[0]

    rng = np.random.default_rng(0)
    kickstart_neurons = rng.choice(n, size=max(1, n // 20), replace=False)

    params = LIFParams(scale_exc=SCALE_EXC, scale_inh=SCALE_INH, tau_syn_ms=TAU_SYN)
    net = LIFNetwork(W, params)

    def input_fn(t, neurons=kickstart_neurons):
        if t < KICKSTART_STEPS:
            ext = np.zeros(n, dtype=np.float32)
            ext[neurons] = KICKSTART_STRENGTH
            return ext
        return None

    raster = net.run(N_STEPS, input_fn)

    # Look only at steady-state (after kickstart), pick the 5 most active neurons
    steady = raster[KICKSTART_STEPS:]
    rates = steady.mean(axis=0)
    top5 = np.argsort(-rates)[:5]

    print("Inter-spike interval stats for the 5 most active neurons (steady-state):\n")
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)

    for ax, neuron_idx in zip(axes, top5):
        spike_steps = np.nonzero(steady[:, neuron_idx])[0]
        isis = np.diff(spike_steps)

        if len(isis) < 2:
            print(f"  neuron {neuron_idx}: too few spikes to analyze")
            ax.set_title(f"neuron {neuron_idx}\n(too few spikes)")
            continue

        mean_isi = isis.mean()
        std_isi = isis.std()
        cv_isi = std_isi / mean_isi if mean_isi > 0 else float("nan")

        print(f"  neuron {neuron_idx}: rate={rates[neuron_idx]:.1%}  "
              f"mean_ISI={mean_isi:.2f}  std_ISI={std_isi:.2f}  CV_ISI={cv_isi:.2f}  "
              f"({'REGULAR/stuck' if cv_isi < 0.15 else 'irregular -- good sign'})")

        ax.hist(isis, bins=range(1, int(isis.max()) + 2), color="steelblue", edgecolor="black")
        ax.set_title(f"neuron {neuron_idx}\nCV={cv_isi:.2f}")
        ax.set_xlabel("ISI (timesteps)")

    axes[0].set_ylabel("count")
    plt.suptitle(f"ISI distributions, scale_exc={SCALE_EXC}, scale_inh={SCALE_INH}, tau_syn={TAU_SYN}ms\n"
                 f"(CV_ISI near 0 = fires at a fixed rhythm; higher CV = irregular, more brain-like)")
    plt.tight_layout()

    out_path = FIGURES_DIR / "isi_histogram.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved to {out_path}")
    print("\nInterpretation guide: CV_ISI < 0.15 usually means near-perfectly regular firing")
    print("(a stuck oscillator). Real cortical/insect spiking data typically shows CV_ISI")
    print("in the 0.5-1.5 range. If everything here is well under 0.15, the refractory")
    print("period is currently the dominant force, not the actual input pattern.")


if __name__ == "__main__":
    main()