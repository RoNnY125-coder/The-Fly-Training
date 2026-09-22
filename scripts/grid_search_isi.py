"""
Phase 1, Step 7: real grid search, judged correctly this time.

Previous metrics (population rate, population-count CV) both failed to
detect a stuck, self-sustaining clique of neurons. The only metric that
caught it was per-neuron ISI regularity. This script sweeps scale_exc,
tau_syn_ms, and refractory_ms together, and for EVERY active neuron
(not just top 5) computes ISI CV, then reports the population median.

We are deliberately trying much smaller scale_exc values than before,
since 0.006 still fully saturated a subgroup.

Run: python scripts/grid_search_isi.py
"""

import numpy as np
from pathlib import Path

from lif_simulator import LIFNetwork, LIFParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

N_STEPS = 500
KICKSTART_STEPS = 20
KICKSTART_STRENGTH = 0.5

SCALE_EXC_VALUES = [0.001, 0.002, 0.003, 0.004, 0.006]
TAU_SYN_VALUES = [2.0, 5.0, 10.0]
REFRACTORY_VALUES = [2.0, 5.0]
SCALE_INH = 0.02  # keep inhibition fixed, only searching excitation + timing


def median_isi_cv(raster: np.ndarray) -> float:
    """Median ISI coefficient of variation across all neurons that fired
    at least 3 times in the steady-state window. Returns nan if none did."""
    cvs = []
    for i in range(raster.shape[1]):
        spike_steps = np.nonzero(raster[:, i])[0]
        if len(spike_steps) < 3:
            continue
        isis = np.diff(spike_steps)
        if isis.mean() > 0:
            cvs.append(isis.std() / isis.mean())
    return float(np.median(cvs)) if cvs else float("nan")


def main():
    data = np.load(PROCESSED_DIR / "circuit_matrix.npz", allow_pickle=True)
    W = data["W"]
    n = W.shape[0]

    rng = np.random.default_rng(0)
    kickstart_neurons = rng.choice(n, size=max(1, n // 20), replace=False)

    print(f"{'scale_exc':>10s}  {'tau_syn':>8s}  {'refrac':>7s}  {'rate':>7s}  "
          f"{'n_active':>9s}  {'median CV_ISI':>14s}  verdict")
    print("-" * 90)

    best = None
    for scale_exc in SCALE_EXC_VALUES:
        for tau_syn in TAU_SYN_VALUES:
            for refrac in REFRACTORY_VALUES:
                params = LIFParams(
                    scale_exc=scale_exc, scale_inh=SCALE_INH,
                    tau_syn_ms=tau_syn, refractory_ms=refrac,
                )
                net = LIFNetwork(W, params)

                def input_fn(t, neurons=kickstart_neurons):
                    if t < KICKSTART_STEPS:
                        ext = np.zeros(n, dtype=np.float32)
                        ext[neurons] = KICKSTART_STRENGTH
                        return ext
                    return None

                raster = net.run(N_STEPS, input_fn)
                steady = raster[KICKSTART_STEPS:]
                rate = steady.mean()
                n_active = (steady.sum(axis=0) >= 3).sum()
                cv = median_isi_cv(steady)

                if rate < 0.001:
                    verdict = "silent"
                elif np.isnan(cv):
                    verdict = "too few repeat-firers to assess"
                elif cv < 0.15:
                    verdict = "stuck/regular"
                elif cv > 1.5:
                    verdict = "erratic (possibly still unstable)"
                else:
                    verdict = "GOOD -- irregular, graded firing"
                    if best is None or (0.3 < cv < 1.2 and rate < 0.3):
                        best = (scale_exc, tau_syn, refrac, rate, cv)

                print(f"{scale_exc:10.4f}  {tau_syn:8.1f}  {refrac:7.1f}  {rate:7.1%}  "
                      f"{n_active:9d}  {cv:14.2f}  {verdict}")

    print()
    if best:
        print(f"Best candidate found: scale_exc={best[0]}, tau_syn={best[1]}, "
              f"refractory={best[2]}  (rate={best[3]:.1%}, median CV_ISI={best[4]:.2f})")
    else:
        print("No setting hit the 'GOOD' band in this search. Consider searching "
              "even smaller scale_exc values, or reducing scale_inh proportionally too.")


if __name__ == "__main__":
    main()