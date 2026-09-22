"""
Phase 1: a minimal leaky integrate-and-fire (LIF) simulator, with
separate excitatory/inhibitory scaling and decaying synaptic current.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class LIFParams:
    tau_ms: float = 20.0          # membrane time constant
    tau_syn_ms: float = 5.0       # synaptic current decay time constant
    v_rest: float = 0.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    refractory_ms: float = 2.0
    scale_exc: float = 0.02       # scale applied to excitatory (positive) weights
    scale_inh: float = 0.02       # scale applied to inhibitory (negative) weights
    dt_ms: float = 1.0


class LIFNetwork:
    def __init__(self, W: np.ndarray, params: LIFParams):
        """
        W: (N, N) signed weight matrix, W[post, pre] convention
           (matches circuit_matrix.npz from Phase 0).
        """
        self.p = params
        self.n = W.shape[0]

        self.W_exc = np.where(W > 0, W, 0.0).astype(np.float32)
        self.W_inh = np.where(W < 0, W, 0.0).astype(np.float32)

        self.V = np.full(self.n, params.v_rest, dtype=np.float32)
        self.I_syn = np.zeros(self.n, dtype=np.float32)
        self.refractory_until = np.zeros(self.n, dtype=np.int32)
        self.step_count = 0

        self.refractory_steps = int(round(params.refractory_ms / params.dt_ms))
        self.syn_decay = np.exp(-params.dt_ms / params.tau_syn_ms)

    def step(self, external_input: np.ndarray = None) -> np.ndarray:
        p = self.p
        self.step_count += 1

        incoming_spikes = self._last_spikes if hasattr(self, "_last_spikes") else np.zeros(self.n)
        exc_input = (self.W_exc @ incoming_spikes) * p.scale_exc
        inh_input = (self.W_inh @ incoming_spikes) * p.scale_inh
        self.I_syn = self.I_syn * self.syn_decay + exc_input + inh_input

        total_input = self.I_syn.copy()
        if external_input is not None:
            total_input = total_input + external_input

        not_refractory = self.step_count > self.refractory_until
        dv = (p.dt_ms / p.tau_ms) * (p.v_rest - self.V) + total_input
        self.V = np.where(not_refractory, self.V + dv, p.v_reset)

        spiked = (self.V >= p.v_thresh) & not_refractory
        self.V = np.where(spiked, p.v_reset, self.V)
        self.refractory_until = np.where(spiked, self.step_count + self.refractory_steps, self.refractory_until)

        self._last_spikes = spiked.astype(np.float32)
        return spiked

    def run(self, n_steps: int, external_input_fn=None) -> np.ndarray:
        raster = np.zeros((n_steps, self.n), dtype=bool)
        for t in range(n_steps):
            ext = external_input_fn(t) if external_input_fn else None
            raster[t] = self.step(ext)
        return raster