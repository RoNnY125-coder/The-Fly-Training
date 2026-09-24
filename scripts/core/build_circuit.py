"""
Phase 0, Step 3: build the signed connectivity matrix for our circuit.

What this script does, in order:
  1. Load cell_types.csv.gz, keep only neurons whose primary_type is in
     our CANDIDATE_TYPES list (circuit_definition.py) -> our ~1,370
     root_ids.
  2. Load connections.csv.gz, keep only edges where BOTH pre and post
     are in our circuit. This is a real cut: many of these neurons'
     true synapses go to neurons outside our 1,370 (e.g. into other
     brain regions), and we are deliberately ignoring those. That's an
     assumption worth stating in any writeup, not something to hide.
  3. Load neurotransmitter_types.csv.gz, join it onto our neurons to
     get a predicted sign: ACH/unclear -> +1 (excitatory default),
     GABA -> -1 (inhibitory), GLUT -> flagged as uncertain (see note
     below) but treated as -1 by default, matching the common
     convention in fly connectome work, while remaining aware this is
     a simplification.
  4. Aggregate multiple synapses between the same pre/post pair (there
     can be several rows in connections.csv for one pair, one per
     neuropil) into a single summed synapse count.
  5. Build an (N, N) signed weight matrix, W[post_idx, pre_idx] = sign * syn_count.
  6. Save the matrix, the root_id <-> index mapping, and per-neuron
     metadata (type, region, nt_type) to disk for every later script
     to reuse.

Run: python scripts/build_circuit.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from circuit_definition import ALL_TYPES, TYPE_TO_REGION, REGION_ORDER

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Neurotransmitter -> sign. GLUT is genuinely ambiguous in Drosophila
# (often inhibitory via GluCl receptors, unlike in vertebrates) -- we
# default it to inhibitory, which is the common convention in recent
# fly connectome modeling papers, but this is an assumption, not a
# measurement, and is worth revisiting if results look wrong.
NT_SIGN = {
    "ACH": +1,
    "GABA": -1,
    "GLUT": -1,
    "DA": 0,   # neuromodulatory, not a fast synaptic sign -- excluded (treated as 0)
    "SER": 0,
    "OCT": 0,
}


def load_circuit_neurons() -> pd.DataFrame:
    """Return one row per neuron in our circuit: root_id, primary_type, region."""
    df = pd.read_csv(DATA_DIR / "cell_types.csv.gz", compression="gzip")
    circuit = df[df["primary_type"].isin(ALL_TYPES)].copy()
    circuit["region"] = circuit["primary_type"].map(TYPE_TO_REGION)

    # Sort by region (in our fixed order), then by type, so the matrix
    # comes out block-structured and easy to plot/read later.
    region_rank = {r: i for i, r in enumerate(REGION_ORDER)}
    circuit["region_rank"] = circuit["region"].map(region_rank)
    circuit = circuit.sort_values(["region_rank", "primary_type", "root_id"]).reset_index(drop=True)
    circuit["idx"] = circuit.index  # 0..N-1, the matrix index for this neuron

    return circuit[["root_id", "primary_type", "region", "idx"]]


def load_edges_within_circuit(circuit_ids: set) -> pd.DataFrame:
    """Load connections.csv.gz, keep only edges fully inside our circuit."""
    # connections.csv.gz is ~68MB compressed -- fine to load in full,
    # then filter, rather than trying to stream/chunk it.
    conn = pd.read_csv(DATA_DIR / "connections.csv.gz", compression="gzip")

    mask = conn["pre_root_id"].isin(circuit_ids) & conn["post_root_id"].isin(circuit_ids)
    edges = conn[mask].copy()

    print(f"  total connectome edges: {len(conn):,}")
    print(f"  edges fully inside our {len(circuit_ids)}-neuron circuit: {len(edges):,}")

    return edges


def attach_neurotransmitter(edges: pd.DataFrame) -> pd.DataFrame:
    """
    connections.csv.gz already has an nt_type column (per-connection,
    not per-neuron), which is what we actually want here -- it's more
    specific than the neuron-level file. Use it directly.
    """
    edges = edges.copy()
    edges["sign"] = edges["nt_type"].map(NT_SIGN).fillna(0).astype(int)
    n_zero = (edges["sign"] == 0).sum()
    if n_zero:
        print(f"  note: {n_zero:,} edges have an unrecognized/neuromodulatory "
              f"nt_type and were set to sign=0 (excluded from the matrix)")
    return edges


def build_weight_matrix(circuit: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    n = len(circuit)
    W = np.zeros((n, n), dtype=np.float32)

    root_to_idx = dict(zip(circuit["root_id"], circuit["idx"]))

    # Aggregate: sum syn_count * sign across all rows for the same
    # (pre, post) pair (a pair can appear more than once, e.g. once per
    # neuropil the synapses sit in).
    edges = edges.copy()
    edges["weighted"] = edges["syn_count"] * edges["sign"]
    grouped = edges.groupby(["pre_root_id", "post_root_id"])["weighted"].sum()

    for (pre_id, post_id), w in grouped.items():
        pre_idx = root_to_idx[pre_id]
        post_idx = root_to_idx[post_id]
        W[post_idx, pre_idx] = w   # W[post, pre] convention: post = row, pre = column

    return W


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: selecting circuit neurons...")
    circuit = load_circuit_neurons()
    print(f"  {len(circuit)} neurons selected (expected 1370)\n")

    print("Step 2: filtering connections to within-circuit edges...")
    circuit_ids = set(circuit["root_id"])
    edges = load_edges_within_circuit(circuit_ids)
    print()

    print("Step 3: assigning synaptic sign from nt_type...")
    edges = attach_neurotransmitter(edges)
    print(edges["nt_type"].value_counts().to_string())
    print()

    print("Step 4: building weight matrix...")
    W = build_weight_matrix(circuit, edges)
    n_nonzero = np.count_nonzero(W)
    print(f"  matrix shape: {W.shape}, nonzero entries: {n_nonzero:,} "
          f"({100 * n_nonzero / W.size:.2f}% density)\n")

    print("Step 5: saving...")
    np.savez_compressed(
        OUT_DIR / "circuit_matrix.npz",
        W=W,
        root_ids=circuit["root_id"].to_numpy(),
        primary_types=circuit["primary_type"].to_numpy(),
        regions=circuit["region"].to_numpy(),
    )
    circuit.to_csv(OUT_DIR / "circuit_neurons.csv", index=False)

    print(f"  saved matrix to {OUT_DIR / 'circuit_matrix.npz'}")
    print(f"  saved neuron table to {OUT_DIR / 'circuit_neurons.csv'}")


if __name__ == "__main__":
    main()