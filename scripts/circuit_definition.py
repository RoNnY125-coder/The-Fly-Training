"""
The final circuit definition for Makkhi's Phase 0-1 model.

This is deliberately the ONLY place these cell-type names live. Every
other script imports CANDIDATE_TYPES from here instead of redefining it,
so if we ever revise the circuit, there's exactly one place to change it.

Selection rationale (see project log for full discussion):
  - PB: the heading ("compass") representation itself
  - EB ring neurons: visual/landmark input that anchors the compass
  - FB local (hDelta, full A-M): the steering computation -- offsets
    between current heading and a remembered/desired heading
  - FB columnar input (PFN): carries self-motion / optic-flow-like
    velocity signals into the FB
  - FB columnar output (PFL, FC1, FC2): descending-facing output columns
    that would drive steering commands
  - NO: velocity integration input to PFN

Deliberately excluded: vDelta neurons (391 real cells found in the data).
These are well-documented FB local interneurons, but their functional
role is tied to FB processing broadly rather than specifically to
heading/steering behavior in the literature we're relying on. Including
them would have pushed the total past 1,700 without a clear behavioral
justification per neuron, so we left them out. This is a judgment call,
not a fact -- worth revisiting if later results seem to be missing
something vDelta would have provided.
"""

CANDIDATE_TYPES = {
    "PB (heading)": ["EPG", "EPGt", "PEN_a/PEN1", "PEN_b/PEN2", "PEG", "Delta7", "PFGs"],
    "EB (ring/visual input)": [
        "ER1", "ER2",
        "ER3d", "ER3w", "ER3a_a", "ER3a_b", "ER3m", "ER3p_a", "ER3p_b",
        "ER4d", "ER4m", "ER5",
    ],
    "FB local (steering computation)": [
        "hDeltaA", "hDeltaB", "hDeltaC", "hDeltaD", "hDeltaE", "hDeltaF",
        "hDeltaG", "hDeltaH", "hDeltaI", "hDeltaJ", "hDeltaK", "hDeltaL", "hDeltaM",
    ],
    "FB columnar (steering/goal input)": ["PFNa", "PFNd", "PFNm", "PFNp", "PFNv"],
    "FB columnar (output)": [
        "PFL1", "PFL2", "PFL3", "PFR",
        "FC1A", "FC1C", "FC1D",
        "FC2A", "FC2B", "FC2C",
    ],
    "NO (velocity input)": ["LNO1", "LNO2", "LNOa", "GLNO"],
}

# Flat list of every type name, and a lookup from type -> region, used by
# the matrix builder and the plotting script to order neurons by region.
ALL_TYPES = [t for types in CANDIDATE_TYPES.values() for t in types]

TYPE_TO_REGION = {
    t: region for region, types in CANDIDATE_TYPES.items() for t in types
}

# Fixed region order for plotting (PB -> EB -> FB local -> FB in -> FB out -> NO)
REGION_ORDER = list(CANDIDATE_TYPES.keys())