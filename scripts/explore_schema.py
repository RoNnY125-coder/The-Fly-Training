"""
Phase 0, Step 2: look at the raw schema before writing any filtering logic.

"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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

def peek(filename: str, n: int = 5):
    path = DATA_DIR / filename
    print(f"\n{'=' * 70}\n{filename}\n{'=' * 70}")
    df = pd.read_csv(path, compression="gzip", nrows=2000)
    print(f"columns: {list(df.columns)}")
    print(df.head(n).to_string())
    return df


def count_cell_types(cell_types_path: Path):
    print(f"\n{'=' * 70}\nCell type counts for candidate circuit\n{'=' * 70}")
    df = pd.read_csv(cell_types_path, compression="gzip")
    print(f"columns available: {list(df.columns)}")

    type_col_candidates = [
    c for c in df.columns
    if "cell_type" in c.lower() or c.lower() in ("type", "primary_type")
    ]
    if not type_col_candidates:
        print("Could not auto-detect a cell-type column -- inspect columns above manually.")
        return
    type_col = type_col_candidates[0]
    print(f"using column: '{type_col}'\n")

    total = 0
    for region, types in CANDIDATE_TYPES.items():
        subset = df[df[type_col].isin(types)]
        counts = subset[type_col].value_counts()
        region_total = len(subset)
        total += region_total
        print(f"{region}: {region_total} neurons")
        for t in types:
            c = counts.get(t, 0)
            flag = "  <-- not found, check name" if c == 0 else ""
            print(f"    {t:10s} {c:5d}{flag}")
        print()

    print(f"TOTAL across candidate circuit: {total} neurons")


def main():
    peek("classification.csv.gz")
    peek("cell_types.csv.gz")
    peek("neurotransmitter_types.csv.gz")
    peek("connections.csv.gz")
    peek("coordinates.csv.gz")

    count_cell_types(DATA_DIR / "cell_types.csv.gz")


if __name__ == "__main__":
    main()