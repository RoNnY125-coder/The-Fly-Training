"""
Phase 0, Step 2b: find the real names for cell types that came back empty.

"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# substrings for the categories that came back empty or suspiciously low
SEARCH_TERMS = ["PEN", "ER3", "PFN", "LNO", "NO", "hDelta", "vDelta", "FC1", "FC2"]


def main():
    df = pd.read_csv(DATA_DIR / "cell_types.csv.gz", compression="gzip")
    types = df["primary_type"].dropna()
    counts = types.value_counts()

    for term in SEARCH_TERMS:
        print(f"\n--- types containing '{term}' ---")
        matches = counts[counts.index.str.contains(term, case=False, na=False)]
        if matches.empty:
            print("  (no matches at all)")
        else:
            for name, c in matches.items():
                print(f"  {name:15s} {c}")


if __name__ == "__main__":
    main()