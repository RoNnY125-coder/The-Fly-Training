import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_FILES = [
    "connections.csv.gz",
    "classification.csv.gz",
    "cell_types.csv.gz",
    "neurotransmitter_types.csv.gz",
    "coordinates.csv.gz",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    missing = [f for f in EXPECTED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print("Missing files in ./data/ :")
        for f in missing:
            print(f"  - {f}")
        print(
            "\nDownload these from https://codex.flywire.ai/api/download?dataset=fafb "
            "(sign in first), save with the exact names above, then re-run."
        )
        sys.exit(1)

    print(f"Verifying files in {DATA_DIR}\n")
    records = []
    for filename in EXPECTED_FILES:
        path = DATA_DIR / filename
        size_bytes = path.stat().st_size
        digest = sha256_of(path)
        print(f"  {filename:35s} {size_bytes / 1e6:8.1f} MB   sha256: {digest[:16]}...")
        records.append(
            {"filename": filename, "size_bytes": size_bytes, "sha256": digest}
        )

    manifest = {
        "dataset": "FlyWire FAFB v783 (Codex export)",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://codex.flywire.ai/api/download?dataset=fafb (manual, signed-in download)",
        "license": "CC BY-NC 4.0 (non-commercial, attribution required)",
        "citations": [
            "Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. "
            "Nature 634, 124-138 (2024).",
            "Schlegel, P. et al. Whole-brain annotation and multi-connectome "
            "cell typing of Drosophila. Nature 634, 139-152 (2024).",
            "Eckstein, N. et al. Neurotransmitter classification from electron "
            "microscopy images at synaptic sites in Drosophila melanogaster. "
            "Cell 187(10), 2574-2594 (2024).",
        ],
        "files": records,
    }

    manifest_path = MANIFEST_DIR / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll files present and hashed. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()