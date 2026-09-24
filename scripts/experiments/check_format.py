"""
Diagnostic: what format are these files actually in, and do they fully
decompress without error?

Run: python scripts/check_format.py
"""

import gzip
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = [
    "connections.csv.gz",
    "classification.csv.gz",
    "cell_types.csv.gz",
    "neurotransmitter_types.csv.gz",
    "coordinates.csv.gz",
]

SIGNATURES = {
    b"\x1f\x8b": "gzip",
    b"PK\x03\x04": "zip",
    b"PK\x05\x06": "zip (empty)",
    b"BZh": "bzip2",
    b"\xef\xbb\xbf": "UTF-8 BOM (likely plain text/CSV)",
}


def identify(path: Path) -> str:
    with open(path, "rb") as f:
        header = f.read(8)
    for magic, name in SIGNATURES.items():
        if header.startswith(magic):
            return name
    try:
        text_preview = header.decode("utf-8")
        if text_preview.isprintable() or "\n" in text_preview or "," in text_preview:
            return f"looks like plain text -- preview: {text_preview!r}"
    except UnicodeDecodeError:
        pass
    return f"UNKNOWN format (bytes: {header[:8]!r})"


def full_decompress_check(path: Path) -> str:
    try:
        total_bytes = 0
        with gzip.open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                total_bytes += len(chunk)
        return f"OK -- decompresses fully ({total_bytes / 1e6:.1f} MB uncompressed)"
    except Exception as e:
        return f"FAILED -- {type(e).__name__}: {e}"


def main():
    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"{filename:35s} MISSING")
            continue
        fmt = identify(path)
        print(f"{filename:35s} header: {fmt}")
        if fmt == "gzip":
            result = full_decompress_check(path)
            print(f"{'':35s} full read: {result}")
        print()


if __name__ == "__main__":
    main()