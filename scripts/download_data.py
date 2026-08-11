"""Fetch the small public datasets used by the runnable part of the suite.

Downloads go through ``curl`` because the stock macOS Python certificate store
is frequently incomplete for ``urllib``.  Everything here is small (<60 MB
total); the large DomainBed / WILDS benchmarks are handled separately by
``sadl/data/real.py`` and are never fetched automatically.
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DSPRITES_URL = (
    "https://github.com/google-deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.relative_to(ROOT)}")
        return
    print(f"[get ] {url}")
    subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(dest), url], check=True)


def main() -> int:
    curl(DSPRITES_URL, DATA / "dsprites" / "dsprites.npz")
    raw = DATA / "mnist" / "raw"
    for name in MNIST_FILES:
        curl(f"{MNIST_BASE}/{name}", raw / name)
        out = raw / name[:-3]
        if not out.exists():
            with gzip.open(raw / name, "rb") as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
