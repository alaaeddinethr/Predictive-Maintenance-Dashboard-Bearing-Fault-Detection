"""
download_data.py
----------------
Downloads and organizes the CWRU Bearing Dataset from Case Western Reserve University.
Dataset URL: https://engineering.case.edu/bearingdatacenter

Structure after download:
    data/raw/
        normal/       → healthy bearing signals
        inner_race/   → inner race fault signals
        outer_race/   → outer race fault signals
        ball/         → ball fault signals

Usage:
    python src/data/download_data.py
    python src/data/download_data.py --output data/raw --fault-size 0.007
"""

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
from tqdm import tqdm

# ─── CWRU Dataset URLs (Drive End, 12kHz) ────────────────────────────────────
# Source: https://engineering.case.edu/bearingdatacenter/download-data-file

CWRU_FILES = {
    # Normal baseline (0, 1, 2, 3 HP)
    "normal": [
        ("97.mat",  "https://engineering.case.edu/sites/default/files/97.mat"),
        ("98.mat",  "https://engineering.case.edu/sites/default/files/98.mat"),
        ("99.mat",  "https://engineering.case.edu/sites/default/files/99.mat"),
        ("100.mat", "https://engineering.case.edu/sites/default/files/100.mat"),
    ],
    # Inner Race Fault — 0.007 inch
    "inner_race_007": [
        ("105.mat", "https://engineering.case.edu/sites/default/files/105.mat"),
        ("106.mat", "https://engineering.case.edu/sites/default/files/106.mat"),
        ("107.mat", "https://engineering.case.edu/sites/default/files/107.mat"),
        ("108.mat", "https://engineering.case.edu/sites/default/files/108.mat"),
    ],
    # Inner Race Fault — 0.014 inch
    "inner_race_014": [
        ("169.mat", "https://engineering.case.edu/sites/default/files/169.mat"),
        ("170.mat", "https://engineering.case.edu/sites/default/files/170.mat"),
        ("171.mat", "https://engineering.case.edu/sites/default/files/171.mat"),
        ("172.mat", "https://engineering.case.edu/sites/default/files/172.mat"),
    ],
    # Outer Race Fault — 0.007 inch (centered @ 6 o'clock)
    "outer_race_007": [
        ("130.mat", "https://engineering.case.edu/sites/default/files/130.mat"),
        ("131.mat", "https://engineering.case.edu/sites/default/files/131.mat"),
        ("132.mat", "https://engineering.case.edu/sites/default/files/132.mat"),
        ("133.mat", "https://engineering.case.edu/sites/default/files/133.mat"),
    ],
    # Ball Fault — 0.007 inch
    "ball_007": [
        ("118.mat", "https://engineering.case.edu/sites/default/files/118.mat"),
        ("119.mat", "https://engineering.case.edu/sites/default/files/119.mat"),
        ("120.mat", "https://engineering.case.edu/sites/default/files/120.mat"),
        ("121.mat", "https://engineering.case.edu/sites/default/files/121.mat"),
    ],
}

# Class mapping for labels
CLASS_MAP = {
    "normal": 0,
    "inner_race_007": 1,
    "inner_race_014": 1,
    "outer_race_007": 2,
    "ball_007": 3,
}


def _progress_hook(pbar):
    """Hook for tqdm progress bar during urllib download."""
    last_b = [0]

    def update(b=1, bsize=1, tsize=None):
        if tsize is not None:
            pbar.total = tsize
        pbar.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update


def download_file(url: str, dest: Path) -> bool:
    """Download a single .mat file with progress bar."""
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return True
    try:
        with tqdm(unit="B", unit_scale=True, miniters=1, desc=f"  ↓ {dest.name}") as t:
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook(t))
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {dest.name}: {e}")
        return False


def load_mat_signal(mat_path: Path) -> np.ndarray:
    """
    Load vibration signal from CWRU .mat file.
    Returns drive-end 12kHz accelerometer signal.
    """
    mat = scipy.io.loadmat(str(mat_path))
    # Find the drive-end 12kHz key (format varies by file)
    for key in mat:
        if "DE_time" in key and "12" in key:
            return mat[key].squeeze()
    # Fallback: find any DE_time key
    for key in mat:
        if "DE_time" in key:
            return mat[key].squeeze()
    raise KeyError(f"Could not find DE_time signal in {mat_path.name}")


def download_all(output_dir: str = "data/raw") -> None:
    """Download all CWRU files and organize by fault class."""
    base = Path(output_dir)

    print("\n🔩 CWRU Bearing Dataset Downloader")
    print("=" * 45)

    total_ok = 0
    for fault_type, files in CWRU_FILES.items():
        folder = base / fault_type
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 {fault_type.upper()}")
        for filename, url in files:
            dest = folder / filename
            ok = download_file(url, dest)
            if ok:
                total_ok += 1

    print(f"\n✅ Done! {total_ok} files ready in {base}/")


def build_numpy_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    window_size: int = 2048,
    overlap: float = 0.5,
) -> None:
    """
    Convert .mat files into windowed numpy arrays.

    Args:
        raw_dir: Directory with downloaded .mat files
        output_dir: Where to save X.npy and y.npy
        window_size: Number of samples per window
        overlap: Fraction overlap between consecutive windows
    """
    raw = Path(raw_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    step = int(window_size * (1 - overlap))
    X_list, y_list = [], []

    print("\n🔧 Building windowed dataset...")
    for fault_type, label in CLASS_MAP.items():
        folder = raw / fault_type
        if not folder.exists():
            print(f"  ⚠ Skipping {fault_type} (not downloaded)")
            continue

        n_windows = 0
        for mat_file in sorted(folder.glob("*.mat")):
            try:
                signal = load_mat_signal(mat_file)
                # Sliding window segmentation
                starts = range(0, len(signal) - window_size, step)
                for s in starts:
                    window = signal[s : s + window_size]
                    X_list.append(window.astype(np.float32))
                    y_list.append(label)
                    n_windows += 1
            except Exception as e:
                print(f"  ⚠ Error reading {mat_file.name}: {e}")

        print(f"  ✓ {fault_type}: {n_windows} windows (label={label})")

    if not X_list:
        print("❌ No data found. Run download_data.py first.")
        return

    X = np.array(X_list)   # shape: (N, window_size)
    y = np.array(y_list)   # shape: (N,)

    np.save(out / "X_raw.npy", X)
    np.save(out / "y.npy", y)

    print(f"\n✅ Dataset saved: X={X.shape}, y={y.shape}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CWRU Bearing Dataset")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--process", action="store_true", help="Also build numpy dataset")
    args = parser.parse_args()

    download_all(args.output)
    if args.process:
        build_numpy_dataset(args.output)
