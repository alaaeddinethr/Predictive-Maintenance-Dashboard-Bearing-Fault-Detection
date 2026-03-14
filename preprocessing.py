"""
preprocessing.py
----------------
Signal preprocessing utilities:
  - Bandpass filtering (remove DC + anti-alias)
  - Normalization (zero-mean, unit-variance)
  - Sliding window segmentation with overlap
  - Train/test/val stratified split

These steps are applied BEFORE feature extraction.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    lowcut: float = 10.0,
    highcut: float = 5000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.

    Removes DC offset (low-freq drift) and high-frequency aliasing noise.
    Typical range for bearing diagnostics: 10 Hz – 5 kHz.

    Args:
        signal: 1D vibration signal
        fs: Sampling frequency (Hz)
        lowcut: Lower cutoff frequency (Hz)
        highcut: Upper cutoff frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal (same shape as input)
    """
    nyq = fs / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def normalize(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize a signal window.

    Args:
        signal: 1D or 2D array. If 2D, normalizes each row independently.
        method: 'zscore' (zero mean, unit variance) or 'minmax' ([-1, 1] range)

    Returns:
        Normalized array (same shape)
    """
    eps = 1e-10

    if signal.ndim == 1:
        if method == "zscore":
            return ((signal - signal.mean()) / (signal.std() + eps)).astype(np.float32)
        elif method == "minmax":
            lo, hi = signal.min(), signal.max()
            return (2 * (signal - lo) / (hi - lo + eps) - 1).astype(np.float32)
        else:
            raise ValueError(f"Unknown method: {method}")

    elif signal.ndim == 2:
        if method == "zscore":
            mean = signal.mean(axis=1, keepdims=True)
            std  = signal.std(axis=1, keepdims=True) + eps
            return ((signal - mean) / std).astype(np.float32)
        elif method == "minmax":
            lo   = signal.min(axis=1, keepdims=True)
            hi   = signal.max(axis=1, keepdims=True)
            return (2 * (signal - lo) / (hi - lo + eps) - 1).astype(np.float32)

    raise ValueError(f"signal must be 1D or 2D, got {signal.ndim}D")


def sliding_window(
    signal: np.ndarray,
    window_size: int,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Segment a 1D signal into overlapping windows.

    Args:
        signal: 1D vibration signal of length N
        window_size: Number of samples per window
        overlap: Fraction of overlap between consecutive windows [0, 1)

    Returns:
        2D array of shape (num_windows, window_size)
    """
    step = int(window_size * (1.0 - overlap))
    if step <= 0:
        raise ValueError(f"Overlap {overlap} too large — step would be ≤ 0")

    starts = range(0, len(signal) - window_size + 1, step)
    windows = np.array([signal[s : s + window_size] for s in starts], dtype=np.float32)
    return windows


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, ...]:
    """
    Stratified split into train / val / test sets.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=random_state
    )
    val_ratio_adj = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adj, stratify=y_trainval, random_state=random_state
    )

    print(f"Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(12000).astype(np.float32)

    filtered  = bandpass_filter(signal, fs=12000)
    normalized = normalize(filtered)
    windows   = sliding_window(normalized, window_size=2048, overlap=0.5)

    print(f"Original signal: {signal.shape}")
    print(f"After filtering:  {filtered.shape}")
    print(f"After sliding window (2048, 50% overlap): {windows.shape}")
    print(f"Windows stats — mean: {windows.mean():.4f}, std: {windows.std():.4f}")
