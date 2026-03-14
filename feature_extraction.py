"""
feature_extraction.py
----------------------
Extracts 52 handcrafted features from vibration signal windows across three domains:
  - Time Domain     (18 features)
  - Frequency Domain(20 features)
  - Envelope Analysis (14 features)

Also computes bearing defect characteristic frequencies from geometry.

Usage:
    from src.features.feature_extraction import FeatureExtractor
    extractor = FeatureExtractor(fs=12000)
    features = extractor.extract(signal_window)   # → dict of 52 features
    X = extractor.transform(X_raw)                # → np.ndarray (N, 52)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq


# ─── Bearing Geometry (CWRU 6205-2RS JEM SKF) ────────────────────────────────

class BearingGeometry:
    """
    Computes characteristic fault frequencies from bearing geometry.

    Formulas:
        BPFI = (N/2) * (1 + d/D * cos(α)) * fr
        BPFO = (N/2) * (1 - d/D * cos(α)) * fr
        BSF  = (D/2d) * (1 - (d/D * cos(α))²) * fr
        FTF  = (1/2)  * (1 - d/D * cos(α)) * fr

    Args:
        n_balls: Number of rolling elements
        d: Ball diameter (inches)
        D: Pitch circle diameter (inches)
        alpha: Contact angle (degrees)
        rpm: Shaft rotation speed
    """

    def __init__(
        self,
        n_balls: int = 9,
        d: float = 0.3126,
        D: float = 1.537,
        alpha: float = 0.0,
        rpm: float = 1797,
    ):
        self.n_balls = n_balls
        self.d = d
        self.D = D
        self.alpha = np.radians(alpha)
        self.fr = rpm / 60.0  # shaft frequency in Hz

        # Precompute ratio
        self._ratio = (d / D) * np.cos(self.alpha)

    @property
    def bpfi(self) -> float:
        """Ball Pass Frequency Inner Race"""
        return (self.n_balls / 2) * (1 + self._ratio) * self.fr

    @property
    def bpfo(self) -> float:
        """Ball Pass Frequency Outer Race"""
        return (self.n_balls / 2) * (1 - self._ratio) * self.fr

    @property
    def bsf(self) -> float:
        """Ball Spin Frequency"""
        return (self.D / (2 * self.d)) * (1 - self._ratio**2) * self.fr

    @property
    def ftf(self) -> float:
        """Fundamental Train Frequency (cage)"""
        return 0.5 * (1 - self._ratio) * self.fr

    def summary(self) -> dict:
        return {
            "shaft_fr_Hz": round(self.fr, 2),
            "BPFI_Hz": round(self.bpfi, 2),
            "BPFO_Hz": round(self.bpfo, 2),
            "BSF_Hz": round(self.bsf, 2),
            "FTF_Hz": round(self.ftf, 2),
        }


# ─── Feature Extractor ───────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts 52 signal features from a 1D vibration window.

    Args:
        fs: Sampling frequency in Hz (default: 12000)
        bearing: BearingGeometry instance (uses CWRU defaults if None)
        n_fft_bins: Number of FFT frequency bins to use
    """

    FEATURE_NAMES: list[str] = []  # populated in __init__

    def __init__(
        self,
        fs: int = 12000,
        bearing: BearingGeometry | None = None,
        n_fft_bins: int = 512,
    ):
        self.fs = fs
        self.bearing = bearing or BearingGeometry()
        self.n_fft_bins = n_fft_bins
        self._build_feature_names()

    def _build_feature_names(self) -> None:
        time_names = [
            "mean", "std", "variance", "rms", "peak", "peak_to_peak",
            "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
            "kurtosis", "skewness", "energy", "log_energy",
            "zero_crossing_rate", "mean_abs", "abs_max", "l2_norm",
        ]
        freq_names = [
            "spectral_centroid", "spectral_bandwidth", "spectral_rolloff_85",
            "spectral_flatness", "dominant_freq",
            "power_band_0_500", "power_band_500_2000", "power_band_2000_4000", "power_band_4000_6000",
            "bpfi_energy", "bpfi_1h", "bpfi_2h", "bpfi_3h",
            "bpfo_energy", "bpfo_1h", "bpfo_2h", "bpfo_3h",
            "bsf_energy", "ftf_energy",
            "total_harmonic_distortion",
        ]
        env_names = [
            "env_rms", "env_kurtosis", "env_skewness", "env_crest_factor",
            "env_bpfi_energy", "env_bpfo_energy", "env_bsf_energy", "env_ftf_energy",
            "env_spectral_centroid", "env_peak_freq",
            "hilbert_mean_inst_freq", "hilbert_std_inst_freq",
            "env_entropy", "modulation_index",
        ]
        self.FEATURE_NAMES = time_names + freq_names + env_names
        assert len(self.FEATURE_NAMES) == 52, f"Expected 52 features, got {len(self.FEATURE_NAMES)}"

    # ── Time Domain ──────────────────────────────────────────────────────────

    def _time_features(self, x: np.ndarray) -> np.ndarray:
        N = len(x)
        abs_x = np.abs(x)
        rms = np.sqrt(np.mean(x**2))
        peak = np.max(abs_x)
        mean_abs = np.mean(abs_x)
        std = np.std(x)
        ptp = np.ptp(x)

        # Guard against zero
        eps = 1e-10
        crest_factor = peak / (rms + eps)
        shape_factor = rms / (mean_abs + eps)
        impulse_factor = peak / (mean_abs + eps)
        clearance_factor = peak / (np.mean(np.sqrt(abs_x + eps)) ** 2)

        zc = np.sum(np.diff(np.sign(x)) != 0) / N

        return np.array([
            np.mean(x),
            std,
            np.var(x),
            rms,
            peak,
            ptp,
            crest_factor,
            shape_factor,
            impulse_factor,
            clearance_factor,
            float(kurtosis(x)),
            float(skew(x)),
            np.sum(x**2),
            np.sum(np.log(x**2 + eps)),
            zc,
            mean_abs,
            np.max(np.abs(x)),
            np.linalg.norm(x),
        ], dtype=np.float32)

    # ── Frequency Domain ─────────────────────────────────────────────────────

    def _freq_features(self, x: np.ndarray) -> np.ndarray:
        N = len(x)
        freqs = rfftfreq(N, d=1.0 / self.fs)
        fft_mag = np.abs(rfft(x)) / N

        eps = 1e-10
        power = fft_mag**2
        total_power = np.sum(power) + eps

        # Spectral descriptors
        centroid = np.sum(freqs * power) / total_power
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power)

        cumpower = np.cumsum(power)
        rolloff_idx = np.searchsorted(cumpower, 0.85 * cumpower[-1])
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

        flatness = np.exp(np.mean(np.log(power + eps))) / (np.mean(power) + eps)
        dom_freq = freqs[np.argmax(fft_mag)]

        # Band powers
        def band_power(f_low, f_high):
            mask = (freqs >= f_low) & (freqs < f_high)
            return np.sum(power[mask]) / total_power

        b0 = band_power(0, 500)
        b1 = band_power(500, 2000)
        b2 = band_power(2000, 4000)
        b3 = band_power(4000, 6000)

        # Fault frequency energies (± 5Hz window)
        def fault_energy(fc, n_harmonics=3):
            energies = []
            for h in range(1, n_harmonics + 1):
                f_center = fc * h
                mask = np.abs(freqs - f_center) <= 5.0
                energies.append(np.sum(power[mask]))
            return energies

        bpfi_e = fault_energy(self.bearing.bpfi)
        bpfo_e = fault_energy(self.bearing.bpfo)
        bsf_e = fault_energy(self.bearing.bsf, n_harmonics=1)
        ftf_e = fault_energy(self.bearing.ftf, n_harmonics=1)

        # Total harmonic distortion
        fund_power = np.max(power)
        harmonic_power = np.sum(power) - fund_power
        thd = np.sqrt(harmonic_power / (fund_power + eps))

        return np.array([
            centroid, bandwidth, rolloff, flatness, dom_freq,
            b0, b1, b2, b3,
            sum(bpfi_e), bpfi_e[0], bpfi_e[1], bpfi_e[2],
            sum(bpfo_e), bpfo_e[0], bpfo_e[1], bpfo_e[2],
            sum(bsf_e), sum(ftf_e),
            thd,
        ], dtype=np.float32)

    # ── Envelope Analysis ────────────────────────────────────────────────────

    def _envelope_features(self, x: np.ndarray) -> np.ndarray:
        analytic = hilbert(x)
        envelope = np.abs(analytic)
        eps = 1e-10

        # Instantaneous frequency
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(inst_phase) / (2.0 * np.pi / self.fs)

        env_rms = np.sqrt(np.mean(envelope**2))
        env_peak = np.max(envelope)
        env_crest = env_peak / (env_rms + eps)

        # Envelope spectrum
        N = len(envelope)
        e_freqs = rfftfreq(N, d=1.0 / self.fs)
        e_mag = np.abs(rfft(envelope)) / N
        e_power = e_mag**2
        e_total = np.sum(e_power) + eps

        # Fault energies in envelope spectrum
        def env_fault_energy(fc):
            mask = np.abs(e_freqs - fc) <= 3.0
            return np.sum(e_power[mask])

        # Spectral entropy
        p_norm = e_power / (e_total + eps)
        entropy = -np.sum(p_norm * np.log(p_norm + eps))

        # Modulation index
        mod_idx = (np.max(envelope) - np.min(envelope)) / (np.mean(envelope) + eps)

        e_centroid = np.sum(e_freqs * e_power) / e_total
        e_peak_freq = e_freqs[np.argmax(e_mag)]

        return np.array([
            env_rms,
            float(kurtosis(envelope)),
            float(skew(envelope)),
            env_crest,
            env_fault_energy(self.bearing.bpfi),
            env_fault_energy(self.bearing.bpfo),
            env_fault_energy(self.bearing.bsf),
            env_fault_energy(self.bearing.ftf),
            e_centroid,
            e_peak_freq,
            np.mean(inst_freq),
            np.std(inst_freq),
            entropy,
            mod_idx,
        ], dtype=np.float32)

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(self, x: np.ndarray) -> dict[str, float]:
        """
        Extract all 52 features from a 1D signal window.

        Args:
            x: Signal array of shape (window_size,)

        Returns:
            Dictionary mapping feature names to values
        """
        x = x.astype(np.float64)
        features = np.concatenate([
            self._time_features(x),
            self._freq_features(x),
            self._envelope_features(x),
        ])
        return dict(zip(self.FEATURE_NAMES, features))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from all windows.

        Args:
            X: Raw signal windows of shape (N, window_size)

        Returns:
            Feature matrix of shape (N, 52)
        """
        print(f"Extracting features from {len(X)} windows...")
        result = np.zeros((len(X), len(self.FEATURE_NAMES)), dtype=np.float32)
        for i, window in enumerate(X):
            feat = self.extract(window)
            result[i] = list(feat.values())
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(X)} done")
        print("✅ Feature extraction complete.")
        return result


if __name__ == "__main__":
    # Quick sanity check
    geometry = BearingGeometry()
    print("Bearing Characteristic Frequencies:")
    for k, v in geometry.summary().items():
        print(f"  {k}: {v} Hz")

    extractor = FeatureExtractor(fs=12000, bearing=geometry)
    test_signal = np.random.randn(2048).astype(np.float32)
    feats = extractor.extract(test_signal)
    print(f"\nExtracted {len(feats)} features from test signal")
    print("Sample features:", {k: round(float(v), 4) for k, v in list(feats.items())[:5]})
