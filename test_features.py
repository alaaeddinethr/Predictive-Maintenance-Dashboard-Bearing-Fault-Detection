"""
test_features.py
----------------
Unit tests for the feature extraction pipeline.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.feature_extraction import FeatureExtractor, BearingGeometry


class TestBearingGeometry:
    """Tests for bearing characteristic frequency calculations."""

    def setup_method(self):
        # CWRU standard parameters
        self.geo = BearingGeometry(n_balls=9, d=0.3126, D=1.537, alpha=0.0, rpm=1797)

    def test_shaft_frequency(self):
        """Shaft frequency = RPM / 60"""
        assert abs(self.geo.fr - 1797 / 60) < 0.01

    def test_bpfi_positive(self):
        assert self.geo.bpfi > 0

    def test_bpfo_positive(self):
        assert self.geo.bpfo > 0

    def test_bpfi_greater_than_bpfo(self):
        """BPFI > BPFO for standard ball bearings (more balls hit inner race per revolution)"""
        assert self.geo.bpfi > self.geo.bpfo

    def test_known_cwru_bpfi(self):
        """CWRU published BPFI ≈ 162.2 Hz at 1797 RPM"""
        assert abs(self.geo.bpfi - 162.2) < 2.0

    def test_known_cwru_bpfo(self):
        """CWRU published BPFO ≈ 107.4 Hz at 1797 RPM"""
        assert abs(self.geo.bpfo - 107.4) < 2.0

    def test_ftf_less_than_shaft(self):
        """Cage frequency (FTF) is always less than shaft frequency"""
        assert self.geo.ftf < self.geo.fr

    def test_summary_keys(self):
        summary = self.geo.summary()
        required_keys = ["shaft_fr_Hz", "BPFI_Hz", "BPFO_Hz", "BSF_Hz", "FTF_Hz"]
        for key in required_keys:
            assert key in summary


class TestFeatureExtractor:
    """Tests for the 52-feature extraction pipeline."""

    def setup_method(self):
        self.extractor = FeatureExtractor(fs=12000)
        self.window_size = 2048

    def _make_signal(self, kind="noise") -> np.ndarray:
        rng = np.random.default_rng(42)
        if kind == "noise":
            return rng.standard_normal(self.window_size).astype(np.float32)
        elif kind == "sine":
            t = np.arange(self.window_size) / 12000
            return np.sin(2 * np.pi * 100 * t).astype(np.float32)
        elif kind == "impulse":
            s = np.zeros(self.window_size, dtype=np.float32)
            s[::200] = 1.0  # periodic impulses
            return s

    def test_returns_52_features(self):
        signal = self._make_signal("noise")
        feats = self.extractor.extract(signal)
        assert len(feats) == 52

    def test_feature_names_correct(self):
        signal = self._make_signal("noise")
        feats = self.extractor.extract(signal)
        assert "rms" in feats
        assert "kurtosis" in feats
        assert "bpfi_energy" in feats
        assert "env_rms" in feats

    def test_all_values_finite(self):
        """No NaN or Inf in any feature."""
        for kind in ["noise", "sine", "impulse"]:
            signal = self._make_signal(kind)
            feats = self.extractor.extract(signal)
            for name, val in feats.items():
                assert np.isfinite(val), f"Non-finite value for {name} with {kind} signal"

    def test_rms_correct(self):
        """RMS of pure sine wave ≈ 1/√2"""
        signal = self._make_signal("sine")
        feats = self.extractor.extract(signal)
        assert abs(feats["rms"] - 1 / np.sqrt(2)) < 0.02

    def test_kurtosis_impulse_high(self):
        """Kurtosis should be high for impulsive signal"""
        impulse = self._make_signal("impulse")
        noise = self._make_signal("noise")
        feats_imp = self.extractor.extract(impulse)
        feats_noise = self.extractor.extract(noise)
        assert feats_imp["kurtosis"] > feats_noise["kurtosis"]

    def test_transform_batch(self):
        """transform() should handle batch input correctly."""
        X = np.random.randn(20, self.window_size).astype(np.float32)
        X_feat = self.extractor.transform(X)
        assert X_feat.shape == (20, 52)
        assert np.all(np.isfinite(X_feat))

    def test_consistent_output(self):
        """Same signal always produces same features."""
        signal = self._make_signal("noise")
        feats1 = self.extractor.extract(signal)
        feats2 = self.extractor.extract(signal)
        for key in feats1:
            assert abs(feats1[key] - feats2[key]) < 1e-5

    def test_crest_factor_pure_sine(self):
        """Crest factor of a pure sine = √2 ≈ 1.414"""
        signal = self._make_signal("sine")
        feats = self.extractor.extract(signal)
        assert abs(feats["crest_factor"] - np.sqrt(2)) < 0.1

    def test_zero_crossing_rate_sine(self):
        """Sine wave has predictable zero crossing rate."""
        signal = self._make_signal("sine")
        feats = self.extractor.extract(signal)
        # 100 Hz sine at 12kHz in 2048 samples:
        # duration = 2048/12000 ≈ 0.1707s → ~17 cycles → ~34 zero crossings
        # ZCR = 34 / 2048 ≈ 0.0166
        n_cycles = 100 * self.window_size / self.extractor.fs
        expected_zcr = (2 * n_cycles) / self.window_size
        assert abs(feats["zero_crossing_rate"] - expected_zcr) < 0.005


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
