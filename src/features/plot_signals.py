"""Figure for the README: raw vibration and envelope spectrum of each bearing condition.

The envelope spectrum is where faults become visible: an inner-race defect
shows a peak at the ball-pass frequency of the inner race (BPFI), an
outer-race defect at BPFO. These are the physical signatures the features
and the PINN are built around.

    python src/features/plot_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.download_data import load_mat_signal  # noqa: E402
from src.features.feature_extraction import BearingGeometry  # noqa: E402

FS = 12000
CASES = [("normal/97.mat", "Normal"), ("inner_race_007/105.mat", "Inner race fault"),
         ("outer_race_007/130.mat", "Outer race fault"), ("ball_007/118.mat", "Ball fault")]


def envelope_spectrum(x: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    env = np.abs(hilbert(x - x.mean()))
    env -= env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
    return np.fft.rfftfreq(len(env), 1 / fs), spec


def main(raw_dir: str = "data/raw", out: str = "assets/envelope_spectra.png") -> None:
    geo = BearingGeometry().summary()
    markers = {"BPFI": geo["BPFI_Hz"], "BPFO": geo["BPFO_Hz"], "BSF": geo["BSF_Hz"]}
    fig, axes = plt.subplots(len(CASES), 2, figsize=(11, 8), gridspec_kw={"width_ratios": [1, 1.4]})
    for (path, title), (ax_t, ax_f) in zip(CASES, axes):
        sig = load_mat_signal(ROOT / raw_dir / path)
        seg = sig[: FS // 10]
        ax_t.plot(np.arange(len(seg)) / FS * 1000, seg, lw=0.6, color="#24292f")
        ax_t.set_ylabel(title, fontsize=9)
        f, s = envelope_spectrum(sig[: FS * 2])
        keep = f <= 400
        ax_f.plot(f[keep], s[keep], lw=0.8, color="#1f6feb")
        for name, fc in markers.items():
            ax_f.axvline(fc, ls="--", lw=0.8, color="#cf222e" if name == "BPFI" else "#9a6700" if name == "BPFO" else "#8250df")
        for ax in (ax_t, ax_f):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
    axes[0, 0].set_title("Raw vibration (first 100 ms)", fontsize=10)
    axes[0, 1].set_title("Envelope spectrum  —  dashed: BPFI (red), BPFO (amber), BSF (purple)", fontsize=10)
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    (ROOT / out).parent.mkdir(exist_ok=True)
    fig.savefig(ROOT / out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
