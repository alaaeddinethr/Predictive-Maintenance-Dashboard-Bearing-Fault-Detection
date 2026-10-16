"""Leave-one-load-out evaluation: does the model generalise to an unseen operating condition?

A random split of overlapping windows puts near-identical windows from the
same recording in both train and test, so it reports ~100 % accuracy that
says nothing about real use. Here every recording is assigned to the motor
load it was measured at (0-3 HP); the model is trained on three loads and
tested on the fourth, which it has never seen.

    python src/models/evaluate_by_load.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.download_data import CLASS_MAP, CWRU_FILES, load_mat_signal  # noqa: E402
from src.features.feature_extraction import (  # noqa: E402
    BearingGeometry,
    FeatureExtractor,
)

WINDOW, STEP, FS = 2048, 1024, 12000


def build(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows + features, labelled with fault class and motor load.

    Within each fault folder the four CWRU files are ordered by load
    (0, 1, 2, 3 HP), which is how the download list is written.
    """
    extractor = FeatureExtractor(fs=FS, bearing=BearingGeometry())
    feats, labels, loads = [], [], []
    for fault, files in CWRU_FILES.items():
        for load, (name, _) in enumerate(files):
            sig = load_mat_signal(raw_dir / fault / name)
            wins = np.stack([sig[s : s + WINDOW] for s in range(0, len(sig) - WINDOW, STEP)]).astype(np.float32)
            feats.append(extractor.transform(wins))
            labels += [CLASS_MAP[fault]] * len(wins)
            loads += [load] * len(wins)
    return np.vstack(feats), np.array(labels), np.array(loads)


def models() -> dict:
    return {
        "XGBoost": Pipeline([("s", StandardScaler()), ("m", xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0))]),
        "Random Forest": Pipeline([("s", StandardScaler()), ("m", RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1))]),
    }


def main(raw_dir: str = "data/raw", out: str = "assets/load_generalisation.json") -> dict:
    X, y, load = build(ROOT / raw_dir)
    results: dict[str, dict] = {}
    for name, model in models().items():
        per_load = {}
        for held_out in range(4):
            tr, te = load != held_out, load == held_out
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            per_load[f"{held_out} HP"] = {
                "accuracy": round(float(accuracy_score(y[te], pred)), 4),
                "f1_macro": round(float(f1_score(y[te], pred, average="macro")), 4),
            }
        accs = [v["accuracy"] for v in per_load.values()]
        results[name] = {"per_held_out_load": per_load,
                         "mean_accuracy": round(float(np.mean(accs)), 4),
                         "min_accuracy": round(float(np.min(accs)), 4)}
        print(f"{name:14s} " + "  ".join(f"{k}: {v['accuracy']:.1%}" for k, v in per_load.items())
              + f"  | mean {np.mean(accs):.1%}")
    Path(ROOT / out).parent.mkdir(exist_ok=True)
    (ROOT / out).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
