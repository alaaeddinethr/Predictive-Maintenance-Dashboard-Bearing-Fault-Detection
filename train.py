"""
train.py
--------
Trains and evaluates all models (XGBoost, Random Forest, PINN) on extracted features.

Usage:
    python src/models/train.py                         # Train all models
    python src/models/train.py --model xgboost --save  # Train XGBoost only
    python src/models/train.py --model pinn --epochs 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.feature_extraction import FeatureExtractor, BearingGeometry
from src.models.pinn import BearingPINN, PhysicsConstraint, PINNTrainer

CLASS_NAMES = ["Normal", "Inner Race", "Outer Race", "Ball Fault"]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(processed_dir: str = "data/processed") -> tuple[np.ndarray, np.ndarray]:
    """Load pre-extracted features or raw windows."""
    feat_path = Path(processed_dir) / "X_features.npy"
    raw_path = Path(processed_dir) / "X_raw.npy"
    y_path = Path(processed_dir) / "y.npy"

    if not y_path.exists():
        raise FileNotFoundError("No processed data found. Run download_data.py first.")

    y = np.load(y_path)

    if feat_path.exists():
        print(f"✓ Loading pre-extracted features: {feat_path}")
        X = np.load(feat_path)
    elif raw_path.exists():
        print("⚙ Extracting features from raw signals...")
        X_raw = np.load(raw_path)
        extractor = FeatureExtractor(fs=12000, bearing=BearingGeometry())
        X = extractor.transform(X_raw)
        np.save(feat_path, X)
        print(f"✓ Features saved to {feat_path}")
    else:
        raise FileNotFoundError("No data found in processed_dir.")

    print(f"Dataset: X={X.shape}, y={y.shape}")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def train_xgboost(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """Train XGBoost classifier with cross-validation."""
    print("\n🌲 Training XGBoost...")
    cfg = config["models"]["xgboost"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            random_state=cfg["random_state"],
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
        )),
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"  CV F1 (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = _evaluate(y_test, y_pred, "XGBoost")
    return {"model": model, "metrics": metrics}


def train_random_forest(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """Train Random Forest classifier."""
    print("\n🌳 Training Random Forest...")
    cfg = config["models"]["random_forest"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"  CV F1 (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = _evaluate(y_test, y_pred, "Random Forest")
    return {"model": model, "metrics": metrics}


def train_pinn(X_train, y_train, X_test, y_test, config: dict, epochs: int = None) -> dict:
    """Train Physics-Informed Neural Network."""
    print("\n⚛ Training PINN...")
    cfg = config["models"]["pinn"]

    # Normalize features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)

    model = BearingPINN(
        input_dim=X_tr.shape[1],
        hidden_layers=cfg["hidden_layers"],
        n_classes=4,
    )
    physics = PhysicsConstraint(lambda_physics=cfg["physics_weight"])
    trainer = PINNTrainer(model, physics, lr=cfg["learning_rate"])

    trainer.train(
        X_tr, y_train, X_te, y_test,
        epochs=epochs or cfg["epochs"],
        batch_size=cfg["batch_size"],
    )

    y_pred, _ = trainer.predict(X_te)
    metrics = _evaluate(y_test, y_pred, "PINN")

    return {"model": trainer, "scaler": scaler, "metrics": metrics}


def _evaluate(y_true, y_pred, model_name: str) -> dict:
    """Compute and print evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'─'*45}")
    print(f"  {model_name} Results")
    print(f"{'─'*45}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")

    return {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm}


def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: str = "assets") -> None:
    """Save confusion matrix heatmap."""
    Path(output_dir).mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = f"{output_dir}/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved: {path}")


def save_models(results: dict, output_dir: str = "models") -> None:
    """Persist trained models."""
    Path(output_dir).mkdir(exist_ok=True)
    for name, result in results.items():
        model = result["model"]
        if name == "pinn":
            model.save(f"{output_dir}/pinn.pt")
        else:
            joblib.dump(model, f"{output_dir}/{name.replace(' ', '_')}.pkl")
            print(f"  💾 Saved {name} to {output_dir}/")


def print_summary(results: dict) -> None:
    """Print a comparison table of all models."""
    print(f"\n{'═'*50}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'═'*50}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1 (macro)':>12}")
    print(f"  {'─'*20} {'─'*10} {'─'*12}")
    for name, result in results.items():
        m = result["metrics"]
        print(f"  {name:<20} {m['accuracy']*100:>9.2f}% {m['f1_macro']:>12.4f}")
    print(f"{'═'*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Train bearing fault detection models")
    parser.add_argument("--model", choices=["xgboost", "rf", "pinn", "all"], default="all")
    parser.add_argument("--save", action="store_true", help="Save trained models")
    parser.add_argument("--epochs", type=int, default=None, help="PINN epochs override")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    config = load_config(args.config)
    X, y = load_data(args.data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - config["data"]["train_ratio"],
        stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    results = {}
    to_train = ["xgboost", "rf", "pinn"] if args.model == "all" else [args.model]

    if "xgboost" in to_train:
        res = train_xgboost(X_train, y_train, X_test, y_test, config)
        results["XGBoost"] = res
        plot_confusion_matrix(res["metrics"]["confusion_matrix"], "XGBoost")

    if "rf" in to_train:
        res = train_random_forest(X_train, y_train, X_test, y_test, config)
        results["Random Forest"] = res
        plot_confusion_matrix(res["metrics"]["confusion_matrix"], "Random Forest")

    if "pinn" in to_train:
        res = train_pinn(X_train, y_train, X_test, y_test, config, args.epochs)
        results["PINN"] = res
        plot_confusion_matrix(res["metrics"]["confusion_matrix"], "PINN")

    print_summary(results)

    if args.save:
        save_models(results)


if __name__ == "__main__":
    main()
