"""
evaluate.py
-----------
Post-training evaluation suite:
  - Detailed classification report per class
  - Confusion matrix with normalized view
  - SHAP feature importance (global + per-class)
  - Learning curves
  - Fault frequency feature contribution analysis
  - Model comparison radar chart

Usage:
    python src/models/evaluate.py --model xgboost
    python src/models/evaluate.py --model all --output assets/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.feature_extraction import FeatureExtractor, BearingGeometry

CLASS_NAMES  = ["Normal", "Inner Race", "Outer Race", "Ball Fault"]
CLASS_COLORS = ["#3fb950", "#ff7b72", "#e3b341", "#d2a8ff"]

# ── Style ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "text.color":       "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
})


# ── Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix_full(y_true, y_pred, model_name: str, output_dir: str) -> None:
    """Plots both raw-count and row-normalized confusion matrices side by side."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, fmt, title in [
        (ax1, cm,      "d",    "Raw Counts"),
        (ax2, cm_norm, ".2f",  "Row-Normalized"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5, linecolor="#30363d",
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, pad=12)
        ax.set_xlabel("Predicted", labelpad=8)
        ax.set_ylabel("True", labelpad=8)
        ax.tick_params(axis="x", rotation=20)

    acc = accuracy_score(y_true, y_pred)
    fig.suptitle(
        f"Confusion Matrix — {model_name}  |  Accuracy: {acc*100:.2f}%",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    path = Path(output_dir) / f"cm_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix saved: {path}")


# ── ROC Curves ────────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_proba, model_name: str, output_dir: str) -> None:
    """One-vs-Rest ROC curves for all 4 classes."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(7, 6))

    for cls, (color, name) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_bin[:, cls], y_proba[:, cls])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    path = Path(output_dir) / f"roc_{model_name.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 ROC curves saved: {path}")


# ── SHAP Analysis ─────────────────────────────────────────────────────────

def plot_shap_importance(model, X_test: np.ndarray, feature_names: list, output_dir: str) -> None:
    """
    SHAP summary plot: global feature importance across all classes.
    Only applicable to tree-based models (XGBoost, RF).
    """
    try:
        import shap
    except ImportError:
        print("  ⚠ shap not installed. Skipping SHAP analysis.")
        return

    try:
        clf = model.named_steps["clf"]
        scaler = model.named_steps["scaler"]
        X_scaled = scaler.transform(X_test)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_scaled)

        # For multi-class, shap_values is (n_classes, n_samples, n_features)
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values)

        global_importance = mean_abs_shap.mean(axis=0)

        # Top 20 features
        top_idx = np.argsort(global_importance)[-20:]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = global_importance[top_idx]

        fig, ax = plt.subplots(figsize=(9, 7))
        bars = ax.barh(top_names, top_vals, color="#58a6ff", edgecolor="#30363d")

        # Color bars by feature domain
        domain_colors = {
            "env_": "#d2a8ff",
            "bpfi": "#ff7b72",
            "bpfo": "#e3b341",
            "bsf":  "#79c0ff",
        }
        for bar, name in zip(bars, top_names):
            for prefix, color in domain_colors.items():
                if prefix in name:
                    bar.set_color(color)
                    break

        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Global Feature Importance (SHAP)", pad=12)
        ax.grid(axis="x", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color="#58a6ff", label="Time Domain"),
            Patch(color="#ff7b72", label="BPFI (Inner Race freq)"),
            Patch(color="#e3b341", label="BPFO (Outer Race freq)"),
            Patch(color="#79c0ff", label="BSF (Ball freq)"),
            Patch(color="#d2a8ff", label="Envelope"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

        plt.tight_layout()
        path = Path(output_dir) / "shap_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  🧠 SHAP plot saved: {path}")

    except Exception as e:
        print(f"  ⚠ SHAP failed: {e}")


# ── Feature Importance (built-in) ─────────────────────────────────────────

def plot_builtin_importance(model, feature_names: list, top_n: int, output_dir: str, model_name: str) -> None:
    """Built-in feature importances for tree models (Gini / gain)."""
    clf = model.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return

    imp = clf.feature_importances_
    top_idx = np.argsort(imp)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in top_idx],
        imp[top_idx],
        color="#3fb950", edgecolor="#30363d"
    )
    ax.set_xlabel("Importance (Gini / Gain)")
    ax.set_title(f"Feature Importances — {model_name}", pad=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Feature importance saved: {path}")


# ── Fault Frequency Contribution Analysis ─────────────────────────────────

def plot_fault_freq_contribution(
    X_feat: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    output_dir: str,
) -> None:
    """
    For each fault class, plot the mean energy at its characteristic frequency
    versus other fault frequencies. This provides physical interpretability.
    """
    freq_features = {
        "BPFI Energy": "bpfi_energy",
        "BPFO Energy": "bpfo_energy",
        "BSF Energy":  "bsf_energy",
        "FTF Energy":  "ftf_energy",
    }

    # Get indices in feature vector
    idx_map = {v: feature_names.index(v) for v in freq_features.values() if v in feature_names}
    if not idx_map:
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)

    for cls, (ax, color, cls_name) in enumerate(zip(axes, CLASS_COLORS, CLASS_NAMES)):
        means, labels = [], []
        for display_name, feat_name in freq_features.items():
            if feat_name in idx_map:
                fi = idx_map[feat_name]
                class_mean = X_feat[y == cls, fi].mean()
                means.append(class_mean)
                labels.append(display_name.replace(" Energy", ""))

        bars = ax.bar(labels, means, color=color, alpha=0.85, edgecolor="#30363d")
        ax.set_title(cls_name, color=color, fontsize=10)
        ax.set_xlabel("Fault Freq.", fontsize=8)
        ax.set_ylabel("Mean Energy" if cls == 0 else "", fontsize=8)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Mean Fault Frequency Energy by Class", fontsize=12, y=1.02)
    plt.tight_layout()
    path = Path(output_dir) / "fault_freq_energy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Fault frequency contribution saved: {path}")


# ── Summary Report ─────────────────────────────────────────────────────────

def print_classification_report(y_true, y_pred, model_name: str) -> None:
    print(f"\n{'═'*55}")
    print(f"  {model_name} — Classification Report")
    print(f"{'═'*55}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate bearing fault detection models")
    parser.add_argument("--model", choices=["xgboost", "rf", "all"], default="xgboost")
    parser.add_argument("--output", default="assets/eval")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    feat_path = Path(args.data_dir) / "X_features.npy"
    y_path    = Path(args.data_dir) / "y.npy"

    if not feat_path.exists():
        print("❌ No feature data found. Run train.py first to extract features.")
        return

    X = np.load(feat_path)
    y = np.load(y_path)

    extractor = FeatureExtractor(fs=12000, bearing=BearingGeometry())
    feature_names = extractor.FEATURE_NAMES

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model_files = {
        "xgboost": "models/xgboost.pkl",
        "rf":      "models/random_forest.pkl",
    }
    to_eval = list(model_files.keys()) if args.model == "all" else [args.model]

    for model_key in to_eval:
        path = model_files[model_key]
        if not Path(path).exists():
            print(f"⚠ Model not found: {path}. Run train.py --save first.")
            continue

        model = joblib.load(path)
        model_name = "XGBoost" if model_key == "xgboost" else "Random Forest"

        print(f"\n🔍 Evaluating {model_name}...")
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        print_classification_report(y_test, y_pred, model_name)
        plot_confusion_matrix_full(y_test, y_pred, model_name, output_dir)
        plot_roc_curves(y_test, y_proba, model_name, output_dir)
        plot_builtin_importance(model, feature_names, args.top_n, output_dir, model_name)
        plot_shap_importance(model, X_test, feature_names, output_dir)

    # Physical analysis (model-agnostic)
    plot_fault_freq_contribution(X, y, feature_names, output_dir)

    print(f"\n✅ All evaluation artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()
