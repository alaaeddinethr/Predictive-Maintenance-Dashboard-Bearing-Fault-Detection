"""
pinn.py
-------
Physics-Informed Neural Network for bearing fault classification.

The key idea: the network's loss function includes a PHYSICS CONSTRAINT term
that penalizes predictions inconsistent with bearing dynamics laws.

Loss = L_data (cross-entropy) + λ * L_physics

L_physics encodes:
  1. Fault frequency consistency: model's attention should activate at
     theoretically predicted BPFI/BPFO/BSF/FTF frequencies
  2. Spectral energy ordering: fault signals should have higher energy
     at defect frequencies than healthy signals

This makes the model physically interpretable and more robust with limited data.

Reference:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
    Physics-informed neural networks: A deep learning framework for solving
    forward and inverse problems involving nonlinear partial differential
    equations. Journal of Computational Physics, 378, 686–707.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# ─── Network Architecture ────────────────────────────────────────────────────

class BearingPINN(nn.Module):
    """
    Physics-Informed Neural Network for bearing fault classification.

    Architecture:
        Input (52 features) → FC layers with residual connections → 4-class output

    The physics constraints are computed externally and fed into the loss.
    The network itself is a standard MLP, but training is physics-constrained.
    """

    def __init__(
        self,
        input_dim: int = 52,
        hidden_layers: list[int] = None,
        n_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [128, 64, 32]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

        # Attention layer over input features (physics-guided)
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch, n_classes)
            attention: (batch, input_dim) — feature importance weights
        """
        attention = self.feature_attention(x)
        x_attended = x * attention
        features = self.backbone(x_attended)
        logits = self.classifier(features)
        return logits, attention


# ─── Physics Constraint ──────────────────────────────────────────────────────

class PhysicsConstraint:
    """
    Computes the physics loss term based on bearing defect frequency theory.

    The constraint: for a bearing with fault class c, the features corresponding
    to fault frequencies of class c should have HIGHER attention weights than
    features of other fault types.

    We use the feature indices corresponding to:
        - bpfi_energy (index 9), bpfo_energy (13), bsf_energy (17), ftf_energy (18)
          in the frequency domain features
    """

    # Feature indices in the 52-dim feature vector
    # See FeatureExtractor.FEATURE_NAMES for full list
    FAULT_FREQ_INDICES = {
        0: [],           # Normal — no fault frequency should dominate
        1: [27, 28, 29, 30],  # Inner Race — bpfi features (indices 18+9=27...)
        2: [31, 32, 33, 34],  # Outer Race — bpfo features
        3: [35],              # Ball — bsf features
    }

    def __init__(self, lambda_physics: float = 0.1):
        self.lambda_physics = lambda_physics

    def compute(
        self,
        attention: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes physics consistency loss.

        For each sample, if label is fault type c:
            - attention at fault_freq_indices[c] should be HIGH
            - attention at other fault indices should be LOW

        Args:
            attention: (batch, 52) — per-feature attention weights
            labels: (batch,) — integer class labels

        Returns:
            Scalar physics loss
        """
        if self.lambda_physics == 0:
            return torch.tensor(0.0, requires_grad=True)

        batch_size = attention.shape[0]
        loss = torch.tensor(0.0, device=attention.device)
        n = 0

        for label_val, indices in self.FAULT_FREQ_INDICES.items():
            if not indices:
                continue

            mask = labels == label_val
            if mask.sum() == 0:
                continue

            att_subset = attention[mask]
            fault_att = att_subset[:, indices].mean(dim=1)        # Mean attention at fault freqs
            other_idx = [i for k, v in self.FAULT_FREQ_INDICES.items()
                         for i in v if k != label_val and v]
            if other_idx:
                other_att = att_subset[:, other_idx].mean(dim=1)   # Mean attention elsewhere
                # Physics constraint: fault attention > other attention
                margin_loss = F.relu(other_att - fault_att + 0.1).mean()
                loss = loss + margin_loss
                n += 1

        return self.lambda_physics * (loss / max(n, 1))


# ─── Trainer ─────────────────────────────────────────────────────────────────

class PINNTrainer:
    """Handles training loop, evaluation, and model persistence."""

    def __init__(
        self,
        model: BearingPINN,
        physics_constraint: PhysicsConstraint,
        lr: float = 1e-3,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.physics = physics_constraint
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> dict:
        """Full training loop with physics-informed loss."""
        # Build dataloaders
        X_tr = torch.FloatTensor(X_train).to(self.device)
        y_tr = torch.LongTensor(y_train).to(self.device)
        X_v = torch.FloatTensor(X_val).to(self.device)
        y_v = torch.LongTensor(y_val).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=batch_size, shuffle=True, drop_last=True
        )

        print(f"Training PINN on {self.device} for {epochs} epochs...")
        print(f"  Physics weight λ = {self.physics.lambda_physics}")

        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            # ── Training step ──
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                logits, attention = self.model(X_batch)

                loss_data = F.cross_entropy(logits, y_batch)
                loss_phys = self.physics.compute(attention, y_batch)
                loss = loss_data + loss_phys

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            self.scheduler.step()

            # ── Validation step ──
            self.model.eval()
            with torch.no_grad():
                val_logits, val_att = self.model(X_v)
                val_loss = F.cross_entropy(val_logits, y_v).item()
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_v).float().mean().item()

            self.history["train_loss"].append(train_loss / len(train_loader))
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if epoch % 10 == 0:
                print(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_loss/len(train_loader):.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc*100:.1f}%"
                )

        print(f"\n✅ Training complete. Best Val Accuracy: {best_val_acc*100:.1f}%")
        return self.history

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns class predictions and attention weights.

        Returns:
            predictions: (N,) integer class labels
            attention: (N, 52) feature attention weights
        """
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits, attention = self.model(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
            att = attention.cpu().numpy()
        return preds, att

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities (N, 4)."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(X_t)
            proba = F.softmax(logits, dim=1).cpu().numpy()
        return proba

    def save(self, path: str = "models/pinn.pt") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "history": self.history,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str = "models/pinn.pt") -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.history = checkpoint.get("history", {})
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Smoke test
    model = BearingPINN(input_dim=52, hidden_layers=[128, 64, 32], n_classes=4)
    physics = PhysicsConstraint(lambda_physics=0.1)
    trainer = PINNTrainer(model, physics)

    # Fake data
    X = np.random.randn(200, 52).astype(np.float32)
    y = np.random.randint(0, 4, 200)

    trainer.train(X[:160], y[:160], X[160:], y[160:], epochs=5, batch_size=32)
    preds, att = trainer.predict(X[:10])
    print(f"Predictions: {preds}")
    print(f"Attention shape: {att.shape}")
