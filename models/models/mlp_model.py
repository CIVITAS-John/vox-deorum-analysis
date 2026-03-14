#!/usr/bin/env python3
"""
Multi-Layer Perceptron (MLP) neural network for victory prediction.
Uses PyTorch with GPU support. Same _UtilityNet architecture as grouped MLP,
but with standard per-sample binary cross-entropy loss (no grouping).
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_predictor import BasePredictor
from .grouped_mlp_model import _UtilityNet


class MLPPredictor(BasePredictor):
    """
    PyTorch MLP for victory prediction with GPU support.
    Uses the same residual _UtilityNet architecture as GroupedMLPPredictor,
    but trains with standard binary cross-entropy on individual samples.
    """

    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = None
    REQUIRED_FEATURES = None
    DISABLE_RESAMPLING = True

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = [],
        random_state: int = 42,
        layer_sizes: tuple = (74,74,74,74),
        dropout: float = 0.14676450270006072,
        lr: float = 0.0016508487822090463,
        weight_decay: float = 0.0017063017923362206,
        epochs: int = 7,
        batch_size: int = 4096,
        device: Optional[str] = None,
    ):
        super().__init__(include_features, exclude_features, random_state)

        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

        if device:
            self.device = device
        else:
            try:
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
            except (ImportError, RuntimeError):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Optional[_UtilityNet] = None
        self.feature_names: Optional[List[str]] = None
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        return (X - self._mu) / self._sigma

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None or self._sigma is None:
            return X
        return (X - self._mu) / self._sigma

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> 'MLPPredictor':
        # Filter features
        X_filtered = self._filter_features(X)
        self.feature_names = list(X_filtered.columns)

        # Standardize
        Xmat = X_filtered.to_numpy(dtype=np.float32)
        Xmat = self._standardize_fit(Xmat)
        ymat = y.to_numpy(dtype=np.float32)

        d = Xmat.shape[1]
        n = Xmat.shape[0]

        # Create model
        self.model = _UtilityNet(d_in=d, layer_sizes=self.layer_sizes, dropout=self.dropout).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Log device
        print(f"[MLP] Training on device: {self.device}")
        if 'xla' in str(self.device):
            print(f"[MLP] TPU available via torch_xla")
        elif torch.cuda.is_available():
            print(f"[MLP] GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[MLP] GPU not available, using CPU")

        # Move data to device
        X_all = torch.tensor(Xmat, dtype=torch.float32, device=self.device)
        y_all = torch.tensor(ymat, dtype=torch.float32, device=self.device)

        # Train loop
        self.model.train()
        is_xla = 'xla' in str(self.device)
        gen_device = 'cpu' if is_xla or not torch.cuda.is_available() else self.device
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(self.random_state)
        n_batches = max(1, (n + self.batch_size - 1) // self.batch_size)

        for epoch in range(self.epochs):
            idx = torch.randperm(n, generator=gen, device=self.device)
            total_loss_t = torch.zeros(1, device=self.device)

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]

                X_batch = X_all[batch_idx]  # (B, D)
                y_batch = y_all[batch_idx]  # (B,)

                logits = self.model(X_batch)  # (B,)

                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)
                loss.backward()
                opt.step()
                if is_xla:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()

                total_loss_t += loss.detach()

            total_loss = total_loss_t.item() / n_batches
            print(f"[MLP] epoch={epoch} loss={total_loss:.4f} samples={n}")

            if epoch_callback is not None and not epoch_callback(epoch, total_loss):
                break

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        if self.selected_features_ is None:
            raise ValueError("Model was not properly fitted (selected_features_ is None)")

        X_filtered = X[self.selected_features_]
        Xmat = X_filtered.to_numpy(dtype=np.float32)
        Xmat = self._standardize_apply(Xmat)

        self.model.eval()
        X_t = torch.tensor(Xmat, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X_t)  # (N,)
            p_win = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([1.0 - p_win, p_win])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        if len(self.layer_sizes) == 0:
            weights = self.model.net.weight.detach().cpu().numpy()
            importances = np.abs(weights).flatten()
        elif len(self.layer_sizes) == 1:
            weights = self.model.net[0].weight.detach().cpu().numpy()
            importances = np.abs(weights).mean(axis=0)
        else:
            weights = self.model.proj.weight.detach().cpu().numpy()
            importances = np.abs(weights).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': importances,
            'abs_coefficient': np.abs(importances)
        })
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        return importance_df

    def get_model_summary(self) -> dict:
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            'model_type': 'MLP (PyTorch)',
            'n_features': len(self.feature_names or []),
            'feature_names': self.feature_names,
            'layer_sizes': self.layer_sizes,
            'dropout': self.dropout,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'device': str(self.device),
        }
