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

from .base_torch_predictor import BaseTorchPredictor
from .grouped_mlp_model import _UtilityNet


class MLPPredictor(BaseTorchPredictor):
    """
    PyTorch MLP for victory prediction with GPU support.
    Uses the same residual _UtilityNet architecture as GroupedMLPPredictor,
    but trains with standard binary cross-entropy on individual samples.
    """

    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = None
    REQUIRED_FEATURES = None

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
        batch_size: int = 32768,
        loss_tp_alpha: float = 0.0,
        device: Optional[str] = None,
    ):
        super().__init__(
            include_features=include_features,
            exclude_features=exclude_features,
            random_state=random_state,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            loss_tp_alpha=loss_tp_alpha,
            device=device,
        )
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size

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
        self._compile_model()
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._log_device("MLP")

        # Move data to device
        X_all = torch.tensor(Xmat, dtype=torch.float32, device=self.device)
        y_all = torch.tensor(ymat, dtype=torch.float32, device=self.device)
        if self.loss_tp_alpha != 0 and "turn_progress" in X.columns:
            tp_all = torch.tensor(X["turn_progress"].values, dtype=torch.float32, device=self.device)
        else:
            tp_all = None

        # Train loop
        self.model.train()
        gen = self._make_generator()
        scaler = self._create_scaler()
        n_batches = max(1, (n + self.batch_size - 1) // self.batch_size)

        for epoch in range(self.epochs):
            idx = torch.randperm(n, generator=gen, device=self.device)
            total_loss_t = torch.zeros(1, device=self.device)

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]

                X_batch = X_all[batch_idx]  # (B, D)
                y_batch = y_all[batch_idx]  # (B,)

                opt.zero_grad()
                with torch.amp.autocast('cuda', enabled=self._amp_enabled):
                    logits = self.model(X_batch)  # (B,)

                    if tp_all is not None:
                        tp_batch = tp_all[batch_idx]
                        weight = tp_batch ** self.loss_tp_alpha
                        raw_loss = F.binary_cross_entropy_with_logits(logits, y_batch, reduction='none')
                        loss = (raw_loss * weight).mean()
                    else:
                        loss = F.binary_cross_entropy_with_logits(logits, y_batch)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                self._xla_mark_step()
                total_loss_t += loss.detach()

            total_loss = total_loss_t.item() / n_batches
            print(f"[MLP] epoch={epoch} loss={total_loss:.4f} samples={n}")

            if epoch_callback is not None and not epoch_callback(epoch, total_loss):
                break

        self._restore_model()
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
            'loss_tp_alpha': self.loss_tp_alpha,
            'device': str(self.device),
        }
