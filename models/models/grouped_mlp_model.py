#!/usr/bin/env python3
"""
Grouped MLP predictor for real-time winrate.

Key idea:
- Group = (game_id, turn)
- Score each player with u(x) via MLP
- Softmax u across players in the same group => winrate table
- Train with group-wise cross entropy on the eventual winner
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from .base_torch_predictor import GroupedTorchPredictor, _BatchedGroups  # noqa: F401 (_BatchedGroups re-exported)


class _UtilityNet(nn.Module):
    """Residual MLP that maps features -> scalar utility.

    Architecture by depth:
    - 0 layers: Linear(d_in, 1) — pure linear scoring
    - 1 layer:  Linear → GELU → Dropout → Linear — single hidden layer
    - 2+ layers: Linear projection → N residual blocks → LayerNorm → Linear
                 Each block: LayerNorm → Linear → GELU → Dropout + skip connection
    """
    def __init__(self, d_in: int, layer_sizes: Tuple[int, ...] = (64,), dropout: float = 0.0):
        super().__init__()
        if len(layer_sizes) == 0:
            # Linear: features → scalar directly
            self.net = nn.Linear(d_in, 1)
        elif len(layer_sizes) == 1:
            # Single hidden layer (no skip connection needed)
            self.net = nn.Sequential(
                nn.Linear(d_in, layer_sizes[0]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0], 1),
            )
        else:
            # Deep residual: project → N residual blocks → output
            hidden = layer_sizes[0]  # constant width
            self.proj = nn.Linear(d_in, hidden)
            self.blocks = nn.ModuleList()
            for _ in layer_sizes:
                self.blocks.append(nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
            self.norm_out = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, 1)
            self.net = None  # flag for forward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_players, d_in) -> (n_players,)
        if self.net is not None:
            return self.net(x).squeeze(-1)
        # Residual path
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.norm_out(x)
        return self.head(x).squeeze(-1)


class GroupedMLPPredictor(GroupedTorchPredictor):
    """
    Grouped MLP model: scores each player with an MLP, then applies
    group-wise softmax over (game_id, turn) to produce normalized win probabilities.

    IMPORTANT:
    - X passed to fit/predict_proba must include columns: game_id, turn
    - These id columns are NOT used as numeric features
    """

    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = [
        # City-adjusted per-turn rates
        'science_share', 'culture_share', 'tourism_share', 'gold_share',
        'food_share', 'military_share',
        # Raw share variants
        'faith_raw_share', 'production_raw_share',
        # Raw counts
        'cities', 'population', 'votes',
        # Share variant
        'minor_allies_share',
        # Gaps from leader
        'technologies_gap', 'policies_gap',
        # Percentage / ratio metrics
        'happiness_percentage', 'military_utilization', 'religion_percentage'
        # Progress
        'turn_progress',
    ]
    REQUIRED_FEATURES = None

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        group_cols: Tuple[str, str] = ("game_id", "turn"),
        id_cols: Tuple[str, ...] = ("experiment", "game_id", "player_id", "turn"),
        layer_sizes: Tuple[int, ...] = (45,45,45,45,45,45),
        dropout: float = 0.10726123057010115,
        lr: float = 0.00320236867338213,
        weight_decay: float = 0.007788566976755283,
        epochs: int = 29,
        batch_size_groups: int = 4096,
        loss_tp_alpha: float = 0.747520815240142,
        device: Optional[str] = None,
    ):
        super().__init__(
            include_features=include_features,
            exclude_features=exclude_features,
            random_state=random_state,
            group_cols=group_cols,
            id_cols=id_cols,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size_groups=batch_size_groups,
            loss_tp_alpha=loss_tp_alpha,
            device=device,
        )
        self.layer_sizes = layer_sizes

    def _model_display_name(self) -> str:
        return "GroupedMLP"

    def _build_model(self, d_in: int) -> nn.Module:
        return _UtilityNet(d_in=d_in, layer_sizes=self.layer_sizes, dropout=self.dropout).to(self.device)

    def _forward_train(self, X_batch: torch.Tensor, mask_batch: torch.Tensor) -> torch.Tensor:
        B, P, D = X_batch.shape
        flat = X_batch.reshape(B * P, D)
        return self.model(flat).reshape(B, P)

    def _forward_inference(self, X_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        N, P, D = X_t.shape
        flat = X_t.reshape(N * P, D)
        return self.model(flat).reshape(N, P)

    def get_model_summary(self) -> dict:
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            "model_type": "GroupedMLP (group-softmax)",
            "group_cols": self.group_cols,
            "n_features": len(self.feature_names or []),
            "feature_names": self.feature_names,
            "layer_sizes": self.layer_sizes,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "loss_tp_alpha": self.loss_tp_alpha,
            "device": self.device,
        }

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
        return importance_df.sort_values('abs_coefficient', ascending=False)
