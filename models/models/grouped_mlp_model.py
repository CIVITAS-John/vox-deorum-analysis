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

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from .base_predictor import BasePredictor


@dataclass
class _TrainGroup:
    # One training example: all players at a (game_id, turn)
    X: np.ndarray          # shape (n_players, n_features)
    y_index: int           # winner index within this group


class _UtilityNet(nn.Module):
    """Simple MLP that maps features -> scalar utility."""
    def __init__(self, d_in: int, layer_sizes: Tuple[int, ...] = (64,), dropout: float = 0.0):
        super().__init__()
        if len(layer_sizes) == 0:
            self.net = nn.Linear(d_in, 1)
        else:
            layers: list[nn.Module] = []
            prev = d_in
            for size in layer_sizes:
                layers.extend([nn.Linear(prev, size), nn.ReLU(), nn.Dropout(dropout)])
                prev = size
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_players, d_in) -> (n_players,)
        return self.net(x).squeeze(-1)


class GroupedMLPPredictor(BasePredictor):
    """
    Grouped MLP model: scores each player with an MLP, then applies
    group-wise softmax over (game_id, turn) to produce normalized win probabilities.

    IMPORTANT:
    - X passed to fit/predict_proba must include columns: game_id, turn
    - These id columns are NOT used as numeric features
    """

    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = None
    DISABLE_RESAMPLING = True
    REQUIRED_FEATURES = None
    REQUIRES_ID_COLUMNS = ['game_id', 'turn', 'player_id']

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        group_cols: Tuple[str, str] = ("game_id", "turn"),
        id_cols: Tuple[str, ...] = ("experiment", "game_id", "player_id", "turn"),
        layer_sizes: Tuple[int, ...] = (118,),
        dropout: float = 0.46,
        lr: float = 0.00032,
        weight_decay: float = 0.0024,
        epochs: int = 5,
        batch_size_groups: int = 128,
        device: Optional[str] = None,
    ):
        super().__init__(include_features, exclude_features, random_state)
        self.group_cols = group_cols
        self.id_cols = id_cols

        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size_groups = batch_size_groups

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

        # Simple standardization (helps stability)
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

    def _build_groups(self, df: pd.DataFrame, y: pd.Series) -> List[_TrainGroup]:
        """
        Convert row-wise data into grouped training samples.

        We assume y is row-wise is_winner (0/1). Winner for a game is fixed.
        For each (game_id, turn), y_index is the index of the winner player in that group.
        """
        g_game, g_turn = self.group_cols

        # winner per game (player_id)
        # We use y==1 rows to determine winner player_id for each game_id
        winner_map = (
            df.loc[y == 1, ["game_id", "player_id"]]
            .drop_duplicates()
            .set_index("game_id")["player_id"]
            .to_dict()
        )

        groups: List[_TrainGroup] = []
        for (game_id, turn), gdf in df.groupby([g_game, g_turn], sort=False):
            if game_id not in winner_map:
                continue
            winner_pid = winner_map[game_id]
            players = gdf["player_id"].tolist()
            if winner_pid not in players:
                continue

            y_index = players.index(winner_pid)
            Xg = gdf[self.selected_features_].to_numpy(dtype=np.float32)
            groups.append(_TrainGroup(X=Xg, y_index=y_index))

        return groups

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None) -> "GroupedMLPPredictor":
        """
        Fit grouped MLP using group-wise cross entropy.

        X MUST include:
        - game_id
        - turn
        - player_id (needed to identify winner index inside each group)
        """
        # 1) Filter features (BasePredictor sets selected_features_)
        X_filtered = self._filter_features(X)

        # Make sure grouping/id columns exist in original X for grouping
        missing = [c for c in self.REQUIRES_ID_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(
                f"GroupedMLPPredictor requires columns {missing} in X. "
                f"These should be automatically injected by the evaluator. "
                f"If calling fit() directly, ensure X includes: {self.REQUIRES_ID_COLUMNS}"
            )

        # selected_features_ should not include IDs; enforce that here
        self.selected_features_ = [c for c in self.selected_features_ if c not in self.id_cols]
        self.feature_names = list(self.selected_features_)

        # 2) Standardize numeric feature matrix (fit on all rows)
        Xmat = X[self.selected_features_].to_numpy(dtype=np.float32)
        Xmat = self._standardize_fit(Xmat)

        # Put standardized values back for group building
        # Put standardized values into a fresh DF to avoid dtype issues (int cols vs float)
        X_std = pd.DataFrame(Xmat, columns=self.selected_features_, index=X.index)
        X_std = pd.concat([X[["game_id", "turn", "player_id"]], X_std], axis=1)

        # 3) Build grouped samples
        train_groups = self._build_groups(X_std, y)
        if len(train_groups) == 0:
            raise ValueError("No valid (game_id, turn) groups constructed. Check your data.")

        # 4) Create model
        d = len(self.selected_features_)
        self.model = _UtilityNet(d_in=d, layer_sizes=self.layer_sizes, dropout=self.dropout).to(self.device)

        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Log device information
        print(f"[GroupedMLP] Training on device: {self.device}")
        if 'xla' in str(self.device):
            print(f"[GroupedMLP] TPU available via torch_xla")
        elif torch.cuda.is_available():
            print(f"[GroupedMLP] GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[GroupedMLP] GPU not available, using CPU")

        # 5) Train loop over groups (mini-batches of groups)
        self.model.train()
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            idx = rng.permutation(len(train_groups))
            total_loss = 0.0

            for start in range(0, len(train_groups), self.batch_size_groups):
                batch_idx = idx[start:start + self.batch_size_groups]
                opt.zero_grad()

                batch_loss = 0.0
                for k in batch_idx:
                    g = train_groups[k]
                    Xg = torch.tensor(g.X, dtype=torch.float32, device=self.device)
                    logits = self.model(Xg)  # (n_players,)
                    logp = logits - torch.logsumexp(logits, dim=0)
                    batch_loss = batch_loss + (-logp[g.y_index])

                batch_loss = batch_loss / max(1, len(batch_idx))
                batch_loss.backward()
                opt.step()
                if 'xla' in str(self.device):
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()

                total_loss += float(batch_loss.detach().cpu().item())

            total_loss /= max(1, len(range(0, len(train_groups), self.batch_size_groups)))
            print(f"[GroupedMLP] epoch={epoch} loss={total_loss:.4f} groups={len(train_groups)}")

        return self

    def predict_group_winrate(self, X: pd.DataFrame) -> pd.Series:
        """
        Return normalized winrates within each (game_id, turn).

        Output: Series aligned to X.index, containing P(win) for each row,
        where probabilities sum to 1 within each group.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        if self.selected_features_ is None:
            raise ValueError("Model was not properly fitted (selected_features_ is None)")
        for c in ["game_id", "turn"]:
            if c not in X.columns:
                raise ValueError(f"predict_group_winrate requires column '{c}' in X")

        self.model.eval()

        # Standardize features
        Xmat = X[self.selected_features_].to_numpy(dtype=np.float32)
        Xmat = self._standardize_apply(Xmat)

        probs = np.zeros(len(X), dtype=np.float32)

        # group-wise softmax
        # NOTE: assumes X has one row per player at the same (game_id, turn)
        gb = X.groupby(list(self.group_cols), sort=False)
        with torch.no_grad():
            for _, idx in gb.indices.items():
                idx = np.array(idx, dtype=np.int64)
                Xg = torch.tensor(Xmat[idx], dtype=torch.float32, device=self.device)
                logits = self.model(Xg)  # (n_players,)
                pg = torch.softmax(logits, dim=0).detach().cpu().numpy()
                probs[idx] = pg.astype(np.float32)

        return pd.Series(probs, index=X.index, name="p_win_group")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Framework-compatible predict_proba:
        returns (n_samples, 2) = [P(loss), P(win)].

        Here, P(win) is the group-normalized winrate.
        """
        p_win = self.predict_group_winrate(X).to_numpy()
        return np.column_stack([1.0 - p_win, p_win])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Binary prediction: 1 if highest winrate in its group, else 0.
        """
        p = self.predict_group_winrate(X)
        preds = np.zeros(len(X), dtype=np.int64)

        for _, gdf in X.assign(_p=p.values).groupby(list(self.group_cols), sort=False):
            idx = gdf.index.to_numpy()
            winner_idx = gdf["_p"].values.argmax()
            preds[np.where(X.index.values == idx[winner_idx])[0][0]] = 1

        return preds

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
            "device": self.device,
        }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance based on input layer weights.

        For linear utility (layer_sizes=()): Uses weights from single Linear layer
        For MLP utility (layer_sizes non-empty): Averages absolute weights from first Linear layer

        Returns:
            DataFrame with columns ['feature', 'coefficient', 'abs_coefficient']
            sorted by abs_coefficient descending
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        # Extract input layer weights based on architecture
        if len(self.layer_sizes) == 0:
            # Linear utility: self.model.net is nn.Linear(d_in, 1)
            weights = self.model.net.weight.detach().cpu().numpy()  # shape: (1, d_in)
            importances = np.abs(weights).flatten()  # shape: (d_in,)
        else:
            # MLP utility: self.model.net[0] is first nn.Linear layer
            weights = self.model.net[0].weight.detach().cpu().numpy()  # shape: (layer_sizes[0], d_in)
            # Average absolute weight across hidden units
            importances = np.abs(weights).mean(axis=0)  # shape: (d_in,)

        # Create DataFrame matching expected format
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': importances,
            'abs_coefficient': np.abs(importances)
        })

        # Sort by importance (descending)
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        return importance_df

