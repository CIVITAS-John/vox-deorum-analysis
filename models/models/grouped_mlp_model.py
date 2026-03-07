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
import torch.nn.functional as F
import torch.optim as optim

from .base_predictor import BasePredictor


@dataclass
class _BatchedGroups:
    """Pre-padded tensors for vectorized training/inference."""
    X: np.ndarray          # (n_groups, max_players, n_features)
    y_indices: np.ndarray  # (n_groups,) winner index per group
    mask: np.ndarray       # (n_groups, max_players) True for real players
    n_groups: int


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

    def _build_groups(self, df: pd.DataFrame, y: pd.Series) -> _BatchedGroups:
        """
        Convert row-wise data into padded, batched tensors for vectorized training.

        Returns _BatchedGroups with pre-padded arrays ready for GPU.
        """
        g_game, g_turn = self.group_cols

        # winner per game (player_id)
        winner_map = (
            df.loc[y == 1, ["game_id", "player_id"]]
            .drop_duplicates()
            .set_index("game_id")["player_id"]
            .to_dict()
        )

        # First pass: collect group data and determine max_players
        group_data: List[Tuple[np.ndarray, int]] = []  # (X_group, winner_idx)
        max_players = 0

        for (game_id, turn), gdf in df.groupby([g_game, g_turn], sort=False):
            if game_id not in winner_map:
                continue
            winner_pid = winner_map[game_id]
            players = gdf["player_id"].tolist()
            if winner_pid not in players:
                continue

            y_index = players.index(winner_pid)
            Xg = gdf[self.selected_features_].to_numpy(dtype=np.float32)
            group_data.append((Xg, y_index))
            max_players = max(max_players, len(players))

        n_groups = len(group_data)
        n_features = len(self.selected_features_)

        # Second pass: pad into dense arrays
        X_padded = np.zeros((n_groups, max_players, n_features), dtype=np.float32)
        y_indices = np.zeros(n_groups, dtype=np.int64)
        mask = np.zeros((n_groups, max_players), dtype=bool)

        for i, (Xg, y_idx) in enumerate(group_data):
            n_p = Xg.shape[0]
            X_padded[i, :n_p, :] = Xg
            y_indices[i] = y_idx
            mask[i, :n_p] = True

        return _BatchedGroups(X=X_padded, y_indices=y_indices, mask=mask, n_groups=n_groups)

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

        # 3) Build grouped samples (padded tensors)
        batched = self._build_groups(X_std, y)
        if batched.n_groups == 0:
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

        # Move padded data to device as tensors
        X_all = torch.tensor(batched.X, dtype=torch.float32, device=self.device)     # (N, P, D)
        y_all = torch.tensor(batched.y_indices, dtype=torch.long, device=self.device) # (N,)
        mask_all = torch.tensor(batched.mask, device=self.device)                     # (N, P)
        max_players = X_all.shape[1]

        # 5) Vectorized train loop
        self.model.train()
        rng = np.random.default_rng(self.random_state)
        n_batches = max(1, (batched.n_groups + self.batch_size_groups - 1) // self.batch_size_groups)

        for epoch in range(self.epochs):
            idx = rng.permutation(batched.n_groups)
            total_loss = 0.0

            for start in range(0, batched.n_groups, self.batch_size_groups):
                batch_idx = idx[start:start + self.batch_size_groups]
                B = len(batch_idx)
                batch_idx_t = torch.tensor(batch_idx, dtype=torch.long, device=self.device)

                X_batch = X_all[batch_idx_t]     # (B, P, D)
                y_batch = y_all[batch_idx_t]     # (B,)
                mask_batch = mask_all[batch_idx_t]  # (B, P)

                # Single forward pass: flatten players, then reshape
                flat = X_batch.reshape(B * max_players, d)  # (B*P, D)
                logits = self.model(flat).reshape(B, max_players)  # (B, P)

                # Mask padded positions with -inf before cross-entropy
                logits = logits.masked_fill(~mask_batch, float('-inf'))

                opt.zero_grad()
                loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                opt.step()
                if 'xla' in str(self.device):
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()

                total_loss += loss.detach().item()

            total_loss /= n_batches
            print(f"[GroupedMLP] epoch={epoch} loss={total_loss:.4f} groups={batched.n_groups}")

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

        # Collect groups and their original indices
        gb = X.groupby(list(self.group_cols), sort=False)
        group_indices: List[np.ndarray] = []
        max_players = 0
        for _, idx in gb.indices.items():
            idx_arr = np.array(idx, dtype=np.int64)
            group_indices.append(idx_arr)
            max_players = max(max_players, len(idx_arr))

        n_groups = len(group_indices)
        n_features = Xmat.shape[1]

        # Pad into batched arrays
        X_padded = np.zeros((n_groups, max_players, n_features), dtype=np.float32)
        mask = np.zeros((n_groups, max_players), dtype=bool)
        for i, idx_arr in enumerate(group_indices):
            n_p = len(idx_arr)
            X_padded[i, :n_p, :] = Xmat[idx_arr]
            mask[i, :n_p] = True

        # Single batched forward pass
        X_t = torch.tensor(X_padded, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, device=self.device)

        with torch.no_grad():
            flat = X_t.reshape(n_groups * max_players, n_features)
            logits = self.model(flat).reshape(n_groups, max_players)
            logits = logits.masked_fill(~mask_t, float('-inf'))
            pg = torch.softmax(logits, dim=1).cpu().numpy()  # (n_groups, max_players)

        # Scatter probabilities back to original row positions
        for i, idx_arr in enumerate(group_indices):
            n_p = len(idx_arr)
            probs[idx_arr] = pg[i, :n_p].astype(np.float32)

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

