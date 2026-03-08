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
        layer_sizes: Tuple[int, ...] = (204,),
        dropout: float = 0.237563,
        lr: float = 0.0026367,
        weight_decay: float = 1.68821e-05,
        epochs: int = 3,
        batch_size_groups: int = 1024,
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

        Uses vectorized pandas/numpy ops instead of Python for-loops.
        Returns _BatchedGroups with pre-padded arrays ready for GPU.
        """
        g_game, g_turn = self.group_cols

        # Vectorized winner mapping
        winner_map = (
            df.loc[y == 1, ["game_id", "player_id"]]
            .drop_duplicates()
            .set_index("game_id")["player_id"]
        )
        df = df.copy()
        df["_winner_pid"] = df["game_id"].map(winner_map)

        # Drop rows where no winner is known for that game
        df = df.dropna(subset=["_winner_pid"])

        # Group IDs and within-group position
        grp = df.groupby([g_game, g_turn], sort=False)
        df["_gid"] = grp.ngroup()
        df["_pos"] = grp.cumcount()
        df["_is_winner"] = (df["player_id"] == df["_winner_pid"]).astype(int)

        # Filter to groups that contain the winner
        valid_gids = df.loc[df["_is_winner"] == 1, "_gid"].unique()
        df = df[df["_gid"].isin(valid_gids)]

        # Re-number groups contiguously after filtering
        unique_gids = df["_gid"].unique()
        gid_remap = pd.Series(np.arange(len(unique_gids)), index=unique_gids)
        df["_gid"] = df["_gid"].map(gid_remap)

        n_groups = len(unique_gids)
        max_players = int(df["_pos"].max()) + 1
        n_features = len(self.selected_features_)

        # Direct array fill using fancy indexing (no Python loop)
        gids = df["_gid"].values
        pos = df["_pos"].values

        X_padded = np.zeros((n_groups, max_players, n_features), dtype=np.float32)
        X_padded[gids, pos, :] = df[self.selected_features_].to_numpy(dtype=np.float32)

        mask = np.zeros((n_groups, max_players), dtype=bool)
        mask[gids, pos] = True

        # Winner index per group
        y_indices = np.zeros(n_groups, dtype=np.int64)
        winner_df = df[df["_is_winner"] == 1]
        y_indices[winner_df["_gid"].values] = winner_df["_pos"].values

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
        is_xla = 'xla' in str(self.device)
        gen_device = 'cpu' if is_xla or not torch.cuda.is_available() else self.device
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(self.random_state)
        n_batches = max(1, (batched.n_groups + self.batch_size_groups - 1) // self.batch_size_groups)

        for epoch in range(self.epochs):
            idx = torch.randperm(batched.n_groups, generator=gen, device=self.device)
            total_loss_t = torch.zeros(1, device=self.device)

            for start in range(0, batched.n_groups, self.batch_size_groups):
                batch_idx_t = idx[start:start + self.batch_size_groups]
                B = batch_idx_t.shape[0]

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
                if is_xla:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()

                total_loss_t += loss.detach()

            total_loss = (total_loss_t.item()) / n_batches
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

        n_features = Xmat.shape[1]

        # Vectorized group indexing (no Python loop)
        gb = X.groupby(list(self.group_cols), sort=False)
        gids = gb.ngroup().values
        pos = gb.cumcount().values
        n_groups = gids.max() + 1
        max_players = pos.max() + 1

        # Direct array fill using fancy indexing
        X_padded = np.zeros((n_groups, max_players, n_features), dtype=np.float32)
        X_padded[gids, pos, :] = Xmat
        mask = np.zeros((n_groups, max_players), dtype=bool)
        mask[gids, pos] = True

        # Single batched forward pass
        X_t = torch.tensor(X_padded, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, device=self.device)

        with torch.no_grad():
            flat = X_t.reshape(n_groups * max_players, n_features)
            logits = self.model(flat).reshape(n_groups, max_players)
            logits = logits.masked_fill(~mask_t, float('-inf'))
            pg = torch.softmax(logits, dim=1).cpu().numpy()  # (n_groups, max_players)

        # Scatter probabilities back using fancy indexing (no loop)
        probs = pg[gids, pos].astype(np.float32)

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

        # Vectorized: find argmax within each group
        winner_indices = p.groupby(
            [X[c] for c in self.group_cols], sort=False
        ).idxmax()

        # Use index-based lookup for O(n) assignment
        idx_to_pos = pd.Series(np.arange(len(X)), index=X.index)
        preds[idx_to_pos[winner_indices].values] = 1

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
        elif len(self.layer_sizes) == 1:
            # Single hidden layer: self.model.net[0] is first nn.Linear layer
            weights = self.model.net[0].weight.detach().cpu().numpy()  # shape: (layer_sizes[0], d_in)
            importances = np.abs(weights).mean(axis=0)  # shape: (d_in,)
        else:
            # Deep residual: self.model.proj is the input projection layer
            weights = self.model.proj.weight.detach().cpu().numpy()  # shape: (hidden, d_in)
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

