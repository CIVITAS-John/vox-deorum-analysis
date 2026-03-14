#!/usr/bin/env python3
"""
Interaction MLP predictor (DeepSets architecture) for real-time winrate.

Key idea:
- Group = (game_id, turn)
- Encode each player's features into an embedding (shared encoder)
- Pool embeddings across all players in the group (masked mean)
- Decode [player_embed, pool_context] → scalar logit per player (shared decoder)
- Softmax across players => winrate table

Unlike GroupedMLPPredictor where each player is scored independently,
this model allows cross-player interaction through the pooled context,
while remaining permutation equivariant by construction.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_predictor import BasePredictor
from .grouped_mlp_model import _BatchedGroups


class _ResidualMLP(nn.Module):
    """Residual MLP block with configurable output dimension.

    Architecture by depth:
    - 0 layers: Linear(d_in, d_out) — pure linear projection
    - 1 layer:  Linear → GELU → Dropout → Linear — single hidden layer
    - 2+ layers: Linear projection → N residual blocks → LayerNorm → Linear
                 Each block: LayerNorm → Linear → GELU → Dropout + skip connection
    """
    def __init__(self, d_in: int, d_out: int, layer_sizes: Tuple[int, ...] = (64,), dropout: float = 0.0):
        super().__init__()
        if len(layer_sizes) == 0:
            self.net = nn.Linear(d_in, d_out)
        elif len(layer_sizes) == 1:
            self.net = nn.Sequential(
                nn.Linear(d_in, layer_sizes[0]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0], d_out),
            )
        else:
            hidden = layer_sizes[0]
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
            self.head = nn.Linear(hidden, d_out)
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is not None:
            return self.net(x)
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.norm_out(x)
        return self.head(x)


class _DeepSetsNet(nn.Module):
    """DeepSets-style network for cross-player interaction.

    Architecture:
    1. Encoder: per-player features → H-dim embedding (shared weights)
    2. Pool: masked mean across players → context vector
    3. Decoder: [player_embed, context] → scalar logit (shared weights)
    """
    def __init__(
        self,
        d_in: int,
        encoder_sizes: Tuple[int, ...] = (64,),
        decoder_sizes: Tuple[int, ...] = (64,),
        dropout: float = 0.0,
    ):
        super().__init__()
        # Embedding dimension = first encoder layer size (or d_in if no layers)
        self.embed_dim = encoder_sizes[0] if encoder_sizes else d_in

        self.encoder = _ResidualMLP(d_in, self.embed_dim, encoder_sizes, dropout)
        # Decoder input: player embedding + pooled context = 2 * embed_dim
        self.decoder = _ResidualMLP(self.embed_dim * 2, 1, decoder_sizes, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, D) player features
            mask: (B, P) True for real players

        Returns:
            (B, P) logits per player
        """
        B, P, D = x.shape

        # 1. Encode each player (shared weights)
        flat = x.reshape(B * P, D)
        h = self.encoder(flat).reshape(B, P, self.embed_dim)  # (B, P, H)

        # 2. Masked mean pool across players
        mask_f = mask.unsqueeze(-1).float()  # (B, P, 1)
        h_masked = h * mask_f
        pool = h_masked.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, H)

        # 3. Broadcast context and concat with player embeddings
        pool_expanded = pool.unsqueeze(1).expand(-1, P, -1)  # (B, P, H)
        combined = torch.cat([h, pool_expanded], dim=-1)  # (B, P, 2H)

        # 4. Decode to scalar logit per player
        flat_combined = combined.reshape(B * P, self.embed_dim * 2)
        logits = self.decoder(flat_combined).reshape(B, P)  # (B, P)

        return logits


class InteractionMLPPredictor(BasePredictor):
    """
    DeepSets-based MLP model: encodes each player, pools across the group
    for cross-player context, then decodes to per-player logits with
    group-wise softmax for normalized win probabilities.

    IMPORTANT:
    - X passed to fit/predict_proba must include columns: game_id, turn, player_id
    - These id columns are NOT used as numeric features
    """

    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = [
        # City-adjusted per-turn rates (not relative to opponents)
        'tourism_adj', 'gold_adj',
        'production_adj', 'military_adj',
        # Non-adjusted
        'population', 'cities', 'votes', 'minor_allies',
        # Gaps from leader
        'technologies_gap', 'policies_gap',
        # Percentage / ratio metrics
        'happiness_percentage', 'religion_percentage',
        'military_utilization',
        # Temporal
        'turn_progress',
    ]
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
        encoder_sizes: Tuple[int, ...] = (128,),
        decoder_sizes: Tuple[int, ...] = (128,),
        dropout: float = 0.3,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        epochs: int = 10,
        batch_size_groups: int = 4096,
        loss_tp_alpha: float = 0.0,
        device: Optional[str] = None,
    ):
        super().__init__(include_features, exclude_features, random_state)
        self.group_cols = group_cols
        self.id_cols = id_cols

        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size_groups = batch_size_groups
        self.loss_tp_alpha = loss_tp_alpha

        if device:
            self.device = device
        else:
            try:
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
            except (ImportError, RuntimeError):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Optional[_DeepSetsNet] = None
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

    def _build_groups(self, df: pd.DataFrame, y: pd.Series, raw_tp: Optional[np.ndarray] = None) -> _BatchedGroups:
        """
        Convert row-wise data into padded, batched tensors for vectorized training.
        Reuses _BatchedGroups from grouped_mlp_model.
        """
        g_game, g_turn = self.group_cols

        winner_map = (
            df.loc[y == 1, ["game_id", "player_id"]]
            .drop_duplicates()
            .set_index("game_id")["player_id"]
        )
        df = df.copy()
        df["_winner_pid"] = df["game_id"].map(winner_map)
        df = df.dropna(subset=["_winner_pid"])

        grp = df.groupby([g_game, g_turn], sort=False)
        df["_gid"] = grp.ngroup()
        df["_pos"] = grp.cumcount()
        df["_is_winner"] = (df["player_id"] == df["_winner_pid"]).astype(int)

        valid_gids = df.loc[df["_is_winner"] == 1, "_gid"].unique()
        df = df[df["_gid"].isin(valid_gids)]

        unique_gids = df["_gid"].unique()
        gid_remap = pd.Series(np.arange(len(unique_gids)), index=unique_gids)
        df["_gid"] = df["_gid"].map(gid_remap)

        n_groups = len(unique_gids)
        max_players = int(df["_pos"].max()) + 1
        n_features = len(self.selected_features_)

        gids = df["_gid"].values
        pos = df["_pos"].values

        X_padded = np.zeros((n_groups, max_players, n_features), dtype=np.float32)
        X_padded[gids, pos, :] = df[self.selected_features_].to_numpy(dtype=np.float32)

        mask = np.zeros((n_groups, max_players), dtype=bool)
        mask[gids, pos] = True

        y_indices = np.zeros(n_groups, dtype=np.int64)
        winner_df = df[df["_is_winner"] == 1]
        y_indices[winner_df["_gid"].values] = winner_df["_pos"].values

        tp = np.ones(n_groups, dtype=np.float32)
        if raw_tp is not None:
            tp_vals = raw_tp.loc[df.index].values.astype(np.float32)
            tp_per_row = pd.Series(tp_vals, index=df.index)
            tp_df = df[["_gid"]].assign(_tp=tp_per_row).groupby("_gid")["_tp"].first()
            tp[tp_df.index.values] = tp_df.values

        return _BatchedGroups(X=X_padded, y_indices=y_indices, mask=mask, tp=tp, n_groups=n_groups)

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> "InteractionMLPPredictor":
        """
        Fit interaction MLP using group-wise cross entropy with DeepSets architecture.
        """
        X_filtered = self._filter_features(X)

        missing = [c for c in self.REQUIRES_ID_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(
                f"InteractionMLPPredictor requires columns {missing} in X. "
                f"These should be automatically injected by the evaluator. "
                f"If calling fit() directly, ensure X includes: {self.REQUIRES_ID_COLUMNS}"
            )

        self.selected_features_ = [c for c in self.selected_features_ if c not in self.id_cols]
        self.feature_names = list(self.selected_features_)

        Xmat = X[self.selected_features_].to_numpy(dtype=np.float32)
        Xmat = self._standardize_fit(Xmat)

        X_std = pd.DataFrame(Xmat, columns=self.selected_features_, index=X.index)
        X_std = pd.concat([X[["game_id", "turn", "player_id"]], X_std], axis=1)

        raw_tp = X["turn_progress"] if "turn_progress" in X.columns and self.loss_tp_alpha != 0 else None
        batched = self._build_groups(X_std, y, raw_tp=raw_tp)
        if batched.n_groups == 0:
            raise ValueError("No valid (game_id, turn) groups constructed. Check your data.")

        d = len(self.selected_features_)
        self.model = _DeepSetsNet(
            d_in=d,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            dropout=self.dropout,
        ).to(self.device)

        is_xla = 'xla' in str(self.device)
        import sys
        if not is_xla and sys.platform != 'win32':
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        print(f"[InteractionMLP] Training on device: {self.device}")
        if 'xla' in str(self.device):
            print(f"[InteractionMLP] TPU available via torch_xla")
        elif torch.cuda.is_available():
            print(f"[InteractionMLP] GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[InteractionMLP] GPU not available, using CPU")

        X_all = torch.tensor(batched.X, dtype=torch.float32, device=self.device)
        y_all = torch.tensor(batched.y_indices, dtype=torch.long, device=self.device)
        mask_all = torch.tensor(batched.mask, device=self.device)
        tp_all = torch.tensor(batched.tp, dtype=torch.float32, device=self.device) if self.loss_tp_alpha != 0 else None

        self.model.train()
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

                X_batch = X_all[batch_idx_t]       # (B, P, D)
                y_batch = y_all[batch_idx_t]       # (B,)
                mask_batch = mask_all[batch_idx_t]  # (B, P)

                # DeepSets forward: network sees full group with mask
                logits = self.model(X_batch, mask_batch)  # (B, P)

                # Mask padded positions with -inf before cross-entropy
                logits = logits.masked_fill(~mask_batch, float('-inf'))

                opt.zero_grad()
                if tp_all is not None:
                    tp_batch = tp_all[batch_idx_t]
                    weight = tp_batch ** self.loss_tp_alpha
                    raw_loss = F.cross_entropy(logits, y_batch, reduction='none')
                    loss = (raw_loss * weight).mean()
                else:
                    loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                opt.step()
                if is_xla:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()

                total_loss_t += loss.detach()

            total_loss = (total_loss_t.item()) / n_batches
            print(f"[InteractionMLP] epoch={epoch} loss={total_loss:.4f} groups={batched.n_groups}")

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

        Xmat = X[self.selected_features_].to_numpy(dtype=np.float32)
        Xmat = self._standardize_apply(Xmat)

        n_features = Xmat.shape[1]

        gb = X.groupby(list(self.group_cols), sort=False)
        gids = gb.ngroup().values
        pos = gb.cumcount().values
        n_groups = gids.max() + 1
        max_players = pos.max() + 1

        X_padded = np.zeros((n_groups, max_players, n_features), dtype=np.float32)
        X_padded[gids, pos, :] = Xmat
        mask = np.zeros((n_groups, max_players), dtype=bool)
        mask[gids, pos] = True

        X_t = torch.tensor(X_padded, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, device=self.device)

        with torch.no_grad():
            logits = self.model(X_t, mask_t)  # (n_groups, max_players)
            logits = logits.masked_fill(~mask_t, float('-inf'))
            pg = torch.softmax(logits, dim=1).cpu().numpy()

        probs = pg[gids, pos].astype(np.float32)

        return pd.Series(probs, index=X.index, name="p_win_group")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Framework-compatible predict_proba:
        returns (n_samples, 2) = [P(loss), P(win)].
        """
        p_win = self.predict_group_winrate(X).to_numpy()
        return np.column_stack([1.0 - p_win, p_win])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Binary prediction: 1 if highest winrate in its group, else 0.
        """
        proba = self.predict_proba(X)
        p = pd.Series(proba[:, 1], index=X.index, name="p_win_group")
        preds = np.zeros(len(X), dtype=np.int64)

        winner_indices = p.groupby(
            [X[c] for c in self.group_cols], sort=False
        ).idxmax()

        idx_to_pos = pd.Series(np.arange(len(X)), index=X.index)
        preds[idx_to_pos[winner_indices].values] = 1

        return preds

    def get_model_summary(self) -> dict:
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            "model_type": "InteractionMLP (DeepSets group-softmax)",
            "group_cols": self.group_cols,
            "n_features": len(self.feature_names or []),
            "feature_names": self.feature_names,
            "encoder_sizes": self.encoder_sizes,
            "decoder_sizes": self.decoder_sizes,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "loss_tp_alpha": self.loss_tp_alpha,
            "device": self.device,
        }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance based on encoder input layer weights.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        encoder = self.model.encoder
        if len(self.encoder_sizes) == 0:
            weights = encoder.net.weight.detach().cpu().numpy()
            importances = np.abs(weights).mean(axis=0)
        elif len(self.encoder_sizes) == 1:
            weights = encoder.net[0].weight.detach().cpu().numpy()
            importances = np.abs(weights).mean(axis=0)
        else:
            weights = encoder.proj.weight.detach().cpu().numpy()
            importances = np.abs(weights).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': importances,
            'abs_coefficient': np.abs(importances)
        })

        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        return importance_df
