#!/usr/bin/env python3
"""
Attention-based predictor for real-time winrate.

Key idea:
- Group = (game_id, turn)
- Encode each player's features into an embedding (shared encoder)
- Self-attention across players: each player attends to all others
- Decode attended embeddings → scalar logit per player (shared decoder)
- Softmax across players => winrate table

Unlike the DeepSets model (InteractionMLP) which uses fixed mean/max pooling,
this model learns which opponents are most relevant for each player's prediction
through multi-head self-attention.
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
from .interaction_mlp_model import _ResidualMLP


class _AttentionBlock(nn.Module):
    """Pre-norm self-attention block with residual connection.

    LayerNorm → MultiheadAttention → residual add.
    """
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, H) player embeddings
            key_padding_mask: (B, P) True = ignore (padded position)

        Returns:
            (B, P, H) attended embeddings with residual
        """
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        return x + attn_out


class _AttentionNet(nn.Module):
    """Attention-based network for cross-player interaction.

    Architecture:
    1. Encoder: per-player features → H-dim embedding (shared weights)
    2. Self-attention: N layers of multi-head self-attention with residual
    3. Decoder: attended embedding → scalar logit per player (shared weights)
    """
    def __init__(
        self,
        d_in: int,
        encoder_sizes: Tuple[int, ...] = (128,),
        decoder_sizes: Tuple[int, ...] = (128,),
        dropout: float = 0.0,
        num_heads: int = 4,
        n_attn_layers: int = 1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = encoder_sizes[0] if encoder_sizes else d_in

        self.encoder = _ResidualMLP(d_in, self.embed_dim, encoder_sizes, dropout)
        self.attn_layers = nn.ModuleList([
            _AttentionBlock(self.embed_dim, num_heads, attn_dropout)
            for _ in range(n_attn_layers)
        ])
        self.decoder = _ResidualMLP(self.embed_dim, 1, decoder_sizes, dropout)

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

        # 2. Self-attention (mask inverted: True = ignore for key_padding_mask)
        key_padding_mask = ~mask
        for layer in self.attn_layers:
            h = layer(h, key_padding_mask)  # (B, P, H)

        # 3. Decode to scalar logit per player
        flat_h = h.reshape(B * P, self.embed_dim)
        logits = self.decoder(flat_h).reshape(B, P)  # (B, P)

        return logits


class AttentionMLPPredictor(BasePredictor):
    """
    Attention-based MLP model: encodes each player, applies self-attention
    across the group for cross-player context, then decodes to per-player
    logits with group-wise softmax for normalized win probabilities.

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
        num_heads: int = 4,
        n_attn_layers: int = 1,
        attn_dropout: float = 0.1,
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
        self.num_heads = num_heads
        self.n_attn_layers = n_attn_layers
        self.attn_dropout = attn_dropout
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

        self.model: Optional[_AttentionNet] = None
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

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> "AttentionMLPPredictor":
        """
        Fit attention MLP using group-wise cross entropy.
        """
        X_filtered = self._filter_features(X)

        missing = [c for c in self.REQUIRES_ID_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(
                f"AttentionMLPPredictor requires columns {missing} in X. "
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
        self.model = _AttentionNet(
            d_in=d,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            dropout=self.dropout,
            num_heads=self.num_heads,
            n_attn_layers=self.n_attn_layers,
            attn_dropout=self.attn_dropout,
        ).to(self.device)

        is_xla = 'xla' in str(self.device)
        import sys
        if not is_xla and sys.platform != 'win32':
            try:
                torch.set_float32_matmul_precision('high')
                self.model = torch.compile(self.model)
            except Exception:
                pass

        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        print(f"[AttentionMLP] Training on device: {self.device}")
        if 'xla' in str(self.device):
            print(f"[AttentionMLP] TPU available via torch_xla")
        elif torch.cuda.is_available():
            print(f"[AttentionMLP] GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[AttentionMLP] GPU not available, using CPU")

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
            print(f"[AttentionMLP] epoch={epoch} loss={total_loss:.4f} groups={batched.n_groups}")

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
            "model_type": "AttentionMLP (self-attention group-softmax)",
            "group_cols": self.group_cols,
            "n_features": len(self.feature_names or []),
            "feature_names": self.feature_names,
            "encoder_sizes": self.encoder_sizes,
            "decoder_sizes": self.decoder_sizes,
            "num_heads": self.num_heads,
            "n_attn_layers": self.n_attn_layers,
            "attn_dropout": self.attn_dropout,
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
