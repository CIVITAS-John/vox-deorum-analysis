#!/usr/bin/env python3
"""
Common base classes for PyTorch-based victory prediction models.

BaseTorchPredictor: shared device detection, standardization, compile, AMP, logging.
GroupedTorchPredictor: shared grouped training loop, _build_groups, predict_group_winrate.
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple

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
    tp: np.ndarray         # (n_groups,) turn_progress per group
    n_groups: int


class BaseTorchPredictor(BasePredictor):
    """
    Base class for all PyTorch-based predictors.

    Provides: device detection, standardization, torch.compile management,
    device logging, RNG generator, and AMP (mixed precision) infrastructure.
    """

    DISABLE_RESAMPLING = True

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        epochs: int = 10,
        loss_tp_alpha: float = 0.0,
        device: Optional[str] = None,
    ):
        super().__init__(include_features, exclude_features, random_state)

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.loss_tp_alpha = loss_tp_alpha

        # Device detection: XLA (TPU) → CUDA → CPU
        if device:
            self.device = device
        else:
            try:
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
            except (ImportError, RuntimeError):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._is_xla = 'xla' in str(self.device)
        self._amp_enabled = (
            not self._is_xla
            and torch.cuda.is_available()
            and 'cuda' in str(self.device)
        )

        self.model: Optional[nn.Module] = None
        self.feature_names: Optional[List[str]] = None
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        return (X - self._mu) / self._sigma

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None or self._sigma is None:
            return X
        return (X - self._mu) / self._sigma

    # ------------------------------------------------------------------
    # torch.compile helpers
    # ------------------------------------------------------------------

    def _compile_model(self) -> None:
        """Optionally torch.compile the model (skipped on Windows & XLA).

        Stores uncompiled model for safe restoration after training.
        Resets Dynamo state to avoid FX/Dynamo conflicts across Optuna trials.
        """
        self._uncompiled_model = self.model
        if self._is_xla or sys.platform == 'win32':
            return
        try:
            torch.compiler.reset()
            torch.set_float32_matmul_precision('high')
            self.model = torch.compile(self.model)
        except Exception:
            self.model = self._uncompiled_model

    def _restore_model(self) -> None:
        """Restore uncompiled model for inference (avoids FX/Dynamo conflicts)."""
        self.model = self._uncompiled_model

    # ------------------------------------------------------------------
    # Device logging
    # ------------------------------------------------------------------

    def _log_device(self, name: str) -> None:
        print(f"[{name}] Training on device: {self.device}")
        if self._is_xla:
            print(f"[{name}] TPU available via torch_xla")
        elif torch.cuda.is_available():
            print(f"[{name}] GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[{name}] GPU not available, using CPU")

    # ------------------------------------------------------------------
    # RNG generator
    # ------------------------------------------------------------------

    def _make_generator(self) -> torch.Generator:
        gen_device = 'cpu' if self._is_xla or not torch.cuda.is_available() else self.device
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(self.random_state)
        return gen

    # ------------------------------------------------------------------
    # AMP (Automatic Mixed Precision)
    # ------------------------------------------------------------------

    def _create_scaler(self) -> Optional[torch.amp.GradScaler]:
        """Create GradScaler for CUDA AMP. Returns None on CPU/XLA."""
        if self._amp_enabled:
            return torch.amp.GradScaler('cuda')
        return None

    def _xla_mark_step(self) -> None:
        """Call xm.mark_step() if running on XLA."""
        if self._is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()


class GroupedTorchPredictor(BaseTorchPredictor):
    """
    Base class for grouped MLP models that use (game_id, turn) groups
    with group-wise softmax and cross-entropy loss.

    Subclasses must implement:
    - _build_model(d_in) -> nn.Module
    - _forward_train(X_batch, mask_batch) -> logits (B, P)
    - _forward_inference(X_t, mask_t) -> logits (N, P)
    - get_model_summary() -> dict
    """

    REQUIRES_ID_COLUMNS = ['game_id', 'turn', 'player_id']

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        group_cols: Tuple[str, str] = ("game_id", "turn"),
        id_cols: Tuple[str, ...] = ("experiment", "game_id", "player_id", "turn"),
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        epochs: int = 10,
        batch_size_groups: int = 4096,
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
        self.group_cols = group_cols
        self.id_cols = id_cols
        self.batch_size_groups = batch_size_groups

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self, d_in: int) -> nn.Module:
        """Construct and return the nn.Module (already on self.device)."""
        ...

    @abstractmethod
    def _forward_train(self, X_batch: torch.Tensor, mask_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass during training. Returns (B, P) logits."""
        ...

    @abstractmethod
    def _forward_inference(self, X_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        """Forward pass during inference. Returns (N, P) logits."""
        ...

    @abstractmethod
    def _model_display_name(self) -> str:
        """Return display name for logging, e.g. 'GroupedMLP'."""
        ...

    # ------------------------------------------------------------------
    # Group building
    # ------------------------------------------------------------------

    def _build_groups(self, df: pd.DataFrame, y: pd.Series, raw_tp: Optional[pd.Series] = None) -> _BatchedGroups:
        """Convert row-wise data into padded, batched tensors for vectorized training."""
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

    # ------------------------------------------------------------------
    # Template fit()
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None):
        # 1) Filter features
        self._filter_features(X)

        missing = [c for c in self.REQUIRES_ID_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires columns {missing} in X. "
                f"These should be automatically injected by the evaluator. "
                f"If calling fit() directly, ensure X includes: {self.REQUIRES_ID_COLUMNS}"
            )

        self.selected_features_ = [c for c in self.selected_features_ if c not in self.id_cols]
        self.feature_names = list(self.selected_features_)

        # 2) Standardize
        Xmat = X[self.selected_features_].to_numpy(dtype=np.float32)
        Xmat = self._standardize_fit(Xmat)

        X_std = pd.DataFrame(Xmat, columns=self.selected_features_, index=X.index)
        X_std = pd.concat([X[["game_id", "turn", "player_id"]], X_std], axis=1)

        # 3) Build groups
        raw_tp = X["turn_progress"] if "turn_progress" in X.columns and self.loss_tp_alpha != 0 else None
        batched = self._build_groups(X_std, y, raw_tp=raw_tp)
        if batched.n_groups == 0:
            raise ValueError("No valid (game_id, turn) groups constructed. Check your data.")

        # 4) Create model
        d = len(self.selected_features_)
        self.model = self._build_model(d)
        self._compile_model()

        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        name = self._model_display_name()
        self._log_device(name)

        # 5) Move data to device
        X_all = torch.tensor(batched.X, dtype=torch.float32, device=self.device)
        y_all = torch.tensor(batched.y_indices, dtype=torch.long, device=self.device)
        mask_all = torch.tensor(batched.mask, device=self.device)
        tp_all = torch.tensor(batched.tp, dtype=torch.float32, device=self.device) if self.loss_tp_alpha != 0 else None

        # 6) Train loop
        self.model.train()
        gen = self._make_generator()
        scaler = self._create_scaler()
        n_batches = max(1, (batched.n_groups + self.batch_size_groups - 1) // self.batch_size_groups)

        for epoch in range(self.epochs):
            idx = torch.randperm(batched.n_groups, generator=gen, device=self.device)
            total_loss_t = torch.zeros(1, device=self.device)

            for start in range(0, batched.n_groups, self.batch_size_groups):
                batch_idx_t = idx[start:start + self.batch_size_groups]

                X_batch = X_all[batch_idx_t]
                y_batch = y_all[batch_idx_t]
                mask_batch = mask_all[batch_idx_t]

                opt.zero_grad()
                with torch.amp.autocast('cuda', enabled=self._amp_enabled):
                    logits = self._forward_train(X_batch, mask_batch)
                    logits = logits.masked_fill(~mask_batch, float('-inf'))

                    if tp_all is not None:
                        tp_batch = tp_all[batch_idx_t]
                        weight = tp_batch ** self.loss_tp_alpha
                        raw_loss = F.cross_entropy(logits, y_batch, reduction='none')
                        loss = (raw_loss * weight).mean()
                    else:
                        loss = F.cross_entropy(logits, y_batch)

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
            print(f"[{name}] epoch={epoch} loss={total_loss:.4f} groups={batched.n_groups}")

            if epoch_callback is not None and not epoch_callback(epoch, total_loss):
                break

        self._restore_model()
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_group_winrate(self, X: pd.DataFrame) -> pd.Series:
        """Return normalized winrates within each (game_id, turn)."""
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
            logits = self._forward_inference(X_t, mask_t)
            logits = logits.masked_fill(~mask_t, float('-inf'))
            pg = torch.softmax(logits, dim=1).cpu().numpy()

        probs = pg[gids, pos].astype(np.float32)
        return pd.Series(probs, index=X.index, name="p_win_group")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_win = self.predict_group_winrate(X).to_numpy()
        return np.column_stack([1.0 - p_win, p_win])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        p = pd.Series(proba[:, 1], index=X.index, name="p_win_group")
        preds = np.zeros(len(X), dtype=np.int64)

        winner_indices = p.groupby(
            [X[c] for c in self.group_cols], sort=False
        ).idxmax()

        idx_to_pos = pd.Series(np.arange(len(X)), index=X.index)
        preds[idx_to_pos[winner_indices].values] = 1
        return preds

    # ------------------------------------------------------------------
    # Feature importance (default: encoder-based)
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Default feature importance using encoder input weights.

        Works for models with self.model.encoder (InteractionMLP, AttentionMLP).
        Override for models with different architecture (e.g. GroupedMLP).
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        encoder = self.model.encoder
        layer_sizes = self._get_encoder_sizes()

        if len(layer_sizes) == 0:
            weights = encoder.net.weight.detach().cpu().numpy()
            importances = np.abs(weights).mean(axis=0)
        elif len(layer_sizes) == 1:
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
        return importance_df.sort_values('abs_coefficient', ascending=False)

    def _get_encoder_sizes(self) -> Tuple[int, ...]:
        """Return encoder layer sizes for feature importance. Override if needed."""
        return getattr(self, 'encoder_sizes', ())
