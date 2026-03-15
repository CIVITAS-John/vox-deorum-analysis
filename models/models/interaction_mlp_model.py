#!/usr/bin/env python3
"""
Interaction MLP predictor (DeepSets architecture) for real-time winrate.

Key idea:
- Group = (game_id, turn)
- Encode each player's features into an embedding (shared encoder)
- Pool embeddings across all players in the group (masked mean + max)
- Decode [player_embed, mean_context, max_context] → scalar logit per player (shared decoder)
- Softmax across players => winrate table

Unlike GroupedMLPPredictor where each player is scored independently,
this model allows cross-player interaction through the pooled context,
while remaining permutation equivariant by construction.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from .base_torch_predictor import GroupedTorchPredictor


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
    2. Pool: masked mean + max across players → context vectors
    3. Decoder: [player_embed, mean_context, max_context] → scalar logit (shared weights)
    """
    POOL_MODES = ('mean', 'max', 'mean+max')

    def __init__(
        self,
        d_in: int,
        encoder_sizes: Tuple[int, ...] = (64,),
        decoder_sizes: Tuple[int, ...] = (64,),
        dropout: float = 0.0,
        pool_mode: str = 'mean+max',
    ):
        super().__init__()
        if pool_mode not in self.POOL_MODES:
            raise ValueError(f"pool_mode must be one of {self.POOL_MODES}, got '{pool_mode}'")
        self.pool_mode = pool_mode
        # Embedding dimension = first encoder layer size (or d_in if no layers)
        self.embed_dim = encoder_sizes[0] if encoder_sizes else d_in
        n_pools = len(pool_mode.split('+'))

        self.encoder = _ResidualMLP(d_in, self.embed_dim, encoder_sizes, dropout)
        # Decoder input: player embedding + pooled context(s)
        self.decoder = _ResidualMLP(self.embed_dim * (1 + n_pools), 1, decoder_sizes, dropout)

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

        # 2. Pool across players
        mask_f = mask.unsqueeze(-1).float()  # (B, P, 1)
        pools = []
        if 'mean' in self.pool_mode:
            h_masked = h * mask_f
            pools.append(h_masked.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1))  # (B, H)
        if 'max' in self.pool_mode:
            h_for_max = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pools.append(h_for_max.max(dim=1).values)  # (B, H)

        # 3. Broadcast and concat: [player_embed, pool_context(s)]
        parts = [h]
        for pool in pools:
            parts.append(pool.unsqueeze(1).expand(-1, P, -1))
        combined = torch.cat(parts, dim=-1)  # (B, P, (1+n_pools)*H)

        # 4. Decode to scalar logit per player
        flat_combined = combined.reshape(B * P, -1)
        logits = self.decoder(flat_combined).reshape(B, P)  # (B, P)

        return logits


class InteractionMLPPredictor(GroupedTorchPredictor):
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
        'science_adj', 'culture_adj', 'tourism_adj', 'gold_adj',
        'food_adj', 'production_adj', 'military_adj', 'faith_adj',
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
    REQUIRED_FEATURES = None

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        group_cols: Tuple[str, str] = ("game_id", "turn"),
        id_cols: Tuple[str, ...] = ("experiment", "game_id", "player_id", "turn"),
        pool_mode: str = 'mean+max',
        encoder_sizes: Tuple[int, ...] = (66,) * 10,
        decoder_sizes: Tuple[int, ...] = (107,),
        dropout: float = 0.0228996,
        lr: float = 0.000154002,
        weight_decay: float = 0.00325868,
        epochs: int = 14,
        loss_tp_alpha: float = 0.778462,
        batch_size_groups: int = 4096,
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
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.pool_mode = pool_mode

    def _model_display_name(self) -> str:
        return "InteractionMLP"

    def _build_model(self, d_in: int) -> nn.Module:
        return _DeepSetsNet(
            d_in=d_in,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            dropout=self.dropout,
            pool_mode=self.pool_mode,
        ).to(self.device)

    def _forward_train(self, X_batch: torch.Tensor, mask_batch: torch.Tensor) -> torch.Tensor:
        return self.model(X_batch, mask_batch)

    def _forward_inference(self, X_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        return self.model(X_t, mask_t)

    def _get_encoder_sizes(self) -> Tuple[int, ...]:
        return self.encoder_sizes

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
            "pool_mode": self.pool_mode,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "loss_tp_alpha": self.loss_tp_alpha,
            "device": self.device,
        }
