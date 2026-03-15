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

import torch
import torch.nn as nn

from .base_torch_predictor import GroupedTorchPredictor
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


class AttentionMLPPredictor(GroupedTorchPredictor):
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
        encoder_sizes: Tuple[int, ...] = (98,) * 5,
        decoder_sizes: Tuple[int, ...] = (39,) * 4,
        num_heads: int = 2,
        n_attn_layers: int = 2,
        attn_dropout: float = 0.238415,
        dropout: float = 0.410835,
        lr: float = 0.000224245,
        weight_decay: float = 0.00528368,
        epochs: int = 22,
        loss_tp_alpha: float = 0.739902,
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
        self.num_heads = num_heads
        self.n_attn_layers = n_attn_layers
        self.attn_dropout = attn_dropout

    def _model_display_name(self) -> str:
        return "AttentionMLP"

    def _build_model(self, d_in: int) -> nn.Module:
        return _AttentionNet(
            d_in=d_in,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            dropout=self.dropout,
            num_heads=self.num_heads,
            n_attn_layers=self.n_attn_layers,
            attn_dropout=self.attn_dropout,
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
