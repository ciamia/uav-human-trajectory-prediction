from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from models.condition_encoder import ConditionEncoder


class FlowTransformer(nn.Module):
    """Transformer velocity field for serving-trajectory Conditional Flow Matching."""

    def __init__(
        self,
        state_dim: int = 6,
        obstacle_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_horizon: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.max_horizon = max_horizon

        self.trajectory_embedding = nn.Linear(state_dim, hidden_dim)
        self.condition_encoder = ConditionEncoder(hidden_dim=hidden_dim, state_dim=state_dim, obstacle_dim=obstacle_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_horizon, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        time: torch.Tensor,
        condition: Mapping[str, torch.Tensor] | torch.Tensor,
        goal: torch.Tensor | None = None,
        obstacles: torch.Tensor | None = None,
        **extra_condition: Any,
    ) -> torch.Tensor:
        """Predict velocity field with shape [B, T, 6].

        The preferred Milestone 5 API is ``model(x_t, t, condition_dict)``.
        The legacy ``model(x_t, t, start, goal, obstacles)`` form is preserved
        for earlier scripts and tests.
        """
        if noisy_trajectory.ndim != 3:
            raise ValueError("noisy_trajectory must have shape [B, T, D]")
        if time.ndim != 1:
            raise ValueError("time must have shape [B]")

        batch, horizon, state_dim = noisy_trajectory.shape
        if state_dim != self.state_dim:
            raise ValueError(f"expected trajectory state_dim={self.state_dim}, got {state_dim}")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max_horizon {self.max_horizon}")
        if time.shape[0] != batch:
            raise ValueError("time batch dimension must match noisy_trajectory")

        condition_dict = self._condition_dict(condition, goal, obstacles, extra_condition)
        context = self.condition_encoder(time, condition_dict).unsqueeze(1)

        tokens = self.trajectory_embedding(noisy_trajectory)
        tokens = tokens + self.position_embedding[:, :horizon] + context
        encoded = self.encoder(tokens)
        return self.velocity_head(self.output_norm(encoded))

    def _condition_dict(
        self,
        condition: Mapping[str, torch.Tensor] | torch.Tensor,
        goal: torch.Tensor | None,
        obstacles: torch.Tensor | None,
        extra_condition: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if isinstance(condition, Mapping):
            merged = dict(condition)
            merged.update(extra_condition)
            return merged
        if goal is None or obstacles is None:
            raise ValueError("legacy FlowTransformer calls require start, goal, and obstacles")
        if condition.ndim != 2 or goal.ndim != 2:
            raise ValueError("start and goal must have shape [B, D]")
        if obstacles.ndim != 3 or obstacles.shape[-1] != 4:
            raise ValueError("obstacles must have shape [B, N, 4]")
        return {"start": condition, "goal": goal, "obstacles": obstacles}
