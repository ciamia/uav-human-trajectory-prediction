from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Create a small MLP used for condition embeddings."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ConditionEncoder(nn.Module):
    """Encode Milestone 5 serving-task conditions into one context vector."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 6,
        hover_goal_dim: int = 4,
        person_state_dim: int = 6,
        people_prediction_dim: int = 3,
        building_dim: int = 6,
        command_dim: int = 2,
        obstacle_dim: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.hover_goal_dim = hover_goal_dim
        self.person_state_dim = person_state_dim
        self.people_prediction_dim = people_prediction_dim
        self.building_dim = building_dim
        self.command_dim = command_dim
        self.obstacle_dim = obstacle_dim

        self.time_embedding = make_mlp(1, hidden_dim, hidden_dim)
        self.drone_start_embedding = make_mlp(state_dim, hidden_dim, hidden_dim)
        self.hover_goal_embedding = make_mlp(hover_goal_dim, hidden_dim, hidden_dim)
        self.goal_state_embedding = make_mlp(state_dim, hidden_dim, hidden_dim)
        self.target_person_embedding = make_mlp(person_state_dim, hidden_dim, hidden_dim)
        self.people_prediction_embedding = make_mlp(people_prediction_dim, hidden_dim, hidden_dim)
        self.building_embedding = make_mlp(building_dim, hidden_dim, hidden_dim)
        self.legacy_obstacle_embedding = make_mlp(obstacle_dim, hidden_dim, hidden_dim)
        self.command_embedding = make_mlp(command_dim, hidden_dim, hidden_dim)
        self.fusion = make_mlp(hidden_dim * 6, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, time: torch.Tensor, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return encoded condition context with shape [B, hidden_dim]."""
        if time.ndim != 1:
            raise ValueError("time must have shape [B]")
        batch = time.shape[0]
        device = time.device
        dtype = time.dtype

        drone_start = _required_matrix(condition, ("drone_start", "start"), batch, self.state_dim, device, dtype)
        goal = _required_matrix(condition, ("hover_goal", "goal"), batch, None, device, dtype)
        target_person = _optional_matrix(condition, ("target_person_state", "target_person"), batch, self.person_state_dim, device, dtype)
        people_predictions = _optional_sequence_context(
            condition,
            ("future_people_predictions", "people_predictions"),
            batch,
            self.people_prediction_dim,
            device,
            dtype,
        )
        buildings = _optional_sequence_context(
            condition,
            ("building_geometry", "buildings"),
            batch,
            self.building_dim,
            device,
            dtype,
        )
        obstacles = _optional_sequence_context(condition, ("obstacles",), batch, self.obstacle_dim, device, dtype)
        command = _optional_matrix(condition, ("command_embedding", "command"), batch, self.command_dim, device, dtype)

        if goal.shape[-1] == self.hover_goal_dim:
            goal_context = self.hover_goal_embedding(goal)
        elif goal.shape[-1] == self.state_dim:
            goal_context = self.goal_state_embedding(goal)
        else:
            raise ValueError(f"hover_goal/goal must have last dim {self.hover_goal_dim} or {self.state_dim}")

        building_context = self.building_embedding(buildings)
        if _has_key(condition, ("obstacles",)):
            building_context = building_context + self.legacy_obstacle_embedding(obstacles)

        parts = torch.cat(
            [
                self.time_embedding(time[:, None]),
                self.drone_start_embedding(drone_start),
                goal_context,
                self.target_person_embedding(target_person),
                self.people_prediction_embedding(people_predictions),
                building_context + self.command_embedding(command),
            ],
            dim=-1,
        )
        return self.norm(self.fusion(parts))


def _has_key(condition: Mapping[str, torch.Tensor], names: tuple[str, ...]) -> bool:
    return any(name in condition for name in names)


def _first(condition: Mapping[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        if name in condition:
            return condition[name]
    return None


def _as_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=dtype)


def _required_matrix(
    condition: Mapping[str, torch.Tensor],
    names: tuple[str, ...],
    batch: int,
    dim: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = _first(condition, names)
    if value is None:
        raise ValueError(f"condition must include one of {names}")
    tensor = _as_tensor(value, device, dtype)
    if tensor.ndim != 2 or tensor.shape[0] != batch:
        raise ValueError(f"{names[0]} must have shape [B, D]")
    if dim is not None and tensor.shape[-1] != dim:
        raise ValueError(f"{names[0]} must have last dim {dim}")
    return tensor


def _optional_matrix(
    condition: Mapping[str, torch.Tensor],
    names: tuple[str, ...],
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = _first(condition, names)
    if value is None:
        return torch.zeros(batch, dim, device=device, dtype=dtype)
    tensor = _as_tensor(value, device, dtype)
    if tensor.ndim != 2 or tensor.shape != (batch, dim):
        raise ValueError(f"{names[0]} must have shape [B, {dim}]")
    return tensor


def _optional_sequence_context(
    condition: Mapping[str, torch.Tensor],
    names: tuple[str, ...],
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = _first(condition, names)
    if value is None:
        return torch.zeros(batch, dim, device=device, dtype=dtype)
    tensor = _as_tensor(value, device, dtype)
    if tensor.ndim < 3 or tensor.shape[0] != batch or tensor.shape[-1] != dim:
        raise ValueError(f"{names[0]} must have shape [B, ..., {dim}]")
    return tensor.reshape(batch, -1, dim).mean(dim=1)
