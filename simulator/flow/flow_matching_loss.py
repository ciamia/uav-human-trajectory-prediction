from __future__ import annotations

from collections.abc import Mapping as RuntimeMapping
from collections.abc import Sequence as RuntimeSequence
from typing import Mapping, Sequence, Union

import torch
import torch.nn.functional as F

Condition = Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]


def linear_interpolation(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return x_t and the constant target velocity for a linear path."""
    view_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
    t_view = t.view(view_shape)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0
    return x_t, target_velocity


def _model_velocity(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: Condition,
) -> torch.Tensor:
    if isinstance(condition, RuntimeMapping):
        return model(x_t, t, condition)
    if isinstance(condition, RuntimeSequence) and not isinstance(condition, torch.Tensor):
        return model(x_t, t, *condition)
    return model(x_t, t, condition)


def flow_matching_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    condition: Condition,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Conditional Flow Matching MSE loss for a linear interpolation path."""
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {x0.shape} and {x1.shape}")
    if t is None:
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)

    x_t, target_velocity = linear_interpolation(x0, x1, t)
    predicted_velocity = _model_velocity(model, x_t, t, condition)
    return F.mse_loss(predicted_velocity, target_velocity)
