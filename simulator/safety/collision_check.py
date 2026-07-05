from __future__ import annotations

import torch


def _positions(waypoints: torch.Tensor) -> torch.Tensor:
    if waypoints.ndim not in (2, 3):
        raise ValueError("waypoints must have shape [T, D] or [B, T, D]")
    if waypoints.shape[-1] < 3:
        raise ValueError("waypoints last dimension must contain at least x, y, z")
    return waypoints[..., :3]


def _batched_positions(waypoints: torch.Tensor) -> tuple[torch.Tensor, bool]:
    positions = _positions(waypoints)
    if positions.ndim == 2:
        return positions.unsqueeze(0), True
    return positions, False


def _batched_obstacles(obstacles: torch.Tensor, batch_size: int) -> torch.Tensor:
    if obstacles.ndim not in (2, 3) or obstacles.shape[-1] != 4:
        raise ValueError("obstacles must have shape [N, 4] or [B, N, 4]")
    if obstacles.ndim == 2:
        return obstacles.unsqueeze(0).expand(batch_size, -1, -1)
    if obstacles.shape[0] != batch_size:
        raise ValueError("obstacles batch dimension must match waypoints")
    return obstacles


def pairwise_sphere_distances(waypoints: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
    """Signed distance from each waypoint position to each spherical obstacle surface.

    Returns a tensor shaped [T, N] for unbatched input or [B, T, N] for batched input.
    Negative values are inside an obstacle.
    """
    positions, squeezed = _batched_positions(waypoints)
    spheres = _batched_obstacles(obstacles.to(device=positions.device, dtype=positions.dtype), positions.shape[0])

    if spheres.shape[1] == 0:
        distances = torch.empty(*positions.shape[:2], 0, device=positions.device, dtype=positions.dtype)
    else:
        center_distances = torch.linalg.norm(positions[:, :, None, :] - spheres[:, None, :, :3], dim=-1)
        distances = center_distances - spheres[:, None, :, 3]
    return distances.squeeze(0) if squeezed else distances


def min_distance_to_obstacles(waypoints: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
    """Minimum signed obstacle-surface distance for each waypoint position."""
    distances = pairwise_sphere_distances(waypoints, obstacles)
    if distances.shape[-1] == 0:
        return torch.full(distances.shape[:-1], torch.inf, device=distances.device, dtype=distances.dtype)
    return distances.min(dim=-1).values


def collision_mask(waypoints: torch.Tensor, obstacles: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Return True for waypoints closer than radius + margin to any obstacle."""
    return min_distance_to_obstacles(waypoints, obstacles) <= margin
