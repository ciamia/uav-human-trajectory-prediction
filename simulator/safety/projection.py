from __future__ import annotations

import torch

from safety.collision_check import _batched_obstacles, _batched_positions, collision_mask


def _fallback_directions(shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    directions = torch.zeros(*shape, 3, device=device, dtype=dtype)
    directions[..., 2] = 1.0
    return directions


def project_waypoints(
    waypoints: torch.Tensor,
    obstacles: torch.Tensor,
    margin: float = 0.0,
    max_iterations: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Push colliding waypoint positions outside obstacle radius plus margin.

    The first three state dimensions are projected. Any remaining dimensions,
    such as velocities in [x, y, z, vx, vy, vz], are preserved.
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    positions, squeezed = _batched_positions(waypoints)
    spheres = _batched_obstacles(obstacles.to(device=positions.device, dtype=positions.dtype), positions.shape[0])
    projected = waypoints.clone()
    projected_positions = projected[..., :3].unsqueeze(0) if squeezed else projected[..., :3]

    if spheres.shape[1] == 0:
        return projected

    for _ in range(max_iterations):
        offsets = projected_positions[:, :, None, :] - spheres[:, None, :, :3]
        center_distances = torch.linalg.norm(offsets, dim=-1)
        clearances = center_distances - spheres[:, None, :, 3]
        closest = clearances.argmin(dim=-1)
        closest_clearance = clearances.gather(-1, closest[..., None]).squeeze(-1)
        needs_projection = closest_clearance <= margin
        if not bool(needs_projection.any()):
            break

        closest_spheres = spheres.gather(1, closest[..., None].expand(-1, -1, 4))
        closest_centers = closest_spheres[..., :3]
        closest_radii = closest_spheres[..., 3]
        directions = projected_positions - closest_centers
        norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
        safe_directions = torch.where(
            norms > eps,
            directions / norms.clamp_min(eps),
            _fallback_directions(norms.shape[:-1], projected_positions.device, projected_positions.dtype),
        )
        target_distances = closest_radii + margin + eps
        updated_positions = closest_centers + safe_directions * target_distances[..., None]
        projected_positions = torch.where(needs_projection[..., None], updated_positions, projected_positions)

    if squeezed:
        projected[..., :3] = projected_positions.squeeze(0)
    else:
        projected[..., :3] = projected_positions
    return projected


def project_collisions(
    waypoints: torch.Tensor,
    obstacles: torch.Tensor,
    margin: float = 0.0,
    max_iterations: int = 4,
) -> torch.Tensor:
    """Alias for project_waypoints kept readable at call sites."""
    projected = project_waypoints(waypoints, obstacles, margin=margin, max_iterations=max_iterations)
    _ = collision_mask(projected, obstacles, margin=margin)
    return projected
