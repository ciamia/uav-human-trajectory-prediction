from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.serving_dataset import ServingTrajectoryDataset
from flow.flow_matching_loss import linear_interpolation
from flow.sampler import sample
from flow.trajectory_losses import TrajectoryLossWeights, trajectory_objective_losses
from models.flow_transformer import FlowTransformer


@dataclass(frozen=True)
class TrainServingConfig:
    """Configuration for Milestone 7 serving Flow Matching training."""

    dataset_path: str = "outputs/serving_dataset.npz"
    output_dir: str = "outputs/serving_fm"
    seed: int = 0
    device: str = "auto"
    epochs: int = 5
    batch_size: int = 16
    val_fraction: float = 0.2
    lr: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    cfm_weight: float = 1.0
    objective_weight: float = 0.05
    checkpoint_every: int = 1
    sample_every: int = 1
    sample_steps: int = 8
    sample_method: str = "heun"


def train(config: TrainServingConfig) -> dict[str, Any]:
    """Train the Milestone 5 model on a Milestone 4 serving dataset."""
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataset = ServingTrajectoryDataset(config.dataset_path)
    train_dataset, val_dataset = split_dataset(dataset, config.val_fraction, config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    horizon = int(dataset.data["expert_trajectory"].shape[1])
    model = build_model(config, horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    log_path = output_dir / "loss_log.jsonl"
    best_val = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, config, train=True)
        val_metrics = run_epoch(model, val_loader, None, device, config, train=False, deterministic=True)
        row = {"epoch": epoch, **_prefix("train", train_metrics), **_prefix("val", val_metrics)}
        history.append(row)
        append_jsonl(log_path, row)
        print(format_metrics(row))

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, config, epoch, row)
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, config, epoch, row)
        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", model, optimizer, config, epoch, row)
        if config.sample_every > 0 and epoch % config.sample_every == 0:
            save_training_sample(model, val_loader, device, config, sample_dir / f"sample_epoch_{epoch:04d}.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "history": history,
        "output_dir": str(output_dir),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "latest_checkpoint": str(checkpoint_dir / "latest.pt"),
    }


def run_epoch(
    model: FlowTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: TrainServingConfig,
    train: bool,
    deterministic: bool = False,
) -> dict[str, float]:
    """Run one train or validation epoch and return mean losses."""
    model.train(train)
    totals: dict[str, float] = {}
    batches = 0
    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in iterator:
        batch = batch_to_device(batch, device)
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            losses = batch_losses(model, batch, config, noise_seed=config.seed + batches if deterministic else None)
            if train and optimizer is not None:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1
        iterator.set_postfix(total=float(losses["total"].detach().cpu()))
    if batches == 0:
        raise RuntimeError("training loader produced no batches")
    return {key: value / batches for key, value in totals.items()}


def batch_losses(
    model: FlowTransformer,
    batch: dict[str, torch.Tensor],
    config: TrainServingConfig,
    noise_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute CFM and soft trajectory objective losses for one batch."""
    x1 = batch["expert_trajectory"][..., :6]
    if noise_seed is None:
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    else:
        generator = torch.Generator(device=x1.device).manual_seed(int(noise_seed))
        x0 = torch.randn(x1.shape, generator=generator, device=x1.device, dtype=x1.dtype)
        t = torch.rand(x1.shape[0], generator=generator, device=x1.device, dtype=x1.dtype)
    x_t, target_velocity = linear_interpolation(x0, x1, t)
    condition = batch_condition(batch)
    predicted_velocity = model(x_t, t, condition)
    cfm = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)

    t_view = t.view(x1.shape[0], *([1] * (x1.ndim - 1)))
    predicted_final = x_t + (1.0 - t_view) * predicted_velocity
    objective_losses = trajectory_objective_losses(predicted_final, condition)
    objective_total = objective_losses["total"]
    total = config.cfm_weight * cfm + config.objective_weight * objective_total
    if not torch.isfinite(total):
        raise RuntimeError("non-finite training loss")

    return {
        "total": total,
        "cfm": cfm,
        "objective": objective_total,
        "goal": objective_losses["goal"],
        "safe_hover": objective_losses["safe_hover"],
        "smoothness": objective_losses["smoothness"],
        "path_time": objective_losses["path_time"],
        "building_collision": objective_losses["building_collision"],
        "people_collision": objective_losses["people_collision"],
        "altitude": objective_losses["altitude"],
    }


def build_model(config: TrainServingConfig, horizon: int) -> FlowTransformer:
    """Construct the serving Flow Transformer."""
    return FlowTransformer(
        state_dim=6,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_horizon=horizon,
    )


def batch_condition(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract model condition tensors from a serving dataset batch."""
    return {
        "drone_start": batch["drone_start"],
        "hover_goal": batch["hover_goal"],
        "target_person_state": batch["target_person_state"],
        "future_people_predictions": batch["future_people_predictions"],
        "building_geometry": batch["building_geometry"],
        "command_embedding": batch["command_embedding"],
    }


@torch.no_grad()
def save_training_sample(
    model: FlowTransformer,
    loader: DataLoader,
    device: torch.device,
    config: TrainServingConfig,
    path: Path,
) -> None:
    """Periodically sample trajectories from validation conditions."""
    model.eval()
    batch = batch_to_device(next(iter(loader)), device)
    target = batch["expert_trajectory"][..., :6]
    noise = torch.randn_like(target)
    generated = sample(model, noise, batch_condition(batch), steps=config.sample_steps, method=config.sample_method)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generated": generated.detach().cpu(),
            "target": target.detach().cpu(),
            "condition": {key: value.detach().cpu() for key, value in batch_condition(batch).items()},
        },
        path,
    )


def split_dataset(dataset: ServingTrajectoryDataset, val_fraction: float, seed: int) -> tuple[Any, Any]:
    """Split a saved dataset into train and validation subsets."""
    val_count = max(1, int(round(len(dataset) * float(val_fraction))))
    train_count = len(dataset) - val_count
    if train_count <= 0:
        train_count, val_count = len(dataset), len(dataset)
        return dataset, dataset
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_count, val_count], generator=generator)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a DataLoader batch to the requested device."""
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
    return moved


def save_checkpoint(
    path: Path,
    model: FlowTransformer,
    optimizer: torch.optim.Optimizer,
    config: TrainServingConfig,
    epoch: int,
    metrics: dict[str, float | int],
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def resolve_device(name: str) -> torch.device:
    """Resolve a configured torch device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def append_jsonl(path: Path, row: dict[str, float | int]) -> None:
    """Append one JSON metrics row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def format_metrics(row: dict[str, float | int]) -> str:
    """Format a compact epoch metrics line."""
    return (
        f"epoch={int(row['epoch'])} "
        f"train_total={float(row['train_total']):.6f} "
        f"val_total={float(row['val_total']):.6f} "
        f"train_cfm={float(row['train_cfm']):.6f} "
        f"val_cfm={float(row['val_cfm']):.6f}"
    )


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def load_config(path: str | Path | None) -> TrainServingConfig:
    """Load training config from YAML, accepting either flat or training-serving sections."""
    if path is None:
        return TrainServingConfig()
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    values = dict(raw.get("serving_training", raw.get("training", raw)))
    allowed = TrainServingConfig.__dataclass_fields__.keys()
    filtered = {key: value for key, value in values.items() if key in allowed}
    if "dataset_path" not in filtered and "serving_dataset_path" in raw:
        filtered["dataset_path"] = raw["serving_dataset_path"]
    return TrainServingConfig(**filtered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    updates = {}
    if args.dataset is not None:
        updates["dataset_path"] = args.dataset
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.epochs is not None:
        updates["epochs"] = args.epochs
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if updates:
        config = TrainServingConfig(**{**asdict(config), **updates})
    train(config)


if __name__ == "__main__":
    main()
