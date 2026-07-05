from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.synthetic_dataset import SyntheticDroneTrajectoryDataset
from flow.flow_matching_loss import flow_matching_loss
from flow.sampler import sample as sample_ode
from env.genesis_sim import GenesisSimBackend, GenesisSimConfig
from models.flow_transformer import FlowTransformer


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_dataset(cfg: dict[str, Any]) -> SyntheticDroneTrajectoryDataset:
    return SyntheticDroneTrajectoryDataset.from_config(
        cfg["dataset"],
        sim_config=cfg.get("sim"),
        planning_config=cfg.get("planning"),
    )


def build_model(cfg: dict[str, Any], horizon: int) -> FlowTransformer:
    return FlowTransformer(
        state_dim=cfg["state_dim"],
        obstacle_dim=cfg["obstacle_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        max_horizon=max(horizon, cfg["max_horizon"]),
    )


def make_genesis_validator(cfg: dict[str, Any]) -> GenesisSimBackend | None:
    sim_cfg = cfg.get("sim", {})
    if sim_cfg.get("backend") != "genesis":
        return None
    return GenesisSimBackend(
        GenesisSimConfig(
            dt=float(sim_cfg.get("dt", 0.02)),
            show_viewer=False,
            add_ground=True,
            drone_radius=float(sim_cfg.get("drone_radius", 0.15)),
            auto_build=False,
            launch_scene=False,
        )
    )


@torch.no_grad()
def validate_collision_rate(
    model: FlowTransformer,
    loader: DataLoader,
    backend: GenesisSimBackend,
    device: torch.device,
    steps: int,
    method: str,
    max_batches: int = 1,
) -> float:
    model.eval()
    collisions = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        start = batch["start"].to(device)
        goal = batch["goal"].to(device)
        obstacles = batch["obstacles"].to(device)
        condition = {"start": start, "goal": goal, "obstacles": obstacles}
        noise = torch.randn_like(batch["trajectory"].to(device))
        generated = sample_ode(model, noise, condition, steps=steps, method=method).cpu().numpy()
        starts = batch["start"].numpy()
        goals = batch["goal"].numpy()
        obstacle_batch = batch["obstacles"].numpy()
        for idx, trajectory in enumerate(generated):
            trajectory[0] = starts[idx]
            trajectory[-1] = goals[idx]
            backend.reset(starts[idx], goals[idx], obstacle_batch[idx])
            rollout = backend.rollout_trajectory(trajectory)
            collisions += int(rollout["collision"])
            total += 1
    model.train()
    return collisions / max(total, 1)


def save_checkpoint(
    path: Path,
    model: FlowTransformer,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epoch: int,
    step: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def train(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    torch.manual_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])

    dataset = build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        drop_last=False,
    )
    validation_loader = DataLoader(
        dataset,
        batch_size=min(cfg["training"]["batch_size"], cfg["training"].get("validation_batch_size", 16)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = build_model(cfg["model"], horizon=dataset.config.horizon).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    output_dir = Path(cfg["training"]["output_dir"])
    checkpoint_dir = Path(cfg["training"].get("checkpoint_dir", output_dir / "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_loss.csv"

    global_step = 0
    best_loss = float("inf")
    epochs = int(cfg["training"]["epochs"])
    checkpoint_every = int(cfg["training"]["checkpoint_every"])
    log_every = int(cfg["training"]["log_every"])
    validate_every = int(cfg["training"].get("validate_collision_every", 0))
    validation_batches = int(cfg["training"].get("validation_batches", 1))
    genesis_validator = make_genesis_validator(cfg)

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "step", "loss", "collision_rate"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}")

            for batch in progress:
                x1 = batch["trajectory"].to(device)
                x0 = torch.randn_like(x1)
                condition = {
                    "start": batch["start"].to(device),
                    "goal": batch["goal"].to(device),
                    "obstacles": batch["obstacles"].to(device),
                }

                loss = flow_matching_loss(model, x0=x0, x1=x1, condition=condition)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg["training"]["grad_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                optimizer.step()

                global_step += 1
                loss_value = float(loss.detach().cpu())
                running_loss += loss_value
                progress.set_postfix(loss=f"{loss_value:.5f}")

                if global_step % log_every == 0:
                    writer.writerow({"epoch": epoch, "step": global_step, "loss": loss_value, "collision_rate": ""})
                    log_file.flush()

            mean_loss = running_loss / max(len(loader), 1)
            writer.writerow({"epoch": epoch, "step": global_step, "loss": mean_loss, "collision_rate": ""})
            log_file.flush()
            print(f"epoch={epoch} mean_loss={mean_loss:.6f}")

            if validate_every > 0 and genesis_validator is not None and epoch % validate_every == 0:
                collision_rate = validate_collision_rate(
                    model,
                    validation_loader,
                    genesis_validator,
                    device=device,
                    steps=int(cfg.get("sampling", {}).get("steps", 20)),
                    method=cfg.get("sampling", {}).get("method", "heun"),
                    max_batches=validation_batches,
                )
                writer.writerow(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": mean_loss,
                        "collision_rate": collision_rate,
                    }
                )
                log_file.flush()
                print(f"epoch={epoch} genesis_collision_rate={collision_rate:.4f}")

            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                step=global_step,
                loss=mean_loss,
            )
            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    step=global_step,
                    loss=mean_loss,
                )
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    step=global_step,
                    loss=mean_loss,
                )

    return checkpoint_dir / "latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    latest = train(args.config)
    print(f"saved latest checkpoint to {latest}")


if __name__ == "__main__":
    main()
