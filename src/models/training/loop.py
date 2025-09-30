"""
Training loop with full-epoch traversal and TensorBoard logging.

Phase 0: no early stopping; however, track and save best checkpoint by val loss.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
try:
    # Prefer new torch.amp (PyTorch 2.x)
    import torch.amp as torch_amp  # type: ignore[attr-defined]
    _HAS_TORCH_AMP = hasattr(torch_amp, "autocast") and hasattr(torch_amp, "GradScaler")
except Exception:
    torch_amp = None  # type: ignore[assignment]
    _HAS_TORCH_AMP = False
if not _HAS_TORCH_AMP:
    # Fallback to legacy CUDA AMP (PyTorch 1.x/early 2.x)
    try:
        from torch.cuda.amp import autocast as cuda_autocast, GradScaler as CudaGradScaler  # type: ignore
    except Exception:
        cuda_autocast = None  # type: ignore
        CudaGradScaler = None  # type: ignore


@dataclass
class TrainRunConfig:
    epochs: int = 20000
    print_freq: int = 2000
    grad_clip: float = 1.0
    # If >0, save last checkpoint every N epochs; 0 = every epoch
    ckpt_interval: int = 0
    # Enable AMP mixed precision on CUDA
    amp: bool = False
    # Validate every N epochs (1 = each epoch)
    val_interval: int = 1


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 2000
    min_delta: float = 0.0


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    warmup_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    run_dir: Path,
    writer: SummaryWriter,
    cfg: TrainRunConfig,
    early_stopping: Optional[EarlyStoppingConfig] = None,
    batch_noise_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    batch_pair_transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> dict:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val: float = float("inf")
    no_improve_epochs: int = 0

    global_step = 0
    use_amp = bool(cfg.amp and torch.cuda.is_available() and device.type == "cuda")
    if use_amp and _HAS_TORCH_AMP:
        scaler = torch_amp.GradScaler("cuda", enabled=True)  # type: ignore[attr-defined]
    elif use_amp and not _HAS_TORCH_AMP and CudaGradScaler is not None:
        scaler = CudaGradScaler(enabled=True)  # type: ignore[call-arg]
    else:
        scaler = None  # type: ignore[assignment]
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for xb, yb in train_loader:

            optimizer.zero_grad(set_to_none=True)
            # Optional pairwise batch transform (e.g., flip both X and Y)
            if batch_pair_transform is not None:
                xb, yb = batch_pair_transform(xb, yb)
            # Optional noise injector on inputs
            if batch_noise_fn is not None:
                xb = batch_noise_fn(xb)

            if use_amp:
                if _HAS_TORCH_AMP:
                    ctx = torch_amp.autocast("cuda", dtype=torch.float16, enabled=True)  # type: ignore[attr-defined]
                else:
                    ctx = cuda_autocast(dtype=torch.float16, enabled=True)  # type: ignore[misc]
                with ctx:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                assert scaler is not None
                scaler.scale(loss).backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1
            global_step += 1

        # End epoch: optionally evaluate on val
        do_val = (cfg.val_interval is None) or (cfg.val_interval <= 1) or (epoch % int(cfg.val_interval) == 0)
        val_loss = float("nan")
        model.eval()
        if do_val:
            with torch.no_grad():
                val_loss_sum = 0.0
                val_batches = 0
                for xb, yb in val_loader:
                    if use_amp:
                        if _HAS_TORCH_AMP:
                            ctx = torch_amp.autocast("cuda", dtype=torch.float16, enabled=True)  # type: ignore[attr-defined]
                        else:
                            ctx = cuda_autocast(dtype=torch.float16, enabled=True)  # type: ignore[misc]
                        with ctx:
                            pred = model(xb)
                            vloss = criterion(pred, yb)
                    else:
                        pred = model(xb)
                        vloss = criterion(pred, yb)
                    val_loss_sum += float(vloss.item())
                    val_batches += 1
            val_loss = val_loss_sum / max(1, val_batches)

        train_loss = train_loss_sum / max(1, train_batches)

        # Scheduler step: warmup (if any) then main scheduler
        import torch.optim.lr_scheduler as lrs
        if warmup_scheduler is not None:
            try:
                warmup_scheduler.step()
            finally:
                # When LinearLR reaches total_iters, it keeps returning last factor; we null it after warmup_steps epochs
                if hasattr(warmup_scheduler, "last_epoch") and hasattr(warmup_scheduler, "total_iters"):
                    if warmup_scheduler.last_epoch >= getattr(warmup_scheduler, "total_iters", 0):
                        warmup_scheduler = None  # type: ignore[assignment]

        if scheduler is not None:
            if isinstance(scheduler, lrs.ReduceLROnPlateau):
                # Step only when we computed a new validation loss
                if do_val:
                    scheduler.step(val_loss)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        writer.add_scalar("loss/train", train_loss, epoch)
        if do_val:
            writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)

        if epoch == 1 or epoch % cfg.print_freq == 0:
            val_str = f"{val_loss:.4e}" if do_val else "NA"
            print(f"Epoch [{epoch:5d}/{cfg.epochs}] | Train MSE: {train_loss:.4e} | Val MSE: {val_str} | LR: {lr:.4e}")

        # Save last checkpoint based on interval
        save_last = True
        if getattr(cfg, "ckpt_interval", 0):
            interval = max(1, int(cfg.ckpt_interval))
            save_last = (epoch % interval == 0)
        if save_last:
            torch.save({"model_state_dict": model.state_dict()}, ckpt_dir / "last.pth")

        # Save best checkpoint
        if do_val:
            if val_loss < best_val - (early_stopping.min_delta if early_stopping else 0.0):
                best_val = val_loss
                torch.save({"model_state_dict": model.state_dict()}, ckpt_dir / "best.pth")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

        # Early stopping check
        if early_stopping and early_stopping.enabled and no_improve_epochs >= early_stopping.patience:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {no_improve_epochs} epochs)."
            )
            break

    # Write history.csv
    hist_csv = run_dir / "history.csv"
    with hist_csv.open("w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "val_loss", "lr"])
        for e, tr, va, l in zip(history["epoch"], history["train_loss"], history["val_loss"], history["lr"]):
            writer_csv.writerow([e, tr, va, l])

    writer.flush()
    return history
