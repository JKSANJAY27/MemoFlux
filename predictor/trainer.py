"""
predictor/trainer.py — Training Loop with TopK-CE Loss

Loss: CE + rank-penalty (ensures true label in top-3, not just top-1).
Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2).
Optimizer: AdamW (lr=3e-3, weight_decay=1e-4).
Early stopping: patience=7 on val HR@3.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .evaluator import evaluate

# ── Loss ──────────────────────────────────────────────────

class TopKCrossEntropyLoss(nn.Module):
    """
    Standard CE + rank-penalty that extra-penalises when
    true label falls outside top-K predictions.

    rank_penalty = mean(max(0, rank_of_true - K + 1))
    total_loss   = CE + α * rank_penalty
    """

    def __init__(self, k: int = 3, alpha: float = 0.3):
        super().__init__()
        self.k     = k
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, labels)

        # Rank of the true label (0-indexed)
        _, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        # Find column index of true label in sorted output
        match  = (sorted_idx == labels.unsqueeze(1))              # (B, V) bool
        ranks  = match.float().argmax(dim=1).float()               # (B,) 0-indexed
        penalty = torch.clamp(ranks - self.k + 1, min=0.0).mean()

        return ce_loss + self.alpha * penalty


# ── One-epoch train ────────────────────────────────────────

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device: str,
    scaler=None,
) -> float:
    model.train()
    total_loss  = 0.0
    total_steps = 0

    for batch in dataloader:
        app_ids  = batch["app_ids"].to(device)
        ctx_vecs = batch["ctx_vecs"].to(device)
        user_ids = batch["user_id"].to(device)
        labels   = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _ = model(app_ids, ctx_vecs, user_ids)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(app_ids, ctx_vecs, user_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss  += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


# ── Main training function ─────────────────────────────────

def train(
    model,
    train_dl,
    val_dl,
    device:          str   = "cpu",
    n_epochs:        int   = 50,
    lr:              float = 3e-3,
    weight_decay:    float = 1e-4,
    patience:        int   = 7,
    checkpoint_dir:  str   = "checkpoints",
    use_amp:         bool  = False,     # AMP only meaningful on CUDA
    verbose:         bool  = True,
) -> Dict:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=weight_decay, betas=(0.9, 0.98),
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = TopKCrossEntropyLoss(k=3, alpha=0.3)
    scaler    = (torch.cuda.amp.GradScaler()
                 if (use_amp and torch.cuda.is_available()) else None)

    history: Dict = {
        "train_loss": [], "val_hr1": [], "val_hr3": [],
        "val_mrr3": [], "lr": [],
    }
    best_hr3        = 0.0
    patience_counter= 0

    try:
        from rich.console import Console
        from rich.table   import Table
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    if verbose:
        print(f"\nTraining ContextAwareLSTM — {model.count_parameters():,} params")
        print(f"Device: {device}  |  Train batches: {len(train_dl)}  |  Epochs: {n_epochs}\n")

    for epoch in range(1, n_epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device, scaler)
        val_m      = evaluate(model, val_dl, device)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 5))
        history["val_hr1"].append(round(val_m["hr1"], 5))
        history["val_hr3"].append(round(val_m["hr3"], 5))
        history["val_mrr3"].append(round(val_m["mrr3"], 5))
        history["lr"].append(round(optimizer.param_groups[0]["lr"], 8))

        elapsed = time.time() - t0

        # Console output every 5 epochs or on improvement
        if verbose and (epoch % 5 == 0 or epoch == 1 or val_m["hr3"] > best_hr3):
            line = (f"Ep {epoch:3d}/{n_epochs}  loss={train_loss:.4f}  "
                    f"HR@1={val_m['hr1']:.1%}  HR@3={val_m['hr3']:.1%}"
                    f"{'  ← BEST ✓' if val_m['hr3'] > best_hr3 else ''}  "
                    f"({elapsed:.1f}s)")
            print(line)

        # Checkpoint
        if val_m["hr3"] > best_hr3:
            best_hr3 = val_m["hr3"]
            patience_counter = 0
            torch.save(
                {
                    "epoch":             epoch,
                    "model_state_dict":  model.state_dict(),
                    "optimizer_state":   optimizer.state_dict(),
                    "val_hr3":           best_hr3,
                    "val_metrics":       val_m,
                    "train_loss":        train_loss,
                    "model_config": {
                        "vocab_size":    model.vocab_size,
                        "n_users":       model.user_profile.embedding.num_embeddings - 1,
                        "hidden_dim":    model.hidden_dim,
                        "seq_len":       model.seq_len,
                    },
                },
                f"{checkpoint_dir}/best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch} (patience={patience})")
                break

    # Final history file
    with open(f"{checkpoint_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"\nTraining complete.  Best Val HR@3 = {best_hr3:.1%}")
        print(f"Checkpoint: {checkpoint_dir}/best_model.pt")
        print(f"History:    {checkpoint_dir}/training_history.json\n")

    return history
