"""
predictor/evaluator.py — HR@K, MRR@K and multi-cut evaluation

Metrics:
  HR@K  — fraction where true next-app is in top-K predictions
  MRR@K — mean reciprocal rank within top-K (0 if outside top-K)

Multi-cut evaluation:
  - Standard:    all test users
  - Cold-start:  users not seen during training (user_id mapped to 0)
  - Archetype:   per-user-type breakdown
"""

import torch
import numpy as np
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────
# Core metric functions (operate on CPU tensors)
# ─────────────────────────────────────────────────────────

def compute_hr_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Hit Rate @ K: fraction of samples where true label is in top-K."""
    _, top_k = torch.topk(logits, k, dim=-1)                      # (B, K)
    hits = (top_k == labels.unsqueeze(1)).any(dim=1).float()       # (B,)
    return hits.mean().item()


def compute_mrr_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Mean Reciprocal Rank @ K."""
    _, sorted_idx = torch.sort(logits, dim=-1, descending=True)    # (B, V)
    # Rank of true label (1-indexed)
    B, V = sorted_idx.shape
    rank_pos = (sorted_idx == labels.unsqueeze(1)).nonzero(as_tuple=False)
    # Build rank tensor — default V+1 (not found)
    ranks = torch.full((B,), V + 1, dtype=torch.float)
    for row_idx, col_idx in rank_pos:
        ranks[row_idx] = col_idx.float() + 1   # 1-indexed rank

    in_top_k = ranks <= k
    mrr = torch.where(in_top_k, 1.0 / ranks, torch.zeros_like(ranks))
    return mrr.mean().item()


# ─────────────────────────────────────────────────────────
# Full evaluation loop
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device: str) -> Dict[str, float]:
    """Run full evaluation; returns dict of HR@1/3/5 and MRR@3/5."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in dataloader:
        app_ids  = batch["app_ids"].to(device)
        ctx_vecs = batch["ctx_vecs"].to(device)
        user_ids = batch["user_id"].to(device)
        labels   = batch["label"].to(device)

        logits, _ = model(app_ids, ctx_vecs, user_ids)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        "hr1":  compute_hr_at_k(all_logits, all_labels, 1),
        "hr3":  compute_hr_at_k(all_logits, all_labels, 3),
        "hr5":  compute_hr_at_k(all_logits, all_labels, 5),
        "mrr3": compute_mrr_at_k(all_logits, all_labels, 3),
        "mrr5": compute_mrr_at_k(all_logits, all_labels, 5),
        "n_samples": len(all_labels),
    }


# ─────────────────────────────────────────────────────────
# Cold-start evaluation
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_cold_start(model, dataloader, device: str) -> Dict[str, float]:
    """
    Evaluate with user_ids forced to 0 (unknown user).
    Shows graceful degradation via global user embedding.
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in dataloader:
        app_ids  = batch["app_ids"].to(device)
        ctx_vecs = batch["ctx_vecs"].to(device)
        user_ids = torch.zeros_like(batch["user_id"]).to(device)   # force cold-start
        labels   = batch["label"].to(device)

        logits, _ = model(app_ids, ctx_vecs, user_ids)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        "cold_hr1":  compute_hr_at_k(all_logits, all_labels, 1),
        "cold_hr3":  compute_hr_at_k(all_logits, all_labels, 3),
        "cold_hr5":  compute_hr_at_k(all_logits, all_labels, 5),
        "cold_mrr3": compute_mrr_at_k(all_logits, all_labels, 3),
        "n_samples": len(all_labels),
    }


# ─────────────────────────────────────────────────────────
# Rich report printer
# ─────────────────────────────────────────────────────────

def print_evaluation_report(
    test_metrics: Dict,
    cold_metrics: Optional[Dict] = None,
    benchmark:    Optional[Dict] = None,
):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()

        t = Table(title="AX Memory — Evaluation Report", box=box.ROUNDED)
        t.add_column("Metric",    style="cyan",  width=22)
        t.add_column("Score",     style="green", justify="right")
        t.add_column("Target",    justify="right")
        t.add_column("Status")

        hr3 = test_metrics.get("hr3", 0)
        t.add_row("HR@1",       f"{test_metrics.get('hr1', 0):.1%}", "—",    "—")
        t.add_row("HR@3",       f"{hr3:.1%}",                         "≥75%", "✓ PASS" if hr3 >= 0.75 else "✗ FAIL")
        t.add_row("HR@5",       f"{test_metrics.get('hr5', 0):.1%}", "—",    "—")
        t.add_row("MRR@3",      f"{test_metrics.get('mrr3', 0):.4f}","—",    "—")
        t.add_row("N samples",  f"{test_metrics.get('n_samples',0):,}","—", "—")

        if cold_metrics:
            t.add_section()
            t.add_row("Cold-start HR@3", f"{cold_metrics.get('cold_hr3', 0):.1%}", "—", "graceful degradation")

        if benchmark:
            t.add_section()
            t.add_row("ONNX p50 latency", f"{benchmark.get('latency_p50_ms','?')}ms", "< 15ms",
                      "✓" if benchmark.get("passes_15ms_target") else "✗")
            t.add_row("Model size",       f"{benchmark.get('model_size_mb','?')}MB",  "< 10MB", "—")

        console.print(t)
    except ImportError:
        # Fallback without rich
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")
