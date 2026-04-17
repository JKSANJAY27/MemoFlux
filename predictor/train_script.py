"""
predictor/train_script.py — End-to-End Training Entrypoint

Usage:
    python -m predictor.train_script
    python -m predictor.train_script --epochs 30 --device cpu
    python -m predictor.train_script --quick       # 3 epochs, smoke test
    python -m predictor.train_script --no-export   # skip ONNX export
"""

import argparse
import json
import sys
from pathlib import Path

# ── ensure project root is on path ──────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import box

from predictor.model               import ContextAwareLSTM
from predictor.dataset             import (
    build_vocab, build_dataloaders, save_vocab, split_sessions_by_user,
)
from predictor.trainer             import train
from predictor.evaluator           import evaluate, evaluate_cold_start, print_evaluation_report
from predictor.predictor_interface import PredictorInterface

console = Console()


# ─────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────

def load_all_sessions():
    """Try LSApp first, fall back to synthetic data."""
    try:
        from data_pipeline.lsapp_loader import LSAppLoader
        loader = LSAppLoader()
        train_s, val_s, test_s = loader.load_splits()
        if len(train_s) > 0:
            console.print(f"[green]PASS[/green] LSApp loaded: "
                          f"{len(train_s)} train / {len(val_s)} val / {len(test_s)} test sessions")
            return train_s + val_s + test_s   # merge then re-split by user
    except Exception as e:
        console.print(f"[yellow]WARN[/yellow] LSApp unavailable ({e}), using synthetic data")

    from data_pipeline.synthetic_generator import SyntheticGenerator
    gen = SyntheticGenerator(n_users=200, days_per_user=30, seed=42)
    sessions = gen.generate_all()
    console.print(f"[green]PASS[/green] Synthetic: {len(sessions)} sessions generated")
    return sessions


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AX Memory — Week 2 Training")
    parser.add_argument("--device",    default="cpu",
                        help="cuda or cpu (default: auto-detect)")
    parser.add_argument("--epochs",    type=int, default=40)
    parser.add_argument("--lr",        type=float, default=3e-3)
    parser.add_argument("--batch",     type=int, default=64)
    parser.add_argument("--hidden",    type=int, default=128)
    parser.add_argument("--patience",  type=int, default=7)
    parser.add_argument("--quick",     action="store_true",
                        help="3-epoch smoke test on 10%% of data")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip ONNX export step")
    args = parser.parse_args()

    # Auto-select device
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        console.print("[cyan]GPU detected → switching to cuda[/cyan]")

    console.print(Panel.fit(
        "[bold blue]AX Memory — Week 2 Training[/bold blue]\n"
        "[dim]ContextAwareLSTM · Temporal Gating · App Attention · User Profile[/dim]",
        border_style="blue",
    ))

    # ── 1. Load data ───────────────────────────────
    sessions = load_all_sessions()
    if args.quick:
        sessions = sessions[: max(len(sessions) // 10, 30)]
        args.epochs  = 3
        args.patience= 3

    train_s, val_s, test_s = split_sessions_by_user(sessions, seed=42)
    console.print(f"Users split → train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)} sessions")

    # ── 2. Vocabularies ───────────────────────────
    all_train = train_s   # build vocab from training users only
    app_vocab, user_vocab = build_vocab(all_train, min_count=3)
    save_vocab(app_vocab, user_vocab)
    V = len(app_vocab)
    N = len(user_vocab)
    console.print(f"Vocabulary: {V} apps · {N} users")

    # ── 3. DataLoaders ────────────────────────────
    train_dl, val_dl, test_dl = build_dataloaders(
        train_s, val_s, test_s, app_vocab, user_vocab,
        batch_size=args.batch, num_workers=0,
    )
    console.print(f"Dataset sizes: train={len(train_dl.dataset):,} · "
                  f"val={len(val_dl.dataset):,} · test={len(test_dl.dataset):,}")

    if len(train_dl.dataset) == 0:
        console.print("[red]ERROR: No training samples — check sessions and vocabulary.[/red]")
        sys.exit(1)

    # ── 4. Model ──────────────────────────────────
    model = ContextAwareLSTM(
        vocab_size    = V,
        n_users       = N,
        hidden_dim    = args.hidden,
        n_lstm_layers = 2,
        dropout       = 0.3,
    )
    console.print(f"\nModel: {model}")

    # ── 5. Train ──────────────────────────────────
    history = train(
        model, train_dl, val_dl,
        device        = device,
        n_epochs      = args.epochs,
        lr            = args.lr,
        patience      = args.patience,
        checkpoint_dir= "checkpoints",
        verbose       = True,
    )

    # ── 6. Test evaluation ────────────────────────
    console.print("\n[bold]Loading best checkpoint for test evaluation...[/bold]")
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics     = evaluate(model, test_dl, device)
    cold_metrics     = evaluate_cold_start(model, test_dl, device)
    print_evaluation_report(test_metrics, cold_metrics)

    # ── 7. ONNX Export ────────────────────────────
    benchmark = {}
    if not args.no_export:
        console.print("\n[bold]Exporting to ONNX...[/bold]")
        try:
            from predictor.export import run_full_export_pipeline
            benchmark = run_full_export_pipeline(model, model.seq_len)
        except Exception as e:
            console.print(f"[yellow]ONNX export failed: {e}[/yellow]")
            try:
                from predictor.export import benchmark_pytorch
                benchmark = {"pytorch": benchmark_pytorch(model, model.seq_len)}
            except Exception:
                pass

    # ── 8. Final summary ──────────────────────────
    best_hr3 = max(history.get("val_hr3", [0]))
    console.print()
    console.print(Panel.fit(
        f"[bold green]Training Complete[/bold green]\n\n"
        f"  Best Val HR@3  : [cyan]{best_hr3:.1%}[/cyan]  "
        f"{'PASS TARGET MET' if best_hr3 >= 0.75 else '— below 75% (add more epochs)'}\n"
        f"  Test HR@3      : [cyan]{test_metrics.get('hr3', 0):.1%}[/cyan]\n"
        f"  Parameters     : [cyan]{model.count_parameters():,}[/cyan]  "
        f"{'PASS' if model.count_parameters() < 2_000_000 else 'FAIL'} (< 2M)\n"
        f"  Checkpoint     : checkpoints/best_model.pt\n"
        f"  History        : checkpoints/training_history.json",
        border_style="green",
    ))

    # Save combined results
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/week2_results.json", "w") as f:
        json.dump({
            "test_metrics":  test_metrics,
            "cold_metrics":  cold_metrics,
            "best_val_hr3":  best_hr3,
            "params":        model.count_parameters(),
            "benchmark":     benchmark,
        }, f, indent=2)
    console.print("[dim]Full results → checkpoints/week2_results.json[/dim]")


if __name__ == "__main__":
    main()
