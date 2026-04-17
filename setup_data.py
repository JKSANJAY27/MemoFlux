"""
setup_data.py — AX Memory Dataset Downloader & Setup
Samsung AX Hackathon 2026 | PS-03

Run with: python setup_data.py
"""

import os
import sys
import gzip
import shutil
import hashlib
import requests
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn

console = Console()

LSAPP_URL = "https://github.com/aliannejadi/LSApp/raw/main/lsapp.tsv.gz"
LSAPP_EXPECTED_MIN_SIZE = 1_000_000  # at least 1MB

DIRS = [
    "data/raw",
    "data/processed",
    "data/synthetic",
    "results",
]


def make_dirs():
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
        gitkeep = Path(d) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    console.print("[green]✓[/green] Directory structure created")


def download_lsapp(target_dir: str = "data/raw") -> str:
    """Download and decompress the LSApp dataset."""
    gz_path = os.path.join(target_dir, "lsapp.tsv.gz")
    tsv_path = os.path.join(target_dir, "lsapp.tsv")

    if os.path.exists(tsv_path) and os.path.getsize(tsv_path) > LSAPP_EXPECTED_MIN_SIZE:
        console.print(f"[green]✓[/green] LSApp already downloaded at [cyan]{tsv_path}[/cyan]")
        return tsv_path

    console.print("[yellow]↓[/yellow] Downloading LSApp dataset (~7MB)...")
    try:
        r = requests.get(LSAPP_URL, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
        ) as progress:
            task = progress.add_task("Downloading lsapp.tsv.gz", total=total)
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.advance(task, len(chunk))

        console.print("[yellow]⊕[/yellow] Decompressing...")
        with gzip.open(gz_path, "rb") as f_in, open(tsv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_path)
        size_mb = os.path.getsize(tsv_path) / 1_048_576
        console.print(f"[green]✓[/green] LSApp saved to [cyan]{tsv_path}[/cyan] ({size_mb:.1f} MB)")
    except Exception as e:
        console.print(f"[red]✗[/red] Download failed: {e}")
        console.print("[yellow]  Generating synthetic data as fallback...[/yellow]")
        _generate_synthetic_fallback()
        return None

    return tsv_path


def _generate_synthetic_fallback():
    """Generate synthetic sessions if real data cannot be downloaded."""
    console.print("[yellow]⊕[/yellow] Generating 200-user synthetic fallback dataset...")
    try:
        # Add project root to path
        sys.path.insert(0, str(Path(__file__).parent))
        from data_pipeline.synthetic_generator import SyntheticGenerator
        gen = SyntheticGenerator(n_users=200, days_per_user=30, seed=42)
        sessions = gen.generate_all()
        console.print(f"[green]✓[/green] Generated {len(sessions)} synthetic sessions")
    except Exception as e:
        console.print(f"[red]  Could not generate synthetic data: {e}[/red]")


def verify_data():
    """Quick sanity check on downloaded data."""
    tsv_path = "data/raw/lsapp.tsv"
    if os.path.exists(tsv_path):
        size = os.path.getsize(tsv_path)
        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            lines = sum(1 for _ in f) + 1
        console.print(f"[green]✓[/green] LSApp: {lines:,} rows | Columns: {header}")
    else:
        console.print("[yellow]⚠[/yellow] LSApp TSV not found — using synthetic data")


def print_banner():
    console.print("""
[bold blue]╔══════════════════════════════════════════════════════╗
║          AX Memory — Samsung AX Hackathon 2026        ║
║      Context-Aware Adaptive Memory | PS-03 Setup      ║
╚══════════════════════════════════════════════════════╝[/bold blue]
""")


if __name__ == "__main__":
    print_banner()
    make_dirs()
    download_lsapp()
    verify_data()

    console.print("""
[bold green]✓ Setup complete! Next steps:[/bold green]

  [cyan]python -m pytest tests/ -v[/cyan]                    # Run all tests
  [cyan]python -m evaluation.benchmark_runner[/cyan]         # Run 500-ep LRU benchmark
  [cyan]streamlit run dashboard/app.py[/cyan]                # Launch dashboard
  [cyan]python data_pipeline/synthetic_generator.py --n_users 100[/cyan]
""")
