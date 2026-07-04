"""
Script 2: PEFT LoRA Fine-Tuning with MLX-LM

Fine-tunes Qwen2.5-1.5B-Instruct on the Medical QA dataset using LoRA
via Apple's MLX framework (optimized for Apple Silicon).

LoRA Config:
  - rank: 8       (low-rank decomposition dimension)
  - alpha: 16     (scaling factor = 2 × rank)
  - layers: 16    (number of transformer layers to apply LoRA to)
  - targets: attention projection layers (q_proj, v_proj)

Estimated training time on M1 8GB: ~30–60 minutes for 100 iterations.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DATA_DIR = "./data"
ADAPTER_DIR = "./adapters"
RESULTS_DIR = "./results"

# LoRA Hyperparameters
LORA_RANK = 8
LORA_ALPHA = 16        # MLX uses this via the model's own scaling
LORA_LAYERS = 28       # Apply LoRA to all 28 layers of Qwen2.5-1.5B
LEARNING_RATE = 1e-4
BATCH_SIZE = 2         # Set to 2 to fit within 8GB physical RAM without swap
NUM_ITERS = 300        # Train for 300 steps (about 10 mins with batch size 2 & seq len 512)
STEPS_PER_EVAL = 50    # Evaluate on validation set every 50 steps
STEPS_PER_SAVE = 50    # Save checkpoint every 50 steps
MAX_SEQ_LEN = 512      # Protects memory & speeds up training on M1 8GB

console = Console()


def check_prerequisites():
    """Verify data files exist before starting training."""
    required_files = [
        os.path.join(DATA_DIR, "train.jsonl"),
        os.path.join(DATA_DIR, "valid.jsonl"),
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        console.print("[bold red]ERROR:[/bold red] Missing data files:")
        for f in missing:
            console.print(f"  • {f}")
        console.print("\nRun [cyan]python data/prepare_dataset.py[/cyan] first.")
        sys.exit(1)

    # Count examples
    with open(os.path.join(DATA_DIR, "train.jsonl")) as f:
        train_count = sum(1 for _ in f)
    with open(os.path.join(DATA_DIR, "valid.jsonl")) as f:
        valid_count = sum(1 for _ in f)

    return train_count, valid_count


def print_lora_config():
    """Print LoRA configuration summary."""
    table = Table(title="LoRA Fine-Tuning Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="bold white")
    table.add_column("Description", style="dim")

    config = [
        ("Model", MODEL_ID, "Base model (4-bit quantized)"),
        ("LoRA Rank (r)", str(LORA_RANK), "Low-rank decomposition size"),
        ("LoRA Alpha (α)", str(LORA_ALPHA), "Scaling factor for LoRA weights"),
        ("LoRA Layers", str(LORA_LAYERS), "Number of transformer layers adapted"),
        ("Learning Rate", str(LEARNING_RATE), "AdamW optimizer LR"),
        ("Batch Size", str(BATCH_SIZE), "Training batch size"),
        ("Iterations", str(NUM_ITERS), "Total training steps"),
        ("Eval Every", f"{STEPS_PER_EVAL} steps", "Validation loss frequency"),
        ("Max Seq Length", str(MAX_SEQ_LEN), "Token truncation limit"),
        ("Adapter Save Path", ADAPTER_DIR, "Where LoRA weights are stored"),
    ]

    for param, value, desc in config:
        table.add_row(param, value, desc)

    console.print(table)


CONFIG_FILE = "lora_config.yaml"


def write_lora_config() -> str:
    """Write LoRA config YAML file for mlx-lm 0.21.x.
    In this version, rank/scale/dropout are set via config file, not CLI flags.
    """
    import yaml  # part of PyYAML, installed with mlx-lm
    config = {
        "lora_parameters": {
            "rank": LORA_RANK,
            "alpha": float(LORA_ALPHA),
            "dropout": 0.05,
            "scale": float(LORA_ALPHA / LORA_RANK),  # scale = alpha / rank
        }
    }
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f)
    return CONFIG_FILE


def build_mlx_command(config_path: str) -> list[str]:
    """Build the mlx_lm.lora training command."""
    return [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", MODEL_ID,
        "--train",
        "--data", DATA_DIR,
        "--adapter-path", ADAPTER_DIR,
        "--num-layers", str(LORA_LAYERS),
        "--batch-size", str(BATCH_SIZE),
        "--iters", str(NUM_ITERS),
        "--val-batches", "5",
        "--learning-rate", str(LEARNING_RATE),
        "--steps-per-eval", str(STEPS_PER_EVAL),
        "--save-every", str(STEPS_PER_SAVE),
        "--max-seq-length", str(MAX_SEQ_LEN),
        "-c", config_path,   # LoRA rank/scale/dropout via config file
    ]


def main():
    console.rule("[bold blue]Step 2: PEFT LoRA Fine-Tuning")

    # 1. Check prerequisites
    console.print("\n[bold]Checking prerequisites...[/bold]")
    train_count, valid_count = check_prerequisites()
    console.print(f"  [green]✓[/green] Training data: [cyan]{train_count:,}[/cyan] examples")
    console.print(f"  [green]✓[/green] Validation data: [cyan]{valid_count:,}[/cyan] examples\n")

    # 2. Show config
    print_lora_config()
    console.print()

    # 3. Create output dirs
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 4. Explain what LoRA does
    console.print(Panel(
        "[bold]What is LoRA?[/bold]\n\n"
        "LoRA (Low-Rank Adaptation) freezes the pre-trained model weights and injects "
        "small [bold cyan]trainable rank decomposition matrices[/bold cyan] into the attention "
        "layers.\n\n"
        f"Instead of training all [bold]{1_500_000_000:,}[/bold] parameters, LoRA trains only "
        f"[bold cyan]~{LORA_RANK * LORA_LAYERS * 2 * 2048 * 2 // 1000}K[/bold cyan] adapter parameters "
        f"([cyan]<0.1%[/cyan] of total), making training feasible on 8GB RAM.",
        title="[bold yellow]LoRA Theory",
        border_style="yellow",
    ))
    console.print()

    # 5. Write LoRA config file and build command
    console.print("[bold]Writing LoRA config file...[/bold]")
    config_path = write_lora_config()
    console.print(f"  [green]✓[/green] LoRA config saved to [cyan]{config_path}[/cyan]")
    console.print(f"  [dim]rank={LORA_RANK}, alpha={LORA_ALPHA}, scale={LORA_ALPHA/LORA_RANK}, dropout=0.05[/dim]\n")

    cmd = build_mlx_command(config_path)
    console.print("[bold]Training command:[/bold]")
    console.print(f"  [dim]{' '.join(cmd)}[/dim]\n")

    # 6. Save training metadata
    metadata = {
        "started_at": datetime.now().isoformat(),
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_layers": LORA_LAYERS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_iters": NUM_ITERS,
        "train_examples": train_count,
        "valid_examples": valid_count,
        "adapter_path": ADAPTER_DIR,
    }
    with open(os.path.join(RESULTS_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    console.print("[bold]Starting LoRA fine-tuning...[/bold]")
    console.print("[dim]Training loss will be printed every step. Validation loss every 20 steps.[/dim]")
    console.print("[dim]You should see training loss decrease from ~2.5 → ~0.8 over 100 iterations.[/dim]\n")
    console.rule()

    # 7. Run training
    start_time = time.time()

    try:
        # Run with live output streamed to terminal
        process = subprocess.run(
            cmd,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        console.print(f"\n[bold red]Training failed with exit code {e.returncode}[/bold red]")
        console.print("Common fixes:")
        console.print("  1. Ensure mlx-lm is installed: [cyan]pip install mlx-lm[/cyan]")
        console.print("  2. Check you have enough RAM (close other apps)")
        console.print("  3. Try reducing batch size: edit BATCH_SIZE to 2 in this script")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
        console.print("Partial adapters may have been saved to [cyan]./adapters/[/cyan]")
        sys.exit(0)

    # 8. Training complete
    elapsed = time.time() - start_time
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    console.rule()
    console.print(f"\n[bold green]✓ LoRA fine-tuning complete![/bold green]")
    console.print(f"  Total training time: [cyan]{elapsed_str}[/cyan]")
    console.print(f"  Adapter weights saved to: [cyan]{ADAPTER_DIR}/[/cyan]")

    # 9. List saved adapter files
    if os.path.exists(ADAPTER_DIR):
        adapter_files = os.listdir(ADAPTER_DIR)
        if adapter_files:
            console.print(f"\n[bold]Adapter files:[/bold]")
            for f in sorted(adapter_files):
                fpath = os.path.join(ADAPTER_DIR, f)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                console.print(f"  • {f}  [dim]({size_mb:.1f} MB)[/dim]")

    # 10. Update metadata
    metadata["completed_at"] = datetime.now().isoformat()
    metadata["total_time_seconds"] = round(elapsed, 1)
    with open(os.path.join(RESULTS_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    console.print("\n[bold]Next step:[/bold] Run [cyan]python3 03_after_eval.py[/cyan]")


if __name__ == "__main__":
    main()
