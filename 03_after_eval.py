"""
Script 3: After Fine-Tuning Evaluation

Runs inference on the FINE-TUNED Qwen2.5-1.5B-Instruct model
(with LoRA adapters loaded) against the same 20 held-out medical
QA test questions used in Script 1.

Results are saved to results/after_results.json for comparison.
"""

import json
import os
import sys
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER_DIR = "./adapters"
TEST_FILE = os.path.join("data", "test_questions.json")
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "after_results.json")
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a knowledgeable and compassionate medical expert. "
    "Provide clear, accurate, and helpful answers to medical questions. "
    "Always be thorough but concise in your explanations."
)

console = Console()


def check_prerequisites():
    """Check that adapters and test data exist."""
    if not os.path.exists(TEST_FILE):
        console.print(f"[bold red]ERROR:[/bold red] {TEST_FILE} not found.")
        console.print("Run [cyan]python data/prepare_dataset.py[/cyan] first.")
        sys.exit(1)

    if not os.path.exists(ADAPTER_DIR):
        console.print(f"[bold red]ERROR:[/bold red] Adapters not found at [cyan]{ADAPTER_DIR}/[/cyan]")
        console.print("Run [cyan]python 02_finetune_lora.py[/cyan] first.")
        sys.exit(1)

    adapter_files = os.listdir(ADAPTER_DIR)
    if not adapter_files:
        console.print(f"[bold red]ERROR:[/bold red] No adapter files found in [cyan]{ADAPTER_DIR}/[/cyan]")
        console.print("Training may not have completed. Run [cyan]python 02_finetune_lora.py[/cyan] again.")
        sys.exit(1)

    return adapter_files


def load_model_with_adapters():
    """Load the base model WITH LoRA adapters applied."""
    from mlx_lm import load
    console.print(f"  Base model  : [cyan]{MODEL_ID}[/cyan]")
    console.print(f"  LoRA adapter: [cyan]{ADAPTER_DIR}/[/cyan]")
    model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_DIR)
    return model, tokenizer


def run_inference(model, tokenizer, question: str) -> tuple[str, float]:
    """Run inference on a single question. Returns (response, latency_seconds)."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    processors = make_logits_processors(repetition_penalty=1.15, repetition_context_size=64)

    start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        logits_processors=processors,
        verbose=False,
    )
    elapsed = time.time() - start

    # Strip the prompt from the response if included
    if prompt in response:
        response = response.replace(prompt, "").strip()

    return response.strip(), elapsed


def main():
    console.rule("[bold blue]Step 3: After Fine-Tuning Evaluation")
    console.print(f"\n[bold]Model:[/bold] {MODEL_ID} [green](+ LoRA adapters)[/green]")
    console.print(f"[bold]Task:[/bold]  Medical QA — same 20 test questions as before\n")

    # 1. Prerequisites
    adapter_files = check_prerequisites()
    console.print(f"[green]✓[/green] Found LoRA adapters in [cyan]{ADAPTER_DIR}/[/cyan]:")
    for f in sorted(adapter_files):
        size_mb = os.path.getsize(os.path.join(ADAPTER_DIR, f)) / (1024 * 1024)
        console.print(f"  • {f}  [dim]({size_mb:.1f} MB)[/dim]")

    with open(TEST_FILE, "r") as f:
        test_data = json.load(f)

    console.print(f"\n[green]✓[/green] Loaded {len(test_data)} test questions\n")

    # 2. Load model + adapters
    console.print("[bold]Loading fine-tuned model (base + LoRA adapters)...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()) as p:
        task = p.add_task("Loading Qwen2.5-1.5B-Instruct + LoRA adapters...", total=None)
        model, tokenizer = load_model_with_adapters()
        p.update(task, description="[green]Fine-tuned model loaded!")

    console.print()

    # 3. Run inference on all test questions
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    console.print("[bold]Running inference on 20 medical questions...[/bold]\n")

    for i, item in enumerate(test_data, start=1):
        question = item["instruction"]
        ground_truth = item["output"]

        console.print(f"[bold][{i:02d}/20][/bold] [yellow]{question[:100]}...[/yellow]")

        response, latency = run_inference(model, tokenizer, question)

        console.print(f"  [dim]Answer:[/dim] {response[:150]}...")
        console.print(f"  [dim]Latency: {latency:.1f}s[/dim]\n")

        results.append({
            "id": i,
            "question": question,
            "ground_truth": ground_truth,
            "model_response": response,
            "model": MODEL_ID,
            "adapter": ADAPTER_DIR,
            "stage": "after_finetune",
            "latency_seconds": round(latency, 2),
        })

    # 4. Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print()
    console.rule("[bold green]After Evaluation Complete")
    console.print(f"\n[green]✓[/green] Results saved → [cyan]{OUTPUT_FILE}[/cyan]")
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    console.print(f"[green]✓[/green] Average latency: [cyan]{avg_latency:.1f}s[/cyan] per question\n")

    # 5. Show sample table
    table = Table(title="After Fine-Tuning — Sample Responses", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Question (truncated)", style="yellow", max_width=40)
    table.add_column("Response (truncated)", style="green", max_width=60)

    for r in results[:5]:
        table.add_row(
            str(r["id"]),
            r["question"][:80] + "...",
            r["model_response"][:120] + "...",
        )

    console.print(table)
    console.print("\n[bold]Next step:[/bold] Run [cyan]python 04_compare_results.py[/cyan]")


if __name__ == "__main__":
    main()
