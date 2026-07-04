"""
Script 1: Before Fine-Tuning Evaluation

Runs inference on the BASE Qwen2.5-1.5B-Instruct model (no LoRA adapters)
against 20 held-out medical QA test questions.

Results are saved to results/before_results.json for comparison.
"""

import json
import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
TEST_FILE = os.path.join("data", "test_questions.json")
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "before_results.json")
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a knowledgeable and compassionate medical expert. "
    "Provide clear, accurate, and helpful answers to medical questions. "
    "Always be thorough but concise in your explanations."
)

console = Console()


def load_model():
    """Load the base model WITHOUT any LoRA adapters."""
    from mlx_lm import load
    console.print(f"  Loading model: [cyan]{MODEL_ID}[/cyan]")
    console.print("  (First run will download ~900MB — subsequent runs use cache)")
    model, tokenizer = load(MODEL_ID)
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
    console.rule("[bold blue]Step 1: Before Fine-Tuning Evaluation")
    console.print(f"\n[bold]Model:[/bold] {MODEL_ID} [dim](base, no adapters)[/dim]")
    console.print(f"[bold]Task:[/bold]  Medical QA — 20 test questions\n")

    # 1. Check test data exists
    if not os.path.exists(TEST_FILE):
        console.print(f"[bold red]ERROR:[/bold red] {TEST_FILE} not found.")
        console.print("Please run [cyan]python data/prepare_dataset.py[/cyan] first.")
        sys.exit(1)

    with open(TEST_FILE, "r") as f:
        test_data = json.load(f)

    console.print(f"[green]✓[/green] Loaded {len(test_data)} test questions from [cyan]{TEST_FILE}[/cyan]\n")

    # 2. Load model
    console.print("[bold]Loading base model...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()) as p:
        task = p.add_task("Initializing Qwen2.5-1.5B-Instruct (4-bit)...", total=None)
        model, tokenizer = load_model()
        p.update(task, description="[green]Model loaded!")

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
            "stage": "before_finetune",
            "latency_seconds": round(latency, 2),
        })

    # 4. Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print()
    console.rule("[bold green]Before Evaluation Complete")
    console.print(f"\n[green]✓[/green] Results saved → [cyan]{OUTPUT_FILE}[/cyan]")
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    console.print(f"[green]✓[/green] Average latency: [cyan]{avg_latency:.1f}s[/cyan] per question\n")

    # 5. Show summary table
    table = Table(title="Before Fine-Tuning — Sample Responses", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Question (truncated)", style="yellow", max_width=40)
    table.add_column("Response (truncated)", style="white", max_width=60)

    for r in results[:5]:
        table.add_row(
            str(r["id"]),
            r["question"][:80] + "...",
            r["model_response"][:120] + "...",
        )

    console.print(table)
    console.print("\n[bold]Next step:[/bold] Run [cyan]python 02_finetune_lora.py[/cyan]")


if __name__ == "__main__":
    main()
