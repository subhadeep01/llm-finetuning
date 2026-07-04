"""
Script 4: Compare Before vs After Fine-Tuning Results

Loads before_results.json and after_results.json, computes evaluation metrics,
and prints a detailed side-by-side comparison using rich terminal formatting.

Metrics computed:
  - ROUGE-L: Measures longest common subsequence overlap (0–1, higher = better)
  - ROUGE-1: Unigram overlap between response and ground truth
  - Response Length: Average number of words in response
  - Improvement: Percentage gain in ROUGE-L after fine-tuning

Output:
  - Terminal: Beautiful comparison tables with side-by-side examples
  - results/comparison_report.json: Full metrics for reproducibility
"""

import json
import os
import sys
from datetime import datetime

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rouge_score import rouge_scorer

# ─── Config ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
BEFORE_FILE = os.path.join(RESULTS_DIR, "before_results.json")
AFTER_FILE = os.path.join(RESULTS_DIR, "after_results.json")
REPORT_FILE = os.path.join(RESULTS_DIR, "comparison_report.json")

NUM_EXAMPLES_TO_SHOW = 3  # How many full side-by-side examples to display

console = Console()
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1 and ROUGE-L scores."""
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def word_count(text: str) -> int:
    return len(text.split())


def compute_all_metrics(results: list[dict]) -> list[dict]:
    """Compute per-example metrics."""
    enriched = []
    for r in results:
        metrics = compute_rouge(r["model_response"], r["ground_truth"])
        metrics["word_count"] = word_count(r["model_response"])
        enriched.append({**r, "metrics": metrics})
    return enriched


def average_metrics(results_with_metrics: list[dict]) -> dict:
    """Compute average metrics across all examples."""
    rouge1 = [r["metrics"]["rouge1_f"] for r in results_with_metrics]
    rougeL = [r["metrics"]["rougeL_f"] for r in results_with_metrics]
    wc = [r["metrics"]["word_count"] for r in results_with_metrics]
    n = len(results_with_metrics)
    return {
        "avg_rouge1": round(sum(rouge1) / n, 4),
        "avg_rougeL": round(sum(rougeL) / n, 4),
        "avg_word_count": round(sum(wc) / n, 1),
        "n_examples": n,
    }


# ─── Display ──────────────────────────────────────────────────────────────────

def print_summary_table(before_avg: dict, after_avg: dict):
    """Print the main metrics comparison table."""
    table = Table(
        title="📊  Before vs After Fine-Tuning — Aggregate Metrics",
        show_header=True,
        header_style="bold white on dark_blue",
        show_lines=True,
        min_width=70,
    )
    table.add_column("Metric", style="bold cyan", width=22)
    table.add_column("Before (Base)", style="red", justify="center", width=18)
    table.add_column("After (LoRA)", style="green", justify="center", width=18)
    table.add_column("Improvement", style="bold yellow", justify="center", width=14)

    def improvement(before: float, after: float) -> str:
        if before == 0:
            return "N/A"
        pct = ((after - before) / before) * 100
        symbol = "▲" if pct > 0 else "▼"
        color = "green" if pct > 0 else "red"
        return f"[{color}]{symbol} {abs(pct):.1f}%[/{color}]"

    table.add_row(
        "ROUGE-1 (F1)",
        f"{before_avg['avg_rouge1']:.4f}",
        f"{after_avg['avg_rouge1']:.4f}",
        improvement(before_avg["avg_rouge1"], after_avg["avg_rouge1"]),
    )
    table.add_row(
        "ROUGE-L (F1)",
        f"{before_avg['avg_rougeL']:.4f}",
        f"{after_avg['avg_rougeL']:.4f}",
        improvement(before_avg["avg_rougeL"], after_avg["avg_rougeL"]),
    )
    table.add_row(
        "Avg Response Length",
        f"{before_avg['avg_word_count']} words",
        f"{after_avg['avg_word_count']} words",
        improvement(before_avg["avg_word_count"], after_avg["avg_word_count"]),
    )
    table.add_row(
        "Test Examples",
        str(before_avg["n_examples"]),
        str(after_avg["n_examples"]),
        "—",
    )

    console.print(table)


def print_per_example_table(before_metrics: list[dict], after_metrics: list[dict]):
    """Print per-question ROUGE-L scores."""
    table = Table(
        title="📋  Per-Question ROUGE-L Scores",
        show_header=True,
        show_lines=False,
        header_style="bold white",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Question (truncated)", style="yellow", max_width=35)
    table.add_column("Before ROUGE-L", justify="center", style="red", width=15)
    table.add_column("After ROUGE-L", justify="center", style="green", width=15)
    table.add_column("Delta", justify="center", style="bold", width=10)

    for b, a in zip(before_metrics, after_metrics):
        before_l = b["metrics"]["rougeL_f"]
        after_l = a["metrics"]["rougeL_f"]
        delta = after_l - before_l
        delta_str = f"[green]+{delta:.3f}[/green]" if delta > 0 else f"[red]{delta:.3f}[/red]"

        table.add_row(
            str(b["id"]),
            b["question"][:60] + "...",
            f"{before_l:.4f}",
            f"{after_l:.4f}",
            delta_str,
        )

    console.print(table)


def print_side_by_side(before_item: dict, after_item: dict, idx: int):
    """Print a side-by-side comparison for a single example."""
    console.print(f"\n[bold]Example {idx}: [yellow]{before_item['question'][:120]}...[/yellow][/bold]\n")

    # Ground truth
    console.print(Panel(
        before_item["ground_truth"][:600] + ("..." if len(before_item["ground_truth"]) > 600 else ""),
        title="[bold white]📋 Ground Truth (Reference Answer)",
        border_style="white",
    ))

    # Before / After side by side
    before_score = before_item["metrics"]["rougeL_f"]
    after_score = after_item["metrics"]["rougeL_f"]

    before_panel = Panel(
        before_item["model_response"][:500] + ("..." if len(before_item["model_response"]) > 500 else ""),
        title=f"[bold red]❌ Before Fine-Tuning  (ROUGE-L: {before_score:.4f})",
        border_style="red",
    )
    after_panel = Panel(
        after_item["model_response"][:500] + ("..." if len(after_item["model_response"]) > 500 else ""),
        title=f"[bold green]✅ After Fine-Tuning  (ROUGE-L: {after_score:.4f})",
        border_style="green",
    )

    console.print(Columns([before_panel, after_panel]))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]Step 4: Before vs After Comparison Report")

    # 1. Load results
    for fpath in [BEFORE_FILE, AFTER_FILE]:
        if not os.path.exists(fpath):
            console.print(f"[bold red]ERROR:[/bold red] {fpath} not found.")
            console.print("Make sure both scripts 01 and 03 have been run successfully.")
            sys.exit(1)

    with open(BEFORE_FILE) as f:
        before_results = json.load(f)
    with open(AFTER_FILE) as f:
        after_results = json.load(f)

    console.print(f"\n[green]✓[/green] Loaded {len(before_results)} before-results")
    console.print(f"[green]✓[/green] Loaded {len(after_results)} after-results\n")

    # Align by ID
    before_map = {r["id"]: r for r in before_results}
    after_map = {r["id"]: r for r in after_results}
    common_ids = sorted(set(before_map.keys()) & set(after_map.keys()))

    before_aligned = [before_map[i] for i in common_ids]
    after_aligned = [after_map[i] for i in common_ids]

    # 2. Compute metrics
    console.print("[bold]Computing ROUGE metrics...[/bold]")
    before_with_metrics = compute_all_metrics(before_aligned)
    after_with_metrics = compute_all_metrics(after_aligned)

    before_avg = average_metrics(before_with_metrics)
    after_avg = average_metrics(after_with_metrics)

    console.print("[green]✓[/green] Metrics computed\n")

    # 3. Print summary table
    print_summary_table(before_avg, after_avg)
    console.print()

    # 4. Per-question table
    print_per_example_table(before_with_metrics, after_with_metrics)
    console.print()

    # 5. Side-by-side qualitative examples
    console.rule("[bold]Qualitative Examples: Side-by-Side Comparison")
    for i in range(min(NUM_EXAMPLES_TO_SHOW, len(common_ids))):
        print_side_by_side(before_with_metrics[i], after_with_metrics[i], i + 1)
        console.print()

    # 6. Save report
    report = {
        "generated_at": datetime.now().isoformat(),
        "before_model": before_results[0].get("model", "unknown"),
        "after_model": after_results[0].get("model", "unknown"),
        "after_adapter": after_results[0].get("adapter", "unknown"),
        "n_test_examples": len(common_ids),
        "aggregate_metrics": {
            "before": before_avg,
            "after": after_avg,
            "rougeL_improvement_pct": round(
                ((after_avg["avg_rougeL"] - before_avg["avg_rougeL"])
                 / max(before_avg["avg_rougeL"], 1e-9)) * 100, 2
            ),
        },
        "per_example": [
            {
                "id": common_ids[i],
                "question": before_with_metrics[i]["question"][:200],
                "before_rougeL": before_with_metrics[i]["metrics"]["rougeL_f"],
                "after_rougeL": after_with_metrics[i]["metrics"]["rougeL_f"],
                "delta_rougeL": round(
                    after_with_metrics[i]["metrics"]["rougeL_f"]
                    - before_with_metrics[i]["metrics"]["rougeL_f"], 4
                ),
            }
            for i in range(len(common_ids))
        ],
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    console.rule("[bold green]Comparison Complete!")
    console.print(f"\n[green]✓[/green] Full report saved → [cyan]{REPORT_FILE}[/cyan]")

    # 7. Final verdict
    improvement = report["aggregate_metrics"]["rougeL_improvement_pct"]
    direction = "▲" if improvement > 0 else "▼"
    color = "green" if improvement > 0 else "red"

    console.print(Panel(
        f"[bold]ROUGE-L Improvement: [{color}]{direction} {abs(improvement):.1f}%[/{color}][/bold]\n\n"
        f"Before fine-tuning avg ROUGE-L: [red]{before_avg['avg_rougeL']:.4f}[/red]\n"
        f"After fine-tuning  avg ROUGE-L: [green]{after_avg['avg_rougeL']:.4f}[/green]\n\n"
        "[dim]ROUGE-L measures how well the model's response overlaps with the\n"
        "reference answer using longest common subsequence. Higher = better.[/dim]",
        title="[bold yellow]🏆 Final Verdict",
        border_style="yellow",
    ))


if __name__ == "__main__":
    main()
