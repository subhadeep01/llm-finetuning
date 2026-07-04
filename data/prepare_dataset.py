"""
Script 1: Prepare the Medical QA Dataset for MLX-LM fine-tuning.

Downloads 'keivalya/MedQuad-MedicalQnADataset' from HuggingFace —
the HuggingFace mirror of MedQuAD (Medical Question Answering Dataset),
the same dataset available at: https://www.kaggle.com/datasets/jpmiller/layoutlm

Formats it into MLX-compatible chat JSONL and splits into:
  - data/train.jsonl         (up to 2,000 training examples)
  - data/valid.jsonl         (200 validation examples)
  - data/test_questions.json (20 held-out examples for before/after eval)
"""

import json
import os
import random

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SIZE = 2000
VALID_SIZE = 200
TEST_SIZE = 20
TOTAL_NEEDED = TRAIN_SIZE + VALID_SIZE + TEST_SIZE
SEED = 42

SYSTEM_PROMPT = (
    "You are a knowledgeable and compassionate medical expert. "
    "Provide clear, accurate, and helpful answers to medical questions. "
    "Always be thorough but concise in your explanations."
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def format_as_chat(instruction: str, output: str) -> dict:
    """Format a Q&A pair as MLX-LM chat messages."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": output.strip()},
        ]
    }


def write_jsonl(records: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"  [green]✓[/green] Saved {len(records)} examples → [cyan]{path}[/cyan]")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]Medical QA Dataset Preparation")

    # 1. Load dataset
    console.print("\n[bold]Step 1:[/bold] Loading [cyan]keivalya/MedQuad-MedicalQnADataset[/cyan]...")
    console.print("  (MedQuAD — Medical Question Answering Dataset, ~22MB, no auth required)")

    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    console.print(f"  [green]✓[/green] Loaded {len(ds):,} total examples")
    console.print(f"  [dim]Columns: {ds.column_names}[/dim]")

    # 2. Filter: keep only examples with non-empty Question + Answer
    # Dataset columns: 'qtype', 'Question', 'Answer'
    console.print("\n[bold]Step 2:[/bold] Filtering and cleaning examples...")
    valid_examples = []
    for example in track(ds, description="  Filtering..."):
        question = (example.get("Question") or "").strip()
        answer = (example.get("Answer") or "").strip()
        # Keep examples where both fields are meaningful and answer is substantial
        if question and answer and len(answer) > 80:
            valid_examples.append({"instruction": question, "output": answer})

    console.print(f"  [green]✓[/green] {len(valid_examples):,} examples passed quality filter")

    # 3. Sample & shuffle
    sample_size = min(TOTAL_NEEDED, len(valid_examples))
    console.print(f"\n[bold]Step 3:[/bold] Sampling {sample_size} examples (seed={SEED})...")
    random.seed(SEED)
    sampled = random.sample(valid_examples, sample_size)

    # 4. Split
    test_raw = sampled[:TEST_SIZE]
    valid_raw = sampled[TEST_SIZE : TEST_SIZE + VALID_SIZE]
    train_raw = sampled[TEST_SIZE + VALID_SIZE :]

    console.print(f"  Train : {len(train_raw):,} examples")
    console.print(f"  Valid : {len(valid_raw):,} examples")
    console.print(f"  Test  : {len(test_raw):,} examples (held-out for evaluation)")

    # 5. Format and write JSONL for MLX-LM
    console.print("\n[bold]Step 4:[/bold] Formatting and writing JSONL files...")

    train_records = [format_as_chat(ex["instruction"], ex["output"]) for ex in train_raw]
    valid_records = [format_as_chat(ex["instruction"], ex["output"]) for ex in valid_raw]

    write_jsonl(train_records, os.path.join(DATA_DIR, "train.jsonl"))
    write_jsonl(valid_records, os.path.join(DATA_DIR, "valid.jsonl"))

    # 6. Write test questions (separate format — we need ground truth for eval)
    test_path = os.path.join(DATA_DIR, "test_questions.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_raw, f, indent=2, ensure_ascii=False)
    console.print(f"  [green]✓[/green] Saved {len(test_raw)} test Q&A pairs → [cyan]{test_path}[/cyan]")

    # 7. Preview a sample
    console.print("\n[bold]Step 5:[/bold] Sample preview (first test question):")
    sample = test_raw[0]
    console.print(f"\n  [bold yellow]Question:[/bold yellow]\n  {sample['instruction'][:200]}...")
    console.print(f"\n  [bold green]Expected Answer:[/bold green]\n  {sample['output'][:200]}...")

    console.rule("[bold green]Dataset preparation complete!")
    console.print("\n[bold]Files created in [cyan]./data/[/cyan]:[/bold]")
    console.print("  • train.jsonl          → MLX-LM training data")
    console.print("  • valid.jsonl          → MLX-LM validation data")
    console.print("  • test_questions.json  → Held-out eval questions\n")
    console.print("[bold]Next step:[/bold] Run [cyan]python3 01_before_eval.py[/cyan]")


if __name__ == "__main__":
    main()
