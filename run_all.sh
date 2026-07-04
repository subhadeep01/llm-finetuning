#!/usr/bin/env bash
# run_all.sh — One-command pipeline runner
# Runs all 5 steps in sequence: setup → before → train → after → compare

set -e  # Exit on any error

echo "=============================================="
echo "  LLM Fine-Tuning Pipeline: Medical QA"
echo "  Model: Qwen2.5-1.5B-Instruct + LoRA"
echo "=============================================="
echo ""

# Step 0: Activate venv and install dependencies
echo ">>> Activating virtual environment..."
source venv/bin/activate
echo "✓ venv activated: $(which python3)"
echo ""

echo ">>> Installing/verifying dependencies..."
pip3 install -r requirements.txt -q
echo "✓ Dependencies ready"
echo ""

# Step 1: Prepare dataset
echo ">>> Step 1/4: Preparing Medical QA dataset..."
python3 data/prepare_dataset.py
echo ""

# Step 2: Before evaluation
echo ">>> Step 2/4: Running before fine-tuning evaluation..."
python3 01_before_eval.py
echo ""

# Step 3: Fine-tune with LoRA
echo ">>> Step 3/4: Running LoRA fine-tuning (~30-60 min on M1)..."
python3 02_finetune_lora.py
echo ""

# Step 4: After evaluation
echo ">>> Step 4/4: Running after fine-tuning evaluation..."
python3 03_after_eval.py
echo ""

# Step 5: Compare
echo ">>> Final: Generating comparison report..."
python3 04_compare_results.py
echo ""

echo "=============================================="
echo "  Pipeline complete! Check results/ folder."
echo "=============================================="
