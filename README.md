# LLM Fine-Tuning: Before vs After — Medical QA with PEFT LoRA

A project demonstrating how **PEFT LoRA fine-tuning** improves an LLM's performance on domain-specific Medical Question Answering. Uses Apple MLX for efficient training on M1 Mac.

## 🏗️ Architecture

```
Base Model: mlx-community/Qwen2.5-1.5B-Instruct-4bit  (open, no auth needed)
Dataset:    keivalya/MedQuad-MedicalQnADataset          (16K doctor-patient Q&A, MedQuAD mirror)
Method:     PEFT LoRA (rank=8, alpha=16, 28 layers)
Framework:  Apple MLX (optimized for Apple Silicon)
```

## 🔬 What is LoRA?

**LoRA (Low-Rank Adaptation)** freezes the pre-trained model weights and injects small trainable matrices into the attention layers.

```
Full fine-tuning: train 1,500,000,000 parameters
LoRA fine-tuning: train ~500,000 parameters  (<0.1% of total!)
```

This makes fine-tuning feasible on consumer hardware (M1 8GB RAM).

## 📁 Project Structure

```
llm-finetuning/
├── data/
│   ├── prepare_dataset.py     # Download & format dataset
│   ├── train.jsonl            # 2,000 training examples (auto-generated)
│   ├── valid.jsonl            # 200 validation examples (auto-generated)
│   └── test_questions.json    # 20 held-out test questions
│
├── 01_before_eval.py          # Evaluate base model (no adapters)
├── 02_finetune_lora.py        # LoRA fine-tuning
├── 03_after_eval.py           # Evaluate fine-tuned model (+ adapters)
├── 04_compare_results.py      # Side-by-side metrics & examples
│
├── adapters/                  # LoRA adapter weights (auto-generated)
├── results/
│   ├── before_results.json    # Base model outputs
│   ├── after_results.json     # Fine-tuned model outputs
│   └── comparison_report.json # Final metrics
│
├── run_all.sh                 # Run entire pipeline in one command
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3) — 8GB+ RAM
- Python 3.10+
- No HuggingFace account needed (model is fully open)

### Option A: Run everything at once
```bash
cd llm-finetuning
chmod +x run_all.sh
./run_all.sh
```

### Option B: Step by step

**1. Install dependencies**
```bash
pip3 install -r requirements.txt
```

**2. Prepare the dataset**
```bash
python3 data/prepare_dataset.py
```
Downloads `keivalya/MedQuad-MedicalQnADataset`, samples 2,220 examples,
and splits them into `train.jsonl`, `valid.jsonl`, and `test_questions.json`.

**3. Before evaluation** (base model, no fine-tuning)
```bash
python3 01_before_eval.py
```
Runs Qwen2.5-1.5B on 20 medical questions. Saves to `results/before_results.json`.

**4. Fine-tune with LoRA**
```bash
python3 02_finetune_lora.py
```
Trains LoRA adapters for 600 iterations (~40–50 min on M1 8GB). Saves weights to `adapters/`.

**5. After evaluation** (fine-tuned model)
```bash
python3 03_after_eval.py
```
Runs same 20 questions with LoRA adapters loaded. Saves to `results/after_results.json`.

**6. Compare results**
```bash
python3 04_compare_results.py
```
Prints side-by-side comparison tables and ROUGE metrics.

## ⚙️ LoRA Hyperparameters

| Parameter       | Value  | Description                              |
|----------------|--------|------------------------------------------|
| Rank (r)        | 8      | Low-rank decomposition dimension         |
| Alpha (α)       | 16     | Scaling factor (= 2 × rank)              |
| LoRA Layers     | 28     | All 28 transformer layers adapted        |
| Learning Rate   | 1e-4   | AdamW optimizer                          |
| Batch Size      | 2      | Reduces peak memory to prevent RAM swap  |
| Iterations      | 300    | ~15–20 min training on M1               |

## 📊 Expected Results

After fine-tuning, you should see:

| Metric       | Before | After (expected) |
|-------------|--------|------------------|
| ROUGE-L F1  | ~0.10  | ~0.25–0.40       |
| ROUGE-1 F1  | ~0.15  | ~0.30–0.45       |

**Qualitative improvement**: Base model gives generic answers like *"consult a doctor"*. Fine-tuned model gives specific, medically-grounded responses aligned with the training data.

## ❓ Why MLX instead of bitsandbytes/QLoRA?

`bitsandbytes` (the standard QLoRA library) **does not support Apple Silicon MPS**. `mlx-lm` is Apple's own ML framework, purpose-built for M1/M2/M3 unified memory architecture — it's the correct and stable choice for LoRA on Mac.

## 📚 References

- [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) — Apple's LLM fine-tuning library
- [PEFT Library](https://huggingface.co/docs/peft) — HuggingFace parameter-efficient fine-tuning
- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [MedQuAD Medical QA Dataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — Base model
