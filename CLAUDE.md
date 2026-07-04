# CLAUDE.md — LLM Fine-Tuning Project Guide

> **This file explains how to use this codebase, what each script does internally,
> and the full implementation details of PEFT LoRA fine-tuning on Apple Silicon.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites & Setup](#2-prerequisites--setup)
3. [How to Use — Step-by-Step](#3-how-to-use--step-by-step)
4. [Implementation Details](#4-implementation-details)
   - [What is PEFT LoRA?](#41-what-is-peft-lora)
   - [Why MLX instead of bitsandbytes/QLoRA?](#42-why-mlx-instead-of-bitsandbytesqlora)
   - [Dataset & Formatting](#43-dataset--formatting)
   - [Script-by-Script Walkthrough](#44-script-by-script-walkthrough)
   - [LoRA Hyperparameters Explained](#45-lora-hyperparameters-explained)
   - [Evaluation Metrics Explained](#46-evaluation-metrics-explained)
5. [Expected Outputs](#5-expected-outputs)
6. [File Structure Reference](#6-file-structure-reference)
7. [Troubleshooting](#7-troubleshooting)
8. [How to Extend This Project](#8-how-to-extend-this-project)

---

## 1. Project Overview

This project demonstrates **PEFT LoRA fine-tuning** of a large language model (LLM)
on a **domain-specific Medical QA** task. The goal is to show a clear, measurable
improvement in response quality before and after fine-tuning.

| | Detail |
|---|---|
| **Base Model** | `Qwen2.5-1.5B-Instruct` (4-bit quantized via MLX) |
| **Fine-Tuning Method** | PEFT LoRA (Parameter-Efficient Fine-Tuning) |
| **Dataset** | `medalpaca/medical_meadow_healthcaremagic` (doctor-patient Q&A) |
| **Framework** | Apple MLX (`mlx-lm`) — optimized for Apple Silicon |
| **Hardware Target** | M1 Mac, 8GB unified RAM |
| **Auth Required** | ❌ None — fully open model and dataset |

**What the project proves:**
- A base 1.5B model gives vague, generic medical answers
- After LoRA fine-tuning on ~2,000 medical Q&A examples, the same model gives
  specific, medically-grounded responses
- The improvement is measured with ROUGE-1 and ROUGE-L metrics

---

## 2. Prerequisites & Setup

### System Requirements
- macOS with **Apple Silicon** (M1, M2, or M3)
- **8GB+ unified RAM** (16GB recommended for comfort)
- **Python 3.10, 3.11, or 3.12** (Python 3.13 has known issues with mlx-lm)
- ~4GB free disk space (model download + dataset)
- Internet connection (first run only)

### Check your Python version
```bash
python3 --version
# Should output: Python 3.10.x / 3.11.x / 3.12.x
```

### Install dependencies
```bash
cd /Users/subhadeepdas/llm-finetuning
pip3 install -r requirements.txt
```

**What gets installed:**
| Package | Version | Purpose |
|---|---|---|
| `mlx-lm` | 0.21.5 | Apple Silicon LoRA training + inference |
| `transformers` | 4.46.3 | Tokenizer & model utilities |
| `datasets` | ≥2.18 | HuggingFace dataset loading |
| `rouge-score` | ≥0.1.2 | ROUGE-1 and ROUGE-L evaluation metrics |
| `rich` | ≥13.7 | Beautiful terminal output / tables |
| `huggingface_hub` | ≥0.21 | Model downloading from HF Hub |

---

## 3. How to Use — Step-by-Step

### Option A: Run everything at once (recommended for first run)
```bash
chmod +x run_all.sh
./run_all.sh
```

### Option B: Run each step manually

#### Step 0 — Install dependencies
```bash
pip3 install -r requirements.txt
```

#### Step 1 — Prepare the dataset
```bash
python3 data/prepare_dataset.py
```
**What happens:**
- Downloads `medalpaca/medical_meadow_healthcaremagic` from HuggingFace (~500MB, cached after first run)
- Filters 226,000 examples down to high-quality ones (output length > 80 chars)
- Randomly samples 2,220 examples
- Saves:
  - `data/train.jsonl` — 2,000 training examples
  - `data/valid.jsonl` — 200 validation examples
  - `data/test_questions.json` — 20 held-out test questions (used for before/after eval)

**Time:** ~2–5 minutes (mostly download on first run)

---

#### Step 2 — Before evaluation (base model, no fine-tuning)
```bash
python3 01_before_eval.py
```
**What happens:**
- Downloads `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~900MB, cached after first run)
- Loads the model WITHOUT any LoRA adapters (pure base model)
- Runs inference on all 20 medical test questions
- Prints responses to terminal
- Saves all results to `results/before_results.json`

**Time:** ~5–10 minutes (mostly model download on first run, then ~15s/question inference)

---

#### Step 3 — LoRA Fine-Tuning
```bash
python3 02_finetune_lora.py
```
**What happens:**
- Validates that `data/train.jsonl` and `data/valid.jsonl` exist
- Displays the full LoRA configuration table
- Launches `mlx_lm.lora` training subprocess with these settings:
  - Rank 8 adapters on all 28 transformer layers
  - 300 iterations over the training set (equivalent to ~0.6 epochs)
  - Validation loss checked every 50 steps
  - Checkpoint saved every 50 steps
- Streams live training output to terminal (watch the loss decrease!)
- Saves LoRA adapter weights to `adapters/` directory
- Saves training metadata to `results/training_metadata.json`

**Time:** ~15–20 minutes on M1 8GB
**Expected loss curve:** starts ~2.5, should drop to ~0.7–1.1 by step 300

> ⚠️ **Do NOT close the terminal during training.** If interrupted, partial
> adapters may be saved — you can re-run and it will overwrite them.

---

#### Step 4 — After evaluation (fine-tuned model)
```bash
python3 03_after_eval.py
```
**What happens:**
- Verifies that `adapters/` directory exists with saved weights
- Loads the base model AND applies the saved LoRA adapter weights on top
- Runs inference on the **exact same 20 test questions** as Step 2
- Prints responses to terminal
- Saves all results to `results/after_results.json`

**Time:** ~5–10 minutes (model already cached, no download)

---

#### Step 5 — Compare results
```bash
python3 04_compare_results.py
```
**What happens:**
- Loads `results/before_results.json` and `results/after_results.json`
- Aligns results by question ID (same question, different model)
- Computes per-question and aggregate metrics:
  - ROUGE-1 F1 score
  - ROUGE-L F1 score
  - Average response word count
- Prints 3 tables:
  1. Aggregate metrics (before vs after side-by-side)
  2. Per-question ROUGE-L scores with delta (▲ green / ▼ red)
  3. Side-by-side qualitative examples (question + ground truth + before + after)
- Saves full report to `results/comparison_report.json`

**Time:** ~10 seconds

---

## 4. Implementation Details

### 4.1 What is PEFT LoRA?

**PEFT** = Parameter-Efficient Fine-Tuning. Instead of updating all model weights
(which would require enormous GPU memory), PEFT techniques update only a tiny
fraction of parameters.

**LoRA** = Low-Rank Adaptation (Hu et al., 2021 — [paper](https://arxiv.org/abs/2106.09685))

#### The core idea:
For each attention weight matrix `W` (e.g. query projection `W_q`):

```
Full fine-tuning:   W_new = W + ΔW           # ΔW is same size as W (huge)
LoRA:               W_new = W + (A × B)       # A and B are tiny low-rank matrices
```

Where:
- `W` is frozen (not updated during training)
- `A` has shape `[d_model × rank]`
- `B` has shape `[rank × d_model]`
- `rank` is a small number (we use 8)

#### Why this works:
Research shows that the change in weights during fine-tuning has a low intrinsic
rank — meaning you don't need full-rank updates. A rank-8 approximation captures
most of the useful adaptation signal.

#### Parameter count comparison for Qwen2.5-1.5B:
```
Full fine-tuning:  1,543,714,816 trainable parameters  (1.5B)
LoRA (rank=8):     ~786,432 trainable parameters        (0.05% of total!)
```

This is why LoRA can run on 8GB RAM when full fine-tuning would need 40GB+.

---

### 4.2 Why MLX instead of bitsandbytes/QLoRA?

**QLoRA** is the most common fine-tuning approach on NVIDIA GPUs. It uses:
- `bitsandbytes` library for 4-bit NF4 quantization
- CUDA backend for computation

**Problem:** `bitsandbytes` is a CUDA-only library. Apple Silicon uses MPS
(Metal Performance Shaders), not CUDA. Attempting to use bitsandbytes on M1
causes import errors or silent computation failures.

**Solution:** Apple's `mlx-lm` library:
- Built natively for Apple Silicon's unified memory architecture
- Supports LoRA fine-tuning natively (similar concept to QLoRA, different backend)
- Handles model quantization internally (the `-4bit` model we use is already 4-bit)
- Significantly more stable and performant than PyTorch/MPS for training

**Conceptual equivalence:**
```
NVIDIA GPU:   QLoRA  = bitsandbytes (4-bit) + PEFT LoRA + PyTorch CUDA
Apple M1:     MLX-LM = mlx quantization (4-bit) + LoRA + MLX Metal
```

The results and concepts are the same — only the underlying compute backend differs.

---

### 4.3 Dataset & Formatting

**Source:** `medalpaca/medical_meadow_healthcaremagic`
- 226,000+ real anonymized doctor-patient conversations from HealthcareMagic
- Columns: `instruction` (patient question), `input` (empty), `output` (doctor response)

**Why this dataset:**
- Shows dramatic improvement (base model is generic, fine-tuned is specific)
- Large enough to demonstrate learning (226K examples)
- Publicly available with no auth

**MLX-LM data format** (chat template JSONL):
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a knowledgeable and compassionate medical expert..."
    },
    {
      "role": "user",
      "content": "I have been having chest pain for 3 days..."
    },
    {
      "role": "assistant",
      "content": "Chest pain lasting 3 days requires immediate evaluation..."
    }
  ]
}
```

Each line in `train.jsonl` / `valid.jsonl` is one such JSON object.
`mlx_lm.lora` reads these files automatically when given the `--data ./data` flag.

**Sampling strategy:**
```python
# We sample 2,220 total:
# - First 20  → test_questions.json (held-out, NEVER seen during training)
# - Next 200  → valid.jsonl (validation, checks for overfitting)
# - Last 2000 → train.jsonl (actual training data)
```

The held-out split ensures the before/after comparison is on truly unseen questions.

---

### 4.4 Script-by-Script Walkthrough

#### `data/prepare_dataset.py`
```
load_dataset()          # HuggingFace datasets library
  → filter()            # Remove short/empty examples
    → random.sample()   # Sample 2,220 examples with seed=42
      → split()         # 20 test | 200 valid | 2000 train
        → write_jsonl() # Save in MLX chat format
```

#### `01_before_eval.py`
```
mlx_lm.load(MODEL_ID)                    # Load base model, no adapters
  → tokenizer.apply_chat_template()      # Format prompt with system + user roles
    → mlx_lm.generate()                  # Run inference (pure base model)
      → save results/before_results.json
```

#### `02_finetune_lora.py`
```
check_prerequisites()                     # Verify train.jsonl, valid.jsonl exist
  → print_lora_config()                   # Display hyperparameter table
    → subprocess.run([                    # Launch mlx_lm.lora subprocess
        "python3", "-m", "mlx_lm.lora",
        "--model", MODEL_ID,
        "--train",
        "--data", DATA_DIR,
        "--adapter-path", ADAPTER_DIR,
        "--num-layers", "28",
        "--iters", "300",
        "-c", "lora_config.yaml",
        ...
      ])                                  # Live output streams to terminal
        → save adapters/                  # adapter_config.json + adapters.npz
```

#### `03_after_eval.py`
```
mlx_lm.load(MODEL_ID, adapter_path=ADAPTER_DIR)  # Load model + inject LoRA weights
  → tokenizer.apply_chat_template()               # Same prompt format as before
    → mlx_lm.generate()                            # Run inference (fine-tuned model)
      → save results/after_results.json
```

#### `04_compare_results.py`
```
load before_results.json + after_results.json
  → rouge_scorer.score(prediction, reference)     # ROUGE-1 and ROUGE-L
    → compute averages across 20 questions
      → rich.Table() × 3                           # Print 3 formatted tables
        → Panel() side-by-side examples            # Qualitative comparison
          → save results/comparison_report.json
```

---

### 4.5 LoRA Hyperparameters Explained

| Parameter | Value | What it means | Impact |
|---|---|---|---|
| `lora_rank` | 8 | Rank of adapter matrices A and B | Higher = more capacity, more memory |
| `lora_alpha` | 16 | Scaling: effective LR = α/r × lr | We use 2× rank (standard practice) |
| `lora_layers` | 28 | How many transformer layers get adapters | Adapts all 28 layers of Qwen2.5-1.5B |
| `learning_rate` | 1e-4 | AdamW step size | Too high → unstable; too low → slow |
| `batch_size` | 2 | Examples per gradient step | Reduces peak memory usage to prevent swap on 8GB RAM |
| `iters` | 300 | Total training steps | ~15-20 min on M1; allows for actual convergence |
| `steps_per_eval` | 50 | Validate every N steps | Monitor overfitting |
| `max_seq_length` | 512 | Truncate long sequences | Protects against OOM & speeds up training |
| `lora_dropout` | 0.05 | Dropout on LoRA matrices | Light regularization |

**To improve results (if you have time/resources):**
```python
NUM_ITERS = 300          # More training steps
LORA_RANK = 16           # Higher capacity adapters
LORA_LAYERS = 32         # Adapt all layers
LEARNING_RATE = 5e-5     # More conservative LR for longer runs
```

---

### 4.6 Evaluation Metrics Explained

#### ROUGE-L (Primary Metric)
- **Full name:** Recall-Oriented Understudy for Gisting Evaluation — Longest Common Subsequence
- **What it measures:** How much the model's answer overlaps with the reference answer,
  using the longest common subsequence (order-aware)
- **Range:** 0.0 (no overlap) → 1.0 (exact match)
- **Example:**
  ```
  Reference:  "Take ibuprofen for pain and rest for 3 days"
  Before:     "You should consult a doctor"               → ROUGE-L: 0.08
  After:      "Take ibuprofen 400mg for pain, rest well"  → ROUGE-L: 0.54
  ```

#### ROUGE-1
- Measures **unigram (single word) overlap** between prediction and reference
- Less strict than ROUGE-L (ignores word order)
- Good for checking if the model uses the right vocabulary/terminology

#### What to expect:
| Scenario | ROUGE-L |
|---|---|
| Random text | 0.00–0.05 |
| Generic/vague answer | 0.05–0.15 |
| Partially correct answer | 0.15–0.35 |
| Good specific answer | 0.35–0.60 |
| Near-perfect match | 0.60–1.00 |

> **Note:** Medical QA is open-ended — there's no single "correct" answer. ROUGE
> measures lexical overlap with one reference, so even good answers may score 0.30–0.45.
> The important thing is the **relative improvement** (before → after), not the absolute score.

---

## 5. Expected Outputs

### Terminal output from `04_compare_results.py`:
```
┌─────────────────────────────────────────────────────────────────────┐
│          📊  Before vs After Fine-Tuning — Aggregate Metrics         │
├────────────────────────┬──────────────────┬──────────────┬──────────┤
│ Metric                 │ Before (Base)    │ After (LoRA) │ Improvement │
├────────────────────────┼──────────────────┼──────────────┼──────────┤
│ ROUGE-1 (F1)           │ 0.1234           │ 0.3456       │ ▲ 180%   │
│ ROUGE-L (F1)           │ 0.0987           │ 0.2890       │ ▲ 193%   │
│ Avg Response Length    │ 42 words         │ 118 words    │ ▲ 181%   │
│ Test Examples          │ 20               │ 20           │ —        │
└────────────────────────┴──────────────────┴──────────────┴──────────┘
```

### Files produced:
```
results/
├── before_results.json       # Array of 20 {question, ground_truth, model_response, latency}
├── after_results.json        # Same structure, fine-tuned model responses
├── training_metadata.json    # LoRA config + timestamps + training duration
└── comparison_report.json    # Aggregate + per-question ROUGE scores

adapters/
├── adapter_config.json       # LoRA architecture config (rank, layers, targets)
└── adapters.npz              # Saved LoRA weight matrices (A and B for each layer)
```

---

## 6. File Structure Reference

```
llm-finetuning/
│
├── CLAUDE.md                  ← You are here
├── README.md                  ← Public-facing project documentation
├── requirements.txt           ← Pinned Python dependencies
├── run_all.sh                 ← One-command full pipeline runner
│
├── data/
│   ├── prepare_dataset.py     ← Download + format + split dataset
│   ├── train.jsonl            ← 2,000 training examples (MLX chat format)
│   ├── valid.jsonl            ← 200 validation examples
│   └── test_questions.json    ← 20 held-out Q&A pairs for evaluation
│
├── 01_before_eval.py          ← Inference: base model, no adapters
├── 02_finetune_lora.py        ← Training: LoRA fine-tuning via mlx_lm.lora
├── 03_after_eval.py           ← Inference: base model + LoRA adapters
├── 04_compare_results.py      ← Metrics + comparison report
│
├── adapters/                  ← Generated by 02_finetune_lora.py
│   ├── adapter_config.json
│   └── adapters.npz
│
└── results/                   ← Generated by eval scripts
    ├── before_results.json
    ├── after_results.json
    ├── training_metadata.json
    └── comparison_report.json
```

---

## 7. Troubleshooting

### ❌ `AttributeError: 'str' object has no attribute '__module__'`
**Cause:** `mlx-lm` latest version is incompatible with `transformers` 5.x
**Fix:**
```bash
pip3 install "mlx-lm==0.21.5" "transformers==4.46.3"
```

### ❌ `ModuleNotFoundError: No module named 'mlx_lm'`
**Fix:**
```bash
pip3 install -r requirements.txt
```

### ❌ Dataset download fails / hangs
**Fix:** Check internet connection. The dataset is ~500MB. Try:
```bash
python3 -c "from datasets import load_dataset; ds = load_dataset('medalpaca/medical_meadow_healthcaremagic', split='train'); print(len(ds))"
```

### ❌ Training crashes with memory error (OOM)
**Fix:** Reduce batch size in `02_finetune_lora.py`:
```python
BATCH_SIZE = 2      # Change from 4 → 2
MAX_SEQ_LEN = 512   # Change from 1024 → 512
```

### ❌ `adapters/` folder is empty after training
**Cause:** Training was interrupted before the first checkpoint save (every 50 steps)
**Fix:** Re-run `python3 02_finetune_lora.py`. It overwrites adapters cleanly.

### ❌ Inference is very slow (>60s per question)
**This is normal on CPU.** MLX uses Metal GPU acceleration automatically on M1.
Make sure you're using the `4bit` model variant (it's in the MODEL_ID already).
Close other memory-heavy apps (Chrome, etc.) to free up unified memory.

### ❌ ROUGE scores are very low (both before and after ~0.05)
**This can happen** with open-ended medical Q&A — the reference answer may use
different phrasing than the model's correct answer. Focus on the **relative improvement**
(after > before) and the **qualitative side-by-side examples** printed by script 04.

---

## 8. How to Extend This Project

### Change the domain (e.g. Legal QA, Financial QA)
1. Replace the dataset in `data/prepare_dataset.py`:
   ```python
   ds = load_dataset("nguyen2010/legal_qa", split="train")        # Legal
   ds = load_dataset("virattt/financial-qa-10K", split="train")   # Financial
   ```
2. Update the `SYSTEM_PROMPT` in all eval scripts to match the new domain
3. Re-run the full pipeline

### Train for longer (better results)
In `02_finetune_lora.py`:
```python
NUM_ITERS = 300    # More iterations → lower loss → better responses
BATCH_SIZE = 4     # Keep at 4 for 8GB RAM
```

### Use a larger model (if you have more RAM)
```python
MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"   # 3B model (needs ~16GB RAM)
MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"   # 7B model (needs ~32GB RAM)
```

### Save and share your fine-tuned model
After training, fuse LoRA weights into the base model:
```bash
python3 -m mlx_lm.fuse \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --adapter-path ./adapters \
    --save-path ./fused_model
```
The `fused_model/` directory can then be used directly or uploaded to HuggingFace.

### Add more evaluation metrics
In `04_compare_results.py`, you can add:
```python
from nltk.translate.bleu_score import sentence_bleu
# BLEU score for n-gram precision
bleu = sentence_bleu([reference.split()], prediction.split())
```

---

*Generated for the LLM Fine-Tuning project — Qwen2.5-1.5B + LoRA on Medical QA*
*Framework: Apple MLX | Hardware: M1 Mac 8GB | Task: Domain-Specific QA*
