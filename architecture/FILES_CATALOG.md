# Codebase File Catalog & Rationale: Prashn-Setu

This document lists every file in the clean, modularized codebase, explaining **what it is**, **why it is present**, and **how it functions** within the pipeline.

---

## 1. Unified Entry Point & Execution Launchers

### 📄 [`run.py`](file:///mnt/a/iiits/5/NLP/PROJECT/run.py)
* **What it is:** Unified single command center CLI for the entire project.
* **Why it is present:** Replaces cluttered loose root scripts with a clean, subcommand-driven command line interface (`python run.py [download|prune|train-stage1|adapt-stage2|infer|evaluate|update-tracker]`).

### 📁 [`scripts/`](file:///mnt/a/iiits/5/NLP/PROJECT/scripts)
* **What it is:** Shell automation scripts for step-by-step pipeline execution (`01_download_data.sh` through `06_interactive_inference.sh`).
* **Why it is present:** Provides single-click shell launchers for headless servers, batch jobs, and SLURM/HPC cluster execution.

---

## 2. Core Python Source Packages (`src/`)

### 📄 [`src/config.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/config.py)
* **What it is:** Centralized configuration module using Python `@dataclass` structures (`ModelConfig`, `DataConfig`, `TrainingConfig`).
* **Why it is present:** 
  * Prevents hardcoding of hyperparameters, model identifiers, and paths across modules.
  * Dynamically resolves absolute paths relative to `PROJECT_ROOT`.
  * Enforces uniform maximum sequence lengths (`max_source_length=512`, `max_target_length=64`) and language tags (`<2hi>`).

### 📁 [`src/data/`](file:///mnt/a/iiits/5/NLP/PROJECT/src/data)
* **[`src/data/download.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/data/download.py):** Downloads only the Hindi subset of `ai4bharat/IndicQuestionGeneration`, caches the Arrow dataset, exports JSONL files (`hi_train.jsonl`, `hi_validation.jsonl`, `hi_test.jsonl`), and creates the NCERT gold curriculum template (`data/ncert_gold_hindi.jsonl`).
* **[`src/data/dataset.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/data/dataset.py):** PyTorch `Dataset` wrapper (`IndicQuestionGenerationDataset`). Formats inputs into IndicBART syntax (`"{answer} </s> {context} </s> <2hi>"`), tokenizes with `add_special_tokens=False`, and sets label padding to `-100` for loss masking.

### 📁 [`src/models/`](file:///mnt/a/iiits/5/NLP/PROJECT/src/models)
* **[`src/models/prune_vocab.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/models/prune_vocab.py):** Vocabulary pruner and embedding slicer. Discards non-Hindi Indic Unicode blocks (Bengali, Tamil, Telugu, Malayalam, Gujarati, Punjabi, Odia, Kannada), reducing vocabulary from 64,014 down to ~18,000 Hindi/Latin/Math tokens. Saves ~50M parameters and eliminates cross-script hallucinations.

### 📁 [`src/training/`](file:///mnt/a/iiits/5/NLP/PROJECT/src/training)
* **[`src/training/metrics.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/training/metrics.py):** Modular metric computation engine computing SacreBLEU, ROUGE-1/2/L, and **chrF++** (character and word n-grams).
* **[`src/training/train_stage1.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/training/train_stage1.py):** Stage 1 fine-tuning pipeline on the IndicNLG Hindi split using `Seq2SeqTrainer`, gradient accumulation, FP16/BF16, and validation checkpoints.
* **[`src/training/adapt_stage2.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/training/adapt_stage2.py):** Stage 2 domain adaptation engine on 300–500 verified NCERT textbook questions with **LoRA (PEFT)** on `q_proj` and `v_proj` to prevent catastrophic forgetting.

### 📁 [`src/evaluation/`](file:///mnt/a/iiits/5/NLP/PROJECT/src/evaluation)
* **[`src/evaluation/inference.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/evaluation/inference.py):** Real-time generation class (`QuestionGenerator`) and interactive terminal session. Supports beam search ($k=4$), repetition penalty, and automatic LoRA adapter loading.
* **[`src/evaluation/evaluate.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/evaluation/evaluate.py):** Benchmarking engine that generates predictions across test splits, computes metrics, and exports qualitative side-by-side JSON reports (`eval_results.json`).

### 📁 [`src/utils/`](file:///mnt/a/iiits/5/NLP/PROJECT/src/utils)
* **[`src/utils/tracker.py`](file:///mnt/a/iiits/5/NLP/PROJECT/src/utils/tracker.py):** State inspection engine. Scans all files, computes SHA-256 hashes, counts lines of code, detects data/checkpoint creation, and regenerates `architecture/SYSTEM_STATE_TRACKER.md`.

---

## 3. Directory Hierarchy

```
PROJECT/
├── architecture/                             # System design specifications
│   ├── README.md                             # Architecture index & guide
│   ├── FILES_CATALOG.md                      # (This file) Complete file inventory & rationale
│   ├── COMPONENT_INTERACTIONS.md             # Diagrams & interaction workflows
│   ├── SOLUTION_ANALYSIS_AND_ALTERNATIVES.md # Benchmark against other models & paradigms
│   ├── PREREQUISITES_AND_ENVIRONMENT_SETUP.md# Hardware & software dependency specs
│   ├── OPERATIONAL_GUIDELINES.md             # Rules for agents & developers
│   └── SYSTEM_STATE_TRACKER.md               # Live automated change log & state
├── src/                                      # Clean Python source package
│   ├── config.py                             # Centralized settings & configurations
│   ├── data/
│   │   ├── download.py                       # Data acquisition & JSONL export
│   │   └── dataset.py                        # Dataset loader & sequence formatting
│   ├── models/
│   │   └── prune_vocab.py                    # Script filtering & embedding slicer
│   ├── training/
│   │   ├── train_stage1.py                   # Stage 1 fine-tuning engine
│   │   ├── adapt_stage2.py                   # Stage 2 NCERT Gold LoRA adaptation
│   │   └── metrics.py                        # Shared evaluation metrics
│   ├── evaluation/
│   │   ├── inference.py                      # Interactive CLI & inference API
│   │   └── evaluate.py                       # Benchmarking against test splits
│   └── utils/
│       └── tracker.py                        # Architecture state tracker
├── scripts/                                  # Shell automation runners
│   ├── 01_download_data.sh
│   ├── 02_prune_vocab.sh
│   ├── 03_train_stage1.sh
│   ├── 04_adapt_stage2.sh
│   ├── 05_evaluate.sh
│   └── 06_interactive_inference.sh
├── run.py                                    # Unified CLI Command Center
├── requirements.txt                          # Python library dependencies
└── README.md                                 # Top-level quickstart guide
```
