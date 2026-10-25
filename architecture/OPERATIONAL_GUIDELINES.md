# Operational Guidelines & Governance Rules

This document establishes the **rules, constraints, and protocols** governing how AI assistants, autonomous agents, and human developers must read, inspect, modify, and write files in the **Prashn-Setu** codebase.

---

## 1. Reading Protocols (How Agents/Models Must Read Files)

### Rule 1.1: Single Source of Truth for Configuration
* **Never assume or hardcode default hyperparameters** in downstream scripts.
* Before proposing or executing any script, read [`config.py`](file:///mnt/a/iiits/5/NLP/PROJECT/config.py) to check the active values of:
  * Sequence lengths (`max_source_length`, `max_target_length`)
  * Target language tags (`<2hi>`)
  * Batch sizes and learning rates.

### Rule 1.2: Strict UTF-8 Hindi Text Encoding
* When reading any dataset file (`.jsonl`, `.json`, `.csv`) containing Hindi or Devanagari characters:
  * Always explicitly specify `encoding="utf-8"`.
  * Never use standard Python string decoding without verifying character integrity (watch for corrupted Unicode escape sequences like `\u092a\u094c\u0927\u0947`).

### Rule 1.3: Tokenizer Special Token Integrity
* When inspecting or parsing input/target sequences:
  * The delimiter tokens `<s>`, `</s>`, and `<2hi>` are **functional semantic markers** required by IndicBART.
  * Do not strip them out during dataset loading or batch preparation.
  * Only strip special tokens during post-processing for human readability or metric evaluation.

### Rule 1.4: Checkpoint Inspection Protocol
* When loading a checkpoint to perform inference or evaluation:
  * First inspect if `adapter_config.json` exists in the checkpoint directory.
  * If `adapter_config.json` exists: It is a **PEFT/LoRA adapter**. You must instantiate the base model first and wrap it via `PeftModel.from_pretrained()`.
  * If `adapter_config.json` is absent: It is a **full model checkpoint**. Load directly via `AutoModelForSeq2SeqLM.from_pretrained()`.

---

## 2. Modifying & Editing Protocols (How to Alter Existing Files)

### Rule 2.1: Configuration Updates
* Any global change to paths, batch sizes, or model identifiers must be made inside `config.py`.
* Do not scatter custom path overrides or magic numbers across `train.py`, `adapt_gold.py`, or `dataset.py`.

### Rule 2.2: Data Immutability
* Files inside `data/indic_question_generation_hi/` and exported splits (`hi_train.jsonl`, `hi_validation.jsonl`, `hi_test.jsonl`) are **immutable raw references**.
* Never overwrite or append experimental or synthetic data directly into the raw splits.
* Place custom or curated textbook data into `data/ncert_gold_hindi.jsonl` or separate explicitly named files (e.g., `data/gold_eval_split.jsonl`).

### Rule 2.3: Safeguards Against Catastrophic Forgetting
* When modifying `adapt_gold.py`:
  * **Never perform unconstrained full-parameter fine-tuning with a high learning rate ($> 2\times 10^{-5}$)** on small gold datasets ($\le 500$ samples).
  * Always enforce either:
    1. **LoRA (PEFT)** on query/value attention projections (`q_proj`, `v_proj`), OR
    2. Freeze the encoder and only fine-tune the top decoder layers with an early stopping patience of $\le 2$ epochs.

### Rule 2.4: Vocabulary & Model Alignment
* If modifying `prune_vocab.py`:
  * Ensure that Latin letters, numerals (`0-9`), math operators, and punctuation are **always preserved** in the retained token set. School textbooks rely heavily on digits, dates, formulas, and units.
  * If the vocabulary is pruned, ensure that downstream training scripts are passed the `--use_pruned` flag or read from `checkpoints/indicbart_hi_pruned`.

---

## 3. Writing Protocols (How to Generate New Files or Artifacts)

### Rule 3.1: Python Code Standards
* All new Python modules must:
  * Include module-level docstrings explaining purpose, inputs, and outputs.
  * Utilize Python type hints (`from typing import List, Dict, Optional, Tuple`).
  * Follow PEP 8 standards with descriptive variable naming.
  * Support command-line argument overrides via `argparse`.

### Rule 3.2: Native Devanagari JSON Serialization
* Whenever saving JSON or JSONL files containing Hindi strings:
  ```python
  # CORRECT:
  json.dump(data, f, ensure_ascii=False, indent=2)

  # FORBIDDEN (Escapes Devanagari into unreadable ascii codes):
  json.dump(data, f, ensure_ascii=True)
```

### Rule 3.3: Metric Computation Requirements
* Never report only BLEU or ROUGE-1 for Hindi generation.
* Any new evaluation script or benchmarking experiment **must compute and report chrF++**:
  * Hindi is agglutinative and morphologically rich. `chrF++` evaluates character n-grams and word n-grams, preventing spurious zero-scores on valid inflectional variations.

### Rule 3.4: Automated Architecture Tracking
* After creating, deleting, or significantly modifying any file in the project:
  * Execute `python update_architecture_state.py`
  * This regenerates [`architecture/SYSTEM_STATE_TRACKER.md`](file:///mnt/a/iiits/5/NLP/PROJECT/architecture/SYSTEM_STATE_TRACKER.md), maintaining an up-to-date ledger of code lines, checksums, and dataset checkpoints.

---

## 4. Summary of Golden Rules

```
┌────────────────────────────────────────────────────────────────────────┐
│                          THE 5 GOLDEN RULES                            │
│                                                                        │
│ 1. ALL HYPERPARAMETERS RESIDE IN config.py                             │
│ 2. PRESERVE UTF-8 (ensure_ascii=False) ON ALL HINDI TEXT               │
│ 3. NEVER FULL-FINETUNE ON SMALL GOLD SETS WITHOUT LoRA / LOW LR       │
│ 4. ALWAYS REPORT chrF++ ALONGSIDE BLEU FOR HINDI EVALUATION            │
│ 5. RE-RUN update_architecture_state.py AFTER STRUCTURAL EDITS          │
└────────────────────────────────────────────────────────────────────────┘
```
