# Operational Guidelines & Governance Rules

This document establishes the **rules, constraints, and protocols** governing how AI assistants, autonomous agents, and human developers must read, inspect, modify, and write files in the **Prashn-Setu** codebase.

---

## 1. Reading & Inspection Protocols

### Rule 1.1: Single Source of Truth for Configuration
* **Never assume or hardcode default hyperparameters** in downstream modules.
* Read [`src/config.py`](../src/config.py) to check active values of:
  * Sequence lengths (`max_source_length=512`, `max_target_length=64`)
  * Target language tags (`<2hi>`)
  * Model identifiers and default directories.

### Rule 1.2: Strict UTF-8 Hindi Text Encoding & Devanagari Integrity
* When reading or writing any dataset or artifact containing Hindi or Devanagari characters:
  * Always explicitly specify `encoding="utf-8"`.
  * Ensure Unicode `NFC` normalization is maintained.
  * Beware of decomposed visual-order short-i matras (`ि` U+093F placed before consonants) or space-detached lower matras (`ु` U+0941). When combining marks are preceded by spaces or newlines, renderers display empty dotted-circle placeholders `◌`. Always sanitize text through `repair_hindi_devanagari_text()`.

### Rule 1.3: Tokenizer Special Token Integrity
* In IndicBART, `<s>` (64000), `</s>` (64001), and `<2hi>` (64006) are **functional syntactic tokens**.
* Never strip them out during dataset loading or generation conditioning.
* Only strip special tokens during post-processing for human readability or metric calculation.

### Rule 1.4: Checkpoint Inspection Protocol
* Check if `adapter_config.json` exists in the checkpoint directory:
  * If present: It is a **PEFT/LoRA adapter**. Load the base model first (`checkpoints/stage1_indicnlg_qg`), then attach the adapter (`PeftModel.from_pretrained()`).
  * If absent: It is a full model checkpoint. Load directly via `AutoModelForSeq2SeqLM.from_pretrained()`.
* Use `ModelManager` in `src/evaluation/model_registry.py` to maintain a singleton model lifecycle and avoid GPU memory leaks.

---

## 2. Modifying & Editing Protocols

### Rule 2.1: Model & Adapter Isolation
* Stage 2 LoRA weights must attach non-destructively over Stage 1 base weights.
* When testing or benchmarking Stage 1, use PEFT's context manager:
  ```python
  with model.disable_adapter():
      # Evaluates Stage 1 base weights without reloading from disk
  ```
* Never overwrite, retrain, or delete existing checkpoint weights in `checkpoints/`.

### Rule 2.2: Research Integrity & Honest Reporting
* **Never fabricate evaluation scores, training logs, or human ratings.**
* If human ratings have not been collected, report them explicitly as **Pending Validation**. Both `results/human_eval_sample_100.csv` and `results/human_eval_blinded_comparison.csv` must remain clean unrated templates until genuine annotators score them.
* Synthetic test fixtures (e.g., in `mock_human_eval_summary.md`) must be clearly labeled as test fixtures, never described as real native-speaker evaluations.
* Always cite the exact sample count ($N$) and artifact file when reporting metrics.

### Rule 2.3: Devanagari ROUGE Tokenization Rule
* Never evaluate Hindi text with default Hugging Face `evaluate.load("rouge")` without specifying a native Hindi tokenizer (`tokenizer=lambda x: x.split()`).
* Default ROUGE strips all non-ASCII Devanagari script, creating a spurious metric defect.

---

## 3. Writing Protocols (Generating New Features & Code)

### Rule 3.1: Python Code Standards
* All new Python modules must:
  * Include module docstrings detailing purpose, inputs, and outputs.
  * Utilize Python type hints (`from typing import List, Dict, Optional, Tuple, Any`).
  * Adhere to PEP 8 standards with descriptive, clean variable naming.

### Rule 3.2: Native Devanagari JSON & CSV Serialization
* When saving JSON files containing Hindi text:
  ```python
  # CORRECT:
  json.dump(data, f, ensure_ascii=False, indent=2)

  # FORBIDDEN (Escapes Devanagari into unreadable ascii hex codes):
  json.dump(data, f, ensure_ascii=True)
  ```
* When exporting CSV files containing Hindi text for Microsoft Excel:
  * Always write a UTF-8 Byte Order Mark (`\ufeff`) or use `encoding="utf-8-sig"`.

### Rule 3.3: Bilingual UI Buttons
* Every interactive button in the browser web demo must include its English translation in brackets alongside its Hindi label, ensuring usability for both Hindi educators and English-speaking researchers (e.g. `💾 सभी प्रश्न बैंक में सुरक्षित करें (Save All to Question Bank)`).

---

## 4. The 5 Golden Rules of Prashn-Setu

```
┌────────────────────────────────────────────────────────────────────────┐
│                          THE 5 GOLDEN RULES                            │
│                                                                        │
│ 1. ALL PATHS & LENGTH SETTINGS RESIDE IN src/config.py                 │
│ 2. PRESERVE UTF-8 (ensure_ascii=False) ON ALL HINDI TEXT               │
│ 3. USE NATIVE HINDI TOKENIZATION FOR ROUGE & ALWAYS REPORT chrF++      │
│ 4. NEVER FABRICATE BENCHMARK RESULTS OR HUMAN EVALUATION SCORES        │
│ 5. PRESERVE DEVANAGARI UNICODE INTEGRITY (NO EMPTY PLACEHOLDER MATRAS) │
└────────────────────────────────────────────────────────────────────────┘
```
