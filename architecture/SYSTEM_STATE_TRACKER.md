# System State & Change Tracker

> **Last Automatically Refreshed:** `2026-10-16 07:09:44 UTC`  
> **Total Tracked Files:** `110` | **Total Code & Doc Lines:** `1908186`

This file is dynamically updated by running `python run.py update-tracker`.
It maintains the live inventory of codebase files, SHA-256 checksums, dataset splits, and model checkpoints.

---
## 1. Tracked Files Inventory & Integrity Checksums

| File Path | Category | Size (KB) | Lines (LOC) | SHA-256 (12 chars) | Last Modified |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `architecture/COMPONENT_INTERACTIONS.md` | Architecture Doc | 7.36 | 132 | `6167c41991e6` | 2026-10-14 15:10:27 |
| `architecture/FILES_CATALOG.md` | Architecture Doc | 6.72 | 71 | `6138b30ae90e` | 2026-10-14 15:10:10 |
| `architecture/OPERATIONAL_GUIDELINES.md` | Architecture Doc | 6.01 | 80 | `422b2b458336` | 2026-10-14 15:02:41 |
| `architecture/PREREQUISITES_AND_ENVIRONMENT_SETUP.md` | Architecture Doc | 5.23 | 130 | `4362fdc4465a` | 2026-10-14 15:02:19 |
| `architecture/README.md` | Architecture Doc | 2.24 | 24 | `9d18774044a6` | 2026-10-14 15:00:25 |
| `architecture/SOLUTION_ANALYSIS_AND_ALTERNATIVES.md` | Architecture Doc | 7.93 | 81 | `9a4267c741e1` | 2026-10-14 15:01:56 |
| `architecture/SYSTEM_STATE_TRACKER.md` | Architecture Doc | 17.71 | 167 | `3d1f1e8a5cd8` | 2026-10-16 12:15:22 |
| `src/config.py` | Configuration | 3.22 | 69 | `7b05b064578b` | 2026-10-14 19:21:45 |
| `src/data/__init__.py` | Data Module | 0.24 | 6 | `3366e1be5c1d` | 2026-10-14 15:04:59 |
| `src/data/dataset.py` | Data Module | 4.32 | 103 | `f3f2d39704f6` | 2026-10-16 12:11:25 |
| `src/data/download.py` | Data Module | 5.21 | 75 | `6c4e93f49128` | 2026-10-16 12:11:40 |
| `data/hi_test.jsonl` | Dataset Artifact | 7469.83 | 10553 | `6646723e0ffc` | 2026-10-14 18:19:17 |
| `data/hi_train.jsonl` | Dataset Artifact | 47811.2 | 69979 | `2ced7cf1de75` | 2026-10-14 18:19:17 |
| `data/hi_validation.jsonl` | Dataset Artifact | 11926.55 | 17495 | `e4954f4b88b9` | 2026-10-14 18:19:17 |
| `data/indic_question_generation_hi/dataset_dict.json` | Dataset Artifact | 0.04 | 1 | `2bf46fe547f1` | 2026-10-14 18:19:17 |
| `data/indic_question_generation_hi/test/dataset_info.json` | Dataset Artifact | 2.62 | 74 | `8247f6c1a0aa` | 2026-10-14 18:19:17 |
| `data/indic_question_generation_hi/test/state.json` | Dataset Artifact | 0.24 | 13 | `796f130f6242` | 2026-10-14 18:19:17 |
| `data/indic_question_generation_hi/train/dataset_info.json` | Dataset Artifact | 2.62 | 74 | `8247f6c1a0aa` | 2026-10-14 18:19:17 |
| `data/indic_question_generation_hi/train/state.json` | Dataset Artifact | 0.24 | 13 | `2d476f677d7e` | 2026-10-14 18:19:17 |
| `data/indic_question_generation_hi/validation/dataset_info.json` | Dataset Artifact | 2.62 | 74 | `8247f6c1a0aa` | 2026-10-14 18:19:17 |
| `data/indic_question_generation_hi/validation/state.json` | Dataset Artifact | 0.25 | 13 | `7e42600ae425` | 2026-10-14 18:19:17 |
| `data/ncert_gold_hindi.jsonl` | Dataset Artifact | 1372.02 | 846 | `99610868f306` | 2026-10-16 12:24:22 |
| `data/ncert_gold_hindi_backup_3samples.jsonl` | Dataset Artifact | 2.68 | 3 | `314b2075f55c` | 2026-10-14 18:19:17 |
| `data/ncert_masked_validation_sample.jsonl` | Dataset Artifact | 55.46 | 21 | `2681016198c9` | 2026-10-16 11:09:46 |
| `src/evaluation/__init__.py` | Inference & Eval | 0.22 | 6 | `0b427094e411` | 2026-10-14 15:07:25 |
| `src/evaluation/evaluate.py` | Inference & Eval | 6.96 | 145 | `b783bc68cd63` | 2026-10-16 12:09:22 |
| `src/evaluation/inference.py` | Inference & Eval | 21.55 | 429 | `8dbb7a4b6ae2` | 2026-10-16 12:37:36 |
| `src/models/__init__.py` | Model Module | 0.17 | 5 | `da80a38486d8` | 2026-10-14 15:05:29 |
| `src/models/prune_vocab.py` | Model Module | 5.02 | 117 | `2e6677539da5` | 2026-10-16 12:10:58 |
| `README.md` | Root Entry Point | 13.92 | 185 | `961c7f8538d2` | 2026-10-15 18:48:54 |
| `check.py` | Root Entry Point | 0.36 | 8 | `cfd741c0bcaa` | 2026-10-14 18:07:40 |
| `requirements.txt` | Root Entry Point | 0.24 | 14 | `f9849ca3a2c6` | 2026-10-14 14:52:26 |
| `run.py` | Root Entry Point | 9.86 | 184 | `2afca9cc52c3` | 2026-10-16 12:38:10 |
| `test_stage1.py` | Root Entry Point | 2.58 | 65 | `18820a3879c1` | 2026-10-15 17:19:40 |
| `scripts/01_download_data.sh` | Shell Launcher | 0.09 | 4 | `6f5569d682e5` | 2026-10-14 15:08:12 |
| `scripts/02_prune_vocab.sh` | Shell Launcher | 0.1 | 4 | `4cee4f8d9805` | 2026-10-14 15:08:19 |
| `scripts/03_train_stage1.sh` | Shell Launcher | 0.24 | 5 | `53b96b7c5cde` | 2026-10-14 15:08:23 |
| `scripts/04_adapt_stage2.sh` | Shell Launcher | 0.31 | 9 | `b439513fe510` | 2026-10-14 15:08:28 |
| `scripts/05_evaluate.sh` | Shell Launcher | 0.24 | 8 | `e26d1a0a3d54` | 2026-10-14 15:08:33 |
| `scripts/06_interactive_inference.sh` | Shell Launcher | 0.16 | 4 | `b0d4ac1e22ea` | 2026-10-14 15:08:38 |
| `scripts/analyze_human_eval.py` | Shell Launcher | 15.21 | 347 | `90ab45179a7a` | 2026-10-16 12:13:42 |
| `scripts/bootstrap_ncert_gold.py` | Shell Launcher | 14.7 | 284 | `f13f070d3699` | 2026-10-16 12:24:16 |
| `scripts/extract_and_mask_ncert.py` | Shell Launcher | 12.66 | 263 | `6b538b12c7db` | 2026-10-16 11:09:36 |
| `scripts/validate_dataset.py` | Shell Launcher | 13.16 | 276 | `110ee65e9d41` | 2026-10-16 12:24:06 |
| `checkpoints/stage1_indicnlg_qg/all_results.json` | Source Code | 0.47 | 17 | `898ecf19643a` | 2026-10-15 02:47:12 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21000/config.json` | Source Code | 0.92 | 37 | `47c89d81b5c0` | 2026-10-15 02:12:55 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21000/generation_config.json` | Source Code | 0.91 | 36 | `7043d37b1c69` | 2026-10-15 02:12:55 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21000/tokenizer.json` | Source Code | 4920.74 | 256317 | `5c954f099029` | 2026-10-15 02:12:56 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21000/tokenizer_config.json` | Source Code | 0.58 | 27 | `30a708a3c199` | 2026-10-15 02:12:56 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21000/trainer_state.json` | Source Code | 46.03 | 1777 | `f77eadbfdb7e` | 2026-10-15 02:12:57 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21870/config.json` | Source Code | 0.92 | 37 | `47c89d81b5c0` | 2026-10-15 02:31:27 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21870/generation_config.json` | Source Code | 0.91 | 36 | `7043d37b1c69` | 2026-10-15 02:31:27 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21870/tokenizer.json` | Source Code | 4920.74 | 256317 | `5c954f099029` | 2026-10-15 02:31:28 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21870/tokenizer_config.json` | Source Code | 0.58 | 27 | `30a708a3c199` | 2026-10-15 02:31:28 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-21870/trainer_state.json` | Source Code | 47.79 | 1846 | `071125cffa6b` | 2026-10-15 02:31:29 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-7000/config.json` | Source Code | 0.92 | 37 | `47c89d81b5c0` | 2026-10-14 21:43:54 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-7000/generation_config.json` | Source Code | 0.91 | 36 | `7043d37b1c69` | 2026-10-14 21:43:54 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-7000/tokenizer.json` | Source Code | 4920.74 | 256317 | `5c954f099029` | 2026-10-14 21:43:55 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-7000/tokenizer_config.json` | Source Code | 0.58 | 27 | `30a708a3c199` | 2026-10-14 21:43:55 |
| `checkpoints/stage1_indicnlg_qg/checkpoint-7000/trainer_state.json` | Source Code | 15.91 | 615 | `64456c6143ab` | 2026-10-14 21:43:55 |
| `checkpoints/stage1_indicnlg_qg/config.json` | Source Code | 0.92 | 37 | `47c89d81b5c0` | 2026-10-15 02:31:29 |
| `checkpoints/stage1_indicnlg_qg/eval_results.json` | Source Code | 0.28 | 12 | `04bdeb175e3b` | 2026-10-15 02:47:12 |
| `checkpoints/stage1_indicnlg_qg/generation_config.json` | Source Code | 0.91 | 36 | `7043d37b1c69` | 2026-10-15 02:31:29 |
| `checkpoints/stage1_indicnlg_qg/tokenizer.json` | Source Code | 4920.74 | 256317 | `5c954f099029` | 2026-10-15 02:31:29 |
| `checkpoints/stage1_indicnlg_qg/tokenizer_config.json` | Source Code | 0.58 | 27 | `30a708a3c199` | 2026-10-15 02:31:29 |
| `checkpoints/stage1_indicnlg_qg/train_results.json` | Source Code | 0.2 | 8 | `c3bfc2131ec7` | 2026-10-15 02:31:29 |
| `checkpoints/stage1_indicnlg_qg/trainer_state.json` | Source Code | 48.04 | 1855 | `762e723c19b0` | 2026-10-15 02:31:29 |
| `checkpoints/stage2_ncert_gold_lora/README.md` | Source Code | 5.13 | 110 | `971c0eb4583a` | 2026-10-16 12:26:02 |
| `checkpoints/stage2_ncert_gold_lora/adapter_config.json` | Source Code | 1.12 | 46 | `e347e81d7e71` | 2026-10-16 12:26:02 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-135/README.md` | Source Code | 5.13 | 110 | `971c0eb4583a` | 2026-10-16 12:26:02 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-135/adapter_config.json` | Source Code | 1.12 | 46 | `e347e81d7e71` | 2026-10-16 12:26:02 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-135/tokenizer.json` | Source Code | 4920.74 | 256317 | `c88e334165be` | 2026-10-16 12:26:02 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-135/tokenizer_config.json` | Source Code | 0.76 | 34 | `ce5eaa45ebdb` | 2026-10-16 12:26:02 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-135/trainer_state.json` | Source Code | 4.35 | 173 | `e04ec0c3ad2c` | 2026-10-16 12:26:02 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-45/README.md` | Source Code | 5.13 | 110 | `971c0eb4583a` | 2026-10-16 12:25:26 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-45/adapter_config.json` | Source Code | 1.12 | 46 | `e347e81d7e71` | 2026-10-16 12:25:26 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-45/tokenizer.json` | Source Code | 4920.74 | 256317 | `c88e334165be` | 2026-10-16 12:25:26 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-45/tokenizer_config.json` | Source Code | 0.76 | 34 | `ce5eaa45ebdb` | 2026-10-16 12:25:26 |
| `checkpoints/stage2_ncert_gold_lora/checkpoint-45/trainer_state.json` | Source Code | 2.09 | 84 | `33b0ecd05ef9` | 2026-10-16 12:25:26 |
| `checkpoints/stage2_ncert_gold_lora/tokenizer.json` | Source Code | 4920.74 | 256317 | `c88e334165be` | 2026-10-16 12:26:02 |
| `checkpoints/stage2_ncert_gold_lora/tokenizer_config.json` | Source Code | 0.76 | 34 | `ce5eaa45ebdb` | 2026-10-16 12:26:02 |
| `docs/ARCHITECTURE.md` | Source Code | 9.81 | 163 | `6ba64707bc70` | 2026-10-15 18:48:02 |
| `docs/EXPERIMENT_LOG.md` | Source Code | 9.83 | 154 | `4189b412b3c7` | 2026-10-16 12:26:38 |
| `docs/NCERT_GOLD_CURATION_GUIDE.md` | Source Code | 10.64 | 132 | `cb18b965724a` | 2026-10-15 18:28:14 |
| `docs/PROJECT_STATE_AND_EXECUTION_PLAN.md` | Source Code | 6.75 | 57 | `ca4844effbeb` | 2026-10-15 18:37:20 |
| `docs/TECHNICAL_REPORT.md` | Source Code | 19.42 | 225 | `9d5e782aa3ae` | 2026-10-15 18:47:42 |
| `results/HUMAN_EVALUATION_PROTOCOL.md` | Source Code | 6.28 | 77 | `896eae5c201a` | 2026-10-15 18:14:43 |
| `results/QUALITATIVE_ERROR_ANALYSIS.md` | Source Code | 10.38 | 81 | `0a5036ff1a7b` | 2026-10-15 18:20:05 |
| `results/mock_human_eval_summary.json` | Source Code | 6.1 | 249 | `702dbcef1c32` | 2026-10-15 18:40:44 |
| `results/mock_human_eval_summary.md` | Source Code | 1.41 | 27 | `32bdb5e4294f` | 2026-10-15 18:40:44 |
| `results/stage1_qualitative_analysis.json` | Source Code | 15.16 | 149 | `3a1f3a741747` | 2026-10-15 18:19:26 |
| `results/stage1_smoke_test.json` | Source Code | 4.84 | 78 | `4eabee7c7a24` | 2026-10-16 12:38:30 |
| `results/stage1_test_results.json` | Source Code | 4.88 | 43 | `e309c9e996e7` | 2026-10-15 17:43:27 |
| `results/stage1_test_results_full.json` | Source Code | 13.77 | 117 | `35018cee2402` | 2026-10-15 18:18:20 |
| `src/__init__.py` | Source Code | 0.04 | 3 | `8b7f429dc042` | 2026-10-14 15:04:42 |
| `src/demo/__init__.py` | Source Code | 0.05 | 3 | `edb85ec5b7f1` | 2026-10-15 18:45:10 |
| `src/demo/app.py` | Source Code | 37.48 | 799 | `4f1327217777` | 2026-10-16 12:37:51 |
| `tests/test_baseline_smoke.py` | Source Code | 11.52 | 209 | `67941658d533` | 2026-10-15 18:37:58 |
| `tests/test_dataset_validation.py` | Source Code | 7.45 | 137 | `aa2c2511bb1e` | 2026-10-15 18:41:27 |
| `tests/test_demo_server.py` | Source Code | 6.42 | 144 | `cd3ebc4ecc93` | 2026-10-16 11:32:04 |
| `tests/test_human_eval_analysis.py` | Source Code | 4.59 | 92 | `010f1e8fe549` | 2026-10-15 18:43:45 |
| `tests/test_metrics.py` | Source Code | 4.26 | 86 | `c16f10d79537` | 2026-10-16 12:14:34 |
| `tests/test_preprocessing.py` | Source Code | 6.22 | 139 | `7412e8a3c300` | 2026-10-15 18:42:22 |
| `tests/test_stage2_safety.py` | Source Code | 4.06 | 87 | `dbbb6722b1a8` | 2026-10-16 12:14:26 |
| `src/training/__init__.py` | Training Pipeline | 0.31 | 7 | `1668256636bc` | 2026-10-14 15:06:53 |
| `src/training/adapt_stage2.py` | Training Pipeline | 11.5 | 249 | `8ef52424158a` | 2026-10-16 12:10:39 |
| `src/training/metrics.py` | Training Pipeline | 2.27 | 49 | `f87b840985a7` | 2026-10-14 19:03:43 |
| `src/training/train_stage1.py` | Training Pipeline | 15.71 | 405 | `3c4e95ae8859` | 2026-10-16 12:12:04 |
| `src/utils/__init__.py` | Utility Module | 0.12 | 5 | `8e6ba8e5cae6` | 2026-10-14 15:07:32 |
| `src/utils/tracker.py` | Utility Module | 12.08 | 185 | `0865ffc76a8b` | 2026-10-16 11:24:23 |

---
## 2. Dataset Split Status

| Split / File | Current Status |
| :--- | :--- |
| `hi_train.jsonl` | Present (69979 records) |
| `hi_validation.jsonl` | Present (17495 records) |
| `hi_test.jsonl` | Present (10553 records) |
| `ncert_gold_hindi.jsonl` | Present (846 records) |

---
## 3. Model Checkpoint Status

| Checkpoint Stage | Status & Instruction |
| :--- | :--- |
| `indicbart_hi_pruned` | Pending execution (Pruned Hindi IndicBART (run `python run.py prune`)) |
| `stage1_indicnlg_qg` | Available (Stage 1 Base Checkpoint (run `python run.py train-stage1`)) |
| `stage2_ncert_adapted` | Pending execution (Stage 2 NCERT Gold LoRA Adapter (pending curation of 300-500 verified pairs; run `python run.py adapt-stage2`)) |

---
## 4. Observed Changes & Milestone History

* **2026-10-16 (NCERT Ingestion, Font Decoding, Cloze Masking & Multi-Question Generation):**
  * **NCERT Books Ingestion & Analysis:** Ingested 33 chapter PDFs across Class 10 Hindi textbooks (Kritika, Kshitij, Sparsh) totaling 288 pages and ~69,277 words.
  * **Legacy Font Decoding Engine:** Solved embedded legacy 8-bit `Walkman-Chanakya 905` font mapping using `lipi-aparsoft` & `fonttools` to extract clean Devanagari Unicode.
  * **Cloze Masking Validation Dataset:** Built `scripts/extract_and_mask_ncert.py` producing `data/ncert_masked_validation_sample.csv` (21 masked evaluation samples).
  * **Multi-Question Generation Engine:** Addressed single-question limitation on long passages. Implemented automated candidate answer extraction (`extract_candidate_answers`), sentence-focal conditioning to eliminate self-attention drift, batched generation (`generate_batch`), and semantic deduplication.
  * **Interactive Web Demo Upgrade:** Enhanced `src/demo/app.py` with multi-question dropdown (1, 3, 5, 8 questions), card list display with answer badges, and copy buttons.
  * **Test Suite Verification:** Comprehensive test suite expanded to 40 passing unit and integration tests across data, models, demo, and inference pipelines.

* **2026-10-15 (Stage 1 Audit, Evaluation Parity & Native-Hindi Curation Governance):**
  * **Stage 1 Benchmark Reproduction:** Evaluated all 10,553 held-out test examples in 273.76s (38.55 samples/sec), reproducing historical metrics: SacreBLEU 10.196 (vs 10.186), chrF++ 29.397 (vs 29.427).
  * **Metric Diagnosis & Dual ROUGE:** Diagnosed why default ROUGE was ~4.5% (English-only tokenizer deletes Devanagari characters); integrated native Hindi whitespace tokenization yielding genuine word-level ROUGE-L of 27.332%.
  * **Human Evaluation Suite:** Created rigorous 5-dimension rubric (`results/HUMAN_EVALUATION_PROTOCOL.md`) and populated 100 stratified test predictions with clean rating fields (`results/human_eval_sample_100.csv`, `results/human_eval_clean_template.csv`).
  * **Qualitative Error Analysis:** Documented systematic error taxonomy across 100 samples covering Wh-word mismatch, factual hallucination, and distractor confusion (`results/QUALITATIVE_ERROR_ANALYSIS.md`, `results/stage1_qualitative_analysis.json`).
  * **Textbook Curation Manual:** Penned native-Hindi textbook curation guidelines for NCERT/State Board grades 6-10 (`docs/NCERT_GOLD_CURATION_GUIDE.md`).
  * **Dataset Quality Validator:** Built automated validator with duplicate detection, substring grounding checks, and cross-split leakage prevention (`scripts/validate_dataset.py`, `python run.py validate-data`).
  * **Stage 2 Safety Guard:** Added explicit partition support (`--train_file`, `--val_file`) and 50-sample minimum safety guard in `src/training/adapt_stage2.py` to prevent accidental training on template placeholders.

* **2026-10-14 (Modular Refactor):** Codebase structured into clean packages:
  * Created `src/` modular layout: `src/data/`, `src/models/`, `src/training/`, `src/evaluation/`, `src/utils/`.
  * Created unified root CLI `run.py` to replace scattered scripts.
  * Added dedicated shell scripts in `scripts/` for automated batch running.
  * Maintained full backwards-compatible configuration and architecture documentation.

* **2026-10-14 (Initial Blueprint):** Initial pipeline creation:
  * Built centralized configuration (`src/config.py`).
  * Built Hindi dataset downloader and exporter (`src/data/download.py`).
  * Built vocabulary pruner and embedding slicer for IndicBART (`src/models/prune_vocab.py`).
  * Built PyTorch dataset with IndicBART formatting (`src/data/dataset.py`).
  * Built Stage 1 fine-tuning pipeline with Seq2SeqTrainer and chrF++ (`src/training/train_stage1.py`).
  * Built Stage 2 NCERT Gold adaptation with LoRA protection (`src/training/adapt_stage2.py`).
  * Built interactive CLI and inference pipeline (`src/evaluation/inference.py`).
  * Built quantitative benchmark evaluator (`src/evaluation/evaluate.py`).
  * Created comprehensive `architecture/` specifications, file catalogs, interaction diagrams, and operational rules.
  * Integrated automated state tracking (`src/utils/tracker.py`).
