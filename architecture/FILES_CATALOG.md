# Codebase File Catalog & Architectural Rationale: Prashn-Setu

This document catalogs every component in the active **Prashn-Setu** repository, explaining **what it is**, **why it is present**, and **how it functions** within the end-to-end question generation platform.

---

## 1. Root Entry Points & Dispatchers

### 📄 `run.py`
* **What it is:** Unified CLI entry point for the entire platform.
* **Why it is present:** Provides single-command access to subcommands: `infer` (interactive terminal prompt), `demo` (browser web demo), `evaluate` (test set benchmarking), and `compare` (multi-model comparative benchmark).
* **Inputs/Outputs:** CLI arguments $\to$ delegates to appropriate module in `src/`.

### 📄 `infer.py`
* **What it is:** Standalone, minimal-dependency CLI inference script.
* **Why it is present:** Enables instant verification of checkpoints on sample passages or custom text without running the full web server.
* **Usage:** `python infer.py --context "..." --answer "..."`

### 📄 `requirements.txt`
* **What it is:** Python package dependency specification.
* **Why it is present:** Specifies pinned package versions: `torch`, `transformers`, `peft`, `sentencepiece`, `protobuf`, `accelerate`, `evaluate`, `sacrebleu`, `rouge-score`, `scikit-learn`, `pandas`, `tqdm`, `pypdf`, and `reportlab`.

### 📄 `README.md`
* **What it is:** Master project documentation and user guide.
* **Why it is present:** Documents transfer learning architecture, verified benchmarks, research integrity audit, installation, API references, and future curation guidelines.

---

## 2. Model Checkpoints (`checkpoints/`)

### 📁 `checkpoints/stage1_indicnlg_qg/` (~932 MB)
* **What it is:** Supervised fine-tuned base checkpoint of `ai4bharat/IndicBART` (244M parameters).
* **Contents:** `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`.
* **Provenance:** Fine-tuned for 5 epochs on ~70,000 Hindi question-generation pairs from `ai4bharat/IndicQuestionGeneration`. Serves as the base model for all operations.

### 📁 `checkpoints/stage2_ncert_gold_lora/` (~4.7 MB)
* **What it is:** Parameter-Efficient Fine-Tuning (LoRA) adapter architecture for NCERT curriculum adaptation.
* **Contents:** `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`.
* **Configuration:** Rank $r=16$, Scaling $\alpha=32$, Dropout $p=0.05$, targeting attention projection layers `q_proj` and `v_proj` (1.18M trainable parameters / 0.481%).
* **Role:** Primary model for all standard question generation in the platform. Attaches over Stage 1 base weights.

---

## 3. Core Source Modules (`src/`)

### 📄 `src/config.py`
* **What it is:** Centralized configuration module using typed dataclasses (`ModelConfig`, `DataConfig`, `TrainingConfig`).
* **Why it is present:** Dynamically resolves absolute paths relative to `PROJECT_ROOT`, establishes sequence length limits (`max_source_length=512`, `max_target_length=64`), and sets Hindi language tags (`<2hi>`).

### 📁 `src/data/` (Data Management & Storage)
* **`src/data/question_bank.py`:**
  * **What it is:** SQLite database layer for persistent question storage (`data/question_bank.db`).
  * **Capabilities:** Full CRUD (`save_question`, `get_questions`, `update_question`, `delete_question`), full-text search, filtering by category and difficulty, quiz attempt logging (`quiz_history`), and summary statistics.

### 📁 `src/demo/` (Demonstration Server & Web UI)
* **`src/demo/app.py`:**
  * **What it is:** High-performance, zero-dependency HTTP server built on Python's `http.server`.
  * **Capabilities:** Serves the single-page application (`index.html`), local TrueType fonts (`/fonts/`), and implements all REST endpoints: `/api/health`, `/api/model_status`, `/api/models`, `/api/generate`, `/api/compare`, `/api/evidence`, `/api/upload_textbook`, `/api/question_bank`, `/api/quiz/evaluate`, `/api/paper/*`, `/api/benchmark_results`.
* **`src/demo/index.html`:**
  * **What it is:** Responsive single-page web application.
  * **Capabilities:** 4 unified workspaces (Question Studio, Practice & Quiz, Model Comparison, Research & System Analytics), inline evidence highlighting, Devanagari text normalization, interactive practice test engine, multi-section paper builder, and bilingual button labeling.
* **`src/demo/fonts/`:**
  * **What it is:** Bundled Google TrueType fonts (`NotoSansDevanagari-Regular.ttf` and `NotoSansDevanagari-Bold.ttf`).
  * **Why it is present:** Guarantees crisp Devanagari rendering across all platforms (Windows, Linux, macOS) for both web display and ReportLab PDF exports.

### 📁 `src/evaluation/` (Inference, Generation & Assessment Engines)
* **`src/evaluation/inference.py`:**
  * **What it is:** Core `QuestionGenerator` class wrapping IndicBART inference.
  * **Capabilities:** Prompt formatting (`"{answer} </s> {context} </s> <2hi>"`), beam search decoding, nucleus sampling, candidate answer extraction via Devanagari POS heuristics, and token deduplication.
* **`src/evaluation/model_registry.py`:**
  * **What it is:** Model manager maintaining singleton model lifecycles and adapter isolation.
  * **Capabilities:** Pre-flight verification (`verify_stage2_checkpoint`), shared weights loading, zero-disk-reload switching between Stage 1 and Stage 2 using `with model.disable_adapter():`.
* **`src/evaluation/advanced_generator.py`:**
  * **What it is:** Higher-level pedagogical question generation engine.
  * **Capabilities:** Categorizes questions into Factual, Conceptual, Reasoning, HOTS, and MCQ. Calibrates difficulty (Easy, Medium, Hard). Synthesizes authentic in-passage distractors for multiple-choice questions.
* **`src/evaluation/evidence_checker.py`:**
  * **What it is:** Grounding and answerability verification engine.
  * **Capabilities:** Splits Hindi passages while preserving exact character start/end offsets, computes lexical overlap between question/answer and context sentences, classifies support into `evidence_found`, `needs_review`, or `unsupported`, and detects duplicate questions.
* **`src/evaluation/textbook_processor.py`:**
  * **What it is:** Document parser and chapter chunking engine.
  * **Capabilities:** Extracts text from PDF and TXT files, applies Devanagari Unicode repair (reordering visual-order `ि`, reattaching space/newline-detached matras, repairing broken line wraps), and divides text into sentence-bounded chunks (150–250 words) to prevent GPU out-of-memory errors.
* **`src/evaluation/quiz_engine.py`:**
  * **What it is:** Interactive practice and assessment evaluation engine.
  * **Capabilities:** Grades multiple-choice questions, applies token overlap and morphological matching for Hindi short answers, computes percentage scores, and logs results to the database.
* **`src/evaluation/paper_builder.py`:**
  * **What it is:** Examination paper assembler and multi-format exporter.
  * **Capabilities:** Validates section marks against total marks, renders publication-quality Devanagari PDFs with student answer privacy (suppresses answers in student copy), exports dedicated Answer Key PDFs, exports UTF-8 BOM CSVs for Excel, and produces structured JSON for LMS import.
* **`src/evaluation/analytics_loader.py`:**
  * **What it is:** Research evaluation metrics parser.
  * **Capabilities:** Loads quantitative overlap benchmarks (SacreBLEU, chrF++, ROUGE-L), reports blinded human evaluation status honestly as pending validation, and compiles qualitative error taxonomies.
* **`src/evaluation/compare_models.py`:**
  * **What it is:** CLI multi-model comparative benchmarking harness.
  * **Capabilities:** Runs comparative evaluation across models on held-out test data, tracks per-model latency, and produces blinded evaluation CSVs and blinding keys.
* **`src/evaluation/evaluate.py`:**
  * **What it is:** Standalone test-set evaluation runner.
  * **Capabilities:** Evaluates single checkpoints against test splits with batch acceleration and exports metrics.

---

## 4. Documentation & Research Artifacts (`docs/` & `results/`)

### 📁 `docs/`
* **`docs/ARCHITECTURE.md`:** Comprehensive system architecture, dataflow pipelines, and safety gates.
* **`docs/TECHNICAL_REPORT.md`:** Scientific technical report documenting the Stage 1 baseline, resolution of the Devanagari ROUGE tokenization defect, error typology, and the LoRA framework.
* **`docs/PROJECT_STATE_AND_EXECUTION_PLAN.md`:** Component inventory and execution tracking.
* **`docs/EXPERIMENT_LOG.md`:** Chronological record of training runs, evaluations, and hardware environments.
* **`docs/NCERT_GOLD_CURATION_GUIDE.md`:** Detailed rubric and guidelines for collecting and verifying native-Hindi educational datasets.

### 📁 `results/`
* **`results/stage1_test_results_full.json`:** Verified benchmark across all 10,553 held-out test examples (SacreBLEU: 10.196, chrF++: 29.397, Hindi ROUGE-L: 27.332).
* **`results/model_comparison_results.json`:** 100-sample comparative evaluation artifact comparing Stage 1 and Stage 2.
* **`results/stage1_test_results.json`:** Historical 5-sample diagnostic artifact documenting early debugging runs.
* **`results/human_eval_sample_100.csv`:** Clean, unrated 100-sample human evaluation template.
* **`results/human_eval_blinded_comparison.csv`:** Clean, unrated 50-sample blinded comparison template.
* **`results/human_eval_blind_key.json`:** Secret model blinding key for rater evaluations.
* **`results/HUMAN_EVALUATION_PROTOCOL.md`:** 5-dimension scoring rubric and inter-annotator agreement methodology.
* **`results/QUALITATIVE_ERROR_ANALYSIS.md`:** Systematic qualitative error taxonomy detailing Wh-word mismatches, numeric hallucination, and distractor confusion.
* **`results/mock_human_eval_summary.md` / `.json`:** Synthetic test fixture reports used exclusively to verify the statistical analysis engine.

---

## 5. Automated Test Suites (`tests/`)

### 📄 `tests/test_platform_features.py` (24 Tests)
* Validates Stage 2 LoRA model pre-flight verification and primary status.
* Tests sentence splitting, character offset mapping, and evidence grounding classification.
* Tests MCQ distractor synthesis and category question formatting.
* Tests textbook text normalization, Devanagari repair, and chunking.
* Tests SQLite Question Bank CRUD, search, filtering, and summary statistics.
* Tests quiz evaluation, exact MCQ grading, and fuzzy short-answer scoring.
* Tests question paper marks validation, student answer suppression, and Devanagari PDF rendering.
* Tests REST API endpoints (`/api/health`, `/api/model_status`, `/api/question_bank`).

### 📄 `tests/test_model_comparison.py` (24 Tests)
* Validates model registry initialization and availability detection.
* Tests `ModelManager` adapter isolation, shared weights lifecycle, and zero-reload switching.
* Tests input validation for empty contexts, invalid decoding strategies, and out-of-range counts.
* Tests graceful status reporting for unavailable models.
* Tests comparative generation parity and live adapter switching.
