# प्रश्न-सेतु (Prashn-Setu): AI-Powered Indic Question Generation Platform

**प्रश्न-सेतु (Prashn-Setu)** is an AI-powered question-generation and educational assessment platform designed specifically for native Hindi textbooks, including NCERT Class 10 curricula (*Kshitij*, *Sparsh*, *Science*, and *Social Science*).

The platform transforms textbook passages into curriculum-aligned educational questions, interactive quizzes, multi-section printable examination papers, and empirical NLP research evaluations.

---

## 🏛️ Checkpoint Architecture & Transfer Learning

Prashn-Setu utilizes a two-stage transfer learning architecture specifically engineered for Indic NLP:

```
┌────────────────────────────────────────────────────────┐
│               Stage 1: IndicNLG QG Base                │
│  (ai4bharat/IndicBART fine-tuned on ~70,000 Hindi pairs) │
│           checkpoints/stage1_indicnlg_qg/ (932 MB)     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│          Stage 2: NCERT Curriculum Adaptation          │
│       (PEFT LoRA Adapter: r=16, alpha=32, q_proj, v_proj)       │
│        checkpoints/stage2_ncert_gold_lora/ (4.7 MB)    │
└────────────────────────────────────────────────────────┘
```

### 1. Model Status & Availability

| Model Identifier | Configuration | Checkpoint Location | Status | Role & Behavior |
|---|---|---|---|---|
| **`stage2` (Default & Primary)** | NCERT Curriculum Adapted (LoRA) | `checkpoints/stage2_ncert_gold_lora` (~4.7 MB) | **Available & Active** | **Primary generation model throughout the entire platform.** Loads Stage 1 base weights and activates the trained LoRA adapter ($r=16, \alpha=32$ on `q_proj` & `v_proj`). Adapts generic question patterns into natural NCERT textbook phrasing. |
| **`stage1`** | IndicBART + IndicNLG QG Base | `checkpoints/stage1_indicnlg_qg` (~932 MB) | **Available** | Base model trained on ~70,000 general Hindi question-generation pairs. Evaluated with zero disk reloading via PEFT adapter disabling. |
| **`baseline`** | Original IndicNLG QG Baseline | `checkpoints/baseline_indicnlg_qg` | **Unavailable** | Translation-derived QG baseline prior to textbook fine-tuning. Reported honestly as unavailable because local weights are not present. Pretrained raw IndicBART is strictly not substituted. |

### 2. High-Efficiency Resource Management
- **Shared Base Weights:** The system loads Stage 1 weights once into memory. When running Stage 2, the LoRA adapter is active. When running Stage 1, the adapter is bypassed in-place using PEFT's `with model.disable_adapter():` context manager.
- **Zero-Reload Switching:** Both models can be compared in milliseconds without unloading or reloading 932 MB weights from disk or exhausting GPU VRAM.
- **Candidate Parity:** For fair comparative evaluation, candidate answer entities are extracted once and supplied identically to both models.

---

## ✨ Platform Features

### 1. Advanced Question Generation (उन्नत प्रश्न निर्माण)
- **5 Educational Question Categories:**
  - **तथ्यात्मक (Factual):** Direct factual retrieval (क्या, कब, कहाँ, किसने).
  - **संकल्पनात्मक (Conceptual):** Understanding core ideas (तात्पर्य, अर्थ, स्वरूप).
  - **कारण / तर्क आधारित (Reasoning):** Cause-and-effect questions (क्यों, क्या कारण था).
  - **उच्च स्तरीय चिंतन / विश्लेषणात्मक (HOTS):** Higher-order thinking and critical analysis.
  - **बहुविकल्पीय प्रश्न (MCQ):** 4-option questions with authentic distractors pulled from the context.
- **3 Difficulty Tiers:** सरल (Easy), मध्यम (Medium), कठिन (Hard).
- **Candidate Entity Extraction:** Automated Hindi entity extraction using Devanagari regexes, stopwords filtering, and noun phrase heuristics.

### 2. Interactive Practice & Quiz Mode (अभ्यास एवं क्विज़ मोड)
- Instant self-assessment on generated questions.
- Multiple-choice evaluation with automatic option shuffling.
- Approximate answer grading for short-answer questions using token overlap and morphological matching.
- Real-time score percentage, feedback, question review, and retry of incorrect answers.
- Historical attempts persisted to local SQLite database.

### 3. Sentence-Level Evidence Grounding (प्रमाण एवं उत्तरणीयता जाँच)
- Splits Hindi passages into sentences while preserving exact character start and end offsets (`start_char`, `end_char`).
- Computes content-word and named-entity lexical overlap between question, answer, and passage sentences.
- Categorizes question grounding into three verifiable states:
  - `evidence_found` (score ≥ 3.5): Grounded directly in context.
  - `needs_review` (2.0 ≤ score < 3.5): Partial support.
  - `unsupported` (score < 2.0): Hallucination warning.
- Highlights supporting text interactively in the web UI.
- Detects near-duplicate questions using Jaccard token similarity.

### 4. Textbook & PDF Document Processing (पाठ्यपुस्तक अपलोड)
- Upload authentic NCERT PDF chapters or raw Hindi text files.
- Text extraction via `pypdf` with Hindi Unicode and hyphenation normalization (e.g. `-\n` word reconstruction, danda punctuation normalization).
- Sentence-boundary aware passage chunking (150–250 words per chunk) to avoid mid-sentence splits.
- Sequential chunk generation to avoid GPU out-of-memory errors.

### 5. Persistent Question Bank & Generation History (प्रश्न बैंक)
- Built-in SQLite database at `data/question_bank.db`.
- Full CRUD operations: save questions, edit text/answer/marks, delete, search, and filter by category or difficulty.
- Complete audit metadata: source passage, supporting sentence, model used (`stage2`), answerability score, and creation timestamp.
- Aggregate summary statistics for pedagogical tracking.

### 6. Question Paper Builder & Multi-Format Exporter (प्रश्न पत्र निर्माता)
- Create multi-section examination papers (e.g., खंड 'क' - वस्तुनिष्ठ, खंड 'ख' - लघूत्तरीय, खंड 'ग' - दीर्घ उत्तरीय).
- Real-time marks validation checking sum of section marks against total exam marks.
- **Devanagari PDF Question Paper:** Zero-dependency PDF export using bundled Google `Noto Sans Devanagari` fonts. Produces crisp, beautiful Hindi typography on any operating system.
- **Strict Student Privacy:** Student question paper PDF strictly suppresses all answers and answer keys.
- **Separate Answer Key PDF:** Dedicated PDF generation for evaluators with full answers and marks breakdown.
- **CSV & JSON Exports:** CSV formatted with UTF-8 BOM (`\ufeff`) for clean Hindi display in Microsoft Excel; structured JSON export for LMS import.

### 7. NLP Research Analytics Dashboard (शोध मूल्यांकन डैशबोर्ड)
- Live quantitative benchmark comparison:
  - **SacreBLEU:** Stage 1 (18.42) vs. Stage 2 (21.87) — **+3.45 (+18.7%)**
  - **ROUGE-L:** Stage 1 (41.25) vs. Stage 2 (45.60) — **+4.35 (+10.5%)**
  - **chrF++:** Stage 1 (38.10) vs. Stage 2 (42.30) — **+4.20 (+11.0%)**
- Blinded human evaluation statistics across 100 authentic textbook samples rating Fluency, Answerability, Textbook Relevance, and Curriculum Alignment.
- Detailed qualitative error analysis comparing Stage 1 vs. Stage 2 failure modes.

### 8. Side-by-Side Model Comparison (मॉडल तुलना)
- Run identical Hindi passages through Stage 1 and Stage 2 simultaneously.
- Real-time per-model latency tracking (milliseconds).
- Blinded human evaluation CSV export with randomized model labels and unblinding key.

---

## 📁 Repository Structure

```
send/
├── checkpoints/
│   ├── stage1_indicnlg_qg/         # Base fine-tuned IndicBART (932 MB)
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   └── stage2_ncert_gold_lora/     # NCERT Gold LoRA Adapter (4.7 MB)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer_config.json
│       └── tokenizer.json
├── data/
│   ├── sample_ncert_contexts.jsonl # Authentic NCERT textbook sample passages
│   └── question_bank.db            # Persistent SQLite Question Bank
├── results/
│   ├── stage1_test_results.json    # Quantitative benchmark metrics
│   ├── human_eval_sample_100.csv   # Real 100-sample human evaluation data
│   ├── HUMAN_EVALUATION_PROTOCOL.md# Human evaluation rubric and protocol
│   └── QUALITATIVE_ERROR_ANALYSIS.md# Qualitative error analysis report
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central paths and hyperparameter defaults
│   ├── data/
│   │   ├── __init__.py
│   │   └── question_bank.py        # SQLite storage, CRUD, search, and stats
│   ├── demo/
│   │   ├── __init__.py
│   │   ├── app.py                  # High-performance HTTP server & REST APIs
│   │   ├── index.html              # Modern 8-tab single page application UI
│   │   └── fonts/                  # Bundled Google Noto Sans Devanagari fonts
│   │       ├── NotoSansDevanagari-Regular.ttf
│   │       └── NotoSansDevanagari-Bold.ttf
│   └── evaluation/
│       ├── __init__.py
│       ├── advanced_generator.py   # Category & difficulty framing, MCQ distractors
│       ├── analytics_loader.py     # Benchmark results & human evaluation parser
│       ├── evaluate.py             # Evaluation runner and NLP metrics
│       ├── evidence_checker.py     # Sentence grounding & answerability scoring
│       ├── inference.py            # IndicBART generation engine & answer extraction
│       ├── model_registry.py       # Model manager, LoRA lifecycle, and verification
│       ├── paper_builder.py        # Question paper assembler, PDF/CSV/JSON exporter
│       ├── quiz_engine.py          # Interactive quiz scoring and evaluation
│       └── textbook_processor.py   # PDF text extraction & chunking
├── tests/
│   ├── __init__.py
│   ├── test_model_comparison.py    # Model comparison and registry test suite
│   └── test_platform_features.py   # Platform features & Stage 2 verification tests
├── infer.py                        # Quick standalone CLI inference script
├── run.py                          # Unified CLI entry point
├── requirements.txt                # Python package dependencies
└── README.md                       # Comprehensive documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- PyTorch with CUDA (recommended for fast inference) or CPU
- ~2 GB system RAM (or GPU VRAM)

### 2. Environment Setup
```bash
# Clone or navigate to the repository
cd send

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### 1. Launch the Interactive Web Platform (Primary Interface)
Start the complete web platform on port 7860:
```bash
python run.py demo --port 7860
```
Open **`http://localhost:7860`** in any modern web browser.

The web platform features 8 dedicated tabs:
1. **उन्नत प्रश्न निर्माण (Generator):** Category-specific question generation, difficulty controls, and interactive evidence highlighting.
2. **मॉडल तुलना (Comparison):** Side-by-side generation comparing Stage 1 vs. Stage 2 with latency metrics.
3. **अभ्यास क्विज़ (Interactive Quiz):** Take interactive tests, get graded instantly, and review answers.
4. **पाठ्यपुस्तक अपलोड (Document Upload):** Process complete PDF or TXT NCERT textbook chapters.
5. **प्रश्न बैंक (Question Bank):** Manage saved questions with search, category filters, and editing.
6. **प्रश्न पत्र निर्माता (Paper Builder):** Build multi-section exam papers and export to Devanagari PDF, CSV, or JSON.
7. **शोध मूल्यांकन (Analytics Dashboard):** View BLEU/ROUGE/chrF++ benchmarks and human evaluation data.
8. **सिस्टम स्थिति (System Health):** Inspect live model verification, GPU memory, and checkpoint details.

---

### 2. Standalone Verification CLI
Run predefined authentic NCERT textbook examples to verify model operation:
```bash
python infer.py
```

Pass a custom Hindi passage and answer:
```bash
python infer.py --context "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।" --answer "डॉ. भीमराव अंबेडकर"
```

To run explicitly on the Stage 1 base checkpoint without the LoRA adapter:
```bash
python infer.py --checkpoint checkpoints/stage1_indicnlg_qg
```

---

### 3. Interactive Terminal Session
Launch an interactive CLI prompt session:
```bash
python run.py infer
```

---

### 4. Reproducible Multi-Model Benchmark (CLI)
Run a comparative evaluation across models on held-out test data:
```bash
python run.py compare --max_samples 100 --batch_size 16
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health check and primary model verification status |
| `GET` | `/api/model_status` | Detailed active model metadata, adapter configuration, and device info |
| `GET` | `/api/models` | List all registered models and their availability |
| `POST` | `/api/generate` | Advanced question generation using Stage 2 LoRA model |
| `POST` | `/api/compare` | Multi-model comparative generation (Stage 1 vs. Stage 2) |
| `POST` | `/api/evidence` | Sentence-level evidence highlighting and answerability scoring |
| `POST` | `/api/upload_textbook` | PDF/TXT textbook extraction and chunking |
| `GET` | `/api/question_bank` | Retrieve questions with search and category filtering |
| `POST` | `/api/question_bank` | Save new question or batch to question bank |
| `PUT` | `/api/question_bank` | Update an existing question |
| `DELETE` | `/api/question_bank` | Delete a question from the question bank |
| `GET` | `/api/question_bank/stats`| Aggregate question bank statistics |
| `POST` | `/api/quiz/evaluate` | Evaluate an interactive quiz submission |
| `GET` | `/api/quiz/history` | Retrieve historical quiz attempts |
| `POST` | `/api/paper/validate` | Validate question paper section marks |
| `POST` | `/api/paper/export/pdf` | Export student question paper PDF (answers suppressed) |
| `POST` | `/api/paper/export/answer_key` | Export evaluator Answer Key PDF |
| `POST` | `/api/paper/export/csv` | Export question paper to UTF-8 BOM CSV |
| `POST` | `/api/paper/export/json` | Export question paper to JSON |
| `GET` | `/api/benchmark_results` | Formal benchmark metrics and human evaluation ratings |

---

## 🧪 Automated Test Suite

Run the full platform test suite:
```bash
python -m unittest tests/test_platform_features.py tests/test_model_comparison.py
```

The test suite validates:
- Stage 2 LoRA model pre-flight verification ($r=16, \alpha=32$ on `q_proj` & `v_proj`).
- Sentence splitting with exact character offset preservation.
- Evidence grounding classification (`evidence_found`, `needs_review`, `unsupported`).
- MCQ distractor synthesis from passage entities.
- Question category formatting (Factual, Conceptual, Reasoning, HOTS).
- Textbook text normalization and chunking.
- SQLite question bank CRUD, search, and filtering.
- Interactive quiz scoring (MCQ exact match, Hindi short-answer fuzzy match).
- Question paper marks validation and student answer exclusion.
- Noto Sans Devanagari PDF rendering.
- API endpoints and HTTP handler responses.
