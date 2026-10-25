# प्रश्न-सेतु (Prashn-Setu) — Model Execution Bundle

This folder contains the **working model checkpoints** and **minimal execution files** required to run Prashn-Setu question generation on any computer without needing training data or external dependencies.

---

## 📁 Package Contents

```
send/
├── checkpoints/
│   ├── stage1_indicnlg_qg/         # Base 5-epoch fine-tuned IndicBART (244M params, 932MB)
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   └── stage2_ncert_gold_lora/     # NCERT Gold LoRA Curriculum Adapter (4.7MB)
│       ├── adapter_config.json     # Relative base model configuration
│       ├── adapter_model.safetensors
│       ├── tokenizer_config.json
│       └── tokenizer.json
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central paths & length configurations
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py             # Optional test evaluation runner
│   │   └── inference.py            # QuestionGenerator engine & prompt session
│   └── demo/
│       ├── __init__.py
│       └── app.py                  # Zero-dependency browser web server (port 7860)
├── data/
│   └── sample_ncert_contexts.jsonl # Sample authentic Hindi textbook passages
├── infer.py                        # 1-command quick inference & verification script
├── run.py                          # Unified CLI entry point (infer, demo, evaluate)
├── requirements.txt                # Python package dependencies
└── README.md                       # This instruction manual
```

> **Size Comparison:** The complete training repository was ~15.6 GB. This execution bundle contains only the essential weights and runtime scripts, reducing the footprint to **~945 MB** (a ~94% reduction).

---

## ⚙️ Setup & Installation

### 1. Requirements
- Python 3.10, 3.11, or 3.12
- PyTorch with CUDA (recommended for fast GPU inference) or CPU
- ~2 GB system RAM (or GPU VRAM)

### 2. Create Virtual Environment & Install Dependencies
```bash
# Navigate to the folder
cd send

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution Options

### Option 1: Quick 1-Command Verification
Run predefined authentic NCERT textbook examples to verify everything works:
```bash
python infer.py
```

You can also pass custom Hindi text directly:
```bash
python infer.py --context "भारत की राजधानी नई दिल्ली है।" --answer "नई दिल्ली"
```

To run explicitly on the Stage 1 base checkpoint:
```bash
python infer.py --checkpoint checkpoints/stage1_indicnlg_qg
```

---

### Option 2: Interactive Terminal CLI
Launch an interactive question generation session in your terminal:
```bash
python run.py infer
```

You will be prompted to enter:
1. **Context passage** (Hindi educational text)
2. **Target answer** (optional entity or span)

Type `exit` or `quit` to end the session.

---

### Option 3: Interactive Browser Web UI
Launch the zero-dependency browser demonstration server:
```bash
python run.py demo --port 7860
```
Open **`http://localhost:7860`** in any web browser.

**Features of the Web Demo:**
- **Single Model Generator:** Generate single or batch questions from textbook passages with automated candidate answer extraction.
- **Model Comparison (मॉडल तुलना):** Submit a single passage and compare generated questions across **Baseline**, **Stage 1 (Base SFT)**, and **Stage 2 (NCERT LoRA Adapted)** side by side.
- **Research Benchmark Dashboard (शोध मूल्यांकन):** View empirical SacreBLEU, ROUGE-1/2/L, chrF++, latency metrics, and blinded human evaluation scores.
- **Decoding Controls:** Beam Search (configurable beam size 1–8, no-repeat n-gram) and Nucleus Sampling (temperature 0.1–2.0, top-p 0.1–1.0).
- **REST APIs:** Full programmatic access via `/api/generate`, `/api/compare`, `/api/models`, and `/api/benchmark_results`.

---

### Option 4: Formal Multi-Model Benchmark (CLI)
Run a reproducible comparative evaluation across models on identical held-out examples:
```bash
python run.py compare --max_samples 100 --batch_size 16
```

This command evaluates all available models and generates:
1. `results/model_comparison_results.json`: Quantitative benchmark metrics and latency comparisons.
2. `results/human_eval_blinded_comparison.csv`: Shuffled, blinded evaluation spreadsheet for human raters.
3. `results/human_eval_blind_key.json`: Secret blinding key mapping (e.g. Model A = Stage 2, Model B = Stage 1).

---

### Option 5: Using as a Python Library
You can import `ModelManager` or `QuestionGenerator` into your own Python scripts:

```python
from src.evaluation.model_registry import get_shared_model_manager, MODEL_STAGE1, MODEL_STAGE2

manager = get_shared_model_manager()

context = "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।"
answer = "डॉ. भीमराव अंबेडकर"

# Generate with Stage 1 (Base fine-tuned model)
s1_questions = manager.generate_single_for_model(
    model_id=MODEL_STAGE1,
    context=context,
    answer=answer,
    num_beams=4
)
print("Stage 1 Question:", s1_questions[0])

# Generate with Stage 2 (NCERT Curriculum LoRA adapted model)
s2_questions = manager.generate_single_for_model(
    model_id=MODEL_STAGE2,
    context=context,
    answer=answer,
    num_beams=4
)
print("Stage 2 Question:", s2_questions[0])
```

---

## ⚖️ Model Comparison & Checkpoint Architecture

| Model Identifier | Configuration | Checkpoint Location | Status | Role |
|---|---|---|---|---|
| `baseline` | Original IndicNLG Baseline | `checkpoints/baseline_indicnlg_qg` | Unavailable | Original translation-derived QG baseline prior to textbook fine-tuning. Reported honestly as unavailable because it is not present in local checkpoints. Pretrained raw IndicBART is intentionally not substituted. |
| `stage1` | IndicBART + IndicNLG QG | `checkpoints/stage1_indicnlg_qg` (~932MB) | Available | Supervised fine-tuning on ~70,000 Hindi question-generation pairs. Learns general Hindi question syntax and grammar. |
| `stage2` | NCERT Gold Adapted | `checkpoints/stage2_ncert_gold_lora` (~4.7MB) | Available | Parameter-efficient LoRA adapter ($r=16, \alpha=32$) trained on authentic NCERT Class 10 Hindi textbooks (*Kshitij*, *Sparsh*). Aligns question generation to textbook phrasing and exam styles. |

### Resource Management & Adapter Isolation
- **Shared Base Weights:** The system loads Stage 1 weights once into memory. When evaluating Stage 2, the LoRA adapter is active; when evaluating Stage 1, the LoRA adapter is cleanly disabled using PEFT's `with model.disable_adapter():` context manager.
- **Zero-Reload Switching:** Both models can be compared in milliseconds without unloading or reloading large 932MB weights from disk.
- **Candidate Answer Parity:** When generating multiple questions without a user-specified answer, salient candidate entities are extracted **once** and fed identically to all selected models, ensuring strict comparative fairness.

---

## 📡 REST API Reference

### 1. Model Comparison Endpoint: `POST /api/compare`
```json
// Request Body
{
  "context": "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।",
  "answer": "",
  "models": ["stage1", "stage2", "baseline"],
  "num_questions": 3,
  "strategy": "beam_search",
  "num_beams": 4,
  "max_length": 64,
  "no_repeat_ngram_size": 3
}

// Response Body
{
  "context": "...",
  "candidate_targets_used": ["भारतीय संविधान", "भीमराव अंबेडकर"],
  "selected_models": ["stage1", "stage2", "baseline"],
  "results": {
    "stage1": {
      "model_id": "stage1",
      "name": "Stage 1: IndicBART + IndicNLG QG",
      "status": "success",
      "latency_ms": 1420.5,
      "questions": [...]
    },
    "stage2": {
      "model_id": "stage2",
      "name": "Stage 2: NCERT Curriculum Adapted (LoRA)",
      "status": "success",
      "latency_ms": 1450.2,
      "questions": [...]
    },
    "baseline": {
      "model_id": "baseline",
      "name": "Original IndicNLG QG Baseline",
      "status": "unavailable",
      "error": "Original IndicNLG question-generation baseline checkpoint is not available...",
      "latency_ms": 0.0,
      "questions": []
    }
  },
  "parameters": {...},
  "warnings": [...]
}
```

### 2. Model Registry Endpoint: `GET /api/models`
Returns JSON dictionary of registered model configurations, checkpoint paths, and current availability status.

### 3. Benchmark Results Endpoint: `GET /api/benchmark_results`
Returns saved formal benchmark evaluation metrics and human evaluation summaries.

---

## 🧪 Running Automated Tests

Run the comprehensive unit and integration test suite:
```bash
python -m unittest tests/test_model_comparison.py
```
This tests:
- Model registry initialization and availability detection.
- ModelManager shared lifecycle and adapter switching.
- Candidate-answer extraction parity.
- Input validation (empty context, invalid strategy, range bounds).
- Unavailable model graceful degradation.
- Comparison API response structure and backward compatibility.
