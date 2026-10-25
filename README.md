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
C
```
Open **`http://localhost:7860`** in any web browser.

**Features of the Web Demo:**
- Preloaded authentic NCERT Class 10 Hindi textbook presets (*Kshitij*, *Sparsh*, Science, Civics).
- Single-question and multi-question generation modes (1, 3, 5, 8 questions).
- Automated candidate answer extraction.
- REST API endpoint (`POST /api/generate`).
- Parameter tuning (beams, temperature, top-p).

---

### Option 4: Using as a Python Library
You can import `QuestionGenerator` into your own Python scripts:

```python
from src.evaluation.inference import QuestionGenerator

# Automatically loads Stage 2 NCERT Gold LoRA model (or fallback to Stage 1)
generator = QuestionGenerator()

context = "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।"
answer = "डॉ. भीमराव अंबेडकर"

# Generate natural Hindi question
questions = generator.generate(context=context, answer=answer)
print("Generated Question:", questions[0])
# Output: 'भारतीय संविधान का जनक किसे माना जाता है?'
```

---

## 🎯 Model Details

- **Base Architecture:** IndicBART Seq2Seq (`ai4bharat/IndicBART`, 244M parameters)
- **Target Language:** Hindi (`<2hi>`)
- **Stage 1 Baseline:** Supervised fine-tuning on ~70,000 IndicQuestionGeneration Hindi pairs.
- **Stage 2 Adaptation:** Parameter-efficient LoRA domain adaptation on authentic Class 10 NCERT textbook curriculum (*Sparsh*, *Kshitij*, *Kritika*).
- **Format:** Inputs are formatted as `{answer} </s> {context} </s> <2hi>`. When no answer is given, format defaults to `{context} </s> <2hi>`.
