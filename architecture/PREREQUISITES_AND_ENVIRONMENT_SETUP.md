# Prerequisites & Environment Setup Guide

This document lists all hardware, software, OS packages, and Python libraries required to set up and run **Prashn-Setu**.

---

## 1. Hardware Requirements

| Component | Minimum Specification (Inference / Testing) | Recommended Specification (Full Training) |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (e.g., GTX 1660 Ti, RTX 3060) or CPU | NVIDIA GPU with $\ge 12\text{--}16\text{ GB}$ VRAM (e.g., T4, V100, RTX 3090, RTX 4090, A10) |
| **System RAM** | 8 GB | 16–32 GB |
| **Storage** | 10 GB free disk space | 25 GB free disk space (to cache base checkpoints, datasets, and LoRA adapters) |
| **Architecture**| x86_64 | x86_64 (Linux Ubuntu/Debian or WSL2) |

> [!NOTE]
> Training on a **free Google Colab T4 GPU instance (15 GB VRAM)** is fully supported and sufficient to complete both Stage 1 and Stage 2 training.

---

## 2. Operating System & System-Level Dependencies

### Supported Operating Systems
* **Linux:** Ubuntu 20.04 LTS, Ubuntu 22.04 LTS, Debian 11/12
* **Windows:** Windows 10/11 using **WSL2** (Windows Subsystem for Linux - Ubuntu)
* **Cloud Environments:** Google Colab, Kaggle Kernels, Lambda Labs, AWS EC2 (`g4dn.xlarge` or `g5.xlarge`)

### System Packages Installation (Debian / Ubuntu / WSL2)
Before installing Python packages, ensure the necessary C++ build tools and Python development headers are installed:

```bash
sudo apt update && sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    libprotobuf-dev \
    protobuf-compiler
```

---

## 3. Python Environment Setup

We strongly recommend using an isolated Python virtual environment (`python >= 3.9, <= 3.11` is optimal).

### Option A: Using `python3-venv` (Recommended for Local Linux / WSL)

```bash
# 1. Navigate to project root
cd /mnt/a/iiits/5/NLP/PROJECT

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel
```

### Option B: Using Conda / Miniconda

```bash
# Create and activate conda environment
conda create -n prashn_setu python=3.10 -y
conda activate prashn_setu
```

---

## 4. Python Package Dependencies

### Core Libraries
Install all required libraries via the provided [`requirements.txt`](file:///mnt/a/iiits/5/NLP/PROJECT/requirements.txt):

```bash
pip install -r requirements.txt
```

### Direct Installation Command (Alternative)
If installing manually:

```bash
# 1. Install PyTorch with CUDA 12.1 support (adjust for your CUDA driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install Hugging Face & Sequence-to-Sequence ecosystem
pip install transformers datasets huggingface_hub accelerate evaluate

# 3. Install Tokenizer, Model Pruning & PEFT dependencies
pip install sentencepiece protobuf peft

# 4. Install Metric Calculation packages
pip install sacrebleu rouge-score scikit-learn pandas tqdm
```

---

## 5. Google Colab Setup Instructions

If executing in **Google Colab**:
1. Open a new notebook and select: `Runtime` $\to$ `Change runtime type` $\to$ **`T4 GPU`**.
2. Run the following setup cell:

```python
# Colab Pre-flight Setup
!nvidia-smi

# Clone or navigate to project directory
%cd /content/PROJECT

# Install requirements
!pip install -q -r requirements.txt
```

---

## 6. Pre-flight Verification Script

To verify that your environment, GPU drivers, PyTorch CUDA bindings, and Hugging Face libraries are functional, run this test command:

```bash
python3 -c "
import torch
import transformers
import datasets
import peft
import sentencepiece

print('=== ENVIRONMENT PRE-FLIGHT CHECK ===')
print(f'PyTorch Version:      {torch.__version__}')
print(f'CUDA Available:       {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device Name:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM Available:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print(f'Transformers Version: {transformers.__version__}')
print(f'PEFT Version:         {peft.__version__}')
print('All prerequisite packages verified successfully!')
"
```

---

## 7. Common Pitfalls & Troubleshooting

### 1. `CUDA out of memory` (OOM) during Stage 1
* **Cause:** Batch size is too large for your GPU VRAM.
* **Fix:** In `config.py` or CLI arguments:
  * Reduce `--batch_size` from `8` to `4` or `2`.
  * Increase `gradient_accumulation_steps` from `2` to `4` or `8` to maintain the effective batch size.
  * Ensure `fp16=True` is enabled in `TrainingConfig`.

### 2. SentencePiece / Protobuf Version Conflicts
* **Symptom:** `TypeError: Descriptors cannot not be created directly...`
* **Fix:**
  ```bash
  pip install --upgrade "protobuf>=3.20.0,<5.0.0"
  ```

### 3. IndicBART Tokenizer Warning: Accent Preservation
* **Symptom:** Hindi characters losing matras or diacritics.
* **Fix:** IndicBART's AlbertTokenizer must always be instantiated with:
  ```python
  tokenizer = AutoTokenizer.from_pretrained(
      "ai4bharat/IndicBART",
      do_lower_case=False,
      use_fast=False,
      keep_accents=True,  # Crucial for Indic vowels & matras
  )
  ```
  *(Already enforced across `dataset.py`, `train.py`, `adapt_gold.py`, and `inference.py`)*.
