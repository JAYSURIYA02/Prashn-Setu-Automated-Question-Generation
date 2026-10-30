# Prerequisites & Environment Setup Guide

This document lists all hardware, operating systems, and Python packages required to run **Prashn-Setu**.

---

## 1. Hardware Requirements

| Component | Minimum Specification (Inference & Demo) | Recommended Specification (Batch Benchmarks) |
| :--- | :--- | :--- |
| **Processor** | Dual-Core CPU (Intel / AMD / Apple Silicon) | Quad-Core CPU or NVIDIA GPU (CUDA 11.8+ / 12.x) |
| **System RAM** | 4 GB | 8–16 GB |
| **GPU VRAM** | Not strictly required (runs cleanly on CPU) | 4–8 GB VRAM (e.g., RTX 3060, T4, A10) |
| **Disk Space** | ~2 GB free disk space (weights: ~945 MB) | 5 GB free disk space |
| **Operating System** | Windows 10/11, Ubuntu 20.04+, macOS 12+ | Windows 11 or Ubuntu 22.04 LTS |

---

## 2. Operating System & System-Level Dependencies

### Supported Operating Systems
* **Windows:** Windows 10 / Windows 11 (Native PowerShell, Command Prompt, or WSL2)
* **Linux:** Ubuntu 20.04 LTS, Ubuntu 22.04 LTS, Debian 11/12
* **macOS:** macOS Monterey (12.x) or higher (Apple Silicon M1/M2/M3 or Intel)
* **Cloud Environments:** Google Colab (CPU or T4 GPU), Kaggle Kernels, AWS EC2, Paperspace

---

## 3. Python Environment Setup

We recommend using Python **3.10, 3.11, or 3.12** inside an isolated virtual environment.

### Option A: Setup on Windows (PowerShell)

```powershell
# 1. Navigate to project root
cd send

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.venv\Scripts\activate

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> On Windows PowerShell, ensure console encoding is set to UTF-8 before running Python commands:
> ```powershell
> $env:PYTHONIOENCODING='utf-8'
> ```

---

### Option B: Setup on Linux / macOS (Bash / Zsh)

```bash
# 1. Navigate to project root
cd send

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Python Package Dependencies

All dependencies are defined in [`requirements.txt`](../requirements.txt):

```text
torch>=2.1.0
transformers>=4.38.0
peft>=0.9.0
sentencepiece>=0.2.0
protobuf>=3.20.0
accelerate>=0.27.0
evaluate>=0.4.1
sacrebleu>=2.4.0
rouge-score>=0.1.2
scikit-learn>=1.4.0
pandas>=2.2.0
tqdm>=4.66.0
pypdf>=4.0.0
reportlab>=4.1.0
```

---

## 5. Verification Commands

Verify that the environment and model checkpoints are functional:

```bash
# 1. Quick CLI inference test
python infer.py

# 2. Run automated platform and model test suites
python -m unittest tests/test_platform_features.py tests/test_model_comparison.py

# 3. Launch the web platform
python run.py demo --port 7860
```
Open **`http://localhost:7860`** in any web browser.
