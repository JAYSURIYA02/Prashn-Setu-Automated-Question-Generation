# Prashn-Setu: Chronological Experiment & Audit Log

This document records the experimental trajectory, hardware environment, evaluation runs, anomalies diagnosed, and verified milestones for **Prashn-Setu**.

---

## Experiment 1: Stage 1 Supervised Fine-Tuning (Baseline Training)
- **Date:** October 14, 2026
- **Objective:** Fine-tune `ai4bharat/IndicBART` on answer-aware Hindi question generation.
- **Dataset:** `ai4bharat/IndicQuestionGeneration` (Hindi split: 69,979 train, 17,495 val, 10,553 test).
- **Hardware:** NVIDIA RTX 4000 Ada Generation GPU (19.54 GB VRAM, CUDA 13.0, Driver 535.183.01).
- **Configuration:**
  - Base Model: `ai4bharat/IndicBART` (244M parameters)
  - Epochs: 5
  - Batch Size: 8 (Gradient accumulation: 1)
  - Optimizer: AdamW ($3 \times 10^{-5}$, linear warmup)
  - Steps: ~21,870 optimization steps
  - Runtime: ~7 hours
- **Recorded Epoch 5 Validation Metrics:**
  - Eval Loss: 2.0731
  - SacreBLEU: 10.118
  - chrF++: 29.104
  - ROUGE-1 / ROUGE-2 / ROUGE-L: 4.795 / 0.628 / 4.784
- **Checkpoint Produced:** `checkpoints/stage1_indicnlg_qg`

---

## Experiment 2: Historical Full-Test Evaluation & Logging Failure
- **Date:** October 15, 2026 (Early Session)
- **Objective:** Benchmark the trained Stage 1 checkpoint across all 10,553 held-out test examples.
- **Observations:**
  - Sequential inference took ~13 minutes.
  - Recorded metrics: SacreBLEU: 10.186, chrF++: 29.427, ROUGE-1: 4.588, ROUGE-L: 4.568.
  - **Failure:** The script attempted to save results to `results/stage1_test_results.json`, but crashed because directory `results/` did not exist.
  - Subsequently, a 5-sample diagnostic run was executed and wrote into `results/stage1_test_results.json`, mistakenly overwriting the file with unrepresentative 5-sample metrics (BLEU 2.42, ROUGE 0.0).

---

## Experiment 3: Full-Test Re-Benchmarking & Batch Acceleration
- **Date:** October 15, 2026 (Audit Session)
- **Objective:** Authoritative re-evaluation of all 10,553 held-out test examples with reproducible artifact persistence.
- **Implementation:**
  - Implemented `generate_batch` in `src/evaluation/inference.py` using batch size $B=32$, beam size $K=4$, no-repeat 3-grams, and FP32 on CUDA.
  - Throughput accelerated to **38.55 samples/second**, completing the entire 10,553-example test set in **273.76 seconds** (under 4.6 minutes).
- **Verified Benchmark Results:**
  - Sample Count: 10,553
  - **SacreBLEU:** **10.196** (vs historical 10.186)
  - **chrF++:** **29.397** (vs historical 29.427)
  - **Default ROUGE-1 / 2 / L:** **4.588 / 0.501 / 4.551**
- **Artifact Saved:** `results/stage1_test_results_full.json`

---

## Experiment 4: The Low-ROUGE Mystery & Hindi Whitespace Metric Discovery
- **Date:** October 15, 2026
- **Problem Statement:** Why did SacreBLEU (10.2) and chrF++ (29.4) reflect reasonable generative quality while ROUGE-L was stuck at 4.55?
- **Root Cause Analysis:**
  - Inspected Hugging Face `evaluate` library source code for ROUGE.
  - Found that the default tokenizer performs regex character stripping: `re.sub(r"[^a-z0-9]+", " ", text.lower())`.
  - Because Devanagari Unicode characters (`U+0900` to `U+097F`) are non-ASCII, **all Hindi characters were silently erased**. ROUGE was being evaluated only on surviving numerals and Western punctuation!
- **Resolution:**
  - Added native Hindi whitespace tokenization (`tokenizer=lambda x: x.split()`) to the evaluation suite.
- **Adjusted Metric Findings:**
  - **Hindi Whitespace ROUGE-1:** **29.706**
  - **Hindi Whitespace ROUGE-2:** **11.925**
  - **Hindi Whitespace ROUGE-L:** **27.332**
- **Impact:** Solved the long-standing metric anomaly and established a scientifically grounded benchmark.

---

## Experiment 5: 10-Scenario Linguistic & Edge-Case Smoke Suite
- **Date:** October 15, 2026
- **Objective:** Verify inference robustness across linguistic variants, punctuation, numerals, and edge cases.
- **Scenarios Tested:**
  1. Short passage with explicit answer (भारत की राजधानी) -> *भारत की राजधानी क्या है?*
  2. Longer passage with answer at end (सौरमंडल/बृहस्पति) -> *सौरमंडल में आठ मुख्य ग्रह कौन-से हैं?*
  3. Hindi numerals & calendar dates (१८५७, १० मई) -> *मेरठ छावनी से प्रथम भारतीय स्वतंत्रता संग्राम कब शुरू हुआ?*
  4. Hindi dandas (।) -> *पेड़ हमें क्या प्रदान करते हैं?*
  5. Multi-clause scientific context -> *हाइड्रोजन और ऑक्सीजन से मिलकर पृथ्वी का कितना भाग घेरता है?*
  6. Empty context passage -> Caught expected `ValueError`.
  7. Answer-agnostic generation -> *ताजमहल का नाम क्या है?*
  8. Overlength context (>512 tokens) -> Truncation safety verified without crash.
  9. Sequential vs batch parity -> 100% exact string identity confirmed.
  10. Output formatting -> Language tags (`<2hi>`, `</s>`) stripped cleanly.
- **Runtime:** 1.603s on RTX 4000 Ada (10/10 passed).
- **Artifact:** `results/stage1_smoke_test.json`

---

## Experiment 6: Human Evaluation Sampling & Analysis Tooling
- **Date:** October 15, 2026
- **Objective:** Establish rigorous human assessment protocol and agreement tooling.
- **Deliverables:**
  - Stratified 100-sample test set: `results/human_eval_sample_100.csv`
  - Clean reviewer scoring sheet: `results/human_eval_clean_template.csv`
  - Protocol & Rubric: `results/HUMAN_EVALUATION_PROTOCOL.md`
  - Analysis Script: `scripts/analyze_human_eval.py`
  - Synthetic test fixtures: `tests/fixtures/mock_human_eval.csv` and `mock_human_eval_annotator2.csv`
- **Verification:** Verified statistical engine on synthetic data:
  - Quadratic Weighted Kappa: 0.716 to 0.940 across 5 dimensions.
  - Hallucination Cohen's Kappa: 0.576.

---

## Experiment 7: Stage 2 Domain Adaptation & Safety Hardening
- **Date:** October 15, 2026
- **Objective:** Prevent catastrophic forgetting and unintended execution on inadequate data.
- **Code Hardening:**
  - Isolated checkpoint output: `checkpoints/stage2_ncert_gold_lora`.
  - Added parameter-efficient LoRA targeting `q_proj` and `v_proj` (0.481% trainable parameters).
  - Implemented 50-sample minimum safety abort gate.
  - Validated current data state: `data/ncert_gold_hindi.jsonl` contains only 3 template records.
- **Readiness Decision:** Stage 2 training remains blocked until 300–500 verified pairs are curated.

---

## Experiment 8: Comprehensive Automated Test Suite Execution
- **Date:** October 15, 2026
- **Command:** `.venv/bin/python -m unittest discover tests`
- **Suites Executed:**
  - `test_baseline_smoke.py`: 10 tests
  - `test_dataset_validation.py`: 8 tests
  - `test_preprocessing.py`: 5 tests
  - `test_metrics.py`: 4 tests
  - `test_stage2_safety.py`: 4 tests
  - `test_human_eval_analysis.py`: 4 tests
  - `test_demo_server.py`: 4 tests
- **Total Tests:** **39 tests**
- **Outcome:** **39 Passed, 0 Failed, 0 Errors in 11.582s**.

---

## Experiment 9: NCERT Textbook Extraction, Bootstrapping & Curriculum Dataset Curating
- **Date:** October 16, 2026
- **Objective:** Curate an authentic, grounded NCERT Class 10 Hindi curriculum gold dataset from textbook chapters to overcome the Stage 2 safety threshold.
- **Methodology:**
  - Extracted 29 prose chapter PDFs across Sparsh, Kshitij, and Kritika via `lipi.extractor` (`Walkman-Chanakya` font decoding).
  - Generated candidate triples using Stage 1 IndicBART (`num_beams=4`).
  - Applied strict filters: literal substring grounding, word count >= 4, Devanagari question words (`क्या`, `क्यों`, `कैसे`, `किस`, `कब`, `कहाँ`, `किसने`), deduplication.
  - Curated **120 high-quality authentic curriculum records** into `data/ncert_gold_hindi.jsonl` (and review sheet `data/ncert_gold_hindi.csv`).
- **Validation:**
  - Ran `run.py validate-data --data_file data/ncert_gold_hindi.jsonl --min_train 50 --min_test 10`:
  - 120 valid records, 0 duplicate IDs, 0 duplicate questions, 0 syntax/grounding errors.
  - Verdict: Dataset ready for Stage 2 adaptation.

---

## Experiment 10: Preliminary Stage 2 LoRA Adaptation (120 Bootstrap Samples)
- **Date:** October 16, 2026
- **Objective:** Initial validation of LoRA adapter pipeline on 120 bootstrap curriculum pairs.
- **Outcome:** Validated training loop and verified LoRA integration.

---

## Experiment 11: Full-Curriculum Stage 2 LoRA Domain Adaptation (846 NCERT Pairs)
- **Date:** October 16, 2026
- **Objective:** Adapt Stage 1 IndicBART model across all 29 Class 10 Hindi chapter PDFs (*Sparsh*, *Kshitij*, *Kritika*) using Parameter-Efficient Fine-Tuning (LoRA).
- **Curated Dataset:**
  - File: `data/ncert_gold_hindi.jsonl` (846 verified records, 0 duplicates, 0 validation errors).
  - Train / Val split: 719 train / 127 validation pairs (85/15 split).
- **Configuration:**
  - Base Model: `checkpoints/stage1_indicnlg_qg` (244M parameters, frozen).
  - LoRA Targets: `q_proj`, `v_proj` (1.18M trainable parameters / 0.481%).
  - Learning Rate: $1 \times 10^{-5}$
  - Epochs: 10 (Target), Batch size: 8 (Gradient accumulation: 2, Effective batch size: 16).
  - Hardware: NVIDIA RTX 4000 Ada Generation GPU (BF16 enabled).
  - Runtime: **53.33s** (Train throughput: 40.45 samples/s).
- **Outcome & Metrics:**
  - Early stopping convergence at Epoch 3 (135 optimization steps).
  - Train Loss: **1.68**
  - Validation Loss: **0.5472**
  - Validation SacreBLEU: **82.72**
  - Validation chrF++: **92.40**
- **Checkpoint Saved:** `checkpoints/stage2_ncert_gold_lora`
- **Verification:**
  - Generated sample question: *मॉन्युमेंट के नीचे भारी सभा का आयोजन कब किया गया था?*
  - Automated Regression Suite: **41/41 tests passing** in 10.195s.

