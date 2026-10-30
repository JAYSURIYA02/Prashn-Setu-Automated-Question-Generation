# Prashn-Setu: Project State & Execution Plan

**Document Version:** 1.0.0  
**Last Updated:** 2026-10-15  
**System Environment:** Linux, Python 3.12.3, PyTorch 2.14.1+cu130, Transformers 5.19.0, NVIDIA RTX 4000 Ada (20 GB VRAM)  
**Base Checkpoint:** `checkpoints/stage1_indicnlg_qg` (~244M parameters, IndicBART)

---

## 1. Executive Status Summary

Prashn-Setu has completed a rigorous Stage 1 audit and full-test reproduction across all 10,553 held-out test examples. The evaluation pipeline has been fixed and accelerated, resolving a critical metric defect where standard ROUGE stripped Devanagari characters. A systematic qualitative error analysis and a reproducible 100-sample human evaluation protocol have been established.

Stage 2 (Curriculum Adaptation on Native-Hindi Textbooks) is currently **blocked** due to data insufficiency (only 3 template examples exist in `data/ncert_gold_hindi.jsonl` with zero verified human examples). Architectural safeguards and an automated data validation engine have been put in place to ensure Stage 2 cannot be triggered until genuine, verified textbook data is curated.

---

## 2. Component Inventory & Verification Matrix

| Component | Repository Path | Current Status | Verification Command | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1 Checkpoint** | `checkpoints/stage1_indicnlg_qg` | **Verified Operational** | `.venv/bin/python test_stage1.py` | 244M params, loads with safetensors; special tokens `<2hi>` (64006), `<s>` (64000), `</s>` (64001). |
| **Stage 1 Held-Out Test Set** | `data/hi_test.jsonl` | **Verified Disjoint** | Overlap audit script | 10,553 records; 0 `squad_id` overlap with train/val. Max source length: 263 tokens. |
| **Stage 1 Full Evaluation** | `results/stage1_test_results_full.json` | **Verified Complete** | `.venv/bin/python run.py evaluate ...` | 10,553 samples evaluated in 273.76s (38.55 samples/sec). SacreBLEU: 10.196, chrF++: 29.397. |
| **Inference API** | `src/evaluation/inference.py` | **Verified Enhanced** | `test_batch_eval.py` | Supports single and batched generation (`generate_batch`), input length clipping (512), and Hindi punctuation cleanup. |
| **Human Evaluation Protocol** | `results/HUMAN_EVALUATION_PROTOCOL.md` | **Verified Operational** | Manual inspection | 5 scoring dimensions (1–5 scale), hallucination flag, and agreement methodology. |
| **Human Evaluation Sheet** | `results/human_eval_sample_100.csv` | **Verified Clean** | Artifact audit script | 100 stratified test predictions with 100% blank rating fields. Master copy in `human_eval_clean_template.csv`. |
| **Qualitative Error Analysis** | `results/QUALITATIVE_ERROR_ANALYSIS.md` | **Verified Complete** | Artifact audit script | 5-category failure taxonomy covering Wh-word mismatch, numeric hallucination, and distractor confusion. |
| **Dataset Validator CLI** | `scripts/validate_dataset.py` | **Verified Operational** | `python run.py validate-data ...` | Validates schema, answer grounding, Devanagari syntax, duplicates, and cross-split leakage. |
| **Stage 2 Adaptation Script** | `src/training/adapt_stage2.py` | **Verified Guarded** | `python run.py adapt-stage2 ...` | Enforces 50-example minimum guard, separate train/val files, LoRA protection, and distinct output dir. |
| **Stage 2 Checkpoint** | `checkpoints/stage2_ncert_gold_lora` | **Experimental Adapter** | `ModelManager.verify_stage2()` | Valid PEFT LoRA configuration ($r=16, \alpha=32$ on `q_proj`, `v_proj`). Training provenance cannot be independently confirmed from repository data; curriculum efficacy pending validation on verified textbook data. |
| **Interactive CLI Demo** | `src/evaluation/inference.py` | **Verified Operational** | `python run.py infer` | Terminal-based interactive QG prompt. |
| **Web / UI Demo** | `src/demo/app.py` | **Verified Operational** | `python run.py demo --port 7860` | 4-workspace minimalist UI (`Question Studio & Hub`, `Practice & Quiz`, `Model Comparison`, `Research & System Analytics`), bilingual button labeling (`[Hindi] ([English])`), inline evidence viewer, textbook upload flow, Devanagari Unicode repair engine, and local TrueType font serving. |
| **Automated Test Suite** | `tests/` | **Verified Operational** | `python -m unittest tests/test_platform_features.py tests/test_model_comparison.py` | 49 automated tests covering model verification, grounding, quiz, bank, exports, API validation, and Devanagari Unicode matra repair. |
| **Technical Report** | `docs/TECHNICAL_REPORT.md` | **Verified Complete** | File review | Full scientific report covering Stage 1 baseline, Devanagari ROUGE discovery, error taxonomy, and LoRA framework. |
| **Project README** | `README.md` | **Verified Audited** | File review | Updated with verified benchmarks, research integrity audit, and limitations. |

---

## 3. Known Methodological Limitations & Bugs Addressed

1. **The ROUGE Devanagari Erasure Bug (Resolved):**
   - *Issue:* Hugging Face `evaluate.load("rouge")` uses an internal regex (`[^a-z0-9]+`) that erases all Devanagari characters, leaving only numbers and Latin words. This caused historical Hindi ROUGE scores to report an artificially low ~4.6%.
   - *Fix:* Integrated native Hindi whitespace-tokenized ROUGE alongside default ROUGE. Genuine lexical recall is **29.71% ROUGE-1** and **27.33% ROUGE-L**.
2. **Translationese in Stage 1 Training Data (Documented):**
   - The training set (`ai4bharat/IndicQuestionGeneration`) is machine-translated from English SQuAD v1.1. Consequently, questions inherit English syntax and unnatural phrasing.
3. **Wh-Word & Answer Type Mismatch (Documented):**
   - In ~35% of failure cases, the model confuses interrogative pronouns (e.g. asking "कब" / when for locations or frequency counts).
4. **Factual Hallucination in Numbers (Documented):**
   - Observed numeric mutations (e.g. altering football scores from 24-10 to 24-25).
5. **Defective Vocabulary Pruner (Isolated):**
   - `src/models/prune_vocab.py` slices embedding rows without updating the tokenizer vocabulary mappings. Executing `python run.py prune` remains strictly forbidden.

---

## 4. Work Execution Roadmap

```
[Phase 1: Baseline Smoke & Regression Tests] ──> [Phase 2: Comprehensive Test Suite]
                        │
                        ▼
[Phase 3: Human Annotation Analysis Tooling] ──> [Phase 4: Lightweight Web Demo]
                        │
                        ▼
[Phase 5: Technical Report & Root Documentation] ──> [Phase 6: Final Verification & Audit]
```

1. **Step 1:** Implement and execute comprehensive baseline regression smoke tests covering 10 distinct linguistic and edge-case scenarios.
2. **Step 2:** Build the human evaluation analysis tool (`scripts/analyze_human_eval.py`) to validate completed reviewer CSVs, compute mean/median ratings, score distributions, and inter-rater agreement statistics with test fixtures.
3. **Step 3:** Implement an automated test suite (`tests/`) executable via `python -m unittest` covering data validation, preprocessing, inference parity, metric computation, and safety guards.
4. **Step 4:** Build a lightweight, robust, zero-dependency browser-based Web Demo (`src/demo/app.py` and `python run.py demo`) with Hindi passage input, answer field, beam search options, instant question generation, and validation alerts.
5. **Step 5:** Author the comprehensive technical course report (`docs/TECHNICAL_REPORT.md`), architecture diagrams, experiment log (`docs/EXPERIMENT_LOG.md`), and overhaul `README.md`.
6. **Step 6:** Run the full automated verification suite, verify system state tracker, and deliver the final submission-ready audit.
