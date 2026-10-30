# System State & Change Tracker

> **Last Automatically Refreshed:** `2026-10-30 11:40:14 UTC`  
> **Total Tracked Files:** `56` | **Total Code & Doc Lines:** `524102`

This file is dynamically updated by running `python scripts/update_tracker.py`.
It maintains the live inventory of codebase files, SHA-256 checksums, dataset splits, and model checkpoints.

---
## 1. Tracked Files Inventory & Integrity Checksums

| File Path | Category | Size (KB) | Lines (LOC) | SHA-256 (12 chars) | Last Modified |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `architecture/COMPONENT_INTERACTIONS.md` | Architecture Doc | 7.76 | 142 | `d44059e11aec` | 2026-10-30 11:20:19 |
| `architecture/FILES_CATALOG.md` | Architecture Doc | 10.84 | 139 | `df89fb8269e7` | 2026-10-30 11:19:57 |
| `architecture/OPERATIONAL_GUIDELINES.md` | Architecture Doc | 5.55 | 95 | `7699f76fc8e7` | 2026-10-30 11:21:24 |
| `architecture/PREREQUISITES_AND_ENVIRONMENT_SETUP.md` | Architecture Doc | 2.80 | 114 | `d5f5c85cf59b` | 2026-10-30 11:20:32 |
| `architecture/README.md` | Architecture Doc | 2.16 | 33 | `fb0f3777d394` | 2026-10-30 11:19:28 |
| `architecture/SOLUTION_ANALYSIS_AND_ALTERNATIVES.md` | Architecture Doc | 6.44 | 93 | `5f13ce535139` | 2026-10-30 11:21:03 |
| `architecture/SYSTEM_STATE_TRACKER.md` | Architecture Doc | 17.71 | 180 | `6a9e4aa2e810` | 2026-10-23 20:05:54 |
| `scripts/update_tracker.py` | Automation Script | 6.05 | 149 | `d732ab72f62f` | 2026-10-30 11:40:00 |
| `src/__init__.py` | Core Configuration | 0.04 | 3 | `8b7f429dc042` | 2026-10-23 20:06:04 |
| `src/config.py` | Core Configuration | 3.22 | 84 | `7b05b064578b` | 2026-10-23 20:06:04 |
| `src/data/question_bank.py` | Data Module | 9.89 | 334 | `1572ca8d21cb` | 2026-10-28 13:11:15 |
| `data/sample_ncert_contexts.jsonl` | Dataset Artifact | 2.96 | 5 | `229096a868b3` | 2026-10-23 20:06:04 |
| `src/demo/__init__.py` | Demo Interface | 0.05 | 3 | `edb85ec5b7f1` | 2026-10-23 20:06:04 |
| `src/demo/app.py` | Demo Interface | 30.30 | 649 | `8395a595e27e` | 2026-10-30 11:31:25 |
| `src/demo/index.html` | Demo Interface | 120.50 | 2412 | `2ab729df73c5` | 2026-10-30 11:36:53 |
| `results/HUMAN_EVALUATION_PROTOCOL.md` | Evaluation Result | 6.71 | 102 | `71f6aaeebd9c` | 2026-10-28 17:09:31 |
| `results/QUALITATIVE_ERROR_ANALYSIS.md` | Evaluation Result | 10.38 | 109 | `0a5036ff1a7b` | 2026-10-23 20:05:55 |
| `results/human_eval_blind_key.json` | Evaluation Result | 4.18 | 206 | `020b323d688f` | 2026-10-28 12:26:39 |
| `results/human_eval_blinded_comparison.csv` | Evaluation Result | 46.93 | 51 | `0effdbd72ab5` | 2026-10-28 12:26:39 |
| `results/human_eval_sample_100.csv` | Evaluation Result | 77.09 | 101 | `0a32f53ee1fc` | 2026-10-23 20:05:55 |
| `results/mock_human_eval_summary.json` | Evaluation Result | 6.10 | 249 | `702dbcef1c32` | 2026-10-23 20:05:55 |
| `results/mock_human_eval_summary.md` | Evaluation Result | 2.22 | 42 | `61bfb0a12c39` | 2026-10-28 17:09:25 |
| `results/model_comparison_results.json` | Evaluation Result | 19.73 | 254 | `59b039f0e0c7` | 2026-10-28 12:26:39 |
| `results/stage1_qualitative_analysis.json` | Evaluation Result | 15.16 | 149 | `3a1f3a741747` | 2026-10-23 20:05:55 |
| `results/stage1_smoke_test.json` | Evaluation Result | 4.84 | 78 | `4eabee7c7a24` | 2026-10-23 20:05:54 |
| `results/stage1_test_results.json` | Evaluation Result | 4.88 | 43 | `e309c9e996e7` | 2026-10-23 20:05:54 |
| `results/stage1_test_results_full.json` | Evaluation Result | 13.77 | 117 | `35018cee2402` | 2026-10-23 20:05:55 |
| `src/evaluation/__init__.py` | Inference & Eval | 0.57 | 26 | `27d11d78f836` | 2026-10-25 12:08:59 |
| `src/evaluation/advanced_generator.py` | Inference & Eval | 9.48 | 260 | `6b8f59f94f9e` | 2026-10-28 13:21:41 |
| `src/evaluation/analytics_loader.py` | Inference & Eval | 9.68 | 170 | `5f52b971756a` | 2026-10-28 17:08:43 |
| `src/evaluation/compare_models.py` | Inference & Eval | 18.83 | 472 | `0aa6f54dacf4` | 2026-10-25 12:15:02 |
| `src/evaluation/evaluate.py` | Inference & Eval | 6.96 | 169 | `b783bc68cd63` | 2026-10-23 20:06:04 |
| `src/evaluation/evidence_checker.py` | Inference & Eval | 8.23 | 203 | `dfa996814e2b` | 2026-10-28 13:11:19 |
| `src/evaluation/inference.py` | Inference & Eval | 23.72 | 524 | `0de2bb8479c1` | 2026-10-24 07:31:42 |
| `src/evaluation/model_registry.py` | Inference & Eval | 17.12 | 434 | `630636d44829` | 2026-10-28 13:01:55 |
| `src/evaluation/paper_builder.py` | Inference & Eval | 13.69 | 379 | `ed42be01a37e` | 2026-10-28 13:11:20 |
| `src/evaluation/quiz_engine.py` | Inference & Eval | 3.90 | 98 | `01fa9d45448c` | 2026-10-28 13:11:20 |
| `src/evaluation/textbook_processor.py` | Inference & Eval | 9.69 | 262 | `8d686c4daa55` | 2026-10-30 11:29:43 |
| `checkpoints/stage1_indicnlg_qg/config.json` | Model Checkpoint | 0.92 | 37 | `47c89d81b5c0` | 2026-10-23 20:05:55 |
| `checkpoints/stage1_indicnlg_qg/generation_config.json` | Model Checkpoint | 0.91 | 36 | `7043d37b1c69` | 2026-10-23 20:05:55 |
| `checkpoints/stage1_indicnlg_qg/tokenizer.json` | Model Checkpoint | 4920.74 | 256317 | `5c954f099029` | 2026-10-23 20:06:04 |
| `checkpoints/stage1_indicnlg_qg/tokenizer_config.json` | Model Checkpoint | 0.58 | 27 | `30a708a3c199` | 2026-10-23 20:06:04 |
| `checkpoints/stage2_ncert_gold_lora/adapter_config.json` | Model Checkpoint | 1.09 | 46 | `2762b51d388d` | 2026-10-23 20:06:04 |
| `checkpoints/stage2_ncert_gold_lora/tokenizer.json` | Model Checkpoint | 4920.74 | 256317 | `c88e334165be` | 2026-10-23 20:06:04 |
| `checkpoints/stage2_ncert_gold_lora/tokenizer_config.json` | Model Checkpoint | 0.76 | 34 | `ce5eaa45ebdb` | 2026-10-23 20:06:04 |
| `docs/ARCHITECTURE.md` | Project Documentation | 9.81 | 190 | `6ba64707bc70` | 2026-10-23 20:05:55 |
| `docs/EXPERIMENT_LOG.md` | Project Documentation | 9.83 | 178 | `4189b412b3c7` | 2026-10-23 20:05:55 |
| `docs/NCERT_GOLD_CURATION_GUIDE.md` | Project Documentation | 10.64 | 168 | `cb18b965724a` | 2026-10-23 20:05:55 |
| `docs/PROJECT_STATE_AND_EXECUTION_PLAN.md` | Project Documentation | 7.25 | 73 | `b3a096063ee6` | 2026-10-28 17:09:49 |
| `docs/TECHNICAL_REPORT.md` | Project Documentation | 19.42 | 282 | `9d5e782aa3ae` | 2026-10-23 20:05:55 |
| `README.md` | Root File | 27.90 | 364 | `5cfec518266e` | 2026-10-28 17:11:03 |
| `infer.py` | Root File | 4.68 | 82 | `776f341c55b8` | 2026-10-23 20:05:54 |
| `requirements.txt` | Root File | 0.27 | 15 | `bb36d84a91c9` | 2026-10-28 13:19:34 |
| `run.py` | Root File | 4.49 | 108 | `65dbd9a60bf7` | 2026-10-25 12:10:42 |
| `tests/test_model_comparison.py` | Test Suite | 18.74 | 412 | `01d7ccba2149` | 2026-10-30 11:37:55 |
| `tests/test_platform_features.py` | Test Suite | 21.85 | 483 | `e16eff7870ff` | 2026-10-30 11:38:48 |

---
## 2. Checkpoints & Architecture Status

- **Stage 1 Base Checkpoint:** `checkpoints/stage1_indicnlg_qg/` (Present, 244M parameters)
- **Stage 2 NCERT LoRA Adapter:** `checkpoints/stage2_ncert_gold_lora/` (Present, PEFT LoRA $r=16$, $\alpha=32$)
- **IndicNLG Baseline Checkpoint:** `checkpoints/baseline_indicnlg_qg/` (Honest status: Unavailable, not substituted)

---
## 3. Active Workspaces & Engine Summary

- **Question Studio & Hub:** Single-passage and multi-page textbook chapter question generation with inline context evidence.
- **Practice & Quiz Mode:** Instant interactive evaluation with scoring, percentage, correct/incorrect feedback, and retry mechanism.
- **Model Comparison:** Pure Stage 1 IndicBART vs. Stage 2 NCERT-Adapted LoRA comparison.
- **Research & System Analytics:** English-first dashboard with ROUGE, SacreBLEU, METEOR benchmarks, human evaluation rubric, qualitative error analysis, and hardware status.

