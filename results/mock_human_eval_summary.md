# Prashn-Setu: Human Evaluation Tooling Verification Report (Synthetic Fixture)

> [!IMPORTANT]
> **RESEARCH INTEGRITY NOTICE — SYNTHETIC TEST FIXTURE ONLY:**
> This report was generated using synthetic demonstration fixtures (`tests/fixtures/mock_human_eval.csv` and `mock_human_eval_annotator2.csv`, $N=25$) to test and verify the statistical analysis engine (`scripts/analyze_human_eval.py`).
> **These figures DO NOT represent real human judgments or native-speaker evaluations of Prashn-Setu.**
> Actual human evaluation of Stage 1 and Stage 2 models on the held-out dataset remains **Pending Validation**. Both `results/human_eval_sample_100.csv` and `results/human_eval_blinded_comparison.csv` are clean, unrated templates awaiting rater execution under the protocol defined in `results/HUMAN_EVALUATION_PROTOCOL.md`.

**Date:** 2026-10-15 18:40:44
**Primary Evaluator:** Synthetic Fixture (Annotator 1 Mock)
**Sample Count:** 25 (Synthetic Test Fixture)
**Evaluated Count:** 25 (100.0%)

## 1. Quantitative Ratings Summary (1–5 Likert Scale)

| Dimension | Evaluated N | Mean | Std | Median | % Rated >= 4 (High Quality) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Answer Alignment | 25 | **4.04** | 0.79 | 4.0 | 72.0% |
| Context Faithfulness | 25 | **4.12** | 0.726 | 4.0 | 80.0% |
| Grammaticality & Naturalness | 25 | **4.64** | 0.49 | 5.0 | 100.0% |
| Answerability | 25 | **4.16** | 0.8 | 4.0 | 76.0% |
| Overall Pedagogical Quality | 25 | **4.0** | 0.645 | 4.0 | 80.0% |

## 2. Factual Hallucination Analysis

- **Total Checked:** 25
- **Hallucinatory Questions:** 4 (**16.0%**)
- **Faithful Grounded Questions:** 21 (**84.0%**)

## 3. Inter-Annotator Agreement (Dual Annotation)

Paired evaluation sample size: **N = 25**

| Dimension | Exact Agreement (%) | Within 1 Pt (%) | Quadratic Weighted Kappa | Pearson r |
| :--- | :---: | :---: | :---: | :---: |
| Answer Alignment | 76.0% | 100.0% | **0.834** | 0.849 |
| Context Faithfulness | 92.0% | 100.0% | **0.933** | 0.948 |
| Grammaticality & Naturalness | 84.0% | 100.0% | **0.716** | 0.735 |
| Answerability | 92.0% | 100.0% | **0.94** | 0.947 |
| Overall Pedagogical Quality | 88.0% | 100.0% | **0.857** | 0.873 |

- **Hallucination Agreement:** Exact: 84.0%, Cohen's Kappa: **0.576**
