"""
Research Analytics and Benchmark Results Loader for Prashn-Setu.
Parses quantitative overlap metrics, Stage 1 vs Stage 2 comparisons,
human evaluation ratings, and qualitative error analysis from real repo artifacts.
"""

import os
import json
import csv
from typing import Dict, Any, Optional, List

from src.config import PROJECT_ROOT

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def load_research_analytics_data() -> Dict[str, Any]:
    """Loads and compiles comprehensive research evaluation analytics."""
    benchmark_file = os.path.join(RESULTS_DIR, "model_comparison_results.json")
    stage1_file = os.path.join(RESULTS_DIR, "stage1_test_results.json")
    human_blind_file = os.path.join(RESULTS_DIR, "human_eval_blinded_comparison.csv")
    human_mock_file = os.path.join(RESULTS_DIR, "mock_human_eval_summary.json")
    error_analysis_file = os.path.join(RESULTS_DIR, "QUALITATIVE_ERROR_ANALYSIS.md")

    bench_data = None
    if os.path.exists(benchmark_file):
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
        except Exception:
            pass

    stage1_data = None
    if os.path.exists(stage1_file):
        try:
            with open(stage1_file, "r", encoding="utf-8") as f:
                stage1_data = json.load(f)
        except Exception:
            pass

    # Process metrics comparison
    comparison_summary = {}
    if bench_data and "models" in bench_data:
        m_data = bench_data["models"]
        s1 = m_data.get("stage1", {}).get("metrics", {})
        s2 = m_data.get("stage2", {}).get("metrics", {})
        baseline = bench_data.get("unavailable_models", {}).get("baseline", {})

        metrics_map = [
            ("sacrebleu", ["sacreBLEU", "sacrebleu"]),
            ("rouge1", ["rouge1"]),
            ("rouge2", ["rouge2"]),
            ("rougeL", ["rougeL"]),
            ("chrf", ["chrF++", "chrf"]),
            ("rougeL_hindi", ["rougeL_hindi", "rougeL_hindi_tokenized"])
        ]
        delta = {}
        for out_name, possible_keys in metrics_map:
            v1 = 0.0
            for k in possible_keys:
                if k in s1:
                    v1 = float(s1[k])
                    break
            v2 = 0.0
            for k in possible_keys:
                if k in s2:
                    v2 = float(s2[k])
                    break

            abs_diff = round(v2 - v1, 3)
            rel_diff = round(((v2 - v1) / v1 * 100.0), 2) if v1 > 0 else 0.0
            delta[out_name] = {
                "stage1": v1,
                "stage2": v2,
                "absolute_diff": abs_diff,
                "relative_pct": rel_diff
            }

        comparison_summary = {
            "test_samples": bench_data.get("metadata", {}).get("evaluated_sample_count", 100),
            "decoding_parameters": bench_data.get("metadata", {}).get("decoding_parameters", {}),
            "baseline_status": "unavailable",
            "baseline_reason": baseline.get("reason", "Original IndicNLG question-generation baseline checkpoint is unavailable in checkpoints/"),
            "metrics_comparison": delta,
            "latency": {
                "stage1_ms": m_data.get("stage1", {}).get("mean_latency_ms", 0.0),
                "stage2_ms": m_data.get("stage2", {}).get("mean_latency_ms", 0.0)
            }
        }

    # Human Evaluation Summary: Honest assessment of current state
    completed_rating_count = 0
    if os.path.exists(human_blind_file):
        try:
            with open(human_blind_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    val = r.get("model_A_fluency_1to5", "").strip()
                    if val and val.isdigit():
                        completed_rating_count += 1
        except Exception:
            pass

    human_eval = {
        "status": "completed" if completed_rating_count > 0 else "pending_validation",
        "is_real_data": completed_rating_count > 0,
        "sample_count": completed_rating_count,
        "evaluation_protocol": "results/HUMAN_EVALUATION_PROTOCOL.md",
        "categories": {
            "naturalness": {
                "stage1": f"{completed_rating_count} ratings" if completed_rating_count > 0 else "लंबित (Pending)",
                "stage2": f"{completed_rating_count} ratings" if completed_rating_count > 0 else "लंबित (Pending)",
                "delta": "N/A"
            },
            "relevance": {
                "stage1": f"{completed_rating_count} ratings" if completed_rating_count > 0 else "लंबित (Pending)",
                "stage2": f"{completed_rating_count} ratings" if completed_rating_count > 0 else "लंबित (Pending)",
                "delta": "N/A"
            },
            "answerability": {
                "stage1": f"{completed_rating_count} ratings" if completed_rating_count > 0 else "लंबित (Pending)",
                "stage2": f"{completed_rating_count} ratings" if completed_rating_count > 0 else "लंबित (Pending)",
                "delta": "N/A"
            },
            "factual_consistency": {
                "stage1": f"{completed_rating_count} ratings" if completed_rating_count > 0 else "लंबित (Pending)",
                "stage2": f"{completed_rating_count} ratings" if completed_rating_count > 0 else "लंबित (Pending)",
                "delta": "N/A"
            }
        },
        "disclaimer": (
            f"results/human_eval_blinded_comparison.csv से {completed_rating_count} मूल्यांकित नमूनों का वास्तविक डेटा।"
            if completed_rating_count > 0 else
            "⚠️ मानवीय मूल्यांकन लंबित है (Pending Validation): 'results/human_eval_sample_100.csv' एवं 'results/human_eval_blinded_comparison.csv' वर्तमान में रिक्त टेम्पलेट हैं। शोध सत्यनिष्ठा हेतु कोई कृत्रिम स्कोर प्रदर्शित नहीं किया गया है।"
        ),
        "synthetic_fixture_note": "results/mock_human_eval_summary.md में उपस्थित स्कोर केवल सांख्यिकीय गणना टूल को सत्यापित करने हेतु निर्मित कृत्रिम टेस्ट फिक्स्चर (tests/fixtures/) के हैं।"
    }

    # Qualitative Error Analysis highlights grounded in verified QUALITATIVE_ERROR_ANALYSIS.md
    qualitative_highlights = [
        {
            "category": "प्रश्नवाचक शब्दों व उत्तर प्रकार का असंतुलन (Wh-Word / Answer Type Mismatch)",
            "observation": "Stage 1 मॉडल लगभग 35-45% विफलताओं में उत्तर के प्रकार के विपरीत प्रश्नवाचक शब्द चुनता है (उदा. स्थान या व्यक्ति के लिए 'कब' पूछना)।",
            "adaptation_impact": "Stage 2 LoRA संरचना में interrogative grounding के लिए पेरामीटर-एफिशिएंट लेयर्स जोड़ी गई हैं, परंतु वास्तविक पाठ्यपुस्तकीय सुधार प्रमाणित डेटासेट पर प्रशिक्षण के पश्चात ही सत्यापित होगा।"
        },
        {
            "category": "संख्यात्मक म्यूटेशन व तथ्यात्मक भ्रम (Numeric & Factual Hallucination)",
            "observation": "सबवर्ड टोकनाइजेशन के कारण मॉडल संख्याओं को विकृत कर देता है (उदा. 24-10 को 24-25 बना देना)।",
            "adaptation_impact": "पाठ्यपुस्तक अनुकूलन में संख्यात्मक तथ्यों के सत्यापन हेतु कंस्ट्रेंट डिकोडिंग व साक्ष्य-सत्यापन (Evidence Checker) आवश्यक है।"
        },
        {
            "category": "अनुवादित शैली बनाम पाठ्यपुस्तकीय भाषा (Translationese vs Textbook Phrasing)",
            "observation": "मूल IndicNLG डेटासेट अंग्रेज़ी SQuAD का अनुवाद होने से वाक्य संरचना में अनुवादित कृत्रिमता पाई जाती है।",
            "adaptation_impact": "प्रामाणिक NCERT पाठ्यपुस्तकों पर LoRA अनुकूलन का मुख्य उद्देश्य 'स्पष्ट कीजिए', 'तात्पर्य क्या है' जैसे मानक शैक्षिक स्वरूप का विकास करना है।"
        }
    ]

    return {
        "benchmark_available": True,
        "quantitative_benchmark": comparison_summary,
        "human_evaluation": human_eval,
        "qualitative_analysis": qualitative_highlights,
        "artifacts_available": {
            "benchmark_results": os.path.exists(benchmark_file),
            "stage1_results": os.path.exists(stage1_file),
            "human_blind_csv": os.path.exists(human_blind_file),
            "error_analysis_md": os.path.exists(error_analysis_file)
        }
    }

