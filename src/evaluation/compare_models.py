"""
Formal Research Evaluation and Multi-Model Comparison Suite for Prashn-Setu.
Benchmarks all available model configurations (Baseline, Stage 1, Stage 2)
on identical held-out test examples with consistent preprocessing and decoding parameters.

Computes:
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4
  - SacreBLEU
  - ROUGE-1, ROUGE-2, ROUGE-L (standard & Hindi-tokenized)
  - chrF++
  - Mean inference latency (ms/sample)
  - Throughput (samples/second)

Exports:
  - results/model_comparison_results.json (quantitative benchmark artifact)
  - results/human_eval_blinded_comparison.csv (blinded human eval protocol sample)
  - results/human_eval_blind_key.json (blinding key mapping)
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DataConfig
from src.evaluation.model_registry import (
    get_model_registry,
    get_shared_model_manager,
    MODEL_BASELINE,
    MODEL_STAGE1,
    MODEL_STAGE2,
    ALL_MODEL_IDS
)


def compute_bleu_n_gram_scores(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Computes BLEU-1, BLEU-2, BLEU-3, and BLEU-4 with sentence-level smoothing.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1

    b1_scores, b2_scores, b3_scores, b4_scores = [], [], [], []

    for pred, refs in zip(predictions, references):
        pred_tokens = pred.strip().split()
        ref_tokens_list = [r.strip().split() for r in refs if r.strip()]
        if not pred_tokens or not ref_tokens_list:
            continue

        b1 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
        b2 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        b3 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.333, 0.333, 0.334, 0), smoothing_function=smooth)
        b4 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        b1_scores.append(b1 * 100.0)
        b2_scores.append(b2 * 100.0)
        b3_scores.append(b3 * 100.0)
        b4_scores.append(b4 * 100.0)

    n = max(len(b1_scores), 1)
    return {
        "bleu1": round(sum(b1_scores) / n, 3),
        "bleu2": round(sum(b2_scores) / n, 3),
        "bleu3": round(sum(b3_scores) / n, 3),
        "bleu4": round(sum(b4_scores) / n, 3)
    }


def run_model_comparison_benchmark(
    test_data_path: Optional[str] = None,
    output_path: str = "results/model_comparison_results.json",
    blinded_csv_path: str = "results/human_eval_blinded_comparison.csv",
    blinded_key_path: str = "results/human_eval_blind_key.json",
    max_eval_samples: int = 100,
    batch_size: int = 16,
    num_beams: int = 4,
    max_length: int = 64,
    no_repeat_ngram_size: int = 3,
    models_to_test: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes a formal comparative benchmark across all requested models on identical test items.
    """
    import evaluate

    print("\n" + "=" * 65)
    print(" Prashn-Setu: Multi-Model Research Comparison Suite")
    print("=" * 65)

    registry = get_model_registry()
    print("\n[Audit] Registered Model Configurations:")
    for mid, info in registry.items():
        print(f"  • {mid:<10}: {info['name']} [{info['status'].upper()}]")
        if info["status"] == "unavailable":
            print(f"    Reason: {info.get('unavailable_reason')}")

    # Determine models to evaluate
    if models_to_test is None:
        models_to_test = [m for m in ALL_MODEL_IDS if registry[m]["status"] == "available"]

    print(f"\n[Config] Active models for this benchmark run: {models_to_test}")

    # Load test dataset
    data_cfg = DataConfig()
    test_file = test_data_path or data_cfg.test_jsonl

    test_records = []
    if os.path.exists(test_file) and test_file.endswith(".csv"):
        import csv
        print(f"[Data] Loading local CSV test dataset from: {test_file}")
        with open(test_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_records.append({
                    "context": row.get("context", ""),
                    "answer": row.get("answer", row.get("target_answer", "")),
                    "question": row.get("reference_question", row.get("question", ""))
                })
    elif os.path.exists(test_file):
        print(f"[Data] Loading local test data from: {test_file}")
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_records.append(json.loads(line.strip()))
    else:
        # Check for human eval sample 100 which has ground truth reference questions
        human_csv = os.path.join(PROJECT_ROOT, "results", "human_eval_sample_100.csv")
        sample_ncert = os.path.join(PROJECT_ROOT, "data", "sample_ncert_contexts.jsonl")

        if os.path.exists(human_csv):
            import csv
            print(f"[Data] Using curated gold test benchmark with reference questions from: {human_csv}")
            with open(human_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    test_records.append({
                        "context": row.get("context", ""),
                        "answer": row.get("answer", row.get("target_answer", "")),
                        "question": row.get("reference_question", row.get("question", ""))
                    })
        elif os.path.exists(sample_ncert):
            print(f"[Data] Using curated NCERT sample contexts from: {sample_ncert}")
            with open(sample_ncert, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        test_records.append(json.loads(line.strip()))
        else:
            raise FileNotFoundError(f"Neither {test_file} nor {sample_ncert} could be found.")

    if max_eval_samples and len(test_records) > max_eval_samples:
        print(f"[Data] Subsampling to {max_eval_samples} examples for standardized benchmarking.")
        test_records = test_records[:max_eval_samples]

    num_samples = len(test_records)
    print(f"[Data] Total test examples to evaluate per model: {num_samples}")

    contexts = [s.get("context", "").strip() for s in test_records]
    answers = [s.get("answer", "").strip() for s in test_records]
    references = [[s.get("question", "").strip()] for s in test_records]

    # Initialize evaluate metrics
    print("\n[Metrics] Initializing SacreBLEU, ROUGE, and chrF++...")
    metric_bleu = evaluate.load("sacrebleu")
    metric_rouge = evaluate.load("rouge")
    metric_chrf = evaluate.load("chrf")

    model_manager = get_shared_model_manager()
    comparison_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluated_sample_count": num_samples,
            "test_dataset": test_file,
            "decoding_parameters": {
                "strategy": "beam_search",
                "num_beams": num_beams,
                "max_length": max_length,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "batch_size": batch_size
            },
            "metric_definitions": {
                "sacreBLEU": "Corpus-level SacreBLEU with standard smoothing",
                "bleu_1_to_4": "NLTK sentence-smoothed n-gram BLEU (1 to 4)",
                "rouge": "Standard Hugging Face ROUGE (rouge1, rouge2, rougeL)",
                "rouge_hindi_tokenized": "Whitespace tokenized ROUGE for Devanagari morphemes",
                "chrF++": "chrF++ with word_order=2 character n-gram scoring",
                "mean_latency_ms": "Average generation latency per sample in milliseconds"
            }
        },
        "models": {},
        "unavailable_models": {
            m: {
                "name": registry[m]["name"],
                "reason": registry[m].get("unavailable_reason", "Not available")
            }
            for m in ALL_MODEL_IDS if registry[m]["status"] == "unavailable"
        },
        "relative_improvements": {},
        "qualitative_comparisons": []
    }

    model_predictions: Dict[str, List[str]] = {}

    for model_id in models_to_test:
        model_info = registry[model_id]
        print(f"\n" + "-" * 50)
        print(f"▶ Benchmarking: {model_info['name']} ({model_id})")
        print(f"  Checkpoint: {model_info['checkpoint_path']}")
        print("-" * 50)

        predictions = []
        latencies_ms = []

        t0_total = time.time()
        for i in range(0, num_samples, batch_size):
            b_ctx = contexts[i:i + batch_size]
            b_ans = answers[i:i + batch_size]

            t_batch_start = time.time()
            # Generate predictions batch-by-batch
            batch_preds = []
            for c, a in zip(b_ctx, b_ans):
                t_single = time.time()
                q_list = model_manager.generate_single_for_model(
                    model_id=model_id,
                    context=c,
                    answer=a,
                    num_beams=num_beams,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=False,
                    no_repeat_ngram_size=no_repeat_ngram_size
                )
                latencies_ms.append((time.time() - t_single) * 1000.0)
                batch_preds.append(q_list[0] if q_list else "")

            predictions.extend(batch_preds)
            print(f"  Processed {len(predictions)}/{num_samples} samples...")

        total_runtime = time.time() - t0_total
        samples_per_sec = num_samples / total_runtime if total_runtime > 0 else 0.0
        mean_latency = sum(latencies_ms) / max(len(latencies_ms), 1)

        model_predictions[model_id] = predictions

        # Compute Metrics
        print(f"  Computing metrics for {model_id}...")
        bleu_res = metric_bleu.compute(predictions=predictions, references=references)
        bleu_score = float(bleu_res["score"]) if (bleu_res and "score" in bleu_res) else 0.0

        ngram_bleu = compute_bleu_n_gram_scores(predictions=predictions, references=references)

        rouge_res = metric_rouge.compute(predictions=predictions, references=[ref[0] for ref in references])
        rouge_dict = rouge_res if rouge_res else {}

        rouge_hi_res = metric_rouge.compute(
            predictions=predictions,
            references=[ref[0] for ref in references],
            tokenizer=lambda x: x.split()
        )
        rouge_hi_dict = rouge_hi_res if rouge_hi_res else {}

        chrf_res = metric_chrf.compute(predictions=predictions, references=references, word_order=2)
        chrf_score = float(chrf_res["score"]) if (chrf_res and "score" in chrf_res) else 0.0

        model_result = {
            "model_id": model_id,
            "name": model_info["name"],
            "checkpoint": model_info["checkpoint_path"],
            "total_samples": num_samples,
            "total_runtime_seconds": round(total_runtime, 2),
            "samples_per_second": round(samples_per_sec, 2),
            "mean_latency_ms": round(mean_latency, 2),
            "metrics": {
                "sacreBLEU": round(bleu_score, 3),
                "bleu_1": ngram_bleu["bleu1"],
                "bleu_2": ngram_bleu["bleu2"],
                "bleu_3": ngram_bleu["bleu3"],
                "bleu_4": ngram_bleu["bleu4"],
                "rouge1": round(float(rouge_dict.get("rouge1", 0.0)) * 100, 3),
                "rouge2": round(float(rouge_dict.get("rouge2", 0.0)) * 100, 3),
                "rougeL": round(float(rouge_dict.get("rougeL", 0.0)) * 100, 3),
                "rouge1_hindi": round(float(rouge_hi_dict.get("rouge1", 0.0)) * 100, 3),
                "rouge2_hindi": round(float(rouge_hi_dict.get("rouge2", 0.0)) * 100, 3),
                "rougeL_hindi": round(float(rouge_hi_dict.get("rougeL", 0.0)) * 100, 3),
                "chrF++": round(chrf_score, 3)
            }
        }
        comparison_results["models"][model_id] = model_result

    # Compute relative differences if both Stage 1 and Stage 2 were evaluated
    if MODEL_STAGE1 in comparison_results["models"] and MODEL_STAGE2 in comparison_results["models"]:
        s1_m = comparison_results["models"][MODEL_STAGE1]["metrics"]
        s2_m = comparison_results["models"][MODEL_STAGE2]["metrics"]

        def _diff(metric_key):
            v1 = s1_m.get(metric_key, 0.0)
            v2 = s2_m.get(metric_key, 0.0)
            delta = v2 - v1
            pct = (delta / v1 * 100.0) if v1 > 0 else 0.0
            return {
                "stage1_baseline": v1,
                "stage2_adapted": v2,
                "absolute_delta": round(delta, 3),
                "percentage_change": round(pct, 2)
            }

        comparison_results["relative_improvements"] = {
            "sacreBLEU": _diff("sacreBLEU"),
            "rougeL": _diff("rougeL"),
            "rougeL_hindi": _diff("rougeL_hindi"),
            "chrF++": _diff("chrF++")
        }

    # Build qualitative comparison examples
    qual_count = min(15, num_samples)
    for idx in range(qual_count):
        item = {
            "sample_index": idx + 1,
            "context": contexts[idx],
            "answer": answers[idx],
            "reference_question": references[idx][0],
            "predictions": {}
        }
        for mid in models_to_test:
            if mid in model_predictions:
                item["predictions"][mid] = model_predictions[mid][idx]
        comparison_results["qualitative_comparisons"].append(item)

    # Save quantitative results JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    print(f"\n[Export] Saved model comparison report to: {output_path}")

    # Prepare Blinded Human Evaluation CSV and Key
    if len(models_to_test) >= 2:
        prepare_blinded_human_evaluation(
            contexts=contexts,
            answers=answers,
            references=references,
            model_predictions=model_predictions,
            models=models_to_test,
            output_csv_path=blinded_csv_path,
            output_key_path=blinded_key_path,
            sample_size=min(50, num_samples)
        )

    print_comparison_summary(comparison_results)
    return comparison_results


def prepare_blinded_human_evaluation(
    contexts: List[str],
    answers: List[str],
    references: List[List[str]],
    model_predictions: Dict[str, List[str]],
    models: List[str],
    output_csv_path: str,
    output_key_path: str,
    sample_size: int = 50
):
    """
    Creates a blinded evaluation dataset where reviewers cannot see model identity.
    For each example, model assignments ('Model A', 'Model B') are randomized with a fixed seed.
    """
    import csv

    rng = random.Random(42)
    sample_indices = list(range(min(sample_size, len(contexts))))

    blinded_rows = []
    blinding_key = {}

    for idx in sample_indices:
        ex_id = f"EVAL_{idx+1:03d}"
        ctx = contexts[idx]
        ans = answers[idx]
        ref = references[idx][0]

        # Shuffle model order
        shuffled_models = list(models)
        rng.shuffle(shuffled_models)

        blinding_key[ex_id] = {
            f"model_{chr(65+i)}": shuffled_models[i]
            for i in range(len(shuffled_models))
        }

        row = {
            "example_id": ex_id,
            "context": ctx,
            "target_answer": ans,
            "reference_question": ref,
        }
        for i, m in enumerate(shuffled_models):
            label = f"model_{chr(65+i)}_question"
            row[label] = model_predictions[m][idx] if m in model_predictions else ""
            row[f"model_{chr(65+i)}_fluency_1to5"] = ""
            row[f"model_{chr(65+i)}_answerability_1to5"] = ""
            row[f"model_{chr(65+i)}_faithfulness_1to5"] = ""
            row[f"model_{chr(65+i)}_hallucination_yes_no"] = ""

        row["reviewer_notes"] = ""
        blinded_rows.append(row)

    if blinded_rows:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        fieldnames = list(blinded_rows[0].keys())
        with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(blinded_rows)
        print(f"[Export] Saved blinded human evaluation template to: {output_csv_path}")

        with open(output_key_path, "w", encoding="utf-8") as f:
            json.dump({
                "description": "Blinded key mapping for human evaluation. Keep hidden from raters during scoring.",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mappings": blinding_key
            }, f, indent=2)
        print(f"[Export] Saved blinding key to: {output_key_path}")


def print_comparison_summary(results: Dict[str, Any]):
    """Prints a clean CLI comparison table."""
    print("\n" + "=" * 70)
    print("               FORMAL BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<12s} {'SacreBLEU':<12s} {'ROUGE-L':<12s} {'chrF++':<10s} {'Mean Latency':<14s}")
    print("-" * 70)

    for mid, data in results.get("models", {}).items():
        m = data["metrics"]
        lat = f"{data['mean_latency_ms']:.1f} ms"
        print(f"  {mid:<12s} {m['sacreBLEU']:<12.3f} {m['rougeL']:<12.3f} {m['chrF++']:<10.3f} {lat:<14s}")

    for mid, unavail in results.get("unavailable_models", {}).items():
        print(f"  {mid:<12s} [UNAVAILABLE: {unavail['reason'][:40]}...]")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prashn-Setu Multi-Model Benchmark Runner")
    parser.add_argument("--test_data", type=str, default=None, help="Path to test JSONL")
    parser.add_argument("--output", type=str, default="results/model_comparison_results.json", help="Output JSON")
    parser.add_argument("--max_samples", type=int, default=100, help="Max test samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size")
    parser.add_argument("--max_length", type=int, default=64, help="Max target tokens")
    parser.add_argument("--models", nargs="+", default=None, help="Model IDs to test (e.g. stage1 stage2)")
    args = parser.parse_args()

    run_model_comparison_benchmark(
        test_data_path=args.test_data,
        output_path=args.output,
        max_eval_samples=args.max_samples,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_length=args.max_length,
        models_to_test=args.models
    )
