"""
Benchmarking & Evaluation Suite for Prashn-Setu.
Evaluates model checkpoints (Base, Stage 1, Stage 2) against test datasets.
Computes SacreBLEU, ROUGE-1/2/L, and chrF++, and exports qualitative comparison reports.
"""

import os
import json
import argparse
from typing import Optional, Dict, Any, List, cast
from tqdm import tqdm
import torch
import evaluate
from datasets import load_dataset
import time
from src.evaluation.inference import QuestionGenerator
from src.config import DataConfig
from pathlib import Path

def evaluate_checkpoint(
    checkpoint_path: str = "checkpoints/stage1_indicnlg_qg",
    test_data_path: Optional[str] = None,
    output_report_path: str = "eval_results.json",
    max_eval_samples: Optional[int] = 500,
    batch_size: int = 32
):
    print(f"\nEvaluating Checkpoint: {checkpoint_path}")
    generator = QuestionGenerator(checkpoint_path=checkpoint_path)

    test_records = []
    data_cfg = DataConfig()
    
    test_file = test_data_path or data_cfg.test_jsonl
    if os.path.exists(test_file):
        print(f"Loading test file: {test_file}")
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_records.append(json.loads(line.strip()))
    else:
        print(f"Local file not found. Fetching test split from {data_cfg.hf_dataset_name} ({data_cfg.hf_dataset_subset})...")
        hf_ds: Any = load_dataset(data_cfg.hf_dataset_name, data_cfg.hf_dataset_subset)
        test_records = [dict(ex) for ex in cast(Any, hf_ds)["test"]]

    if max_eval_samples and len(test_records) > max_eval_samples:
        print(f"Subsampling evaluation to first {max_eval_samples} examples for speed.")
        test_records = test_records[:max_eval_samples]

    print(f"Total test examples to evaluate: {len(test_records)}")

    metric_bleu = evaluate.load("sacrebleu")
    metric_rouge = evaluate.load("rouge")
    metric_chrf = evaluate.load("chrf")

    contexts = [sample.get("context", "") for sample in test_records]
    answers = [sample.get("answer", "") for sample in test_records]
    references = [[sample.get("question", "").strip()] for sample in test_records]

    print(f"Generating predictions (batch_size={batch_size})...")
    start_time = time.time()
    
    predictions = []
    with tqdm(total=len(test_records)) as pbar:
        for i in range(0, len(test_records), batch_size):
            b_contexts = contexts[i:i + batch_size]
            b_answers = answers[i:i + batch_size]
            b_preds = generator.generate_batch(
                contexts=b_contexts,
                answers=b_answers,
                batch_size=batch_size,
                num_beams=4,
                max_length=64
            )
            predictions.extend(b_preds)
            pbar.update(len(b_contexts))

    runtime = time.time() - start_time
    samples_per_sec = len(predictions) / runtime if runtime > 0 else 0.0

    qualitative_samples = []
    for idx in range(min(15, len(test_records))):
        qualitative_samples.append({
            "context": contexts[idx],
            "answer": answers[idx],
            "ground_truth": references[idx][0],
            "predicted": predictions[idx]
        })

    print(f"\nCompleted generation in {runtime:.2f}s ({samples_per_sec:.2f} samples/sec).")
    print("Computing quantitative metrics...")

    bleu_res = metric_bleu.compute(predictions=predictions, references=references)
    bleu_score: float = float(bleu_res["score"]) if (bleu_res is not None and "score" in bleu_res) else 0.0
    
    # Standard ROUGE (default tokenizer, for exact parity with training/historical runs)
    rouge_res = metric_rouge.compute(predictions=predictions, references=[ref[0] for ref in references])
    rouge_score: Dict[str, Any] = rouge_res if rouge_res is not None else {}
    
    # Hindi-tokenized ROUGE (whitespace tokenizer, reflecting true Hindi word overlap)
    rouge_hi_res = metric_rouge.compute(
        predictions=predictions,
        references=[ref[0] for ref in references],
        tokenizer=lambda x: x.split()
    )
    rouge_score_hi: Dict[str, Any] = rouge_hi_res if rouge_hi_res is not None else {}
    
    chrf_res = metric_chrf.compute(predictions=predictions, references=references, word_order=2)
    chrf_score: float = float(chrf_res["score"]) if (chrf_res is not None and "score" in chrf_res) else 0.0

    results = {
        "checkpoint": checkpoint_path,
        "test_data": test_file,
        "sample_count": len(predictions),
        "runtime_seconds": round(runtime, 2),
        "samples_per_second": round(samples_per_sec, 2),
        "generation_config": {
            "num_beams": 4,
            "max_length": 64,
            "no_repeat_ngram_size": 3,
            "batch_size": batch_size,
            "decoder_start_token_id": generator.target_lang_id,
            "forced_bos_token_id": generator.target_lang_id
        },
        "metrics": {
            "sacreBLEU": round(bleu_score, 3),
            "rouge1": round(float(rouge_score.get("rouge1", 0.0)) * 100, 3),
            "rouge2": round(float(rouge_score.get("rouge2", 0.0)) * 100, 3),
            "rougeL": round(float(rouge_score.get("rougeL", 0.0)) * 100, 3),
            "chrF++": round(chrf_score, 3),
            "rouge1_hindi_tokenized": round(float(rouge_score_hi.get("rouge1", 0.0)) * 100, 3),
            "rouge2_hindi_tokenized": round(float(rouge_score_hi.get("rouge2", 0.0)) * 100, 3),
            "rougeL_hindi_tokenized": round(float(rouge_score_hi.get("rougeL", 0.0)) * 100, 3)
        },
        "qualitative_examples": qualitative_samples
    }

    print("\n" + "="*55)
    print("                 BENCHMARK RESULTS")
    print("="*55)
    print(f"  {'Metric':<28s}: Value")
    print("-" * 55)
    for k, v in results["metrics"].items():
        print(f"  {k:<28s}: {v}")
    print("="*55)
    
    report_dir = os.path.dirname(os.path.abspath(output_report_path)) or "."
    os.makedirs(report_dir, exist_ok=True)
    
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed evaluation report saved to: {output_report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Question Generation Models")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage1_indicnlg_qg", help="Checkpoint path")
    parser.add_argument("--test_data", type=str, default=None, help="Path to test JSONL file")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file")
    parser.add_argument("--max_samples", type=int, default=500, help="Max test samples (0 for all)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        output_report_path=args.output,
        max_eval_samples=args.max_samples if args.max_samples > 0 else None,
        batch_size=args.batch_size
    )
