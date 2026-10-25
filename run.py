#!/usr/bin/env python3
"""
Prashn-Setu Unified Execution Command Center.
Provides single-entry-point CLI to run inference, interactive sessions, and the web demo.

Usage:
  python run.py infer                 # Launch interactive terminal question generator
  python run.py demo                  # Launch interactive web demo server (default: port 7860)
  python run.py evaluate              # Benchmark checkpoint on test data
"""

import sys
import os
import argparse

# Ensure src/ is accessible from root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def cmd_infer(args):
    from src.evaluation.inference import QuestionGenerator, interactive_session
    generator = QuestionGenerator(checkpoint_path=args.checkpoint)
    interactive_session(generator)

def cmd_demo(args):
    from src.demo.app import run_demo_server
    run_demo_server(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port
    )

def cmd_evaluate(args):
    from src.evaluation.evaluate import evaluate_checkpoint
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        output_report_path=args.output,
        max_eval_samples=args.max_samples if args.max_samples > 0 else None,
        batch_size=args.batch_size
    )

def cmd_compare(args):
    from src.evaluation.compare_models import run_model_comparison_benchmark
    run_model_comparison_benchmark(
        test_data_path=args.test_data,
        output_path=args.output,
        max_eval_samples=args.max_samples,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_length=args.max_length,
        models_to_test=args.models
    )

def main():
    parser = argparse.ArgumentParser(
        description="Prashn-Setu: IndicBART Question Generation Execution Center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Available execution subcommands")

    default_checkpoint = (
        "checkpoints/stage2_ncert_gold_lora"
        if os.path.exists("checkpoints/stage2_ncert_gold_lora")
        else "checkpoints/stage1_indicnlg_qg"
    )

    # 1. Infer
    p_infer = subparsers.add_parser("infer", help="Launch interactive generation CLI")
    p_infer.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Checkpoint path to load")
    p_infer.set_defaults(func=cmd_infer)

    # 2. Web Demo
    p_demo = subparsers.add_parser("demo", help="Launch interactive browser demonstration server")
    p_demo.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Checkpoint path to load")
    p_demo.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    p_demo.add_argument("--port", type=int, default=7860, help="Port to listen on (default 7860)")
    p_demo.set_defaults(func=cmd_demo)

    # 3. Evaluate Single Checkpoint
    p_eval = subparsers.add_parser("evaluate", help="Benchmark checkpoint on test dataset")
    p_eval.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Checkpoint path to load")
    p_eval.add_argument("--test_data", type=str, default=None, help="Path to test JSONL file")
    p_eval.add_argument("--output", type=str, default="eval_results.json", help="Output JSON results file")
    p_eval.add_argument("--max_samples", type=int, default=100, help="Max test samples (0 for all)")
    p_eval.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    p_eval.set_defaults(func=cmd_evaluate)

    # 4. Compare Models
    p_comp = subparsers.add_parser("compare", help="Run formal multi-model comparison benchmark")
    p_comp.add_argument("--test_data", type=str, default=None, help="Path to test JSONL file")
    p_comp.add_argument("--output", type=str, default="results/model_comparison_results.json", help="Output JSON")
    p_comp.add_argument("--max_samples", type=int, default=100, help="Max test samples")
    p_comp.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p_comp.add_argument("--num_beams", type=int, default=4, help="Beam size")
    p_comp.add_argument("--max_length", type=int, default=64, help="Max target tokens")
    p_comp.add_argument("--models", nargs="+", default=None, help="Model IDs to test (e.g. stage1 stage2)")
    p_comp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == "__main__":
    main()
