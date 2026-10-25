"""
Centralized Configuration Module for Prashn-Setu.
Resolves all directories relative to PROJECT_ROOT and defines typed hyperparameter dataclasses.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# Root directory of the repository (/mnt/a/iiits/5/NLP/PROJECT)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@dataclass
class ModelConfig:
    # Model and Tokenizer settings
    model_name_or_path: str = "ai4bharat/IndicBART"
    tokenizer_name_or_path: str = "ai4bharat/IndicBART"
    
    # Target language settings for IndicBART
    target_lang: str = "hi"
    target_lang_tag: str = "<2hi>"
    
    # Token length limits
    max_source_length: int = 512    # Context + Answer
    max_target_length: int = 64     # Generated Question
    min_target_length: int = 6      # Minimum question length
    
    # Pruning options
    pruned_model_output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints", "indicbart_hi_pruned")
    use_pruned_model: bool = False   # Set True after running pruning


@dataclass
class DataConfig:
    # Hugging Face dataset identifier
    hf_dataset_name: str = "ai4bharat/IndicQuestionGeneration"
    hf_dataset_subset: str = "hi"
    
    # Local storage paths
    data_dir: str = os.path.join(PROJECT_ROOT, "data")
    raw_dataset_dir: str = os.path.join(PROJECT_ROOT, "data", "indic_question_generation_hi")
    gold_data_path: str = os.path.join(PROJECT_ROOT, "data", "ncert_gold_hindi.jsonl")
    
    # Split paths
    train_jsonl: str = os.path.join(PROJECT_ROOT, "data", "hi_train.jsonl")
    val_jsonl: str = os.path.join(PROJECT_ROOT, "data", "hi_validation.jsonl")
    test_jsonl: str = os.path.join(PROJECT_ROOT, "data", "hi_test.jsonl")


@dataclass
class TrainingConfig:
    # Stage 1: Fine-tuning on IndicNLG (ai4bharat/IndicQuestionGeneration)
    stage1_output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints", "stage1_indicnlg_qg")
    stage1_num_train_epochs: int = 5
    stage1_learning_rate: float = 3e-5
    stage1_per_device_train_batch_size: int = 8
    stage1_per_device_eval_batch_size: int = 8
    stage1_gradient_accumulation_steps: int = 2
    stage1_warmup_ratio: float = 0.1
    stage1_weight_decay: float = 0.01
    stage1_eval_steps: int = 1000
    stage1_save_steps: int = 1000
    stage1_logging_steps: int = 100
    stage1_save_total_limit: int = 3
    
    # Stage 2: Domain Adaptation on NCERT Gold Data (300-500 samples)
    stage2_output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints", "stage2_ncert_adapted")
    stage2_num_train_epochs: int = 5
    stage2_learning_rate: float = 1e-5   # Low learning rate to prevent catastrophic forgetting
    stage2_per_device_train_batch_size: int = 4
    stage2_per_device_eval_batch_size: int = 4
    stage2_gradient_accumulation_steps: int = 2
    stage2_warmup_ratio: float = 0.15
    stage2_weight_decay: float = 0.01
    
    # Generation parameters during evaluation
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    
    # Hardware acceleration
    fp16: bool = False      # Fallback mixed precision (can overflow in IndicBART)
    bf16: bool = True       # Enable BF16 for Ampere+ / Ada architectures (avoids FP16 gradient overflow)
    seed: int = 42
