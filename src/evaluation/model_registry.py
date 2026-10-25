"""
Model Registry and Shared Model Manager for Prashn-Setu.
Provides structured configuration, availability detection, and lifecycle management
for Baseline, Stage 1 (IndicBART + IndicNLG QG), and Stage 2 (NCERT Gold LoRA).
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
import torch

from src.config import PROJECT_ROOT


# Model Identifiers
MODEL_BASELINE = "baseline"
MODEL_STAGE1 = "stage1"
MODEL_STAGE2 = "stage2"

ALL_MODEL_IDS = [MODEL_BASELINE, MODEL_STAGE1, MODEL_STAGE2]


def get_model_registry() -> Dict[str, Dict[str, Any]]:
    """
    Returns the dynamic model registry with current on-disk availability status.
    Ensures that missing checkpoints are clearly reported rather than silently replaced.
    """
    stage1_dir = os.path.join(PROJECT_ROOT, "checkpoints", "stage1_indicnlg_qg")
    stage2_dir = os.path.join(PROJECT_ROOT, "checkpoints", "stage2_ncert_gold_lora")

    stage1_weights_exist = (
        os.path.exists(os.path.join(stage1_dir, "model.safetensors")) or
        os.path.exists(os.path.join(stage1_dir, "pytorch_model.bin"))
    )
    stage1_config_exists = os.path.exists(os.path.join(stage1_dir, "config.json"))
    stage1_available = stage1_weights_exist and stage1_config_exists

    stage2_adapter_exists = (
        os.path.exists(os.path.join(stage2_dir, "adapter_model.safetensors")) or
        os.path.exists(os.path.join(stage2_dir, "adapter_model.bin"))
    )
    stage2_config_exists = os.path.exists(os.path.join(stage2_dir, "adapter_config.json"))
    stage2_available = stage1_available and stage2_adapter_exists and stage2_config_exists

    # Baseline: Original IndicNLG question-generation checkpoint prior to Stage 1 fine-tuning
    # We check if a dedicated baseline checkpoint exists in checkpoints/
    baseline_candidates = [
        os.path.join(PROJECT_ROOT, "checkpoints", "baseline_indicnlg_qg"),
        os.path.join(PROJECT_ROOT, "checkpoints", "indicnlg_qg_baseline"),
    ]
    baseline_path = None
    for cand in baseline_candidates:
        if os.path.exists(cand) and (
            os.path.exists(os.path.join(cand, "model.safetensors")) or
            os.path.exists(os.path.join(cand, "pytorch_model.bin"))
        ):
            baseline_path = cand
            break

    baseline_available = baseline_path is not None

    registry = {
        MODEL_BASELINE: {
            "id": MODEL_BASELINE,
            "name": "Original IndicNLG QG Baseline",
            "name_hi": "मूल इंडिक-एनएलजी बेसलाइन",
            "tag": "Baseline",
            "checkpoint_path": baseline_path,
            "base_checkpoint": None,
            "description": (
                "Original IndicNLG translation-derived question-generation checkpoint "
                "prior to textbook curriculum fine-tuning."
            ),
            "status": "available" if baseline_available else "unavailable",
            "unavailable_reason": (
                None if baseline_available else
                "Original IndicNLG question-generation baseline checkpoint is not available in "
                "checkpoints/ (only Stage 1 fine-tuned IndicBART and Stage 2 LoRA adapter are present). "
                "Pretrained raw IndicBART is intentionally not substituted to maintain evaluation integrity."
            ),
            "is_adapter": False,
            "badge_color": "#6c757d"
        },
        MODEL_STAGE1: {
            "id": MODEL_STAGE1,
            "name": "Stage 1: IndicBART + IndicNLG QG",
            "name_hi": "स्टेज १: इंडिकबार्ट + इंडिक-एनएलजी",
            "tag": "Stage 1",
            "checkpoint_path": "checkpoints/stage1_indicnlg_qg",
            "base_checkpoint": None,
            "description": (
                "ai4bharat/IndicBART supervised fine-tuned on ~70,000 IndicQuestionGeneration "
                "Hindi pairs. Learns core Hindi syntax and question formation patterns."
            ),
            "status": "available" if stage1_available else "unavailable",
            "unavailable_reason": None if stage1_available else "Stage 1 checkpoint weights not found.",
            "is_adapter": False,
            "badge_color": "#0d6efd"
        },
        MODEL_STAGE2: {
            "id": MODEL_STAGE2,
            "name": "Stage 2: NCERT Curriculum Adapted (LoRA)",
            "name_hi": "स्टेज २: एनसीईआरटी अनुकूलित (LoRA)",
            "tag": "Stage 2 (Adapted)",
            "checkpoint_path": "checkpoints/stage2_ncert_gold_lora",
            "base_checkpoint": "checkpoints/stage1_indicnlg_qg",
            "description": (
                "Stage 1 base model + authentic NCERT Class 10 Hindi curriculum LoRA adapter "
                "(r=16, alpha=32). Adapts question generation to textbook phrasing and exam styles."
            ),
            "status": "available" if stage2_available else "unavailable",
            "unavailable_reason": None if stage2_available else (
                "Stage 2 LoRA adapter or required Stage 1 base model weights not found."
            ),
            "is_adapter": True,
            "badge_color": "#198754"
        }
    }

    return registry


class ModelManager:
    """
    Manages model lifecycle, shared base weights, and LoRA adapter switching
    to support multi-model comparisons without redundant memory consumption.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._generator = None
        self._loaded_checkpoint = None

    def get_generator(self):
        """
        Loads and caches the QuestionGenerator. Prefers loading Stage 2 (which loads
        Stage 1 as base model), enabling zero-reload switching between Stage 1 and Stage 2.
        """
        from src.evaluation.inference import QuestionGenerator
        registry = get_model_registry()

        if self._generator is None:
            # Prefer loading stage2 if available, which contains base model + LoRA
            if registry[MODEL_STAGE2]["status"] == "available":
                print(f"[ModelManager] Loading Stage 2 (Base + LoRA) on {self.device}...")
                self._generator = QuestionGenerator(
                    checkpoint_path=registry[MODEL_STAGE2]["checkpoint_path"],
                    device=self.device
                )
                self._loaded_checkpoint = MODEL_STAGE2
            elif registry[MODEL_STAGE1]["status"] == "available":
                print(f"[ModelManager] Loading Stage 1 Base on {self.device}...")
                self._generator = QuestionGenerator(
                    checkpoint_path=registry[MODEL_STAGE1]["checkpoint_path"],
                    device=self.device
                )
                self._loaded_checkpoint = MODEL_STAGE1
            else:
                raise RuntimeError("No compatible model checkpoints found to initialize ModelManager.")

        return self._generator

    def generate_single_for_model(
        self,
        model_id: str,
        context: str,
        answer: str = "",
        num_beams: int = 4,
        max_length: int = 64,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
        no_repeat_ngram_size: int = 3
    ) -> List[str]:
        """
        Executes single question generation for a specific model identifier
        with safe adapter isolation.
        """
        generator = self.get_generator()
        registry = get_model_registry()

        if model_id not in registry:
            raise ValueError(f"Unknown model identifier '{model_id}'. Available: {ALL_MODEL_IDS}")

        info = registry[model_id]
        if info["status"] != "available":
            raise RuntimeError(f"Model '{model_id}' is unavailable: {info.get('unavailable_reason')}")

        has_peft = hasattr(generator.model, "disable_adapter")

        if model_id == MODEL_STAGE1 and has_peft:
            # Stage 1: evaluate base model by cleanly disabling LoRA adapter
            with generator.model.disable_adapter():
                return generator.generate(
                    context=context,
                    answer=answer,
                    num_beams=num_beams,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    no_repeat_ngram_size=no_repeat_ngram_size
                )
        else:
            # Stage 2 (or non-peft stage 1)
            return generator.generate(
                context=context,
                answer=answer,
                num_beams=num_beams,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size
            )

    def generate_multi_for_model(
        self,
        model_id: str,
        context: str,
        answer: str = "",
        candidate_targets: Optional[List[Dict[str, Any]]] = None,
        num_questions: int = 3,
        num_beams: int = 4,
        max_length: int = 64,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95
    ) -> List[Dict[str, str]]:
        """
        Generates multiple questions for a specified model.
        When candidate_targets are provided, uses the exact same candidate answers
        to guarantee rigorous, strictly fair comparison across models.
        """
        generator = self.get_generator()
        registry = get_model_registry()

        if model_id not in registry:
            raise ValueError(f"Unknown model identifier '{model_id}'. Available: {ALL_MODEL_IDS}")

        info = registry[model_id]
        if info["status"] != "available":
            raise RuntimeError(f"Model '{model_id}' is unavailable: {info.get('unavailable_reason')}")

        has_peft = hasattr(generator.model, "disable_adapter")

        def _do_multi():
            # If candidate_targets was pre-extracted for parity, inject them
            if candidate_targets is not None and not answer:
                return self._generate_with_fixed_targets(
                    generator=generator,
                    context=context,
                    candidate_targets=candidate_targets,
                    num_questions=num_questions,
                    num_beams=num_beams,
                    max_length=max_length,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                return generator.generate_multi_questions(
                    context=context,
                    answer=answer,
                    num_questions=num_questions,
                    num_beams=num_beams,
                    max_length=max_length,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p
                )

        if model_id == MODEL_STAGE1 and has_peft:
            with generator.model.disable_adapter():
                return _do_multi()
        else:
            return _do_multi()

    def _generate_with_fixed_targets(
        self,
        generator,
        context: str,
        candidate_targets: List[Dict[str, Any]],
        num_questions: int,
        num_beams: int,
        max_length: int,
        no_repeat_ngram_size: int,
        do_sample: bool,
        temperature: float,
        top_p: float
    ) -> List[Dict[str, str]]:
        """
        Generates questions for pre-extracted candidate targets to ensure
        all models in a comparison receive IDENTICAL input answers.
        """
        ctx_clean = str(context).strip()
        targets = candidate_targets[:num_questions]
        if not targets:
            return generator.generate_multi_questions(
                context=ctx_clean,
                answer="",
                num_questions=num_questions,
                num_beams=num_beams,
                max_length=max_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )

        b_contexts = [ctx_clean for _ in targets]
        b_answers = [t.get("answer", "") for t in targets]

        generated_qs = generator.generate_batch(
            contexts=b_contexts,
            answers=b_answers,
            batch_size=len(b_contexts),
            num_beams=num_beams,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p
        )

        results = []
        for cand, q in zip(targets, generated_qs):
            results.append({
                "question": q,
                "answer": cand.get("answer", ""),
                "source_sentence": cand.get("sentence", ctx_clean)
            })

        return results


# Global singleton instance
_GLOBAL_MODEL_MANAGER: Optional[ModelManager] = None

def get_shared_model_manager(device: Optional[str] = None) -> ModelManager:
    global _GLOBAL_MODEL_MANAGER
    if _GLOBAL_MODEL_MANAGER is None:
        _GLOBAL_MODEL_MANAGER = ModelManager(device=device)
    return _GLOBAL_MODEL_MANAGER
