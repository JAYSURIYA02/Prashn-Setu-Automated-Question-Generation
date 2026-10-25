"""
Evaluation & Inference package for Prashn-Setu.
"""
from src.evaluation.inference import QuestionGenerator
from src.evaluation.evaluate import evaluate_checkpoint
from src.evaluation.model_registry import (
    ModelManager,
    get_model_registry,
    get_shared_model_manager,
    MODEL_BASELINE,
    MODEL_STAGE1,
    MODEL_STAGE2,
    ALL_MODEL_IDS
)

__all__ = [
    "QuestionGenerator",
    "evaluate_checkpoint",
    "ModelManager",
    "get_model_registry",
    "get_shared_model_manager",
    "MODEL_BASELINE",
    "MODEL_STAGE1",
    "MODEL_STAGE2",
    "ALL_MODEL_IDS"
]
