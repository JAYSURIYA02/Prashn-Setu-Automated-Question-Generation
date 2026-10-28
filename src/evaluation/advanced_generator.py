"""
Advanced Question Generation Engine for Prashn-Setu.
Enriches core Stage 2 (NCERT-adapted LoRA) generation with:
  1. Educational Question Categories (Factual, Conceptual, Reasoning, HOTS, MCQ).
  2. Difficulty calibration (Easy, Medium, Hard).
  3. Plausible in-passage distractor synthesis for authentic MCQs.
  4. Seamless integration with Evidence Highlighting and Answerability Checking.
"""

import re
import uuid
import random
from typing import List, Dict, Any, Optional, Tuple

from src.evaluation.model_registry import (
    get_shared_model_manager,
    MODEL_STAGE2,
    MODEL_STAGE1
)
from src.evaluation.inference import extract_candidate_answers
from src.evaluation.evidence_checker import compute_evidence_support, split_hindi_sentences


CATEGORY_LABELS = {
    "factual": "तथ्यात्मक प्रश्न (Factual)",
    "conceptual": "संकल्पनात्मक प्रश्न (Conceptual)",
    "reasoning": "तर्क / कारण आधारित (Reasoning)",
    "hots": "उच्च स्तरीय चिंतन / विश्लेषणात्मक (HOTS)",
    "mcq": "बहुविकल्पीय प्रश्न (MCQ)"
}

DIFFICULTY_LABELS = {
    "easy": "सरल (Easy)",
    "medium": "मध्यम (Medium)",
    "hard": "कठिन (Hard)"
}


def synthesize_mcq_options(
    correct_answer: str,
    passage: str,
    all_candidates: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, str]], str]:
    """
    Synthesizes four plausible multiple-choice options (A, B, C, D) by pulling
    other salient entities from the same passage as distractors.
    Guarantees that distractors are context-relevant and in Hindi.
    """
    ans_clean = correct_answer.strip()
    distractors = []

    # 1. Pull candidate entities from the passage that are not the correct answer
    for cand in all_candidates:
        c_text = cand.get("answer", "").strip()
        if c_text and c_text != ans_clean and len(c_text) > 1 and c_text not in distractors:
            distractors.append(c_text)
        if len(distractors) >= 6:
            break

    # 2. If fewer than 3 distractors, extract words/phrases from passage
    if len(distractors) < 3:
        words = re.findall(r"[\u0900-\u097F]{4,}", passage)
        for w in words:
            if w != ans_clean and w not in distractors:
                distractors.append(w)
            if len(distractors) >= 6:
                break

    # 3. Fallback generic distractors if passage is very short
    fallbacks = ["उपर्युक्त में से कोई नहीं", "कथन सत्य नहीं है", "अन्य कारण से", "विशेष परिस्थिति में"]
    for fb in fallbacks:
        if len(distractors) < 3:
            distractors.append(fb)

    selected_distractors = distractors[:3]
    choices = [ans_clean] + selected_distractors
    random.shuffle(choices)

    keys = ["A", "B", "C", "D"]
    options = []
    correct_key = "A"

    for k, text in zip(keys, choices):
        options.append({"key": k, "text": text})
        if text == ans_clean:
            correct_key = k

    return options, correct_key


def format_category_question(
    base_question: str,
    category: str,
    difficulty: str,
    answer: str,
    supporting_sentence: str
) -> str:
    """
    Applies pedagogically sound Hindi framing aligned with CBSE/NCERT textbook
    question patterns for conceptual, reasoning, and HOTS categories.
    """
    q = base_question.strip()
    if not q.endswith(("?", "।")):
        q += "?"

    if category == "factual" or category == "mcq":
        return q

    # Reasoning (Why/How)
    if category == "reasoning":
        if not any(w in q for w in ["क्यों", "कारण", "कैसे"]):
            # Transform or prepend reasoning prompt
            if "क्या" in q:
                q = q.replace("क्या", "क्यों")
            else:
                q = f"गद्यांश के अनुसार, {q[:-1]} का क्या कारण है?"

    # Conceptual (Explain / Define / Meaning)
    elif category == "conceptual":
        if difficulty == "hard":
            q = f"प्रसंग के आधार पर स्पष्ट कीजिए कि {q}"
        elif not any(w in q for w in ["अभिप्राय", "आशय", "तात्पर्य", "स्पष्ट"]):
            q = f"पाठ के आधार पर {q[:-1]} से क्या तात्पर्य है?"

    # HOTS (Higher Order Thinking / Analysis)
    elif category == "hots":
        if not any(w in q for w in ["विश्लेषण", "समीक्षा", "महत्व", "मूल्यांकन"]):
            q = f"गद्यांश के परिप्रेक्ष्य में समीक्षा कीजिए कि {q[:-1]} का क्या महत्व है?"

    # Ensure clean punctuation
    q = q.strip()
    if not q.endswith(("?", "।")):
        q += "?"

    return q


def generate_advanced_questions(
    context: str,
    answer: str = "",
    num_questions: int = 3,
    category: str = "factual",
    difficulty: str = "medium",
    model_id: str = MODEL_STAGE2,
    num_beams: int = 4,
    max_length: int = 64,
    no_repeat_ngram_size: int = 3,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.92,
    source_file: str = "",
    page_number: int = 1
) -> List[Dict[str, Any]]:
    """
    Primary API function for generating advanced, curriculum-aligned questions
    using the Stage 2 LoRA model by default.
    """
    if not context or not context.strip():
        raise ValueError("Context passage cannot be empty.")

    mgr = get_shared_model_manager()
    generator = mgr.get_generator()

    # Pre-extract candidates for entity selection and distractor generation
    candidates = extract_candidate_answers(context, max_candidates=max(num_questions * 3, 10))

    # Adjust decoding parameters based on difficulty
    eff_beams = num_beams
    eff_sample = do_sample
    eff_temp = temperature
    eff_max_len = max_length

    if difficulty == "easy":
        eff_beams = max(2, num_beams)
        eff_max_len = min(48, max_length)
    elif difficulty == "hard":
        eff_beams = max(4, num_beams)
        eff_max_len = max(64, max_length)
        if category in ["reasoning", "hots"]:
            eff_sample = True
            eff_temp = 0.75

    # Generate raw questions using ModelManager (which ensures Stage 2 LoRA is active)
    raw_results = mgr.generate_multi_for_model(
        model_id=model_id,
        context=context,
        answer=answer,
        candidate_targets=candidates if not answer else None,
        num_questions=num_questions,
        num_beams=eff_beams,
        max_length=eff_max_len,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=eff_sample,
        temperature=eff_temp,
        top_p=top_p
    )

    enriched_questions: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_results):
        base_q = item.get("question", "").strip()
        ans = item.get("answer", "").strip() or answer
        src_sentence = item.get("source_sentence", context)

        # 1. Evidence Checking & Answerability
        evidence = compute_evidence_support(base_q, ans, context)

        # 2. Category & Difficulty Formatting
        formatted_q = format_category_question(
            base_question=base_q,
            category=category,
            difficulty=difficulty,
            answer=ans,
            supporting_sentence=evidence["supporting_sentence"]
        )

        # 3. MCQ Synthesis if requested
        options = []
        correct_option = ""
        explanation = ""

        if category == "mcq":
            mcq_ans = ans if ans else (candidates[idx]["answer"] if idx < len(candidates) else "मुख्य तथ्य")
            options, correct_option = synthesize_mcq_options(mcq_ans, context, candidates)
            ans = mcq_ans
            explanation = f"गद्यांश के अनुसार: \"{evidence['supporting_sentence']}\""
        else:
            explanation = f"साक्ष्य वाक्य: \"{evidence['supporting_sentence']}\""

        q_obj = {
            "id": f"q_{uuid.uuid4().hex[:10]}",
            "question": formatted_q,
            "answer": ans,
            "category": category,
            "category_label": CATEGORY_LABELS.get(category, category),
            "difficulty": difficulty,
            "difficulty_label": DIFFICULTY_LABELS.get(difficulty, difficulty),
            "source_passage": context,
            "source_sentence": evidence["supporting_sentence"],
            "source_file": source_file,
            "page_number": page_number,
            "model_id": model_id,
            "model_name": "Stage 2: NCERT Curriculum Adapted (LoRA)" if model_id == MODEL_STAGE2 else "Stage 1: IndicBART Base",
            "options": options,
            "correct_option": correct_option,
            "explanation": explanation,
            "evidence": evidence,
            "evidence_status": evidence["status"],
            "evidence_label": evidence["status_label"],
            "evidence_start_char": evidence["start_char"],
            "evidence_end_char": evidence["end_char"],
            "answerability_score": evidence["answerability_score"],
            "overlap_score": evidence["overlap_score"],
            "matched_keywords": evidence["matched_keywords"]
        }

        enriched_questions.append(q_obj)

    return enriched_questions

