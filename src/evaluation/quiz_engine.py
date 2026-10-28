"""
Interactive Quiz and Practice Engine for Prashn-Setu.
Evaluates student responses against Stage 2 generated MCQs and short-answer questions,
computes transparent scores, and provides source-grounded feedback.
"""

import re
from typing import List, Dict, Any, Tuple
from src.data.question_bank import record_quiz_result
from src.evaluation.evidence_checker import HINDI_STOPWORDS


def clean_hindi_tokens(text: str) -> set:
    """Extracts non-stopword tokens for approximate short-answer grading."""
    tokens = re.findall(r"[\u0900-\u097F\w]+", text.lower())
    return {t for t in tokens if t not in HINDI_STOPWORDS and len(t) > 1}


def evaluate_quiz_submission(submission: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates submitted user answers for a set of questions.
    Supports both MCQs (exact option key match) and Short Answer (semantic/lexical match).
    """
    title = submission.get("title", "हिंदी अभ्यास क्विज़ (Hindi Practice Quiz)")
    questions = submission.get("questions", [])
    user_answers = submission.get("user_answers", {})  # map: q_id -> submitted_value

    total_count = len(questions)
    correct_count = 0
    feedback_items = []

    for idx, q in enumerate(questions):
        q_id = q.get("id", f"q_{idx}")
        category = q.get("category", "factual")
        user_val = str(user_answers.get(q_id, "")).strip()

        is_mcq = (category == "mcq") or bool(q.get("options"))
        correct_answer = str(q.get("answer", "")).strip()
        correct_option = str(q.get("correct_option", "")).strip().upper()
        explanation = q.get("explanation", f"साक्ष्य वाक्य: {q.get('source_sentence', '')}")

        is_correct = False
        grading_type = "exact" if is_mcq else "approximate"

        if is_mcq:
            # Check option key (e.g. "A") or exact option text
            u_upper = user_val.upper()
            if u_upper == correct_option:
                is_correct = True
            elif user_val and (user_val.lower() == correct_answer.lower()):
                is_correct = True
        else:
            # Short answer evaluation
            # 1. Exact string match or substring match
            if user_val and (user_val.lower() == correct_answer.lower() or correct_answer.lower() in user_val.lower()):
                is_correct = True
            elif user_val:
                # 2. Token overlap threshold (>= 60% of key answer tokens)
                ans_tokens = clean_hindi_tokens(correct_answer)
                user_tokens = clean_hindi_tokens(user_val)
                if ans_tokens:
                    matched = len(ans_tokens.intersection(user_tokens))
                    ratio = matched / len(ans_tokens)
                    if ratio >= 0.60:
                        is_correct = True

        if is_correct:
            correct_count += 1

        feedback_items.append({
            "question_id": q_id,
            "question_index": idx + 1,
            "question_text": q.get("question", ""),
            "user_answer": user_val,
            "correct_answer": f"विकल्प ({correct_option}): {correct_answer}" if is_mcq else correct_answer,
            "is_correct": is_correct,
            "grading_method": "एमसीक्यू विकल्प मिलान" if is_mcq else "निकटतम अर्थ मिलान (Approximate Lexical Match)",
            "explanation": explanation,
            "source_sentence": q.get("source_sentence", "")
        })

    pct = round((correct_count / max(total_count, 1)) * 100.0, 1)

    result_data = {
        "title": title,
        "total_questions": total_count,
        "score": correct_count,
        "incorrect_count": total_count - correct_count,
        "percentage": pct,
        "answers": feedback_items
    }

    # Persist in SQLite
    quiz_record_id = record_quiz_result(result_data)
    result_data["quiz_id"] = quiz_record_id

    return result_data

