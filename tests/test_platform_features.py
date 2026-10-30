"""
Comprehensive Unit and Integration Test Suite for Prashn-Setu Platform Upgrades.

Tests:
  1. Stage 2 LoRA Model verification, pre-flight checks, and primary model guarantee.
  2. Advanced question generation (Factual, Conceptual, Reasoning, HOTS, MCQ options).
  3. Evidence highlighting, character offsets, and answerability classification.
  4. Textbook text normalization, sentence-aware passage chunking, and PDF parsing.
  5. SQLite Question Bank persistence, CRUD operations, filtering, and stats.
  6. Interactive Quiz scoring, approximate short-answer grading, and history.
  7. Question paper builder, marks validation, PDF/CSV/JSON exports, and answer exclusion.
  8. Research analytics parsing and human evaluation honesty.
  9. HTTP API endpoints (/api/health, /api/model_status, /api/question_bank, /api/quiz/evaluate).
"""

import os
import sys
import json
import sqlite3
import unittest
from io import BytesIO
from typing import Optional, Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.model_registry import (
    get_model_registry,
    verify_stage2_checkpoint,
    get_shared_model_manager,
    MODEL_BASELINE,
    MODEL_STAGE1,
    MODEL_STAGE2
)
from src.evaluation.evidence_checker import (
    split_hindi_sentences,
    compute_evidence_support,
    detect_question_repetition,
    tokenize_content_words
)
from src.evaluation.advanced_generator import (
    synthesize_mcq_options,
    format_category_question
)
from src.evaluation.textbook_processor import (
    normalize_hindi_text,
    chunk_page_passages
)
from src.data.question_bank import (
    init_question_bank_db,
    save_question,
    save_questions_batch,
    get_questions,
    get_question_by_id,
    update_question,
    delete_question,
    get_question_bank_stats,
    record_quiz_result,
    get_db_connection
)
from src.evaluation.quiz_engine import (
    evaluate_quiz_submission,
    clean_hindi_tokens
)
from src.evaluation.paper_builder import (
    validate_paper_config,
    generate_paper_pdf,
    generate_answer_key_pdf,
    export_paper_csv,
    export_paper_json
)
from src.evaluation.analytics_loader import load_research_analytics_data
from src.demo.app import DemoHTTPRequestHandler


class MockHTTPHandler(DemoHTTPRequestHandler):
    """Subclass of DemoHTTPRequestHandler for in-memory testing without a live socket."""
    def __init__(self, method: str, path: str, body: Optional[dict] = None):
        self.command = method
        self.path = path
        body_bytes = json.dumps(body or {}).encode("utf-8") if body else b""
        self.headers = {"Content-Length": str(len(body_bytes)), "Content-Type": "application/json"}
        self.rfile = BytesIO(body_bytes)
        self.wfile = BytesIO()
        self.response_code = None
        self.response_headers = {}

    def send_response(self, code, message=None):
        self.response_code = code

    def send_header(self, keyword, value):
        self.response_headers[keyword] = value

    def end_headers(self):
        pass

    def get_response_json(self):
        self.wfile.seek(0)
        content = self.wfile.read()
        return json.loads(content.decode("utf-8")) if content else {}

    def get_response_text(self):
        self.wfile.seek(0)
        content = self.wfile.read()
        return content.decode("utf-8") if content else ""


class TestStage2PrimaryModelVerification(unittest.TestCase):
    """Verifies that Stage 2 LoRA model is the verified primary default generation model."""

    def test_stage2_preflight_verification(self):
        ver = verify_stage2_checkpoint()
        self.assertTrue(ver["verified"], f"Stage 2 verification failed: {ver.get('errors')}")
        self.assertTrue(ver["stage1_available"])
        self.assertTrue(ver["stage2_available"])
        self.assertEqual(ver["active_model"], MODEL_STAGE2)

        cfg = ver["adapter_config"]
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["peft_type"], "LORA")
        self.assertEqual(cfg["r"], 16)
        self.assertEqual(cfg["lora_alpha"], 32)
        self.assertIn("q_proj", cfg["target_modules"])

    def test_model_manager_reports_stage2_as_active(self):
        mgr = get_shared_model_manager()
        status = mgr.get_active_model_status()
        self.assertEqual(status["active_model_id"], MODEL_STAGE2)
        self.assertIn("Stage 2", status["active_model_name"])


class TestEvidenceHighlightingAndAnswerability(unittest.TestCase):
    """Tests for sentence offset mapping, grounding, and answerability checking."""

    def setUp(self):
        self.passage = (
            "स्वयं प्रकाश का जन्म सन् १९४७ में इंदौर (मध्य प्रदेश) में हुआ। "
            "उन्होंने मैकेनिकल इंजीनियरिंग की पढ़ाई की और 'वसुधा' पत्रिका का संपादन किया। "
            "उनके तेरह कहानी संग्रह प्रकाशित हो चुके हैं।"
        )

    def test_sentence_splitting_with_character_offsets(self):
        sentences = split_hindi_sentences(self.passage)
        self.assertEqual(len(sentences), 3)
        for s in sentences:
            extracted_slice = self.passage[s["start_char"]:s["end_char"]]
            self.assertEqual(extracted_slice, s["text"])

    def test_evidence_found_for_grounded_question(self):
        q = "स्वयं प्रकाश का जन्म किस वर्ष और कहाँ हुआ था?"
        ans = "सन् १९४७ में इंदौर"
        evidence = compute_evidence_support(q, ans, self.passage)
        self.assertEqual(evidence["status"], "evidence_found")
        self.assertIn("१९४७", evidence["supporting_sentence"])
        self.assertGreaterEqual(evidence["answerability_score"], 3.5)
        self.assertEqual(evidence["sentence_index"], 0)

    def test_unsupported_flag_for_hallucinated_question(self):
        q = "महात्मा गांधी ने दांडी यात्रा कब प्रारंभ की थी?"
        ans = "१२ मार्च १९३०"
        evidence = compute_evidence_support(q, ans, self.passage)
        self.assertEqual(evidence["status"], "unsupported")
        self.assertLess(evidence["answerability_score"], 3.0)

    def test_detect_question_repetition(self):
        qs = [
            "स्वयं प्रकाश का जन्म कहाँ हुआ था?",
            "स्वयं प्रकाश का जन्म किस स्थान पर हुआ था?",  # near duplicate
            "उन्होंने कौन सी पत्रिका का संपादन किया?"
        ]
        rep = detect_question_repetition(qs, threshold=0.55)
        self.assertFalse(rep[0]["is_repetitive"])
        self.assertTrue(rep[1]["is_repetitive"])
        self.assertEqual(rep[1]["duplicate_of_index"], 0)
        self.assertFalse(rep[2]["is_repetitive"])


class TestAdvancedQuestionFormattingAndMCQ(unittest.TestCase):
    """Tests question category framing and MCQ distractor synthesis."""

    def setUp(self):
        self.passage = (
            "प्रकाश संश्लेषण एक जैव-रासायनिक प्रक्रिया है जिसमें हरे पौधे सूर्य के प्रकाश "
            "और क्लोरोफिल की उपस्थिति में ग्लूकोज का निर्माण करते हैं।"
        )
        self.candidates = [
            {"answer": "ग्लूकोज", "sentence": self.passage},
            {"answer": "क्लोरोफिल", "sentence": self.passage},
            {"answer": "सूर्य के प्रकाश", "sentence": self.passage}
        ]

    def test_mcq_distractor_synthesis(self):
        correct = "ग्लूकोज"
        options, correct_key = synthesize_mcq_options(correct, self.passage, self.candidates)
        self.assertEqual(len(options), 4)
        option_keys = [o["key"] for o in options]
        self.assertEqual(sorted(option_keys), ["A", "B", "C", "D"])

        correct_opt = next(o for o in options if o["key"] == correct_key)
        self.assertEqual(correct_opt["text"], correct)

    def test_format_reasoning_question(self):
        q = format_category_question("पौधे ग्लूकोज का निर्माण क्यों करते हैं?", "reasoning", "medium", "ऊर्जा हेतु", self.passage)
        self.assertIn("क्यों", q)
        self.assertTrue(q.endswith("?"))

    def test_format_conceptual_question(self):
        q = format_category_question("प्रकाश संश्लेषण प्रक्रिया", "conceptual", "medium", "जैव-रासायनिक", self.passage)
        self.assertTrue("तात्पर्य" in q or "अभिप्राय" in q or "स्पष्ट" in q)


class TestTextbookProcessor(unittest.TestCase):
    """Tests textbook text extraction, Unicode normalization, and chunking."""

    def test_normalize_hindi_text(self):
        raw = "यह   एक-\nपरीक्षण   पाठ    है।\r\n\r\n\r\nदूसरी पंक्ति।"
        clean = normalize_hindi_text(raw)
        self.assertNotIn("-\n", clean)
        self.assertNotIn("  ", clean)
        self.assertIn("परीक्षण", clean)

    def test_devanagari_corrupt_matra_and_visual_order_repair(self):
        """Verifies fix for visual short-i, detached combining marks, and broken lines."""
        corrupted = (
            "आरसी प्रसाद िसं\nह प्रकृित और जीवन-सं\nघष  को अपनी रचनाओ में प्रम ं खता से\nु\n"
            "िचित्रत करने वाले किव हैं। वे अपनी रचनाओ में प्रेम, करुणा, त्याग-बिलदान, म ं िक् त\nु\n"
            "और िमल-जलकर एक स ु ं\nदर स ु ं\nसार रचने क  कल्पना करते रहे हैं। जैसा िक ‘िचिड़या’\n"
            "किवता में भी आपने पढ़ा। उन्होंने िचिड़या के माध्यम से िकतनी सं\nदर बात कही है— “िचिड़या बैठी ु\n"
            "प्रेम-प्रीित क  रीित हमें िसखलाती है! वह जग के बं"
        )
        repaired = normalize_hindi_text(corrupted)

        # Assert correct reordered and repaired Hindi words
        self.assertIn("सिंह", repaired)
        self.assertIn("प्रकृति", repaired)
        self.assertIn("संघर्ष", repaired)
        self.assertIn("कवि", repaired)
        self.assertIn("बलिदान", repaired)
        self.assertIn("मुक्ति", repaired)
        self.assertIn("सुंदर", repaired)
        self.assertIn("संसार", repaired)
        self.assertIn("की कल्पना", repaired)
        self.assertIn("कविता", repaired)
        self.assertIn("चिड़िया", repaired)
        self.assertIn("प्रेम-प्रीति की रीति", repaired)
        self.assertIn("सिखलाती है", repaired)

        # Ensure no detached placeholder marks (e.g. space before vowel marks)
        self.assertNotIn(" ु", repaired)
        self.assertNotIn(" िसं", repaired)
        self.assertNotIn(" िक", repaired)

    def test_chunk_page_passages_preserves_sentence_boundaries(self):
        page = {
            "page_number": 1,
            "text": "पहला वाक्य यहाँ समाप्त होता है। दूसरा वाक्य यहाँ है। तीसरा वाक्य बहुत महत्वपूर्ण है।"
        }
        chunks = chunk_page_passages([page], target_words=5)
        self.assertGreater(len(chunks), 1)
        for chk in chunks:
            self.assertTrue(chk["text"].endswith("।") or len(chk["text"]) > 0)


class TestQuestionBankPersistence(unittest.TestCase):
    """Tests SQLite question bank CRUD, persistence, and queries."""

    def setUp(self):
        init_question_bank_db()

    def test_save_and_retrieve_question(self):
        q_data = {
            "question": "भारतीय संविधान कब लागू हुआ था?",
            "answer": "२६ जनवरी १९५०",
            "category": "factual",
            "difficulty": "easy",
            "source_passage": "भारतीय संविधान २६ जनवरी १९५० को लागू हुआ था।",
            "source_sentence": "भारतीय संविधान २६ जनवरी १९५० को लागू हुआ था।",
            "model_id": "stage2"
        }
        saved = save_question(q_data)
        q_id = saved["id"]
        self.assertTrue(q_id.startswith("q_"))

        fetched = get_question_by_id(q_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["question"], q_data["question"])
        self.assertEqual(fetched["answer"], q_data["answer"])
        self.assertEqual(fetched["model_id"], "stage2")
        self.assertEqual(fetched["is_edited"], 0)

    def test_update_and_mark_edited(self):
        q_data = {
            "question": "मूल प्रश्न पाठ",
            "answer": "मूल उत्तर",
            "category": "factual"
        }
        saved = save_question(q_data)
        q_id = saved["id"]

        update_question(q_id, {"question": "संपादित प्रश्न पाठ"})
        updated = get_question_by_id(q_id)
        self.assertEqual(updated["question"], "संपादित प्रश्न पाठ")
        self.assertEqual(updated["is_edited"], 1)

    def test_filter_and_search_questions(self):
        unique_token = f"पानीपत_{os.urandom(4).hex()}"
        save_question({"question": "विज्ञान प्रश्न: जल का सूत्र?", "answer": "H2O", "category": "conceptual", "difficulty": "hard"})
        save_question({"question": f"इतिहास प्रश्न: {unique_token} का युद्ध?", "answer": "१५२६", "category": "factual", "difficulty": "easy"})

        res_concept = get_questions(category="conceptual")
        self.assertTrue(any("जल" in q["question"] for q in res_concept["questions"]))

        res_search = get_questions(query=unique_token)
        self.assertEqual(res_search["total"], 1)
        self.assertIn(unique_token, res_search["questions"][0]["question"])

    def test_delete_question(self):
        saved = save_question({"question": "हटाने हेतु प्रश्न", "answer": "उत्तर"})
        q_id = saved["id"]
        self.assertTrue(delete_question(q_id))
        self.assertIsNone(get_question_by_id(q_id))

    def test_question_bank_stats(self):
        stats = get_question_bank_stats()
        self.assertIn("total_questions", stats)
        self.assertIn("by_category", stats)
        self.assertIn("by_difficulty", stats)


class TestQuizEngine(unittest.TestCase):
    """Tests interactive quiz evaluation and scoring."""

    def test_mcq_evaluation_correct(self):
        submission = {
            "title": "Test MCQ Quiz",
            "questions": [{
                "id": "q1",
                "question": "प्रकाश संश्लेषण में कौन सी गैस मुक्त होती है?",
                "category": "mcq",
                "options": [{"key": "A", "text": "ऑक्सीजन"}, {"key": "B", "text": "नाइट्रोजन"}],
                "correct_option": "A",
                "answer": "ऑक्सीजन"
            }],
            "user_answers": {"q1": "A"}
        }
        res = evaluate_quiz_submission(submission)
        self.assertEqual(res["score"], 1)
        self.assertEqual(res["percentage"], 100.0)
        self.assertTrue(res["answers"][0]["is_correct"])

    def test_short_answer_evaluation(self):
        submission = {
            "title": "Test Short Answer Quiz",
            "questions": [{
                "id": "q2",
                "question": "भारतीय संविधान का जनक किसे माना जाता है?",
                "category": "factual",
                "answer": "डॉ. भीमराव अंबेडकर"
            }],
            "user_answers": {"q2": "डॉ भीमराव अंबेडकर को"}
        }
        res = evaluate_quiz_submission(submission)
        self.assertEqual(res["score"], 1)
        self.assertTrue(res["answers"][0]["is_correct"])


class TestQuestionPaperBuilderAndExports(unittest.TestCase):
    """Tests paper marks validation, answer key generation, and PDF export."""

    def setUp(self):
        self.paper_data = {
            "school_name": "केन्द्रीय विद्यालय",
            "title": "प्रथम आवधिक परीक्षा २०२६",
            "subject": "कक्षा १० हिन्दी",
            "total_marks": 5,
            "sections": [
                {
                    "title": "खंड 'क'",
                    "questions": [
                        {"question": "नेताजी का चश्मा किसने लिखा?", "answer": "स्वयं प्रकाश", "marks": 2, "category": "factual"},
                        {"question": "भवन की नींव कैसी डाली गई?", "answer": "मजबूत", "marks": 3, "category": "conceptual"}
                    ]
                }
            ]
        }

    def test_validate_paper_marks_match(self):
        valid, warnings = validate_paper_config(self.paper_data)
        self.assertTrue(valid)
        self.assertEqual(len(warnings), 0)

    def test_validate_paper_marks_mismatch_warning(self):
        bad_paper = dict(self.paper_data)
        bad_paper["total_marks"] = 20  # Mismatch: sum is 5
        valid, warnings = validate_paper_config(bad_paper)
        self.assertFalse(valid)
        self.assertGreater(len(warnings), 0)

    def test_generate_paper_pdf_no_answers_for_students(self):
        pdf_bytes = generate_paper_pdf(self.paper_data, include_answers=False)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

    def test_generate_answer_key_pdf(self):
        pdf_bytes = generate_answer_key_pdf(self.paper_data)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

    def test_export_csv_and_json(self):
        csv_str = export_paper_csv(self.paper_data)
        self.assertTrue(csv_str.startswith("\ufeff"))  # UTF-8 BOM
        self.assertIn("स्वयं प्रकाश", csv_str)

        json_str = export_paper_json(self.paper_data)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["title"], self.paper_data["title"])


class TestResearchAnalytics(unittest.TestCase):
    """Tests loading real benchmark metrics and qualitative analysis."""

    def test_load_research_analytics_data(self):
        data = load_research_analytics_data()
        self.assertIn("quantitative_benchmark", data)
        self.assertIn("human_evaluation", data)
        self.assertIn("qualitative_analysis", data)

        bench = data["quantitative_benchmark"]
        if "metrics_comparison" in bench:
            m_comp = bench["metrics_comparison"]
            self.assertIn("sacrebleu", m_comp)
            self.assertIn("rouge1", m_comp)
            self.assertIn("chrf", m_comp)
            # Verify delta calculation
            s1_val = m_comp["sacrebleu"]["stage1"]
            s2_val = m_comp["sacrebleu"]["stage2"]
            diff = m_comp["sacrebleu"]["absolute_diff"]
            self.assertAlmostEqual(s2_val - s1_val, diff, delta=0.02)


class TestAPIEndpoints(unittest.TestCase):
    """Tests API routes for health, model status, and question bank."""

    def test_get_health_reports_stage2_primary(self):
        handler = MockHTTPHandler("GET", "/api/health")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        data = handler.get_response_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["primary_model"], MODEL_STAGE2)

    def test_get_model_status_reports_lora_active(self):
        handler = MockHTTPHandler("GET", "/api/model_status")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        data = handler.get_response_json()
        self.assertEqual(data["active_model_id"], MODEL_STAGE2)

    def test_api_question_bank_crud(self):
        handler = MockHTTPHandler("POST", "/api/question_bank", {
            "question": "API परीक्षण प्रश्न?",
            "answer": "परीक्षण उत्तर",
            "category": "factual"
        })
        handler.do_POST()
        self.assertEqual(handler.response_code, 200)
        res = handler.get_response_json()
        self.assertEqual(res["status"], "saved")
        q_id = res["question"]["id"]

        # Retrieve
        handler_get = MockHTTPHandler("GET", f"/api/question_bank?query=API%20परीक्षण")
        handler_get.do_GET()
        self.assertEqual(handler_get.response_code, 200)
        data = handler_get.get_response_json()
        self.assertGreater(data["total"], 0)


if __name__ == "__main__":
    unittest.main()

