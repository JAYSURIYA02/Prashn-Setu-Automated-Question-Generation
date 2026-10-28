#!/usr/bin/env python3
"""
Prashn-Setu: Complete AI-Powered Indic Question Generation Platform.
Upgraded Web Application & REST Server.

Features:
  1. Primary Stage 2 LoRA Model Generation (verified at startup).
  2. Advanced Question Generation (Factual, Conceptual, Reasoning, HOTS, MCQ).
  3. Interactive Quiz & Practice Mode (instant evaluation, score, review).
  4. Evidence Highlighting & Answerability Checking (interactive passage view).
  5. PDF / Textbook Upload & Chapter-Based Generation.
  6. Persistent SQLite Question Bank & Generation History.
  7. Question Paper Builder & Multi-Format Exporter (PDF with Hindi fonts, CSV, JSON).
  8. NLP Research Analytics Dashboard & Stage 1 vs Stage 2 Comparative Evaluation.
  9. System Status & Verification.

Usage:
  python run.py demo --port 7860
"""

import os
import sys
import json
import time
import base64
import tempfile
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.inference import QuestionGenerator, extract_candidate_answers
from src.evaluation.model_registry import (
    ModelManager,
    get_model_registry,
    get_shared_model_manager,
    verify_stage2_checkpoint,
    MODEL_BASELINE,
    MODEL_STAGE1,
    MODEL_STAGE2,
    ALL_MODEL_IDS
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
    record_quiz_result
)
from src.evaluation.evidence_checker import (
    compute_evidence_support,
    split_hindi_sentences,
    detect_question_repetition
)
from src.evaluation.advanced_generator import (
    generate_advanced_questions,
    CATEGORY_LABELS,
    DIFFICULTY_LABELS
)
from src.evaluation.textbook_processor import process_textbook_and_generate
from src.evaluation.paper_builder import (
    validate_paper_config,
    generate_paper_pdf,
    generate_answer_key_pdf,
    export_paper_csv,
    export_paper_json
)
from src.evaluation.quiz_engine import evaluate_quiz_submission
from src.evaluation.analytics_loader import load_research_analytics_data


# Global instances
_MODEL_MANAGER: Optional[ModelManager] = None
_GENERATOR: Optional[QuestionGenerator] = None

# Preloaded authentic Hindi textbook examples
PRESET_EXAMPLES = [
    {
        "title": "क्षितिज भाग २ (Class 10: नेताजी का चश्मा — स्वयं प्रकाश)",
        "context": "स्वयं प्रकाश का जन्म सन् १९४७ में इंदौर (मध्य प्रदेश) में हुआ। मैकेनिकल इंजीनियरिंग की पढ़ाई करके एक औद्योगिक प्रतिष्ठान में नौकरी करने वाले स्वयं प्रकाश का बचपन और नौकरी का बड़ा हिस्सा राजस्थान में बीता। ये 'वसुधा' पत्रिका के संपादन से जुड़े रहे। उनके तेरह कहानी संग्रह प्रकाशित हो चुके हैं जिनमें 'सूरज कब निकलेगा', 'आएंगे अच्छे दिन भी' और 'संधान' उल्लेखनीय हैं।",
        "answer": ""
    },
    {
        "title": "स्पर्श भाग २ (Class 10: डायरी का एक पन्ना — स्वतंत्रता संग्राम)",
        "context": "२६ जनवरी १९३१ के दिन कोलकाता में स्वतंत्रता दिवस मनाने की पूरी तैयारी की गई थी। पुलिस कमिश्नर ने नोटिस निकाला था कि अमुक-अमुक धारा के तहत कोई भी सभा नहीं हो सकती। काउंसिल के अध्यक्ष सुभाष चंद्र बोस के नेतृत्व में विशाल जुलूस निकाला गया। पुलिस ने मैदान में लाठीचार्ज किया जिसमें सैकड़ों लोग घायल हुए। इसके बावजूद धर्मतल्ले के मोड़ पर स्त्रियों ने राष्ट्रीय ध्वज फहराया।",
        "answer": ""
    },
    {
        "title": "स्पर्श भाग २ (Class 10: बड़े भाई साहब — प्रेमचंद)",
        "context": "मेरे भाई साहब मुझसे पाँच साल बड़े थे, लेकिन केवल तीन दरजे आगे। उन्होंने भी उसी उम्र में पढ़ना शुरू किया था जब मैंने शुरू किया था; लेकिन तालीम जैसे महत्व के मामले में वह जल्दबाज़ी से काम लेना पसंद न करते थे। इस भवन की बुनियाद खूब मज़बूत डालना चाहते थे, जिस पर आलीशान महल बन सके। एक साल का काम दो साल में करते थे।",
        "answer": ""
    },
    {
        "title": "विज्ञान (Class 10 Science: प्रकाश संश्लेषण)",
        "context": "प्रकाश संश्लेषण एक ऐसी जैव-रासायनिक प्रक्रिया है जिसमें हरे पौधे सूर्य के प्रकाश और पर्णहरित (क्लोरोफिल) की उपस्थिति में कार्बन डाइऑक्साइड और जल का उपयोग करके ग्लूकोज का निर्माण करते हैं तथा ऑक्सीजन गैस मुक्त करते हैं।",
        "answer": "सूर्य के प्रकाश और पर्णहरित की उपस्थिति में"
    },
    {
        "title": "नागरिक शास्त्र (Class 9 Civics: भारतीय संविधान)",
        "context": "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है। उन्होंने संविधान प्रारूप समिति के अध्यक्ष के रूप में भारत के सर्वोच्च कानून के निर्माण में ऐतिहासिक योगदान दिया था। भारतीय संविधान २६ जनवरी १९५० को लागू हुआ था। इसमें कुल ३९५ अनुच्छेद और ८ अनुसूचियां थीं।",
        "answer": ""
    }
]


class DemoHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles REST APIs and serves the comprehensive Prashn-Setu Web Platform."""

    def do_GET(self):
        """Routes GET requests to API endpoints or serves the platform HTML UI."""
        path_clean = self.path.split("?")[0]

        if path_clean in ["/", "/index.html"]:
            html_file = os.path.join(PROJECT_ROOT, "src", "demo", "index.html")
            if os.path.exists(html_file):
                with open(html_file, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = "<h1>Prashn-Setu Interface Error: index.html not found.</h1>"

            content = content.replace("__PRESETS_JSON__", json.dumps(PRESET_EXAMPLES, ensure_ascii=False))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))

        elif path_clean == "/api/health":
            mgr = get_shared_model_manager()
            ver = mgr.verify_stage2()
            status = {
                "status": "healthy",
                "model_loaded": _GENERATOR is not None,
                "device": _GENERATOR.device if _GENERATOR else None,
                "primary_model": MODEL_STAGE2 if ver["stage2_available"] else MODEL_STAGE1,
                "stage2_adapter_active": ver.get("adapter_active", False),
                "available_models": [
                    k for k, v in get_model_registry().items() if v["status"] == "available"
                ]
            }
            self._send_json(200, status)

        elif path_clean == "/api/model_status":
            mgr = get_shared_model_manager()
            status = mgr.get_active_model_status()
            self._send_json(200, status)

        elif path_clean == "/api/models":
            self._send_json(200, get_model_registry())

        elif path_clean == "/api/benchmark_results":
            analytics = load_research_analytics_data()
            self._send_json(200, analytics)

        elif path_clean == "/api/question_bank":
            from urllib.parse import parse_qs, urlparse
            query_params = parse_qs(urlparse(self.path).query)
            q_str = query_params.get("query", [None])[0]
            category = query_params.get("category", [None])[0]
            difficulty = query_params.get("difficulty", [None])[0]
            source_file = query_params.get("source_file", [None])[0]
            limit = int(query_params.get("limit", [50])[0])
            offset = int(query_params.get("offset", [0])[0])

            data = get_questions(
                query=q_str,
                category=category,
                difficulty=difficulty,
                source_file=source_file,
                limit=limit,
                offset=offset
            )
            self._send_json(200, data)

        elif path_clean == "/api/question_bank/stats":
            stats = get_question_bank_stats()
            self._send_json(200, stats)

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint Not Found")

    def do_POST(self):
        """Handles question generation, textbook upload, question bank CRUD, and exports."""
        path_clean = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            data = json.loads(body) if body else {}
        except Exception as e:
            self._send_json_error(400, f"Invalid JSON payload: {str(e)}")
            return

        # 1. Advanced Question Generation (Primary: Stage 2 LoRA Model)
        if path_clean in ["/api/generate", "/api/generate_advanced"]:
            context = data.get("context", "").strip()
            if not context:
                self._send_json_error(400, "Field 'context' cannot be empty.")
                return

            answer = data.get("answer", "").strip()
            num_questions = int(data.get("num_questions", 3))
            category = data.get("category", "factual")
            difficulty = data.get("difficulty", "medium")
            model_id = data.get("model_id", MODEL_STAGE2)  # Stage 2 primary by default!

            # Decoding hyperparameters
            num_beams = int(data.get("num_beams", 4))
            max_length = int(data.get("max_length", 64))
            no_repeat_ngram_size = int(data.get("no_repeat_ngram_size", 3))
            do_sample = bool(data.get("do_sample", False))
            temperature = float(data.get("temperature", 0.7))
            top_p = float(data.get("top_p", 0.92))

            t0 = time.time()
            try:
                questions = generate_advanced_questions(
                    context=context,
                    answer=answer,
                    num_questions=num_questions,
                    category=category,
                    difficulty=difficulty,
                    model_id=model_id,
                    num_beams=num_beams,
                    max_length=max_length,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p
                )
                lat_ms = (time.time() - t0) * 1000.0

                first_q = questions[0]["question"] if questions else ""
                resp = {
                    "questions": questions,
                    "question": first_q,
                    "total_count": len(questions),
                    "context": context,
                    "answer": answer,
                    "category": category,
                    "difficulty": difficulty,
                    "model_id": model_id,
                    "latency_ms": round(lat_ms, 2)
                }
                self._send_json(200, resp)
            except Exception as e:
                self._send_json_error(500, f"Question generation failed: {str(e)}")

        # 2. Multi-Model Comparison (Stage 1 vs. Stage 2 vs. Baseline)
        elif path_clean == "/api/compare":
            context = data.get("context", "").strip()
            if not context:
                self._send_json_error(400, "Field 'context' cannot be empty.")
                return

            models = data.get("models", [MODEL_STAGE1, MODEL_STAGE2])
            if not isinstance(models, list) or not models:
                self._send_json_error(400, "Field 'models' must be a non-empty list.")
                return

            # Validate generation strategy if provided
            strategy = data.get("strategy")
            if strategy is not None and strategy not in ["beam_search", "sampling", "beam", "sample"]:
                self._send_json_error(400, f"Invalid generation strategy '{strategy}'. Supported: 'beam_search', 'sampling'.")
                return

            try:
                num_questions = int(data.get("num_questions", 3))
            except (ValueError, TypeError):
                self._send_json_error(400, "Parameter 'num_questions' must be an integer.")
                return

            if num_questions < 1 or num_questions > 10:
                self._send_json_error(400, f"Parameter 'num_questions' ({num_questions}) out of valid range (1-10).")
                return

            answer = data.get("answer", "").strip()
            num_beams = int(data.get("num_beams", 4))
            max_length = int(data.get("max_length", 64))
            no_repeat_ngram_size = int(data.get("no_repeat_ngram_size", 3))
            do_sample = bool(data.get("do_sample", False))
            temperature = float(data.get("temperature", 0.7))
            top_p = float(data.get("top_p", 0.92))

            candidate_targets = None
            if not answer:
                candidate_targets = extract_candidate_answers(context, max_candidates=max(num_questions * 2, 8))

            registry = get_model_registry()
            model_mgr = get_shared_model_manager()
            results = {}
            warnings = []

            for mid in models:
                if mid not in registry:
                    warnings.append(f"Model ID '{mid}' not found in registry.")
                    results[mid] = {"model_id": mid, "status": "error", "error": f"Unknown model '{mid}'"}
                    continue

                info = registry[mid]
                if info["status"] != "available":
                    warnings.append(f"Model '{info['name']}' is unavailable: {info.get('unavailable_reason')}")
                    results[mid] = {
                        "model_id": mid,
                        "name": info["name"],
                        "checkpoint": info["checkpoint_path"],
                        "status": "unavailable",
                        "error": info.get("unavailable_reason"),
                        "latency_ms": 0.0,
                        "questions": []
                    }
                    continue

                t0 = time.time()
                try:
                    q_items = model_mgr.generate_multi_for_model(
                        model_id=mid,
                        context=context,
                        answer=answer,
                        candidate_targets=candidate_targets,
                        num_questions=num_questions,
                        num_beams=num_beams,
                        max_length=max_length,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p
                    )
                    lat_ms = (time.time() - t0) * 1000.0
                    results[mid] = {
                        "model_id": mid,
                        "name": info["name"],
                        "checkpoint": info["checkpoint_path"],
                        "status": "success",
                        "latency_ms": round(lat_ms, 2),
                        "questions": q_items,
                        "total_count": len(q_items)
                    }
                except Exception as e:
                    lat_ms = (time.time() - t0) * 1000.0
                    results[mid] = {
                        "model_id": mid,
                        "name": info["name"],
                        "checkpoint": info["checkpoint_path"],
                        "status": "error",
                        "error": str(e),
                        "latency_ms": round(lat_ms, 2),
                        "questions": []
                    }

            self._send_json(200, {
                "context": context,
                "answer": answer,
                "selected_models": models,
                "results": results,
                "warnings": warnings
            })

        # 3. Evidence Highlighting & Answerability Standalone
        elif path_clean == "/api/evidence":
            question = data.get("question", "").strip()
            answer = data.get("answer", "").strip()
            passage = data.get("passage", "").strip()
            if not question or not passage:
                self._send_json_error(400, "Fields 'question' and 'passage' are required.")
                return

            res = compute_evidence_support(question, answer, passage)
            self._send_json(200, res)

        # 4. Textbook Upload & Chunk Processing
        elif path_clean == "/api/upload_textbook":
            filename = data.get("filename", "textbook.pdf")
            file_base64 = data.get("file_base64", "")
            raw_text = data.get("raw_text", "")
            questions_per_chunk = int(data.get("questions_per_chunk", 2))
            category = data.get("category", "factual")
            difficulty = data.get("difficulty", "medium")

            if not file_base64 and not raw_text:
                self._send_json_error(400, "Provide either 'file_base64' (PDF/TXT) or 'raw_text'.")
                return

            temp_path = None
            try:
                suffix = os.path.splitext(filename)[1].lower() or ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    temp_path = tmp.name
                    if file_base64:
                        tmp.write(base64.b64decode(file_base64))
                    else:
                        tmp.write(raw_text.encode("utf-8"))

                result = process_textbook_and_generate(
                    file_path=temp_path,
                    questions_per_chunk=questions_per_chunk,
                    category=category,
                    difficulty=difficulty,
                    model_id=MODEL_STAGE2
                )
                result["original_filename"] = filename
                self._send_json(200, result)
            except Exception as e:
                self._send_json_error(500, f"Textbook processing failed: {str(e)}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        # 5. Question Bank: Save Question(s)
        elif path_clean == "/api/question_bank":
            if "questions" in data and isinstance(data["questions"], list):
                saved = save_questions_batch(data["questions"])
                self._send_json(200, {"saved_count": len(saved), "questions": saved})
            elif "question" in data:
                saved = save_question(data)
                self._send_json(200, {"status": "saved", "question": saved})
            else:
                self._send_json_error(400, "Payload must include 'question' object or 'questions' list.")

        # 6. Interactive Quiz Evaluation
        elif path_clean == "/api/quiz/evaluate":
            try:
                quiz_eval = evaluate_quiz_submission(data)
                self._send_json(200, quiz_eval)
            except Exception as e:
                self._send_json_error(500, f"Quiz evaluation error: {str(e)}")

        # 7. Paper Builder: Validate Configuration
        elif path_clean == "/api/paper/validate":
            valid, warnings = validate_paper_config(data)
            self._send_json(200, {"valid": valid, "warnings": warnings})

        # 8. Paper Builder: Export PDF Question Paper (Strictly NO answers for students)
        elif path_clean == "/api/paper/export_pdf":
            try:
                pdf_bytes = generate_paper_pdf(data, include_answers=False)
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition", 'attachment; filename="Prashn_Setu_Question_Paper.pdf"')
                self.end_headers()
                self.wfile.write(pdf_bytes)
            except Exception as e:
                self._send_json_error(500, f"PDF generation failed: {str(e)}")

        # 9. Paper Builder: Export PDF Answer Key
        elif path_clean == "/api/paper/export_key_pdf":
            try:
                pdf_bytes = generate_answer_key_pdf(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition", 'attachment; filename="Prashn_Setu_Answer_Key.pdf"')
                self.end_headers()
                self.wfile.write(pdf_bytes)
            except Exception as e:
                self._send_json_error(500, f"Answer Key PDF generation failed: {str(e)}")

        # 10. Paper Builder: Export CSV
        elif path_clean == "/api/paper/export_csv":
            try:
                csv_text = export_paper_csv(data)
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8-sig")
                self.send_header("Content-Disposition", 'attachment; filename="Prashn_Setu_Paper.csv"')
                self.end_headers()
                self.wfile.write(csv_text.encode("utf-8-sig"))
            except Exception as e:
                self._send_json_error(500, f"CSV export failed: {str(e)}")

        # 11. Paper Builder: Export JSON
        elif path_clean == "/api/paper/export_json":
            json_text = export_paper_json(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="Prashn_Setu_Paper.json"')
            self.end_headers()
            self.wfile.write(json_text.encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint Not Found")

    def do_PUT(self):
        """Handles Question Bank update requests."""
        path_clean = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            data = json.loads(body)
        except Exception as e:
            self._send_json_error(400, f"Invalid JSON payload: {str(e)}")
            return

        if path_clean == "/api/question_bank":
            q_id = data.get("id")
            if not q_id:
                self._send_json_error(400, "Missing 'id' for question update.")
                return
            success = update_question(q_id, data)
            self._send_json(200, {"success": success, "id": q_id})
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint Not Found")

    def do_DELETE(self):
        """Handles Question Bank deletion requests."""
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(self.path)
        path_clean = parsed.path
        query_params = parse_qs(parsed.query)

        if path_clean == "/api/question_bank":
            q_id = query_params.get("id", [None])[0]
            if not q_id:
                self._send_json_error(400, "Missing query parameter '?id=...' for deletion.")
                return
            success = delete_question(q_id)
            self._send_json(200, {"success": success, "id": q_id})
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint Not Found")

    def _send_json(self, code: int, payload: Any):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _send_json_error(self, code: int, message: str):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        """Quiet server logging for cleaner terminal experience."""
        pass


def run_demo_server(
    checkpoint_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 7860
):
    """Initializes ModelManager, verifies Stage 2 LoRA model, and starts HTTP server."""
    global _GENERATOR, _MODEL_MANAGER

    print("\n" + "=" * 68)
    print(" प्रश्न-सेतु (Prashn-Setu) — AI-Powered Indic Question Platform")
    print("=" * 68)

    # 1. Initialize SQLite Database
    init_question_bank_db()
    print("📁 [Database] SQLite Question Bank initialized at data/question_bank.db")

    # 2. Stage 2 Verification
    print("🔍 [Pre-flight] Verifying Stage 2 NCERT-Adapted LoRA checkpoint...")
    ver = verify_stage2_checkpoint()
    if ver["verified"]:
        print("✅ [Pre-flight Verified] Stage 1 Base and Stage 2 LoRA Adapter are intact:")
        print(f"   • Base Model Path:    checkpoints/stage1_indicnlg_qg")
        print(f"   • LoRA Adapter Path:  checkpoints/stage2_ncert_gold_lora")
        print(f"   • LoRA Rank (r):      {ver['adapter_config']['r']}, Alpha: {ver['adapter_config']['lora_alpha']}")
        print(f"   • Target Modules:     {ver['adapter_config']['target_modules']}")
    else:
        print("⚠️ [Pre-flight Warning] Stage 2 verification encountered issues:")
        for err in ver["errors"]:
            print(f"   • {err}")

    # 3. Model Loading
    _MODEL_MANAGER = get_shared_model_manager()
    _GENERATOR = _MODEL_MANAGER.get_generator()
    status = _MODEL_MANAGER.get_active_model_status()
    print(f"🚀 [Model Active] {status['active_model_name']} on device: {status['device']}")

    # 4. HTTP Server
    server_address = (host, port)
    HTTPServer.allow_reuse_address = True
    httpd = HTTPServer(server_address, DemoHTTPRequestHandler)

    url_local = f"http://localhost:{port}"
    print(f"\n🌐 Demonstration Platform is LIVE!")
    print(f"👉 Local Access URL:   {url_local}")
    print(f"👉 Network Access:     http://{host}:{port}")
    print(f"👉 API Health Check:   {url_local}/api/health")
    print(f"👉 Model Status API:   {url_local}/api/model_status")
    print(f"👉 Question Bank API:  {url_local}/api/question_bank")
    print("=" * 68)
    print("Press Ctrl+C to terminate the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down platform server gracefully...")
        httpd.server_close()


def main():
    default_ckpt = (
        "checkpoints/stage2_ncert_gold_lora"
        if os.path.exists("checkpoints/stage2_ncert_gold_lora")
        else "checkpoints/stage1_indicnlg_qg"
    )
    parser = argparse.ArgumentParser(description="Prashn-Setu Web Platform Server")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Checkpoint to serve")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    args = parser.parse_args()

    run_demo_server(checkpoint_path=args.checkpoint, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
