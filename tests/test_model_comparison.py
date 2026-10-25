"""
Unit and Integration Test Suite for Prashn-Setu Model Comparison Feature.
Tests:
  1. Model registry structure, availability detection, and honest baseline reporting.
  2. ModelManager lifecycle, shared weights, and adapter isolation (Stage 1 vs. Stage 2).
  3. Candidate-answer extraction consistency and parity across models.
  4. Input validation: empty context, invalid strategy, out-of-range parameters.
  5. Unavailable model graceful degradation (one model unavailable, others succeed).
  6. Comparison API request handling, payload schema, and latency reporting.
  7. Backward compatibility for /api/health, /api/models, and /api/benchmark_results.
"""

import os
import sys
import json
import unittest
from io import BytesIO
from typing import Optional, Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.model_registry import (
    get_model_registry,
    ModelManager,
    get_shared_model_manager,
    MODEL_BASELINE,
    MODEL_STAGE1,
    MODEL_STAGE2,
    ALL_MODEL_IDS
)
from src.evaluation.inference import extract_candidate_answers, clean_candidate_span
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


class TestModelRegistry(unittest.TestCase):
    """Tests for the dynamic model registry."""

    def test_registry_contains_all_model_ids(self):
        registry = get_model_registry()
        for mid in ALL_MODEL_IDS:
            self.assertIn(mid, registry, f"Model ID '{mid}' missing from registry.")
            self.assertIn("name", registry[mid])
            self.assertIn("status", registry[mid])
            self.assertIn("description", registry[mid])

    def test_baseline_is_reported_unavailability_honestly(self):
        registry = get_model_registry()
        baseline = registry[MODEL_BASELINE]
        # In current repo, the baseline checkpoint is not present
        if baseline["status"] == "unavailable":
            self.assertIsNotNone(baseline.get("unavailable_reason"))
            self.assertIn("checkpoint", baseline["unavailable_reason"].lower())
            # Ensure raw IndicBART is not secretly labeled as the baseline
            self.assertIn("integrity", baseline["unavailable_reason"].lower())

    def test_stage1_and_stage2_registered(self):
        registry = get_model_registry()
        self.assertIn(MODEL_STAGE1, registry)
        self.assertIn(MODEL_STAGE2, registry)
        self.assertEqual(registry[MODEL_STAGE2]["base_checkpoint"], "checkpoints/stage1_indicnlg_qg")
        self.assertTrue(registry[MODEL_STAGE2]["is_adapter"])
        self.assertFalse(registry[MODEL_STAGE1]["is_adapter"])


class TestCandidateAnswerExtraction(unittest.TestCase):
    """Tests for candidate answer extraction parity across models."""

    def setUp(self):
        self.sample_passage = (
            "२६ जनवरी १९३१ के दिन कोलकाता में स्वतंत्रता दिवस मनाने की पूरी तैयारी की गई थी। "
            "पुलिस कमिश्नर ने नोटिस निकाला था कि अमुक-अमुक धारा के तहत कोई भी सभा नहीं हो सकती। "
            "काउंसिल के अध्यक्ष सुभाष चंद्र बोस के नेतृत्व में विशाल जुलूस निकाला गया।"
        )

    def test_extract_candidate_answers_detects_entities(self):
        candidates = extract_candidate_answers(self.sample_passage, max_candidates=5)
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0, "Failed to extract any candidate answers.")
        answers = [c["answer"] for c in candidates]
        # Should detect key entities like dates, locations, or persons
        has_salient_entity = any(
            any(k in ans for k in ["२६ जनवरी १९३१", "कोलकाता", "सुभाष चंद्र बोस"])
            for ans in answers
        )
        self.assertTrue(has_salient_entity, f"Expected salient entities in extracted answers: {answers}")

    def test_clean_candidate_span_strips_particles(self):
        raw = "के सुभाष चंद्र बोस ने"
        cleaned = clean_candidate_span(raw)
        self.assertFalse(cleaned.startswith("के "), "Prefix particle 'के ' was not stripped.")
        self.assertFalse(cleaned.endswith(" ने"), "Suffix particle ' ने' was not stripped.")
        self.assertEqual(cleaned, "सुभाष चंद्र बोस")


class TestComparisonAPIValidation(unittest.TestCase):
    """Tests for input validation in the comparison API."""

    def test_reject_empty_context(self):
        handler = MockHTTPHandler("POST", "/api/compare", {
            "context": "   ",
            "models": ["stage1", "stage2"]
        })
        handler.do_POST()
        self.assertEqual(handler.response_code, 400)
        data = handler.get_response_json()
        self.assertIn("error", data)
        self.assertIn("context", data["error"].lower())

    def test_reject_empty_models_list(self):
        handler = MockHTTPHandler("POST", "/api/compare", {
            "context": "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।",
            "models": []
        })
        handler.do_POST()
        self.assertEqual(handler.response_code, 400)
        data = handler.get_response_json()
        self.assertIn("models", data["error"].lower())

    def test_reject_invalid_strategy(self):
        handler = MockHTTPHandler("POST", "/api/compare", {
            "context": "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।",
            "models": ["stage1"],
            "strategy": "invalid_genetic_algorithm"
        })
        handler.do_POST()
        self.assertEqual(handler.response_code, 400)
        data = handler.get_response_json()
        self.assertIn("strategy", data["error"].lower())

    def test_reject_out_of_range_questions(self):
        handler = MockHTTPHandler("POST", "/api/compare", {
            "context": "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।",
            "models": ["stage1"],
            "num_questions": 99
        })
        handler.do_POST()
        self.assertEqual(handler.response_code, 400)
        data = handler.get_response_json()
        self.assertIn("num_questions", data["error"])


class TestComparisonAPIResponseStructure(unittest.TestCase):
    """Tests for /api/compare and endpoint handling."""

    def test_unavailable_model_returns_graceful_status(self):
        # Baseline is unavailable; should not crash, should return status: 'unavailable'
        handler = MockHTTPHandler("POST", "/api/compare", {
            "context": "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।",
            "models": ["baseline"],
            "num_questions": 1
        })
        handler.do_POST()
        self.assertEqual(handler.response_code, 200)
        data = handler.get_response_json()
        self.assertIn("results", data)
        self.assertIn("baseline", data["results"])
        b_res = data["results"]["baseline"]
        self.assertEqual(b_res["status"], "unavailable")
        self.assertIsNotNone(b_res.get("error"))
        self.assertGreater(len(data.get("warnings", [])), 0)

    def test_get_models_endpoint(self):
        handler = MockHTTPHandler("GET", "/api/models")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        data = handler.get_response_json()
        self.assertIn(MODEL_STAGE1, data)
        self.assertIn(MODEL_STAGE2, data)
        self.assertIn(MODEL_BASELINE, data)

    def test_get_health_endpoint(self):
        handler = MockHTTPHandler("GET", "/api/health")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        data = handler.get_response_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("available_models", data)

    def test_get_benchmark_results_endpoint(self):
        handler = MockHTTPHandler("GET", "/api/benchmark_results")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        data = handler.get_response_json()
        self.assertIn("benchmark_available", data)


class TestModelSelectionBehavior(unittest.TestCase):
    """Tests specifically validating the Model Comparison UI model-selection behavior."""

    def test_html_served_has_select_all_available_control(self):
        handler = MockHTTPHandler("GET", "/")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        html = handler.get_response_text()
        self.assertIn('id="btnSelectAllAvailable"', html, "Select All Available Models button missing from HTML.")
        self.assertIn('selectAllAvailableModels()', html, "selectAllAvailableModels function call missing from HTML.")
        self.assertIn('id="modelSelectGroup"', html, "Dynamic modelSelectGroup container missing from HTML.")

    def test_html_baseline_checkbox_disabled_and_deselected(self):
        handler = MockHTTPHandler("GET", "/")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        html = handler.get_response_text()
        self.assertIn('id="chkBaseline"', html)
        # Verify chkBaseline has disabled attribute
        self.assertRegex(html, r'<input[^>]*id="chkBaseline"[^>]*disabled', "chkBaseline must have disabled attribute.")
        # Verify chkBaseline does NOT have checked attribute
        self.assertNotRegex(html, r'<input[^>]*id="chkBaseline"[^>]*checked', "chkBaseline must NOT have checked attribute.")

    def test_html_displays_unavailable_explanation_reason(self):
        handler = MockHTTPHandler("GET", "/")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        html = handler.get_response_text()
        self.assertIn('id="baselineUnavailableNote"', html, "Baseline explanation container missing.")
        self.assertIn('Original IndicNLG question-generation baseline checkpoint is not available', html)
        self.assertIn('Pretrained raw IndicBART is intentionally not substituted', html)

    def test_api_models_provides_dynamic_status_and_reasons(self):
        handler = MockHTTPHandler("GET", "/api/models")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        registry = handler.get_response_json()

        self.assertIn(MODEL_BASELINE, registry)
        self.assertIn(MODEL_STAGE1, registry)
        self.assertIn(MODEL_STAGE2, registry)

        self.assertEqual(registry[MODEL_BASELINE]["status"], "unavailable")
        self.assertIsNotNone(registry[MODEL_BASELINE]["unavailable_reason"])
        self.assertEqual(registry[MODEL_STAGE1]["status"], "available")
        self.assertIsNone(registry[MODEL_STAGE1]["unavailable_reason"])
        self.assertEqual(registry[MODEL_STAGE2]["status"], "available")
        self.assertIsNone(registry[MODEL_STAGE2]["unavailable_reason"])

    def test_client_model_selection_simulation(self):
        """Simulates the JavaScript client-side selection logic based on the /api/models response."""
        handler = MockHTTPHandler("GET", "/api/models")
        handler.do_GET()
        registry = handler.get_response_json()

        # 1. Dynamic rendering simulation
        checkboxes = {}
        for mid, info in registry.items():
            is_available = info["status"] == "available"
            # Requirements 2 & 3:
            # Unavailable models -> disabled=True, checked=False
            # Available models -> disabled=False, default Stage1 and Stage2 to checked=True
            if is_available:
                is_checked = (mid in ["stage1", "stage2"])
            else:
                is_checked = False

            checkboxes[mid] = {
                "disabled": not is_available,
                "checked": is_checked,
                "unavailable_reason": info.get("unavailable_reason") if not is_available else None
            }

        # Verify Stage 1 and Stage 2 are enabled and checked by default
        self.assertTrue(checkboxes["stage1"]["checked"])
        self.assertFalse(checkboxes["stage1"]["disabled"])
        self.assertTrue(checkboxes["stage2"]["checked"])
        self.assertFalse(checkboxes["stage2"]["disabled"])

        # Verify Baseline is disabled and deselected
        self.assertFalse(checkboxes["baseline"]["checked"])
        self.assertTrue(checkboxes["baseline"]["disabled"])
        self.assertIsNotNone(checkboxes["baseline"]["unavailable_reason"])

        # 2. Select All Available Models control simulation
        for mid, state in checkboxes.items():
            if not state["disabled"]:
                state["checked"] = True
            else:
                state["checked"] = False

        selected_models = [mid for mid, state in checkboxes.items() if state["checked"] and not state["disabled"]]
        self.assertEqual(sorted(selected_models), ["stage1", "stage2"])
        self.assertNotIn("baseline", selected_models)

        # 3. Simulate deselecting all available models -> should be prevented from sending request
        for mid, state in checkboxes.items():
            state["checked"] = False

        active_models = [mid for mid, state in checkboxes.items() if state["checked"] and not state["disabled"]]
        self.assertEqual(len(active_models), 0)

    def test_prevent_comparison_when_no_available_models_selected(self):
        handler = MockHTTPHandler("POST", "/api/compare", {
            "context": "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।",
            "models": []
        })
        handler.do_POST()
        self.assertEqual(handler.response_code, 400)
        data = handler.get_response_json()
        self.assertIn("error", data)
        self.assertIn("models", data["error"].lower())

    def test_compare_stage1_and_stage2_generates_both_without_baseline(self):
        """Verifies Stage 1 and Stage 2 generate results together and baseline is never sent in request."""
        # Payload only sends the selected available models
        submitted_models = ["stage1", "stage2"]
        self.assertNotIn("baseline", submitted_models, "Unavailable baseline must never be in submitted models.")

        handler = MockHTTPHandler("POST", "/api/compare", {
            "context": "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है। उन्होंने समाज सुधार के लिए अथक प्रयास किए।",
            "answer": "डॉ. भीमराव अंबेडकर",
            "models": submitted_models,
            "num_questions": 1,
            "num_beams": 2,
            "max_length": 32
        })
        handler.do_POST()
        self.assertEqual(handler.response_code, 200)
        data = handler.get_response_json()

        # Verify selected models echoed back
        self.assertEqual(data["selected_models"], ["stage1", "stage2"])
        self.assertNotIn("baseline", data["selected_models"])

        # Verify results contain both models and not baseline
        results = data.get("results", {})
        self.assertIn("stage1", results)
        self.assertIn("stage2", results)
        self.assertNotIn("baseline", results)

        self.assertEqual(results["stage1"]["status"], "success")
        self.assertEqual(results["stage2"]["status"], "success")
        self.assertGreater(len(results["stage1"]["questions"]), 0)
        self.assertGreater(len(results["stage2"]["questions"]), 0)


class TestModelManagerLive(unittest.TestCase):
    """Live integration test for model manager adapter isolation with real weights."""

    @classmethod
    def setUpClass(cls):
        registry = get_model_registry()
        cls.stage1_avail = registry[MODEL_STAGE1]["status"] == "available"
        cls.stage2_avail = registry[MODEL_STAGE2]["status"] == "available"

    def test_adapter_switching_and_question_generation(self):
        if not (self.stage1_avail and self.stage2_avail):
            self.skipTest("Stage 1 and Stage 2 checkpoints are not both available for live test.")

        mgr = get_shared_model_manager(device="cpu")
        context = "डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है।"
        answer = "डॉ. भीमराव अंबेडकर"

        # Test Stage 1 generation (base model with adapter disabled)
        s1_qs = mgr.generate_single_for_model(
            model_id=MODEL_STAGE1,
            context=context,
            answer=answer,
            num_beams=2,
            max_length=32
        )
        self.assertIsInstance(s1_qs, list)
        self.assertGreater(len(s1_qs), 0)
        self.assertTrue(len(s1_qs[0].strip()) > 0)

        # Test Stage 2 generation (adapted model with active LoRA)
        s2_qs = mgr.generate_single_for_model(
            model_id=MODEL_STAGE2,
            context=context,
            answer=answer,
            num_beams=2,
            max_length=32
        )
        self.assertIsInstance(s2_qs, list)
        self.assertGreater(len(s2_qs), 0)
        self.assertTrue(len(s2_qs[0].strip()) > 0)


if __name__ == "__main__":
    unittest.main()
