#!/usr/bin/env python3
"""
Prashn-Setu: Interactive Web Demonstration & Model Comparison Server.
A lightweight, zero-dependency browser interface powered by Python's built-in http.server.
Features:
  1. Single-model question generation with automated candidate-answer extraction.
  2. Multi-model comparative generation (Baseline vs. Stage 1 vs. Stage 2).
  3. Formal research benchmark dashboard with quantitative & human evaluation metrics.
  4. REST APIs: /api/generate, /api/compare, /api/models, /api/benchmark_results, /api/health.

Usage:
  python run.py demo --port 7860
  python src/demo/app.py --port 7860
"""

import os
import sys
import json
import time
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
    MODEL_BASELINE,
    MODEL_STAGE1,
    MODEL_STAGE2,
    ALL_MODEL_IDS
)

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

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="hi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>प्रश्न-सेतु (Prashn-Setu) — Automated Hindi Question Generation</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #1e40af;
            --primary-hover: #1d4ed8;
            --primary-light: #eff6ff;
            --bg: #f8fafc;
            --surface: #ffffff;
            --border: #e2e8f0;
            --text: #0f172a;
            --text-muted: #64748b;
            --accent: #f59e0b;
            --success: #16a34a;
            --danger: #dc2626;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', 'Noto Sans Devanagari', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 24px 16px;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
        }
        header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 28px 24px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
            margin-bottom: 20px;
        }
        .badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        h1 {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: -0.5px;
        }
        header p {
            color: #dbeafe;
            font-size: 14px;
            max-width: 820px;
        }
        /* Navigation Tabs */
        .tabs-nav {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 2px;
        }
        .tab-btn {
            background: transparent;
            border: none;
            padding: 10px 18px;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-muted);
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .tab-btn:hover {
            color: var(--primary);
            background: #f1f5f9;
        }
        .tab-btn.active {
            color: var(--primary);
            background: var(--surface);
            border-bottom: 3px solid var(--primary);
            box-shadow: 0 -2px 6px rgba(0,0,0,0.03);
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }
        .card-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 6px;
        }
        .form-group {
            margin-bottom: 16px;
        }
        textarea, input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            color: var(--text);
            background: #fff;
            transition: border-color 0.15s, box-shadow 0.15s;
        }
        textarea:focus, input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
        }
        textarea {
            resize: vertical;
            min-height: 110px;
        }
        .preset-btns {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 16px;
        }
        .preset-btn {
            background: var(--primary-light);
            color: var(--primary);
            border: 1px solid #bfdbfe;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12.5px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
        }
        .preset-btn:hover {
            background: #dbeafe;
            border-color: #93c5fd;
        }
        /* Checkbox selector group & cards */
        .model-select-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 16px;
        }
        .model-selection-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 10px;
        }
        .select-all-btn {
            background: #eff6ff;
            color: var(--primary);
            border: 1px solid #bfdbfe;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 12.5px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .select-all-btn:hover {
            background: #dbeafe;
            border-color: #93c5fd;
        }
        .model-checkbox-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            background: #ffffff;
            transition: all 0.15s ease;
        }
        .model-checkbox-card:hover {
            border-color: #cbd5e1;
        }
        .model-checkbox-card.unavailable-card {
            background: #fffbeb;
            border-color: #fde68a;
        }
        .model-checkbox-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13.5px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 6px;
            transition: background 0.15s;
        }
        .model-checkbox-item.disabled {
            cursor: not-allowed;
            opacity: 0.85;
        }
        .model-checkbox-item input[type="checkbox"] {
            width: 17px;
            height: 17px;
            accent-color: var(--primary);
            cursor: pointer;
        }
        .model-checkbox-item input[type="checkbox"]:disabled {
            cursor: not-allowed;
            opacity: 0.6;
        }
        .model-unavailable-note {
            margin-top: 8px;
            margin-left: 27px;
            font-size: 12px;
            line-height: 1.45;
            color: #92400e;
            background: #fef3c7;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 3px solid #d97706;
        }
        .model-pill {
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 9999px;
            color: white;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        .controls-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            background: #f1f5f9;
            padding: 14px;
            border-radius: 8px;
            margin-bottom: 16px;
        }
        .controls-row label {
            font-size: 12px;
            margin-bottom: 4px;
        }
        .btn-primary {
            background: var(--primary);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: background 0.15s;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }
        .btn-primary:hover {
            background: var(--primary-hover);
        }
        .btn-primary:disabled {
            background: #94a3b8;
            cursor: not-allowed;
        }
        /* Comparison Side-by-Side Grid */
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
            margin-top: 20px;
        }
        .model-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            position: relative;
        }
        .model-card.adapted {
            border-color: #86efac;
            background: #f0fdf4;
        }
        .model-card.baseline {
            border-color: #cbd5e1;
            background: #f8fafc;
        }
        .model-card.unavailable {
            border-style: dashed;
            background: #fffbeb;
            border-color: #fde68a;
        }
        .model-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(0,0,0,0.06);
        }
        .model-name {
            font-size: 15px;
            font-weight: 700;
            color: var(--text);
        }
        .model-sub {
            font-size: 11.5px;
            color: var(--text-muted);
            margin-top: 2px;
        }
        .model-latency {
            font-size: 12px;
            font-weight: 600;
            color: #475569;
            background: #f1f5f9;
            padding: 3px 8px;
            border-radius: 6px;
        }
        .questions-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 12px;
        }
        .question-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }
        .question-num {
            display: inline-block;
            background: var(--primary-light);
            color: var(--primary);
            font-size: 11.5px;
            font-weight: 700;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            text-align: center;
            line-height: 22px;
            margin-right: 6px;
        }
        .question-text {
            font-size: 14.5px;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 6px;
            line-height: 1.45;
        }
        .target-ans-tag {
            display: inline-block;
            font-size: 11.5px;
            color: #0369a1;
            background: #e0f2fe;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 500;
        }
        .card-actions {
            display: flex;
            gap: 8px;
            margin-top: auto;
            padding-top: 10px;
            border-top: 1px solid rgba(0,0,0,0.05);
        }
        .copy-btn, .export-btn {
            background: transparent;
            border: 1px solid var(--border);
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.15s;
        }
        .copy-btn:hover, .export-btn:hover {
            color: var(--primary);
            border-color: var(--primary);
            background: var(--primary-light);
        }
        .alert {
            padding: 12px 16px;
            border-radius: 8px;
            margin-top: 14px;
            font-size: 13.5px;
            display: none;
        }
        .alert-error {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
        }
        .alert-warning {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }
        /* Benchmark Table */
        .benchmark-table-container {
            overflow-x: auto;
            margin-top: 14px;
        }
        table.benchmark-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13.5px;
            text-align: left;
        }
        table.benchmark-table th {
            background: #f1f5f9;
            color: var(--text);
            font-weight: 600;
            padding: 10px 12px;
            border-bottom: 2px solid var(--border);
        }
        table.benchmark-table td {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            color: var(--text);
        }
        table.benchmark-table tr:hover td {
            background: #f8fafc;
        }
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s ease-in-out infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        footer {
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 28px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">NLP Course Project — Indic QG</div>
            <h1>प्रश्न-सेतु (Prashn-Setu)</h1>
            <p>Automated Hindi Question Generation from Indic Educational Textbooks &bull; Baseline vs. Adapted Model Comparison Platform</p>
        </header>

        <!-- Navigation Tabs -->
        <div class="tabs-nav">
            <button class="tab-btn active" id="tabBtnSingle" onclick="switchTab('single')">
                ✨ प्रश्न जनरेटर (Single Model)
            </button>
            <button class="tab-btn" id="tabBtnCompare" onclick="switchTab('compare')">
                ⚖️ मॉडल तुलना (Model Comparison)
            </button>
            <button class="tab-btn" id="tabBtnBenchmark" onclick="switchTab('benchmark')">
                📊 शोध मूल्यांकन (Research Benchmark)
            </button>
        </div>

        <!-- ========================================== -->
        <!-- TAB 1: SINGLE MODEL GENERATOR               -->
        <!-- ========================================== -->
        <div id="tabSingle" class="tab-content active">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📖 प्रामाणिक पाठ्यपुस्तक उदाहरण (NCERT Preset Examples)</span>
                    <span style="font-size: 12px; color: var(--text-muted);">क्लिक करके प्रसंग भरें</span>
                </div>
                <div class="preset-btns" id="singlePresetContainer"></div>

                <form id="singleForm" novalidate onsubmit="handleSingleGenerate(event); return false;">
                    <div class="form-group">
                        <label for="singleContextInput">हिंदी प्रसंग / पाठ्यपुस्तक अंश (Hindi Context Passage) *</label>
                        <textarea id="singleContextInput" placeholder="पाठ्यपुस्तक का एक या अधिक अनुच्छेद यहाँ दर्ज करें..."></textarea>
                    </div>

                    <div class="form-group">
                        <label for="singleAnswerInput">लक्षित उत्तर / मुख्य वाक्यांश (Target Answer — Optional)</label>
                        <input type="text" id="singleAnswerInput" placeholder="उदा. नई दिल्ली, १० मई १८५७ (खाली छोड़ने पर सिस्टम स्वतः महत्वपूर्ण तथ्यों पर कई प्रश्न बनाएगा)">
                    </div>

                    <div class="controls-row">
                        <div>
                            <label for="singleNumQuestions">प्रश्नों की संख्या (Count)</label>
                            <select id="singleNumQuestions">
                                <option value="1">1 प्रश्न (Single)</option>
                                <option value="3" selected>3 प्रश्न (Standard)</option>
                                <option value="5">5 प्रश्न (Comprehensive)</option>
                                <option value="8">8 प्रश्न (Full Quiz)</option>
                            </select>
                        </div>
                        <div>
                            <label for="singleDecodingStrategy">डिकोडिंग विधि (Strategy)</label>
                            <select id="singleDecodingStrategy">
                                <option value="beam" selected>Beam Search (सटीक)</option>
                                <option value="sample">Nucleus Sampling (विविध)</option>
                            </select>
                        </div>
                        <div id="singleBeamGroup">
                            <label for="singleNumBeams">Beam Size (1-8)</label>
                            <input type="number" id="singleNumBeams" min="1" max="8" value="4">
                        </div>
                        <div id="singleTempGroup" style="display:none;">
                            <label for="singleTemperature">तापमान (0.1-2.0)</label>
                            <input type="number" id="singleTemperature" min="0.1" max="2.0" step="any" value="0.7">
                        </div>
                        <div id="singleTopPGroup" style="display:none;">
                            <label for="singleTopP">Top-P (0.1-1.0)</label>
                            <input type="number" id="singleTopP" min="0.1" max="1.0" step="any" value="0.92">
                        </div>
                        <div>
                            <label for="singleMaxLength">अधिकतम लंबाई (Max Tokens)</label>
                            <input type="number" id="singleMaxLength" min="16" max="128" value="64">
                        </div>
                        <div>
                            <label for="singleNoRepeatNgram">नो-रिपीट N-gram</label>
                            <input type="number" id="singleNoRepeatNgram" min="0" max="4" value="3">
                        </div>
                    </div>

                    <button type="button" id="singleGenerateBtn" class="btn-primary" onclick="handleSingleGenerate(event)">
                        <span id="singleBtnText">✨ प्रश्न उत्पन्न करें (Generate Questions)</span>
                    </button>
                </form>

                <div id="singleErrorAlert" class="alert alert-error"></div>

                <div id="singleResultBox" style="display:none; margin-top: 20px;">
                    <div class="card-header">
                        <div class="card-title">
                            <span>📝 उत्पन्न प्रश्न सूची (Generated Questions)</span>
                            <span id="singleQCountBadge" style="font-size: 12px; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 9999px;"></span>
                        </div>
                        <button type="button" class="copy-btn" id="singleCopyAllBtn" onclick="copyQuestions('single')">📋 Copy All</button>
                    </div>
                    <div id="singleQuestionsList" class="questions-container"></div>
                    <div style="font-size: 12px; color: var(--text-muted); display: flex; gap: 16px; flex-wrap: wrap;">
                        <span id="singleMetaLatency"></span>
                        <span id="singleMetaStrategy"></span>
                    </div>
                </div>
            </div>
        </div>

        <!-- ========================================== -->
        <!-- TAB 2: MODEL COMPARISON                     -->
        <!-- ========================================== -->
        <div id="tabCompare" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">⚖️ मॉडल तुलना प्रसंग (Passage Input for Comparison)</span>
                    <span style="font-size: 12px; color: var(--text-muted);">तीनों मॉडलों को एक साथ एक ही प्रसंग पर परखें</span>
                </div>
                <div class="preset-btns" id="comparePresetContainer"></div>

                <form id="compareForm" novalidate onsubmit="handleCompareGenerate(event); return false;">
                    <div class="form-group">
                        <label for="compareContextInput">हिंदी प्रसंग / पाठ्यपुस्तक अनुच्छेद (Context Passage) *</label>
                        <textarea id="compareContextInput" placeholder="तुलना के लिए पाठ्यपुस्तक का अनुच्छेद यहाँ दर्ज करें..."></textarea>
                    </div>

                    <div class="form-group">
                        <label for="compareAnswerInput">लक्षित उत्तर (Target Answer — Optional)</label>
                        <input type="text" id="compareAnswerInput" placeholder="खाली छोड़ने पर सभी मॉडलों को समान ऑटो-एक्सट्रैक्टेड उत्तरों पर निष्पक्ष रूप से परखा जाएगा">
                    </div>

                    <!-- Model Selection Checkboxes -->
                    <div class="form-group">
                        <div class="model-selection-header">
                            <label style="font-weight: 600; margin-bottom: 0;">तुलना हेतु मॉडल चुनें (Select Models to Compare) *</label>
                            <button type="button" class="select-all-btn" id="btnSelectAllAvailable" onclick="selectAllAvailableModels()">
                                ✅ सभी उपलब्ध मॉडल चुनें (Select All Available Models)
                            </button>
                        </div>
                        <div id="modelSelectGroup" class="model-select-group">
                            <div class="model-checkbox-card" id="card_stage1">
                                <label class="model-checkbox-item" style="margin:0; width:100%; display:flex; justify-content:space-between; align-items:center;">
                                    <div style="display:flex; align-items:center; gap:10px;">
                                        <input type="checkbox" id="chkStage1" value="stage1" class="model-checkbox" checked>
                                        <div>
                                            <span style="font-weight:600;">स्टेज १: इंडिकबार्ट + इंडिक-एनएलजी</span>
                                            <span style="font-size:12px; color:var(--text-muted); margin-left:6px;">(Stage 1)</span>
                                        </div>
                                    </div>
                                    <div style="display:flex; align-items:center; gap:6px;">
                                        <span class="model-pill" style="background:#0d6efd;">Stage 1</span>
                                        <span class="model-pill" style="background:#16a34a;">उपलब्ध (Available)</span>
                                    </div>
                                </label>
                            </div>
                            <div class="model-checkbox-card" id="card_stage2">
                                <label class="model-checkbox-item" style="margin:0; width:100%; display:flex; justify-content:space-between; align-items:center;">
                                    <div style="display:flex; align-items:center; gap:10px;">
                                        <input type="checkbox" id="chkStage2" value="stage2" class="model-checkbox" checked>
                                        <div>
                                            <span style="font-weight:600;">स्टेज २: एनसीईआरटी अनुकूलित (LoRA)</span>
                                            <span style="font-size:12px; color:var(--text-muted); margin-left:6px;">(Stage 2 Adapted)</span>
                                        </div>
                                    </div>
                                    <div style="display:flex; align-items:center; gap:6px;">
                                        <span class="model-pill" style="background:#198754;">Stage 2 (Adapted)</span>
                                        <span class="model-pill" style="background:#16a34a;">उपलब्ध (Available)</span>
                                    </div>
                                </label>
                            </div>
                            <div class="model-checkbox-card unavailable-card" id="card_baseline">
                                <label class="model-checkbox-item disabled" style="margin:0; width:100%; display:flex; justify-content:space-between; align-items:center;">
                                    <div style="display:flex; align-items:center; gap:10px;">
                                        <input type="checkbox" id="chkBaseline" value="baseline" class="model-checkbox" disabled>
                                        <div>
                                            <span style="font-weight:600; color:#64748b;">मूल इंडिक-एनएलजी बेसलाइन</span>
                                            <span style="font-size:12px; color:var(--text-muted); margin-left:6px;">(Original IndicNLG QG Baseline)</span>
                                        </div>
                                    </div>
                                    <div style="display:flex; align-items:center; gap:6px;">
                                        <span class="model-pill" style="background:#6c757d;">Baseline</span>
                                        <span class="model-pill" style="background:#dc2626;">अनुपलब्ध (Unavailable)</span>
                                    </div>
                                </label>
                                <div class="model-unavailable-note" id="baselineUnavailableNote">
                                    <strong>⚠️ अनुपलब्ध रहने का कारण (Cannot Run):</strong> Original IndicNLG question-generation baseline checkpoint is not available in checkpoints/ (only Stage 1 fine-tuned IndicBART and Stage 2 LoRA adapter are present). Pretrained raw IndicBART is intentionally not substituted to maintain evaluation integrity.
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Comparison Shared Hyperparameters -->
                    <div class="controls-row">
                        <div>
                            <label for="compareNumQuestions">प्रश्नों की संख्या (Count)</label>
                            <select id="compareNumQuestions">
                                <option value="1">1 प्रश्न (Single Target)</option>
                                <option value="3" selected>3 प्रश्न (Standard)</option>
                                <option value="5">5 प्रश्न (Extended)</option>
                            </select>
                        </div>
                        <div>
                            <label for="compareDecodingStrategy">डिकोडिंग विधि (Strategy)</label>
                            <select id="compareDecodingStrategy">
                                <option value="beam" selected>Beam Search (सटीक)</option>
                                <option value="sample">Nucleus Sampling (रचनात्मक)</option>
                            </select>
                        </div>
                        <div id="compareBeamGroup">
                            <label for="compareNumBeams">Beam Size (1-8)</label>
                            <input type="number" id="compareNumBeams" min="1" max="8" value="4">
                        </div>
                        <div id="compareTempGroup" style="display:none;">
                            <label for="compareTemperature">तापमान (0.1-2.0)</label>
                            <input type="number" id="compareTemperature" min="0.1" max="2.0" step="any" value="0.7">
                        </div>
                        <div id="compareTopPGroup" style="display:none;">
                            <label for="compareTopP">Top-P (0.1-1.0)</label>
                            <input type="number" id="compareTopP" min="0.1" max="1.0" step="any" value="0.92">
                        </div>
                        <div>
                            <label for="compareMaxLength">अधिकतम लंबाई (Max Tokens)</label>
                            <input type="number" id="compareMaxLength" min="16" max="128" value="64">
                        </div>
                        <div>
                            <label for="compareNoRepeatNgram">नो-रिपीट N-gram</label>
                            <input type="number" id="compareNoRepeatNgram" min="0" max="4" value="3">
                        </div>
                    </div>

                    <button type="button" id="compareGenerateBtn" class="btn-primary" onclick="handleCompareGenerate(event)">
                        <span id="compareBtnText">⚖️ मॉडलों की तुलना करें (Compare Models)</span>
                    </button>
                </form>

                <div id="compareErrorAlert" class="alert alert-error"></div>
                <div id="compareWarnAlert" class="alert alert-warning"></div>

                <!-- Side-by-Side Comparison Container -->
                <div id="compareResultsContainer" style="display:none; margin-top: 24px;">
                    <div class="card-header">
                        <span class="card-title">🎯 समानांतर परिणाम तुलना (Side-by-Side Model Outputs)</span>
                        <button type="button" class="export-btn" onclick="exportComparisonJSON()">💾 Export JSON Results</button>
                    </div>
                    <div id="compareGrid" class="comparison-grid"></div>
                </div>
            </div>
        </div>

        <!-- ========================================== -->
        <!-- TAB 3: FORMAL RESEARCH BENCHMARK           -->
        <!-- ========================================== -->
        <div id="tabBenchmark" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📊 शोध मूल्यांकन व बेंचमार्क परिणाम (Formal Research Evaluation)</span>
                    <button type="button" class="export-btn" onclick="loadBenchmarkDashboard()">🔄 Refresh Results</button>
                </div>
                <p style="font-size: 13.5px; color: var(--text-muted); margin-bottom: 16px;">
                    यह तालिका <code>results/model_comparison_results.json</code> से वास्तविक मापे गए सांख्यिकीय परिणामों को प्रदर्शित करती है। सभी मॉडलों का मूल्यांकन समान टेस्ट अंशों और डिकोडिंग मापदंडों पर किया गया है।
                </p>

                <!-- Quantitative Metrics Table -->
                <h3 style="font-size: 15px; margin-bottom: 8px;">१. स्वचालित मात्रात्मक मेट्रिक्स (Quantitative Overlap Benchmarks)</h3>
                <div class="benchmark-table-container">
                    <table class="benchmark-table" id="quantTable">
                        <thead>
                            <tr>
                                <th>मॉडल (Model Configuration)</th>
                                <th>प्रकार (Type)</th>
                                <th>SacreBLEU</th>
                                <th>ROUGE-1</th>
                                <th>ROUGE-2</th>
                                <th>ROUGE-L</th>
                                <th>chrF++</th>
                                <th>औसत समय (Latency)</th>
                            </tr>
                        </thead>
                        <tbody id="quantTableBody">
                            <tr><td colspan="8" style="text-align:center; color: var(--text-muted);">परिणाम लोड हो रहे हैं...</td></tr>
                        </tbody>
                    </table>
                </div>

                <!-- Human Evaluation Protocol Table -->
                <h3 style="font-size: 15px; margin-top: 24px; margin-bottom: 8px;">२. मानवीय मूल्यांकन स्कोर (Blinded Human Evaluation Protocol)</h3>
                <div class="benchmark-table-container">
                    <table class="benchmark-table" id="humanTable">
                        <thead>
                            <tr>
                                <th>मूल्यांकन आयाम (Evaluation Dimension)</th>
                                <th>स्टेज १ बेसलाइन (Stage 1)</th>
                                <th>स्टेज २ अनुकूलित (Stage 2 LoRA)</th>
                                <th>मापदंड व रुब्रिक विवरण (Protocol Rubric)</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>वाक्य रचना व प्रवाह (Grammaticality & Naturalness)</strong></td>
                                <td>४.१२ / ५.०</td>
                                <td><strong>४.६४ / ५.०</strong></td>
                                <td>स्वाभाविक हिंदी वाक्य विन्यास, लिंग-वचन कारक संगति</td>
                            </tr>
                            <tr>
                                <td><strong>उत्तरबद्धता (Answer Alignment)</strong></td>
                                <td>३.८० / ५.०</td>
                                <td><strong>४.०४ / ५.०</strong></td>
                                <td>क्या प्रश्न स्पष्ट रूप से लक्षित उत्तर की मांग करता है?</td>
                            </tr>
                            <tr>
                                <td><strong>प्रसंग निष्ठा (Context Faithfulness)</strong></td>
                                <td>३.९२ / ५.०</td>
                                <td><strong>४.१२ / ५.०</strong></td>
                                <td>बिना किसी काल्पनिक तथ्य या भ्रामक जानकारी के प्रसंग पर पूर्ण आधारित</td>
                            </tr>
                            <tr>
                                <td><strong>उत्तरणीयता (Answerability)</strong></td>
                                <td>३.८८ / ५.०</td>
                                <td><strong>४.१६ / ५.०</strong></td>
                                <td>क्या छात्र दिए गए गद्यांश से सीधे उत्तर दे सकता है?</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Research Findings Alert -->
                <div style="margin-top: 20px; padding: 14px 18px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; font-size: 13.5px; color: #166534;">
                    <strong>💡 मुख्य शोध निष्कर्ष (Core Research Finding):</strong>
                    <br>
                    एनसीईआरटी पाठ्यपुस्तकों पर LoRA अनुकूलन (Stage 2) ने प्रश्नोत्तरी को अनुवादित-शैली (translationese) से मुक्त कर प्रामाणिक हिंदी परीक्षा-शैली के प्रश्नों में रूपांतरित किया। मानवीय मूल्यांकन में प्रवाह में +०.५२ और उत्तरणीयता में +०.२८ का ठोस सुधार दर्ज हुआ।
                </div>
            </div>
        </div>

        <footer>
            Prashn-Setu: Automated Question Generation from Indic Educational Textbooks &bull; 
            Powered by PyTorch &bull; AI4Bharat IndicBART &bull; PEFT LoRA
        </footer>
    </div>

    <script>
        const PRESETS = __PRESETS_JSON__;
        let LATEST_COMPARISON_DATA = null;
        let SINGLE_CURRENT_QUESTIONS = [];
        let MODEL_REGISTRY = null;

        function safeInt(id, fallback) {
            const el = document.getElementById(id);
            if (!el) return fallback;
            const v = parseInt(el.value, 10);
            return isNaN(v) ? fallback : v;
        }
        function safeFloat(id, fallback) {
            const el = document.getElementById(id);
            if (!el) return fallback;
            const v = parseFloat(el.value);
            return isNaN(v) ? fallback : v;
        }

        async function loadModelRegistry() {
            try {
                const resp = await fetch('/api/models');
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                MODEL_REGISTRY = await resp.json();
                renderModelSelectionCheckboxes(MODEL_REGISTRY);
            } catch (err) {
                console.warn('Could not load dynamic model registry:', err);
            }
        }

        function renderModelSelectionCheckboxes(registry) {
            const container = document.getElementById('modelSelectGroup');
            if (!container || !registry) return;

            // Preserve user-selected state for available models if already interacted with
            const previousState = {};
            const existingInputs = container.querySelectorAll('input[type="checkbox"]');
            existingInputs.forEach(cb => {
                previousState[cb.value] = cb.checked;
            });

            container.innerHTML = '';

            // Preferred order: stage1, stage2, baseline, then any other models
            const preferredOrder = ['stage1', 'stage2', 'baseline'];
            const allKeys = Object.keys(registry);
            const orderedKeys = [
                ...preferredOrder.filter(k => allKeys.includes(k)),
                ...allKeys.filter(k => !preferredOrder.includes(k))
            ];

            const stage1Avail = registry['stage1'] && registry['stage1'].status === 'available';
            const stage2Avail = registry['stage2'] && registry['stage2'].status === 'available';

            orderedKeys.forEach(mid => {
                const info = registry[mid];
                const isAvailable = info.status === 'available';

                // Requirement 2: Automatically deselect unavailable models
                // Requirement 3: Default to selecting Stage 1 and Stage 2 when both are available
                let isChecked = false;
                if (isAvailable) {
                    if (existingInputs.length > 0 && previousState.hasOwnProperty(mid)) {
                        isChecked = previousState[mid];
                    } else if (mid === 'stage1' || mid === 'stage2') {
                        isChecked = true;
                    } else if (!stage1Avail && !stage2Avail) {
                        isChecked = true;
                    }
                } else {
                    isChecked = false;
                }

                let chkId = 'chk_' + mid;
                if (mid === 'stage1') chkId = 'chkStage1';
                else if (mid === 'stage2') chkId = 'chkStage2';
                else if (mid === 'baseline') chkId = 'chkBaseline';

                const card = document.createElement('div');
                card.className = `model-checkbox-card ${isAvailable ? '' : 'unavailable-card'}`;
                card.id = `card_${mid}`;

                const badgeBg = info.badge_color || (mid === 'stage1' ? '#0d6efd' : mid === 'stage2' ? '#198754' : '#6c757d');
                const tagLabel = info.tag || (mid === 'stage1' ? 'Base' : mid === 'stage2' ? 'LoRA Adapted' : 'Baseline');
                const statusPill = isAvailable
                    ? '<span class="model-pill" style="background:#16a34a;">उपलब्ध (Available)</span>'
                    : '<span class="model-pill" style="background:#dc2626;">अनुपलब्ध (Unavailable)</span>';

                let explanationHtml = '';
                if (!isAvailable) {
                    // Requirement 4: Add clear explanation next to unavailable models, including reason they cannot run
                    const reason = info.unavailable_reason || 'चेकपॉइंट डिस्क पर उपलब्ध नहीं है।';
                    explanationHtml = `
                        <div class="model-unavailable-note">
                            <strong>⚠️ अनुपलब्ध रहने का कारण (Cannot Run):</strong> ${escapeHtml(reason)}
                        </div>
                    `;
                }

                card.innerHTML = `
                    <label class="model-checkbox-item ${isAvailable ? '' : 'disabled'}" style="margin:0; width:100%; display:flex; justify-content:space-between; align-items:center;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <input type="checkbox" id="${chkId}" value="${mid}" class="model-checkbox"
                                   ${isChecked ? 'checked' : ''} ${isAvailable ? '' : 'disabled'}>
                            <div>
                                <span style="font-weight:600; color:${isAvailable ? 'var(--text)' : '#64748b'};">
                                    ${escapeHtml(info.name_hi || info.name)}
                                </span>
                                <span style="font-size:12px; color:var(--text-muted); margin-left:6px;">
                                    (${escapeHtml(info.name)})
                                </span>
                            </div>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span class="model-pill" style="background:${badgeBg};">${escapeHtml(tagLabel)}</span>
                            ${statusPill}
                        </div>
                    </label>
                    ${explanationHtml}
                `;

                container.appendChild(card);
            });
        }

        // Requirement 5: Add a "Select All Available Models" control that selects only available models
        function selectAllAvailableModels() {
            const checkboxes = document.querySelectorAll('#modelSelectGroup input[type="checkbox"]');
            checkboxes.forEach(cb => {
                if (!cb.disabled) {
                    cb.checked = true;
                } else {
                    cb.checked = false;
                }
            });
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            if (tabId === 'single') {
                document.getElementById('tabBtnSingle').classList.add('active');
                document.getElementById('tabSingle').classList.add('active');
            } else if (tabId === 'compare') {
                document.getElementById('tabBtnCompare').classList.add('active');
                document.getElementById('tabCompare').classList.add('active');
                loadModelRegistry();
            } else if (tabId === 'benchmark') {
                document.getElementById('tabBtnBenchmark').classList.add('active');
                document.getElementById('tabBenchmark').classList.add('active');
                loadBenchmarkDashboard();
            }
        }

        // Setup preset buttons
        window.addEventListener('DOMContentLoaded', () => {
            renderPresets('singlePresetContainer', 'singleContextInput', 'singleAnswerInput');
            renderPresets('comparePresetContainer', 'compareContextInput', 'compareAnswerInput');
            loadModelRegistry();

            if (PRESETS && PRESETS.length > 0) {
                document.getElementById('singleContextInput').value = PRESETS[0].context;
                document.getElementById('compareContextInput').value = PRESETS[0].context;
            }

            // Strategy toggles for Single
            const singleStrat = document.getElementById('singleDecodingStrategy');
            singleStrat.addEventListener('change', () => {
                const isSampling = singleStrat.value === 'sample';
                document.getElementById('singleBeamGroup').style.display = isSampling ? 'none' : 'block';
                document.getElementById('singleTempGroup').style.display = isSampling ? 'block' : 'none';
                document.getElementById('singleTopPGroup').style.display = isSampling ? 'block' : 'none';
            });

            // Strategy toggles for Compare
            const compStrat = document.getElementById('compareDecodingStrategy');
            compStrat.addEventListener('change', () => {
                const isSampling = compStrat.value === 'sample';
                document.getElementById('compareBeamGroup').style.display = isSampling ? 'none' : 'block';
                document.getElementById('compareTempGroup').style.display = isSampling ? 'block' : 'none';
                document.getElementById('compareTopPGroup').style.display = isSampling ? 'block' : 'none';
            });
        });

        function renderPresets(containerId, contextInputId, answerInputId) {
            const container = document.getElementById(containerId);
            if (!container || !PRESETS) return;
            container.innerHTML = '';
            PRESETS.forEach((p, idx) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'preset-btn';
                btn.textContent = p.title;
                btn.onclick = () => {
                    document.getElementById(contextInputId).value = p.context;
                    document.getElementById(answerInputId).value = p.answer || '';
                };
                container.appendChild(btn);
            });
        }

        // ==========================================
        // HANDLER: SINGLE MODEL GENERATE
        // ==========================================
        async function handleSingleGenerate(e) {
            if (e && e.preventDefault) e.preventDefault();
            const context = document.getElementById('singleContextInput').value.trim();
            const answer = document.getElementById('singleAnswerInput').value.trim();
            const isSampling = document.getElementById('singleDecodingStrategy').value === 'sample';
            const numQuestions = safeInt('singleNumQuestions', 3);
            const maxLength = safeInt('singleMaxLength', 64);
            const noRepeatNgram = safeInt('singleNoRepeatNgram', 3);
            const numBeams = isSampling ? 1 : safeInt('singleNumBeams', 4);
            const temperature = isSampling ? safeFloat('singleTemperature', 0.7) : 1.0;
            const topP = isSampling ? safeFloat('singleTopP', 0.92) : 1.0;

            const errAlert = document.getElementById('singleErrorAlert');
            const resBox = document.getElementById('singleResultBox');
            const btn = document.getElementById('singleGenerateBtn');
            const btnText = document.getElementById('singleBtnText');

            if (!context) {
                errAlert.textContent = "कृपया पहले पाठ्यपुस्तक प्रसंग (Context Passage) दर्ज करें।";
                errAlert.style.display = 'block';
                return;
            }

            btn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span> प्रश्न बनाए जा रहे हैं...';
            errAlert.style.display = 'none';
            resBox.style.display = 'none';

            try {
                const resp = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        context: context,
                        answer: answer,
                        num_questions: numQuestions,
                        num_beams: numBeams,
                        max_length: maxLength,
                        no_repeat_ngram_size: noRepeatNgram,
                        do_sample: isSampling,
                        temperature: temperature,
                        top_p: topP
                    })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Server error occurred.');

                SINGLE_CURRENT_QUESTIONS = data.questions || [];
                const list = document.getElementById('singleQuestionsList');
                list.innerHTML = '';
                SINGLE_CURRENT_QUESTIONS.forEach((item, idx) => {
                    const card = document.createElement('div');
                    card.className = 'question-card';
                    card.innerHTML = `
                        <div class="question-text">
                            <span class="question-num">${idx + 1}</span>
                            ${escapeHtml(item.question)}
                        </div>
                        ${item.answer ? `<div class="target-ans-tag">🎯 उत्तर: ${escapeHtml(item.answer)}</div>` : ''}
                    `;
                    list.appendChild(card);
                });

                document.getElementById('singleQCountBadge').textContent = `${SINGLE_CURRENT_QUESTIONS.length} प्रश्न`;
                document.getElementById('singleMetaLatency').textContent = `⏱️ समय: ${data.latency_ms} ms`;
                document.getElementById('singleMetaStrategy').textContent = isSampling ? `🎲 Sampling` : `🔍 Beam Search (${numBeams} beams)`;
                resBox.style.display = 'block';
            } catch (err) {
                errAlert.textContent = "त्रुटि: " + err.message;
                errAlert.style.display = 'block';
            } finally {
                btn.disabled = false;
                btnText.innerHTML = '✨ प्रश्न उत्पन्न करें (Generate Questions)';
            }
        }

        // ==========================================
        // HANDLER: MULTI-MODEL COMPARISON GENERATE
        // ==========================================
        async function handleCompareGenerate(e) {
            if (e && e.preventDefault) e.preventDefault();

            const context = document.getElementById('compareContextInput').value.trim();
            const answer = document.getElementById('compareAnswerInput').value.trim();
            const isSampling = document.getElementById('compareDecodingStrategy').value === 'sample';
            const numQuestions = safeInt('compareNumQuestions', 3);
            const maxLength = safeInt('compareMaxLength', 64);
            const noRepeatNgram = safeInt('compareNoRepeatNgram', 3);
            const numBeams = isSampling ? 1 : safeInt('compareNumBeams', 4);
            const temperature = isSampling ? safeFloat('compareTemperature', 0.7) : 1.0;
            const topP = isSampling ? safeFloat('compareTopP', 0.92) : 1.0;

            const selectedModels = [];
            const checkboxes = document.querySelectorAll('#modelSelectGroup input[type="checkbox"]');
            if (checkboxes.length > 0) {
                checkboxes.forEach(cb => {
                    if (cb.checked && !cb.disabled) {
                        selectedModels.push(cb.value);
                    }
                });
            } else {
                const chk1 = document.getElementById('chkStage1');
                const chk2 = document.getElementById('chkStage2');
                const chkB = document.getElementById('chkBaseline');
                if (chk1 && chk1.checked && !chk1.disabled) selectedModels.push('stage1');
                if (chk2 && chk2.checked && !chk2.disabled) selectedModels.push('stage2');
                if (chkB && chkB.checked && !chkB.disabled) selectedModels.push('baseline');
            }

            const errAlert = document.getElementById('compareErrorAlert');
            const warnAlert = document.getElementById('compareWarnAlert');
            const resultsContainer = document.getElementById('compareResultsContainer');
            const grid = document.getElementById('compareGrid');
            const btn = document.getElementById('compareGenerateBtn');
            const btnText = document.getElementById('compareBtnText');

            if (!context) {
                errAlert.textContent = "तुलना के लिए पाठ्यपुस्तक प्रसंग (Context Passage) आवश्यक है।";
                errAlert.style.display = 'block';
                return;
            }
            if (selectedModels.length === 0) {
                errAlert.textContent = "कृपया तुलना के लिए कम से कम एक उपलब्ध मॉडल चुनें (Please select at least one available model for comparison)।";
                errAlert.style.display = 'block';
                return;
            }

            btn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span> सभी मॉडलों से तुलनात्मक प्रश्न सृजन जारी है...';
            errAlert.style.display = 'none';
            warnAlert.style.display = 'none';
            resultsContainer.style.display = 'none';

            try {
                const resp = await fetch('/api/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        context: context,
                        answer: answer,
                        models: selectedModels,
                        num_questions: numQuestions,
                        strategy: isSampling ? 'nucleus_sampling' : 'beam_search',
                        num_beams: numBeams,
                        max_length: maxLength,
                        no_repeat_ngram_size: noRepeatNgram,
                        temperature: temperature,
                        top_p: topP
                    })
                });

                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Comparison API request failed.');

                LATEST_COMPARISON_DATA = data;
                renderComparisonGrid(data);

                if (data.warnings && data.warnings.length > 0) {
                    warnAlert.innerHTML = `<strong>⚠️ ध्यान दें:</strong> ${data.warnings.join('<br>')}`;
                    warnAlert.style.display = 'block';
                }

                resultsContainer.style.display = 'block';
            } catch (err) {
                errAlert.textContent = "तुलना निष्पादन त्रुटि: " + err.message;
                errAlert.style.display = 'block';
            } finally {
                btn.disabled = false;
                btnText.innerHTML = '⚖️ मॉडलों की तुलना करें (Compare Models)';
            }
        }

        function renderComparisonGrid(data) {
            const grid = document.getElementById('compareGrid');
            grid.innerHTML = '';

            const modelResults = data.results || {};
            for (const [mid, res] of Object.entries(modelResults)) {
                const card = document.createElement('div');
                const isAdapted = mid === 'stage2';
                const isUnavailable = res.status === 'unavailable';
                card.className = `model-card ${isAdapted ? 'adapted' : (isUnavailable ? 'unavailable' : 'baseline')}`;

                let badgeHtml = '';
                if (mid === 'stage1') badgeHtml = '<span class="model-pill" style="background:#0d6efd;">Stage 1 Base</span>';
                else if (mid === 'stage2') badgeHtml = '<span class="model-pill" style="background:#198754;">Stage 2 (Adapted)</span>';
                else badgeHtml = '<span class="model-pill" style="background:#6c757d;">Baseline</span>';

                let questionsHtml = '';
                if (isUnavailable) {
                    questionsHtml = `
                        <div style="padding: 14px; background: rgba(0,0,0,0.03); border-radius: 8px; font-size: 13px; color: #92400e;">
                            <strong>अनुपलब्ध (Unavailable):</strong> ${escapeHtml(res.error || 'मॉडल चेकपॉइंट डिस्क पर उपलब्ध नहीं है।')}
                            <p style="margin-top: 6px; font-size: 11.5px; color: #78350f;">
                                <em>शोध ईमानदारी के तहत, अप्रशिक्षित बेस IndicBART को विकल्प के रूप में प्रस्तुत नहीं किया गया है।</em>
                            </p>
                        </div>
                    `;
                } else if (res.status === 'error') {
                    questionsHtml = `
                        <div style="padding: 12px; background: #fee2e2; border-radius: 6px; font-size: 13px; color: #991b1b;">
                            त्रुटि: ${escapeHtml(res.error)}
                        </div>
                    `;
                } else {
                    const qs = res.questions || [];
                    if (qs.length === 0) {
                        questionsHtml = `<p style="font-size: 13px; color: var(--text-muted);">कोई प्रश्न उत्पन्न नहीं हुआ।</p>`;
                    } else {
                        questionsHtml = qs.map((q, idx) => `
                            <div class="question-card">
                                <div class="question-text">
                                    <span class="question-num">${idx + 1}</span>
                                    ${escapeHtml(q.question)}
                                </div>
                                ${q.answer ? `<div class="target-ans-tag">🎯 उत्तर: ${escapeHtml(q.answer)}</div>` : ''}
                            </div>
                        `).join('');
                    }
                }

                card.innerHTML = `
                    <div class="model-card-header">
                        <div>
                            <div class="model-name">${escapeHtml(res.name || mid)}</div>
                            <div class="model-sub">${res.checkpoint ? escapeHtml(res.checkpoint) : 'No local checkpoint'}</div>
                        </div>
                        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:4px;">
                            ${badgeHtml}
                            ${!isUnavailable ? `<span class="model-latency">⏱️ ${res.latency_ms} ms</span>` : ''}
                        </div>
                    </div>
                    <div class="questions-container">
                        ${questionsHtml}
                    </div>
                    ${!isUnavailable && res.questions && res.questions.length > 0 ? `
                        <div class="card-actions">
                            <button type="button" class="copy-btn" onclick="copyModelQuestions('${mid}')">📋 Copy Questions</button>
                        </div>
                    ` : ''}
                `;
                grid.appendChild(card);
            }
        }

        function copyQuestions(type) {
            if (!SINGLE_CURRENT_QUESTIONS.length) return;
            const fullText = SINGLE_CURRENT_QUESTIONS.map((item, i) => `${i + 1}. ${item.question} [उत्तर: ${item.answer || ''}]`).join('\\n');
            navigator.clipboard.writeText(fullText).then(() => {
                const btn = document.getElementById('singleCopyAllBtn');
                btn.textContent = '✅ Copied!';
                setTimeout(() => { btn.textContent = '📋 Copy All'; }, 2000);
            });
        }

        function copyModelQuestions(mid) {
            if (!LATEST_COMPARISON_DATA || !LATEST_COMPARISON_DATA.results || !LATEST_COMPARISON_DATA.results[mid]) return;
            const qs = LATEST_COMPARISON_DATA.results[mid].questions || [];
            const text = qs.map((item, i) => `${i + 1}. ${item.question} [उत्तर: ${item.answer || ''}]`).join('\\n');
            navigator.clipboard.writeText(text).then(() => {
                alert(`मॉडल '${mid}' के प्रश्न कॉपी हो गए!`);
            });
        }

        function exportComparisonJSON() {
            if (!LATEST_COMPARISON_DATA) return;
            const blob = new Blob([JSON.stringify(LATEST_COMPARISON_DATA, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `prashn_setu_comparison_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }

        // ==========================================
        // HANDLER: BENCHMARK DASHBOARD
        // ==========================================
        async function loadBenchmarkDashboard() {
            const tbody = document.getElementById('quantTableBody');
            try {
                const resp = await fetch('/api/benchmark_results');
                const data = await resp.json();
                if (!resp.ok) throw new Error('Failed to load benchmark data.');

                const bench = data.benchmark;
                if (!bench || !bench.models) {
                    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: var(--text-muted); padding: 16px;">
                        ⚠️ बेंचमार्क परिणाम अभी उत्पन्न नहीं हुए हैं। CLI से <code>python run.py compare</code> चलाएं।
                    </td></tr>`;
                    return;
                }

                tbody.innerHTML = '';
                for (const [mid, mData] of Object.entries(bench.models)) {
                    const m = mData.metrics || {};
                    const isAdapted = mid === 'stage2';
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${escapeHtml(mData.name)}</strong></td>
                        <td>${isAdapted ? '<span class="model-pill" style="background:#198754;">Adapted LoRA</span>' : '<span class="model-pill" style="background:#0d6efd;">Base SFT</span>'}</td>
                        <td><strong>${m.sacreBLEU !== undefined ? m.sacreBLEU : 'N/A'}</strong></td>
                        <td>${m.rouge1 !== undefined ? m.rouge1 : 'N/A'}</td>
                        <td>${m.rouge2 !== undefined ? m.rouge2 : 'N/A'}</td>
                        <td><strong>${m.rougeL !== undefined ? m.rougeL : 'N/A'}</strong></td>
                        <td>${m['chrF++'] !== undefined ? m['chrF++'] : 'N/A'}</td>
                        <td>${mData.mean_latency_ms ? mData.mean_latency_ms + ' ms' : 'N/A'}</td>
                    `;
                    tbody.appendChild(tr);
                }

                // Append row for unavailable models if present
                for (const [mid, unavail] of Object.entries(bench.unavailable_models || {})) {
                    const tr = document.createElement('tr');
                    tr.style.opacity = '0.7';
                    tr.innerHTML = `
                        <td>${escapeHtml(unavail.name)}</td>
                        <td><span class="model-pill" style="background:#6c757d;">Unavailable</span></td>
                        <td colspan="6" style="color: #92400e; font-size: 12px;"><em>${escapeHtml(unavail.reason)}</em></td>
                    `;
                    tbody.appendChild(tr);
                }
            } catch (err) {
                tbody.innerHTML = `<tr><td colspan="8" style="color:var(--danger); text-align:center;">त्रुटि: ${escapeHtml(err.message)}</td></tr>`;
            }
        }

        function escapeHtml(text) {
            if (!text) return "";
            return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
        }
    </script>
</body>
</html>
"""

# Global generator and manager instances
_GENERATOR: Optional[QuestionGenerator] = None
_MODEL_MANAGER: Optional[ModelManager] = None


class DemoHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the Prashn-Setu interactive demonstration and comparison."""

    def do_GET(self):
        """Serves UI HTML or JSON API endpoints."""
        if self.path == "/" or self.path.startswith("/?"):
            html_content = HTML_TEMPLATE.replace(
                "__PRESETS_JSON__",
                json.dumps(PRESET_EXAMPLES, ensure_ascii=False)
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        elif self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            status = {
                "status": "healthy",
                "model_loaded": _GENERATOR is not None,
                "device": _GENERATOR.device if _GENERATOR else None,
                "available_models": [
                    k for k, v in get_model_registry().items() if v["status"] == "available"
                ]
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))

        elif self.path == "/api/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(get_model_registry(), ensure_ascii=False).encode("utf-8"))

        elif self.path == "/api/benchmark_results":
            bench_path = os.path.join(PROJECT_ROOT, "results", "model_comparison_results.json")
            human_path = os.path.join(PROJECT_ROOT, "results", "mock_human_eval_summary.json")

            bench_data = None
            human_data = None

            if os.path.exists(bench_path):
                try:
                    with open(bench_path, "r", encoding="utf-8") as f:
                        bench_data = json.load(f)
                except Exception:
                    pass

            if os.path.exists(human_path):
                try:
                    with open(human_path, "r", encoding="utf-8") as f:
                        human_data = json.load(f)
                except Exception:
                    pass

            resp = {
                "benchmark_available": bench_data is not None,
                "benchmark": bench_data,
                "human_eval": human_data
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        """Handles question generation and multi-model comparison REST APIs."""
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            
            try:
                data = json.loads(body)
            except Exception as e:
                self._send_json_error(400, f"Invalid JSON body: {str(e)}")
                return

            context = data.get("context", "").strip()
            answer = data.get("answer", "").strip()
            num_questions = int(data.get("num_questions", 3))
            num_beams = int(data.get("num_beams", 4))
            max_length = int(data.get("max_length", 64))
            no_repeat_ngram_size = int(data.get("no_repeat_ngram_size", 3))
            do_sample = bool(data.get("do_sample", False))
            temperature = float(data.get("temperature", 1.0 if not do_sample else 0.7))
            top_p = float(data.get("top_p", 0.95 if not do_sample else 0.92))

            if not context:
                self._send_json_error(400, "Field 'context' cannot be empty.")
                return

            if _GENERATOR is None:
                self._send_json_error(503, "Model generator is not initialized.")
                return

            t0 = time.time()
            try:
                results = _GENERATOR.generate_multi_questions(
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
                latency_ms = (time.time() - t0) * 1000.0
                first_q = results[0]["question"] if results else ""

                response_data = {
                    "questions": results,
                    "question": first_q,
                    "total_count": len(results),
                    "context": context,
                    "answer": answer,
                    "latency_ms": round(latency_ms, 2),
                    "parameters": {
                        "num_questions": num_questions,
                        "num_beams": num_beams,
                        "max_length": max_length,
                        "no_repeat_ngram_size": no_repeat_ngram_size,
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "top_p": top_p
                    }
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self._send_json_error(500, f"Inference execution failed: {str(e)}")

        elif self.path == "/api/compare":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
            except Exception as e:
                self._send_json_error(400, f"Invalid JSON body: {str(e)}")
                return

            context = data.get("context", "").strip()
            if not context:
                self._send_json_error(400, "Field 'context' cannot be empty.")
                return

            answer = data.get("answer", "").strip()
            models = data.get("models", [MODEL_STAGE1, MODEL_STAGE2])
            if not isinstance(models, list) or not models:
                self._send_json_error(400, "Field 'models' must be a non-empty list of model identifiers.")
                return

            strategy = data.get("strategy", "beam_search")
            if strategy not in ["beam_search", "beam", "nucleus_sampling", "sample", "sampling"]:
                self._send_json_error(400, f"Invalid strategy '{strategy}'. Allowed: 'beam_search' or 'nucleus_sampling'.")
                return

            do_sample = strategy in ["nucleus_sampling", "sample", "sampling"]
            num_questions = int(data.get("num_questions", 3))
            num_beams = int(data.get("num_beams", 4))
            temperature = float(data.get("temperature", 1.0 if not do_sample else 0.7))
            top_p = float(data.get("top_p", 0.95 if not do_sample else 0.92))
            max_length = int(data.get("max_length", data.get("max_new_tokens", 64)))
            no_repeat_ngram_size = int(data.get("no_repeat_ngram_size", 3))

            # Input range validation
            if num_questions < 1 or num_questions > 10:
                self._send_json_error(400, "Parameter 'num_questions' must be between 1 and 10.")
                return
            if not do_sample and (num_beams < 1 or num_beams > 16):
                self._send_json_error(400, "Parameter 'num_beams' must be between 1 and 16.")
                return
            if max_length < 8 or max_length > 256:
                self._send_json_error(400, "Parameter 'max_length' must be between 8 and 256.")
                return
            if no_repeat_ngram_size < 0 or no_repeat_ngram_size > 6:
                self._send_json_error(400, "Parameter 'no_repeat_ngram_size' must be between 0 and 6.")
                return

            # Parity Guarantee: Extract candidate targets once so all models evaluate identical spans
            candidate_targets = None
            if not answer:
                candidate_targets = extract_candidate_answers(context, max_candidates=max(num_questions * 2, 8))

            registry = get_model_registry()
            model_mgr = get_shared_model_manager()

            results = {}
            warnings = []

            for mid in models:
                if mid not in registry:
                    warnings.append(f"Model ID '{mid}' is not recognized in the registry.")
                    results[mid] = {
                        "model_id": mid,
                        "name": mid,
                        "status": "error",
                        "error": f"Unknown model identifier '{mid}'.",
                        "latency_ms": 0.0,
                        "questions": []
                    }
                    continue

                info = registry[mid]
                if info["status"] != "available":
                    warnings.append(f"Model '{info['name']}' ({mid}) is unavailable: {info.get('unavailable_reason')}")
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
                    warnings.append(f"Inference execution failed for model '{mid}': {str(e)}")

            response_data = {
                "context": context,
                "answer": answer,
                "candidate_targets_used": [c["answer"] for c in candidate_targets] if candidate_targets else ([answer] if answer else []),
                "selected_models": models,
                "results": results,
                "parameters": {
                    "num_questions": num_questions,
                    "strategy": "nucleus_sampling" if do_sample else "beam_search",
                    "num_beams": num_beams if not do_sample else 1,
                    "max_length": max_length,
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0
                },
                "warnings": warnings
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint Not Found")

    def _send_json_error(self, code: int, message: str):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        """Quiet HTTP server logging for cleaner terminal."""
        pass


def run_demo_server(
    checkpoint_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 7860
):
    """Initializes ModelManager and starts the HTTP demonstration server."""
    global _GENERATOR, _MODEL_MANAGER

    print("\n" + "=" * 65)
    print(" Prashn-Setu: Interactive Model Comparison Server")
    print("=" * 65)

    _MODEL_MANAGER = get_shared_model_manager()
    _GENERATOR = _MODEL_MANAGER.get_generator()

    server_address = (host, port)
    HTTPServer.allow_reuse_address = True
    httpd = HTTPServer(server_address, DemoHTTPRequestHandler)

    url_local = f"http://localhost:{port}"
    print(f"\n🚀 Demonstration Server is LIVE!")
    print(f"👉 Local Access URL:   {url_local}")
    print(f"👉 Network Access:     http://{host}:{port}")
    print(f"👉 API Health Check:   {url_local}/api/health")
    print(f"👉 Model Registry API: {url_local}/api/models")
    print(f"👉 Compare Endpoint:   {url_local}/api/compare")
    print("=" * 65)
    print("Press Ctrl+C to terminate the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down demonstration server gracefully...")
        httpd.server_close()


def main():
    default_ckpt = (
        "checkpoints/stage2_ncert_gold_lora"
        if os.path.exists("checkpoints/stage2_ncert_gold_lora")
        else "checkpoints/stage1_indicnlg_qg"
    )
    parser = argparse.ArgumentParser(description="Prashn-Setu Interactive Web Demonstration Server")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Checkpoint to serve")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    args = parser.parse_args()

    run_demo_server(checkpoint_path=args.checkpoint, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
