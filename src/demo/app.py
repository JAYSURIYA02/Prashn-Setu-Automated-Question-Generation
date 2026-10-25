#!/usr/bin/env python3
"""
Prashn-Setu: Interactive Web Demonstration Server.
A lightweight, zero-dependency browser interface powered by Python's built-in http.server.
Supports single-question and passage-level multi-question generation from Hindi educational
textbooks with automated candidate answer extraction, parameter tuning, and REST API.

Usage:
  python run.py demo --port 7860
  python src/demo/app.py --port 7860 --checkpoint checkpoints/stage1_indicnlg_qg
"""

import os
import sys
import json
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.inference import QuestionGenerator

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
            max-width: 980px;
            margin: 0 auto;
        }
        header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 32px 28px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
            margin-bottom: 24px;
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
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: -0.5px;
        }
        header p {
            color: #dbeafe;
            font-size: 14.5px;
            max-width: 760px;
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
        textarea, input[type="text"], select {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: inherit;
            font-size: 14.5px;
            color: var(--text);
            background: #fff;
            transition: border-color 0.15s, box-shadow 0.15s;
        }
        textarea:focus, input[type="text"]:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
        }
        textarea {
            resize: vertical;
            min-height: 120px;
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
        .controls-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            background: #f1f5f9;
            padding: 14px;
            border-radius: 8px;
            margin-bottom: 16px;
        }
        .controls-row label {
            font-size: 12px;
            color: var(--text);
            margin-bottom: 4px;
        }
        .controls-row input, .controls-row select {
            padding: 8px 10px;
            font-size: 13.5px;
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
        .result-box {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #e2e8f0;
        }
        .result-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .count-badge {
            background: #dbeafe;
            color: var(--primary);
            font-size: 12px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 12px;
        }
        .questions-container {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .question-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: transform 0.1s, box-shadow 0.1s;
        }
        .question-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        }
        .q-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
        }
        .q-number {
            font-size: 12px;
            font-weight: 700;
            color: var(--primary);
            background: var(--primary-light);
            padding: 2px 8px;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 6px;
        }
        .q-text {
            font-size: 17.5px;
            font-weight: 600;
            color: var(--text);
            line-height: 1.5;
            font-family: 'Noto Sans Devanagari', sans-serif;
            margin-bottom: 8px;
        }
        .q-ans-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 6px;
            font-size: 13.5px;
        }
        .ans-badge {
            background: #fef3c7;
            color: #92400e;
            font-size: 12.5px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 6px;
            border: 1px solid #fde68a;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .q-source {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 6px;
            font-style: italic;
        }
        .copy-btn {
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            color: var(--text);
            white-space: nowrap;
            transition: all 0.15s;
        }
        .copy-btn:hover {
            background: #e2e8f0;
            border-color: #94a3b8;
        }
        .result-meta {
            font-size: 12.5px;
            color: var(--text-muted);
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }
        .alert {
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13.5px;
            margin-top: 12px;
            display: none;
        }
        .alert-error {
            background: #fef2f2;
            color: var(--danger);
            border: 1px solid #fecaca;
        }
        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 3px solid white;
            width: 18px;
            height: 18px;
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        footer {
            text-align: center;
            font-size: 12.5px;
            color: var(--text-muted);
            margin-top: 24px;
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
            <p>Automated Hindi Question Generation from Indic Educational Textbooks &bull; Generates single or multiple curriculum-grounded questions across entire passages.</p>
        </header>

        <div class="card">
            <div class="card-header">
                <span class="card-title">📖 प्रामाणिक पाठ्यपुस्तक उदाहरण (NCERT Preset Examples)</span>
                <span style="font-size: 12px; color: var(--text-muted);">क्लिक करके प्रसंग भरें</span>
            </div>
            <div class="preset-btns" id="presetContainer"></div>

            <form id="qgForm" novalidate onsubmit="handleGenerate(event); return false;">
                <div class="form-group">
                    <label for="contextInput">हिंदी प्रसंग / पाठ्यपुस्तक अंश (Hindi Context Passage) *</label>
                    <textarea id="contextInput" placeholder="पाठ्यपुस्तक का एक या अधिक अनुच्छेद यहाँ दर्ज करें..."></textarea>
                </div>

                <div class="form-group">
                    <label for="answerInput">लक्षित उत्तर / मुख्य वाक्यांश (Target Answer — Optional)</label>
                    <input type="text" id="answerInput" placeholder="उदा. नई दिल्ली, १० मई १८५७ (खाली छोड़ने पर सिस्टम स्वतः महत्वपूर्ण तथ्यों पर कई प्रश्न बनाएगा)">
                </div>

                <div class="controls-row">
                    <div>
                        <label for="numQuestions">प्रश्नों की संख्या (Count)</label>
                        <select id="numQuestions">
                            <option value="1">1 प्रश्न (Single)</option>
                            <option value="3" selected>3 प्रश्न (Standard)</option>
                            <option value="5">5 प्रश्न (Comprehensive)</option>
                            <option value="8">8 प्रश्न (Full Quiz)</option>
                        </select>
                    </div>
                    <div>
                        <label for="decodingStrategy">डिकोडिंग विधि (Strategy)</label>
                        <select id="decodingStrategy">
                            <option value="beam" selected>Beam Search (किरणें / सटीक)</option>
                            <option value="sample">Nucleus Sampling (विविध / रचनात्मक)</option>
                        </select>
                    </div>
                    <div id="beamGroup">
                        <label for="numBeams">Beam Size (किरणें: 1-8)</label>
                        <input type="number" id="numBeams" min="1" max="8" value="4">
                    </div>
                    <div id="tempGroup" style="display:none;">
                        <label for="temperature">तापमान (Temp: 0.1-2.0)</label>
                        <input type="number" id="temperature" min="0.1" max="2.0" step="any" value="0.7">
                    </div>
                    <div id="topPGroup" style="display:none;">
                        <label for="topP">Top-P (0.1-1.0)</label>
                        <input type="number" id="topP" min="0.1" max="1.0" step="any" value="0.92">
                    </div>
                    <div>
                        <label for="maxLength">अधिकतम लंबाई (Max Tokens)</label>
                        <input type="number" id="maxLength" min="16" max="128" value="64">
                    </div>
                    <div>
                        <label for="noRepeatNgram">नो-रिपीट N-gram</label>
                        <input type="number" id="noRepeatNgram" min="0" max="4" value="3">
                    </div>
                </div>

                <div style="font-size: 11.5px; color: var(--text-muted); margin-bottom: 14px;">
                    💡 <em>नोट: 'Beam Search' सर्वाधिक संभावित सटीक प्रश्न चुनता है। विभिन्न वाक्यांशों व रचनात्मक विविधता के लिए 'Nucleus Sampling' चुनें और तापमान (0.7 - 1.0) बढ़ाएं।</em>
                </div>

                <button type="button" id="generateBtn" class="btn-primary" onclick="handleGenerate(event)">
                    <span id="btnText">✨ प्रश्न उत्पन्न करें (Generate Questions)</span>
                </button>
            </form>

            <div id="errorAlert" class="alert alert-error"></div>

            <div id="resultBox" class="result-box">
                <div class="result-header">
                    <div class="result-title">
                        <span>📝 उत्पन्न प्रश्न सूची (Generated Questions)</span>
                        <span id="qCountBadge" class="count-badge"></span>
                    </div>
                    <button type="button" class="copy-btn" id="copyAllBtn" onclick="copyAllQuestions()">📋 Copy All</button>
                </div>

                <div id="questionsList" class="questions-container"></div>

                <div class="result-meta">
                    <span id="metaLatency"></span>
                    <span id="metaCount"></span>
                    <span id="metaStrategy"></span>
                    <span id="metaModel">Model: IndicBART Stage 1 (244M)</span>
                </div>
            </div>
        </div>

        <footer>
            Prashn-Setu: Automated Question Generation from Indic Educational Textbooks &bull; 
            Powered by PyTorch &bull; AI4Bharat IndicBART
        </footer>
    </div>

    <script>
        const PRESETS = __PRESETS_JSON__;
        let CURRENT_QUESTIONS = [];

        // Safe parsers that handle 0 correctly (0 is falsy in JS)
        function safeInt(id, fallback) {
            const v = parseInt(document.getElementById(id).value, 10);
            return isNaN(v) ? fallback : v;
        }
        function safeFloat(id, fallback) {
            const v = parseFloat(document.getElementById(id).value);
            return isNaN(v) ? fallback : v;
        }

        // Preload first preset by default
        window.addEventListener('DOMContentLoaded', () => {
            if (PRESETS && PRESETS.length > 0) {
                const ctxElem = document.getElementById('contextInput');
                if (ctxElem && !ctxElem.value) {
                    ctxElem.value = PRESETS[0].context;
                }
            }
        });

        // Render presets
        const container = document.getElementById('presetContainer');
        PRESETS.forEach((p, idx) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'preset-btn';
            btn.textContent = p.title;
            btn.onclick = () => {
                document.getElementById('contextInput').value = p.context;
                document.getElementById('answerInput').value = p.answer || "";
                document.getElementById('resultBox').style.display = 'none';
                document.getElementById('errorAlert').style.display = 'none';
            };
            container.appendChild(btn);
        });

        // Strategy toggle
        const decodingStrategy = document.getElementById('decodingStrategy');
        const beamGroup = document.getElementById('beamGroup');
        const tempGroup = document.getElementById('tempGroup');
        const topPGroup = document.getElementById('topPGroup');

        decodingStrategy.addEventListener('change', () => {
            const isSampling = decodingStrategy.value === 'sample';
            beamGroup.style.display = isSampling ? 'none' : 'block';
            tempGroup.style.display = isSampling ? 'block' : 'none';
            topPGroup.style.display = isSampling ? 'block' : 'none';
        });

        // Robust Generation Handler
        async function handleGenerate(e) {
            if (e && e.preventDefault) e.preventDefault();

            const contextElem = document.getElementById('contextInput');
            const answerElem = document.getElementById('answerInput');
            const context = contextElem ? contextElem.value.trim() : "";
            const answer = answerElem ? answerElem.value.trim() : "";
            const isSampling = document.getElementById('decodingStrategy').value === 'sample';
            const numQuestions = safeInt('numQuestions', 3);
            const maxLength = safeInt('maxLength', 64);
            const noRepeatNgram = safeInt('noRepeatNgram', 3);
            // Strategy-specific: only read relevant params
            const numBeams = isSampling ? 1 : safeInt('numBeams', 4);
            const temperature = isSampling ? safeFloat('temperature', 0.7) : 1.0;
            const topP = isSampling ? safeFloat('topP', 0.92) : 1.0;

            const generateBtn = document.getElementById('generateBtn');
            const btnText = document.getElementById('btnText');
            const resultBox = document.getElementById('resultBox');
            const errorAlert = document.getElementById('errorAlert');

            if (!context) {
                showError("कृपया हिंदी प्रसंग (Context) दर्ज करें या ऊपर दिए गए NCERT उदाहरणों में से चुनें।");
                if (contextElem) contextElem.focus();
                return;
            }

            // UI state: loading
            generateBtn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span> प्रश्नों का सृजन हो रहा है...';
            errorAlert.style.display = 'none';
            resultBox.style.display = 'none';

            try {
                const response = await fetch('/api/generate', {
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

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Server error occurred during generation.');
                }

                // Render questions
                renderQuestions(data);
                resultBox.style.display = 'block';
            } catch (err) {
                showError("त्रुटि: " + err.message);
            } finally {
                generateBtn.disabled = false;
                btnText.innerHTML = '✨ प्रश्न उत्पन्न करें (Generate Questions)';
            }
        }

        document.getElementById('qgForm').addEventListener('submit', handleGenerate);

        function renderQuestions(data) {
            questionsList.innerHTML = "";
            let items = data.questions || [];
            if (!items.length && data.question) {
                items = [{ question: data.question, answer: data.answer || "" }];
            }
            CURRENT_QUESTIONS = items;

            qCountBadge.textContent = `${items.length} प्रश्न उपलब्ध`;
            document.getElementById('metaLatency').textContent = `⏱️ कुल समय (Latency): ${data.latency_ms.toFixed(1)} ms`;
            document.getElementById('metaCount').textContent = `🎯 उत्पन्न प्रश्न: ${items.length}`;
            const p = data.parameters || {};
            if (p.do_sample) {
                document.getElementById('metaStrategy').textContent = `🎲 Sampling (temp=${p.temperature}, top_p=${p.top_p})`;
            } else {
                document.getElementById('metaStrategy').textContent = `🔍 Beam Search (beams=${p.num_beams})`;
            }

            items.forEach((item, idx) => {
                const card = document.createElement('div');
                card.className = 'question-card';

                const ansHtml = item.answer 
                    ? `<div class="q-ans-row"><span class="ans-badge">🎯 लक्षित उत्तर: <strong>${escapeHtml(item.answer)}</strong></span></div>`
                    : '';

                const srcHtml = item.source_sentence
                    ? `<div class="q-source">📖 संदर्भ: "${escapeHtml(item.source_sentence)}"</div>`
                    : '';

                card.innerHTML = `
                    <div class="q-header">
                        <div>
                            <span class="q-number">प्रश्न ${idx + 1}</span>
                            <div class="q-text">${escapeHtml(item.question)}</div>
                            ${ansHtml}
                            ${srcHtml}
                        </div>
                        <button type="button" class="copy-btn" onclick="copySingleQuestion(${idx}, this)">📋 Copy</button>
                    </div>
                `;
                questionsList.appendChild(card);
            });
        }

        function showError(msg) {
            errorAlert.textContent = msg;
            errorAlert.style.display = 'block';
        }

        function copySingleQuestion(idx, btn) {
            const item = CURRENT_QUESTIONS[idx];
            if (!item) return;
            const textToCopy = item.answer 
                ? `प्रश्न: ${item.question}\\nउत्तर: ${item.answer}`
                : item.question;

            navigator.clipboard.writeText(textToCopy).then(() => {
                const orig = btn.textContent;
                btn.textContent = '✅ Copied!';
                setTimeout(() => { btn.textContent = orig; }, 2000);
            });
        }

        function copyAllQuestions() {
            if (!CURRENT_QUESTIONS.length) return;
            const fullText = CURRENT_QUESTIONS.map((item, i) => {
                let s = `${i + 1}. ${item.question}`;
                if (item.answer) s += `\\n   उत्तर: ${item.answer}`;
                return s;
            }).join("\\n\\n");

            navigator.clipboard.writeText(fullText).then(() => {
                const btn = document.getElementById('copyAllBtn');
                btn.textContent = '✅ All Copied!';
                setTimeout(() => { btn.textContent = '📋 Copy All'; }, 2000);
            });
        }

        function escapeHtml(text) {
            if (!text) return "";
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
        }
    </script>
</body>
</html>
"""

# Global generator instance for the server
_GENERATOR: Optional[QuestionGenerator] = None


class DemoHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the Prashn-Setu interactive demonstration."""

    def do_GET(self):
        """Serves UI HTML or health check endpoint."""
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
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "status": "healthy",
                "model_loaded": _GENERATOR is not None,
                "device": _GENERATOR.device if _GENERATOR else None
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        """Handles question generation REST API."""
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
                # Use multi-question generation engine with complete decoding controls
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
                    "question": first_q,  # Maintained for backward compatibility
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
    """Initializes QuestionGenerator and starts the HTTP demonstration server."""
    global _GENERATOR

    if checkpoint_path is None:
        checkpoint_path = (
            "checkpoints/stage2_ncert_gold_lora"
            if os.path.exists("checkpoints/stage2_ncert_gold_lora")
            else "checkpoints/stage1_indicnlg_qg"
        )

    print("\n" + "=" * 65)
    print(" Prashn-Setu: Interactive Demonstration Server")
    print("=" * 65)
    print(f"Loading checkpoint from: {checkpoint_path}...")
    _GENERATOR = QuestionGenerator(checkpoint_path=checkpoint_path)

    server_address = (host, port)
    HTTPServer.allow_reuse_address = True
    httpd = HTTPServer(server_address, DemoHTTPRequestHandler)

    url_local = f"http://localhost:{port}"
    print(f"\n🚀 Demonstration Server is LIVE!")
    print(f"👉 Local Access URL:   {url_local}")
    print(f"👉 Network Access:     http://{host}:{port}")
    print(f"👉 API Health Check:   {url_local}/api/health")
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
