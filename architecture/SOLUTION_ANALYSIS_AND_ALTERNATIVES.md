# Solution Analysis & Benchmark Against Alternatives

This document addresses the question: **"Is IndicBART the best solution so far, or what other solutions exist?"**

We evaluate the current Prashn-Setu architecture against alternative model families, generation paradigms, and deployment strategies.

---

## 1. Executive Assessment: Is IndicBART the Best Solution?

The answer depends on the operational constraints:

| Objective / Constraint | Best Solution | Verdict on IndicBART |
| :--- | :--- | :--- |
| **Low-resource Academic NLP Project** | **IndicBART** (with LoRA) | **Optimal.** Runs on a free Google Colab T4 GPU (16 GB VRAM) with fast training cycles and low memory overhead. |
| **Edge / Local Deployment (School Tablets/Laptops)** | **IndicBART (Pruned)** | **Optimal.** At ~190M parameters when pruned, it consumes < 800 MB RAM and operates under 200ms per question without requiring expensive GPUs. |
| **Deep Conceptual / Analytical Questions (Bloom's Level 3-5)** | **Sarvam-2B / Llama-3-8B** | **Superior.** 8B autoregressive LLMs exhibit deeper reasoning, enabling questions requiring synthesis rather than span retrieval. |
| **Turnkey Zero-Training Pipeline** | **Gemini 1.5 Flash / GPT-4o** | **Superior in fluency**, but incurs ongoing per-token API costs, requires persistent internet connectivity, and cannot be fine-tuned or hosted on-premise without significant infrastructure. |

**Conclusion:** For a self-contained, reproducible, cost-effective, and fully offline Hindi educational question generator, **IndicBART adapted via a two-stage transfer pipeline (IndicNLG $\to$ NCERT Gold with LoRA) is currently the most practical engineering solution.**

---

## 2. Multi-Model Comparison Matrix

| Model / Approach | Architecture | Parameters | VRAM (Training) | Indic Inductive Bias | Question Quality (Grammar) | Inference Latency | Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IndicBART (Prashn-Setu)** | Encoder-Decoder (Seq2Seq) | 244M (~190M pruned) | ~6–8 GB | High (Pretrained on IndicCorp) | High (Post-NCERT adaptation) | ~150 ms | Free / Open-source |
| **IndicBARTSS** | Encoder-Decoder (Seq2Seq) | 244M | ~8 GB | High (Native script) | High | ~150 ms | Free / Open-source |
| **MultiIndicQuestionGenerationSS** | Pre-fine-tuned IndicBARTSS | 244M | None (Zero-shot) | High | Moderate (Translationese bias) | ~150 ms | Free / Open-source |
| **google/mt5-base** | Encoder-Decoder (T5) | 580M | ~14–16 GB | Moderate (mC4 corpus) | Moderate (High hallucination rate) | ~400 ms | Free / Open-source |
| **Sarvam-2B** | Autoregressive (Decoder) | 2 Billion | ~12 GB (with QLoRA) | Very High (Trained on 10 Indic languages) | Very High | ~350 ms | Free / Open-source |
| **Airavata / Llama-3-8B-Indic** | Autoregressive (Decoder) | 8 Billion | ~18–24 GB (requires A100 or 4-bit) | Moderate to High | Very High | ~800 ms | Heavy GPU required |
| **Commercial APIs (Gemini / GPT-4o)** | Proprietary LLM | Hundreds of Billions | Cloud Hosted | High | Very High | ~1–2 sec | ~$0.001 - $0.01 per query |

---

## 3. In-Depth Analysis of Competing Solutions

### Alternative 1: `ai4bharat/IndicBARTSS` vs. `ai4bharat/IndicBART`
* **Difference:** IndicBART maps all scripts to Devanagari internally, while IndicBARTSS maintains separate native scripts.
* **Why it matters for Hindi:** Hindi is *already* natively written in Devanagari. Therefore, IndicBART incurs **no transliteration overhead** for Hindi. 
* **Advantage of IndicBART over IndicBARTSS for Hindi:** Pruning the 8 non-Devanagari scripts from IndicBART is straightforward and leaves a clean, compact Devanagari model.

---

### Alternative 2: Using the Existing Pre-Trained `ai4bharat/MultiIndicQuestionGenerationSS`
* **What it is:** AI4Bharat already trained IndicBARTSS on `IndicQuestionGeneration` and published the checkpoint.
* **Why not stop there?**
  * `MultiIndicQuestionGenerationSS` was trained **only on the machine-translated IndicNLG benchmark**.
  * Consequently, it suffers directly from the **translationese defect** you identified in your problem statement (e.g., generating literal calques like *"यह प्रक्रिया क्या बुलाया जाता है?"*).
  * **Our Innovation:** Using `MultiIndicQuestionGenerationSS` (or our Stage 1 model) merely as the starting point, and adapting it on a curated gold dataset of real NCERT textbook passages to correct its syntax toward authentic textbook language.

---

### Alternative 3: `google/mt5` (Multilingual T5)
* **Shortcomings for Indic Question Generation:**
  1. **Vocabulary bloat:** mT5 uses a 250,000-token vocabulary. A massive portion of model parameters is tied up in embedding matrices, making it memory-inefficient.
  2. **Accidental English / Cross-lingual drift:** mT5 is notorious for code-switching into English or unrelated languages when generating text for low-to-medium resource languages.
  3. **High resource footprint:** `mt5-base` (580M) requires more than twice the VRAM of IndicBART (244M) without delivering superior Hindi syntax.

---

### Alternative 4: Small Autoregressive Models (Sarvam-2B, Llama-3-8B-Instruct with QLoRA)
* **Strengths:**
  * Decoder-only LLMs understand broader world context and can generate high-order conceptual questions (e.g., *"यदि प्रकाश संश्लेषण न हो, तो पृथ्वी के पारिस्थितिकी तंत्र पर क्या प्रभाव पड़ेगा?"*).
* **Weaknesses:**
  * Autoregressive models are prone to hallucinating facts not present in the textbook passage.
  * Seq2Seq models (IndicBART) have an explicit bidirectional encoder that tightly conditions generation on the passage, significantly reducing hallucinations.
  * 8B models require 16–24 GB VRAM for 4-bit fine-tuning and cannot easily run on standard school hardware.

---

### Alternative 5: Rule-Based / Dependency Parsing QG
* **Mechanism:** Uses an Indic dependency parser to identify Subject-Verb-Object triples and applies grammar transformation templates (e.g., replacing Subject with *"किसने"* or Object with *"किसे"*).
* **Limitations:**
  * Highly brittle. Fails completely on compound or complex Hindi sentences.
  * Incapable of generating conceptual ("*क्यों*", "*कैसे*") or summary questions.

---

## 4. Architectural Trade-offs in Question Generation Formulation

### 1. Answer-Aware vs. Answer-Agnostic
```mermaid
graph LR
    subgraph Answer-Aware (Current Setup)
        A1["Context + Target Answer Span"] --> B1["IndicBART Decoder"] --> C1["Question targeting specific answer"]
    end
    subgraph Answer-Agnostic (Alternative)
        A2["Context Only"] --> B2["IndicBART Decoder"] --> C2["Generic or multiple questions"]
    end
```
* **Answer-Aware (Our approach):**
  * **Pros:** Highly precise; ensures generated question directly targets an examinable fact or concept.
  * **Cons:** Requires the user (teacher) to specify the answer or requires an upstream keyword extractor.
* **Answer-Agnostic:**
  * **Pros:** One-click generation directly from a paragraph.
  * **Cons:** High entropy; the model may generate a question targeting an insignificant trivial date or prepositional phrase rather than the core pedagogical concept.

---

## 5. Summary Recommendation for Your Project

1. **Keep IndicBART as the core generation engine:** It provides the optimal balance of parameter efficiency, native Indic pretraining, low hallucination rate, and edge deployability.
2. **Execute Vocabulary Pruning:** Trimming other scripts reduces model size and prevents cross-script leaks.
3. **Use LoRA for Stage 2 Adaptation:** Guarantees that adapting on 300–500 NCERT samples refines question phrasing without destroying baseline grammatical competency.
4. **Future Expansion (v2.0):** Add a simple upstream keyphrase extractor (e.g., Hindi POS/NER tagger or TF-IDF keyword selector) to automatically supply answer spans to the pipeline, turning Prashn-Setu into a fully autonomous end-to-end textbook test generator.
