# Solution Analysis & Benchmark Against Alternatives

This document addresses the core design question: **"Is IndicBART the best solution so far, or what other solutions exist?"**

We evaluate the current Prashn-Setu architecture against alternative model families, generation paradigms, and deployment strategies.

---

## 1. Executive Assessment: Is IndicBART the Best Solution?

The answer depends on operational and resource constraints:

| Objective / Constraint | Best Solution | Verdict on IndicBART |
| :--- | :--- | :--- |
| **Academic NLP Course Project** | **IndicBART** (with LoRA) | **Optimal.** Runs on commodity hardware or CPU/T4 GPU with low memory footprint (~945 MB on disk) and sub-second generation latency. |
| **Edge / Local Classroom Deployment (School Laptops)** | **IndicBART** | **Optimal.** At 244M parameters, it operates on CPUs without requiring expensive GPU infrastructure. |
| **Deep Conceptual / Analytical Questions (Bloom's Level 3–5)** | **Sarvam-2B / Llama-3-8B** | **Superior in reasoning**, but requires 16–24 GB VRAM for fine-tuning and has higher propensity for factual hallucinations. |
| **Zero-Training Turnkey API** | **Gemini 1.5 Flash / GPT-4o** | **Superior in surface fluency**, but incurs recurring per-query API costs, requires continuous internet access, and lacks fine-grained control over local textbook data privacy. |

**Conclusion:** For a self-contained, reproducible, cost-effective, and fully offline Hindi educational question generation system, **IndicBART adapted via transfer learning and Parameter-Efficient Fine-Tuning (LoRA) is the most practical engineering solution.**

---

## 2. Multi-Model Comparison Matrix

| Model / Approach | Architecture | Parameters | VRAM (Training) | Indic Inductive Bias | Question Quality (Hindi) | Inference Latency | Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IndicBART (Prashn-Setu)** | Encoder-Decoder (Seq2Seq) | 244M | ~6–8 GB (BF16) | High (Pretrained on IndicCorp) | High (Grounded syntax) | ~150–350 ms | Free / Open-source |
| **IndicBARTSS** | Encoder-Decoder (Seq2Seq) | 244M | ~8 GB | High (Native script) | High | ~150–350 ms | Free / Open-source |
| **google/mt5-base** | Encoder-Decoder (T5) | 580M | ~14–16 GB | Moderate (mC4 corpus) | Moderate (Cross-lingual drift) | ~400–600 ms | Free / Open-source |
| **Sarvam-2B** | Autoregressive (Decoder) | 2 Billion | ~12 GB (QLoRA) | Very High (10 Indic languages) | Very High | ~350 ms | Free / Open-source |
| **Llama-3-8B-Indic / Airavata** | Autoregressive (Decoder) | 8 Billion | ~18–24 GB | Moderate to High | Very High | ~800 ms | Heavy GPU required |
| **Commercial APIs (Gemini / GPT-4o)** | Proprietary LLM | Hundreds of Billions | Cloud Hosted | High | Very High | ~1–2 sec | ~$0.001–$0.01 per query |

---

## 3. In-Depth Analysis of Competing Solutions

### Alternative 1: `ai4bharat/IndicBARTSS` vs. `ai4bharat/IndicBART`
* **Difference:** IndicBART maps all scripts to Devanagari internally, while IndicBARTSS maintains separate native scripts.
* **Why it matters for Hindi:** Hindi is already natively written in Devanagari. Therefore, IndicBART incurs **no transliteration overhead** for Hindi.
* **Advantage:** Devanagari vocabulary mapping is compact, eliminating cross-script token bloat.

---

### Alternative 2: Using the Existing Pre-Trained `ai4bharat/MultiIndicQuestionGenerationSS`
* **What it is:** AI4Bharat already trained IndicBARTSS on `IndicQuestionGeneration` and published the checkpoint.
* **Why not stop there?**
  * `MultiIndicQuestionGenerationSS` was trained **only on machine-translated SQuAD**.
  * Consequently, it suffers directly from the **translationese defect** (generating awkward calques like *"यह प्रक्रिया क्या बुलाया जाता है?"*).
  * **Our Approach:** Using the Stage 1 model as a general syntactic base, and injecting a parameter-efficient LoRA adapter trained on authentic textbook phrasing to align generation with curriculum standards.

---

### Alternative 3: `google/mt5` (Multilingual T5)
* **Shortcomings for Indic Question Generation:**
  1. **Vocabulary bloat:** mT5 uses a 250,000-token vocabulary where a large fraction of parameters are tied up in embeddings.
  2. **Cross-lingual drift:** mT5 is prone to code-switching into English or unrelated scripts during generation.
  3. **High resource footprint:** `mt5-base` (580M) requires more than twice the memory of IndicBART (244M) without delivering superior Hindi grammar.

---

### Alternative 4: Small Autoregressive Models (Sarvam-2B, Llama-3-8B-Instruct)
* **Strengths:**
  * Decoder-only LLMs understand broader world context and can generate high-order conceptual questions (Bloom's Taxonomy Level 4–5).
* **Weaknesses:**
  * Prone to hallucinating facts not grounded in the textbook passage.
  * Seq2Seq models (IndicBART) have an explicit bidirectional encoder that tightly conditions generation on the context, drastically reducing hallucinations.
  * 8B models require expensive GPU hardware and cannot run locally on standard classroom devices.

---

## 4. Architectural Trade-offs in Question Generation Formulation

### Answer-Aware vs. Answer-Agnostic Generation

```mermaid
graph LR
    subgraph Answer-Aware (Prashn-Setu Primary Formulation)
        A1["Context + Target Answer Span"] --> B1["IndicBART Decoder"] --> C1["Question targeting specific answer"]
    end
    subgraph Answer-Agnostic (Fallback Formulation)
        A2["Context Only"] --> B2["IndicBART Decoder"] --> C2["Question about main premise"]
    end
```

* **Answer-Aware (Our Primary Approach):**
  * **Pros:** Highly precise; ensures the generated question directly targets an examinable fact, scientific definition, or key date.
  * **Cons:** Requires the user to provide an answer or requires an automated candidate-answer extraction heuristic.
  * **Prashn-Setu Resolution:** Built an automated candidate answer extractor (`extract_candidate_answers`) using Devanagari entity heuristics, dates, and noun-phrase detection. When the user leaves the answer field blank, the system automatically extracts the top salient candidate answers and generates questions for each.

* **Sentence-Level Grounding:**
  * Surface n-gram metrics (BLEU) penalize valid paraphrasing. Prashn-Setu pairs generation with an explicit lexical overlap answerability checker (`evidence_checker.py`), classifying each question into `evidence_found`, `needs_review`, or `unsupported` and providing character-level offset mapping for inline highlighting.
