# Prashn-Setu: Human Evaluation Protocol & Guidelines

## 1. Objective

This protocol provides a reproducible, rigorous framework for human assessment of Indic Question Generation (Hindi) models. Automatic metrics (SacreBLEU, ROUGE, chrF++) measure surface n-gram and character overlap against a single reference question. However:
- A passage often admits multiple valid, pedagogically useful questions for a given answer.
- A model may generate a grammatically fluent, factually sound question that uses different wording than the reference, yielding low BLEU/ROUGE despite high pedagogical value.
- Conversely, a model may hallucinate false facts or mismatch question words (e.g. asking "when" for a team or person) while achieving partial n-gram overlap.

Therefore, human evaluation is essential to assess semantic validity, factual faithfulness, and educational utility.

---

## 2. Sampling Methodology

- **Test Population:** 10,553 held-out test examples from `data/hi_test.jsonl` (IndicQuestionGeneration Hindi split).
- **Sample Size:** 100 examples ($N=100$).
- **Sampling Strategy:** Stratified pseudo-random sampling across passage lengths and answer types using fixed seed `42` (`random.seed(42)`).
- **File Artifact:** `results/human_eval_sample_100.csv`

---

## 3. Evaluation Dimensions & Scoring Rubric

Each evaluator rates each generated question along 5 distinct dimensions:

### 1. Answer Alignment (1–5)
*Does the generated question logically lead to the supplied answer?*
- **5 (Perfect):** The question unambiguously asks for the target answer. The target answer directly and naturally answers the question.
- **4 (Good):** The question asks for the target answer, but might be slightly broader or could admit close synonyms.
- **3 (Partial):** The question is related to the answer topic, but the target answer is only a partial or awkward fit.
- **2 (Poor):** The question asks about something else in the passage; the answer is barely relevant.
- **1 (Unacceptable):** The question asks for an entity type completely incompatible with the answer (e.g., asking "When?" for a person/team, or "Where?" for a date).

### 2. Context Faithfulness & Grounding (1–5)
*Are the premises and relationships in the question fully supported by the passage?*
- **5 (Fully Supported):** Every entity, relation, number, and event mentioned in the question is explicitly verified by the passage.
- **4 (Mostly Supported):** Minor stylistic paraphrasing, but no unsupported facts or assumptions.
- **3 (Extrapolated):** Question introduces minor unverified details not in the context, though not directly contradictory.
- **2 (Partially Contradicted):** Question alters an important detail (e.g., changes a score, date, or role).
- **1 (Fabricated/Contradicted):** Hallucinates entities, events, or facts directly refuted by the passage.

### 3. Grammaticality & Naturalness (1–5)
*Is the question well-formed, natural Hindi?*
- **5 (Native/Flawless):** Natural Devanagari Hindi syntax, correct postpositions (*karak*), agreement in gender/number (*ling/vachan*), and natural phrasing.
- **4 (Minor Flaw):** Minor awkwardness in word order or translationese, but easily understood without ambiguity.
- **3 (Noticeable Errors):** Noticeable inflectional or syntactic errors, but the core meaning remains decipherable.
- **2 (Severely Degraded):** Major grammatical breakdowns, missing essential verbs or arguments.
- **1 (Incomprehensible):** Word salad, severe token repetition, or unreadable fragment.

### 4. Answerability (1–5)
*Can a student answer this question solely using the provided passage?*
- **5 (Directly Answerable):** Context contains clear, sufficient information to answer the question without external knowledge.
- **4 (Inferable):** Requires minor common-sense deduction directly from the context facts.
- **3 (Ambiguous):** Context leaves multiple potential answers plausible.
- **2 (Insufficient Information):** Crucial details required to answer are absent from the context.
- **1 (Unanswerable):** Cannot be answered from the text under any interpretation.

### 5. Factual Hallucination (Binary: Yes / No)
*Did the model introduce, fabricate, or distort a factual entity, date, number, or relation?*
- **No:** All numbers, names, and relations match context.
- **Yes:** Model altered numbers (e.g. 24-10 became 24-25), swapped names, or invented facts.

### 6. Overall Pedagogical Quality (1–5)
*Overall assessment of whether this question could appear on an exam or in a textbook study guide:*
- **5:** Excellent question, ready for classroom/textbook use.
- **4:** Good question, would require only trivial editorial polish.
- **3:** Acceptable for informal practice, but flawed for formal exams.
- **2:** Flawed, confusing, or misleading.
- **1:** Completely unusable.

---

## 4. Evaluator Qualification & Inter-Annotator Agreement

- **Evaluator Criteria:** Native or near-native Hindi speakers with at least undergraduate-level literacy in Hindi.
- **Double Annotation:** At least 20% of the sample (20 items) should be dual-annotated.
- **Agreement Metric:** Cohen's Kappa ($\kappa$) for categorical/binary judgments (Hallucination) and Intraclass Correlation Coefficient (ICC) / Kendall's Tau for ordinal scales (1–5).

---

## 5. CSV Schema Definition

| Column | Type | Permitted Values | Description |
| :--- | :--- | :--- | :--- |
| `example_id` | String | e.g. `1`, `2` | Split-level example identifier |
| `squad_id` | String | Hex string | Unique SQuAD provenance ID |
| `context` | String | Devanagari text | Passage excerpt provided to model |
| `answer` | String | Devanagari text | Supplied target answer entity |
| `reference_question` | String | Devanagari text | Human reference ground truth |
| `generated_question` | String | Devanagari text | Model generation output |
| `answer_alignment_1to5` | Integer | `1`, `2`, `3`, `4`, `5` | Alignment score |
| `context_faithfulness_1to5` | Integer | `1`, `2`, `3`, `4`, `5` | Faithfulness score |
| `grammaticality_1to5` | Integer | `1`, `2`, `3`, `4`, `5` | Fluency and grammar |
| `answerability_1to5` | Integer | `1`, `2`, `3`, `4`, `5` | Text-based answerability |
| `factual_hallucination_yes_no` | Categorical | `Yes`, `No` | Hallucination presence |
| `overall_quality_1to5` | Integer | `1`, `2`, `3`, `4`, `5` | Overall rating |
| `reviewer_notes` | String | Free text | Specific qualitative observations |
