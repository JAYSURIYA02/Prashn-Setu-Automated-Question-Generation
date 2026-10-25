# Prashn-Setu: Native-Hindi Educational Dataset Curation Guide

## 1. Motivation & Scientific Rationale

Prashn-Setu's Stage 1 baseline was fine-tuned on `ai4bharat/IndicQuestionGeneration` (Hindi split), which is derived from automated translations of English SQuAD v1.1. While this provides a strong syntactic initialization on Devanagari text, it carries well-documented limitations:
- **Translationese:** Sentence structures frequently reflect English syntax and idiomatic usage (e.g., awkward phrasing around American football terms, transliterated names, or unnatural relative clauses).
- **Domain Mismatch:** SQuAD passages focus on general encyclopedia topics (Wikipedia), whereas school education requires pedagogical comprehension across Science, Mathematics, History, and Geography curricula.
- **Pedagogical Question Styles:** Native Hindi textbooks use specific interrogative patterns (e.g., `...किसे कहते हैं?`, `...से क्या तात्पर्य है?`, `...का क्या कार्य है?`) that rarely occur in translated SQuAD.

**Stage 2 Adaptation Objective:** Fine-tune the model on authentic, naturally authored Hindi textbook excerpts from NCERT and State Board curricula so that generated questions match classroom examinations and pedagogical standards.

---

## 2. Target Dataset Partitions & Scale

Before training Stage 2, the dataset must be curated and partitioned into strictly disjoint splits:

| Partition | Target Size | File Path | Purpose |
| :--- | :--- | :--- | :--- |
| **Gold Training Set** | **300–500 examples** | `data/ncert_gold_train.jsonl` | Curriculum adaptation via Parameter-Efficient Fine-Tuning (LoRA) |
| **Gold Validation Set** | **50–80 examples** | `data/ncert_gold_val.jsonl` | Early stopping, learning-rate tuning, and checkpoint selection |
| **Held-Out Test Set** | **100–150 examples** | `data/ncert_gold_test.jsonl` | Final benchmark of native-Hindi educational question generation |

> [!IMPORTANT]
> **Data Integrity Rule:** The held-out test set (`ncert_gold_test.jsonl`) must NEVER be seen during training, hyperparameter search, or checkpoint selection. Chapters used in the test set should ideally come from separate textbooks or distinct chapters from those in the training set to prevent context leakage.

---

## 3. Approved Sources & Grade Levels

Curators should source material exclusively from official, published Hindi-medium school textbooks:

1. **NCERT Textbooks (राष्ट्रीय शैक्षिक अनुसंधान और प्रशिक्षण परिषद):**
   - **Science (विज्ञान):** Classes 6, 7, 8, 9, 10
   - **Social Science (सामाजिक विज्ञान):**
     - History (हमारे अतीत / भारत और समकालीन विश्व)
     - Geography (पृथ्वी: हमारा आवास / समकालीन भारत)
     - Civics / Political Science (सामाजिक एवं राजनीतिक जीवन / लोकतांत्रिक राजनीति)
   - **Environmental Studies (आस-पास):** Classes 4, 5
2. **State Board Textbooks (SCERT):**
   - Uttar Pradesh Basic Education Board (बेसिक शिक्षा परिषद)
   - Madhya Pradesh Board (मध्य प्रदेश पाठ्यपुस्तक निगम)
   - Rajasthan Board of Secondary Education (RBSE)
   - Bihar State Textbook Publishing Corporation (BSTBPC)

---

## 4. Record Schema & Specification

Every record in the JSONL files must adhere to the following schema:

```json
{
  "id": "ncert_sci_g07_ch01_001",
  "context": "पौधे सूर्य के प्रकाश की सहायता से अपना भोजन स्वयं बनाते हैं। इस प्रक्रिया को प्रकाश संश्लेषण कहते हैं। इसमें पत्तियां क्लोरोफिल की उपस्थिति में जल और कार्बन डाइऑक्साइड का उपयोग करती हैं।",
  "answer": "प्रकाश संश्लेषण",
  "question": "पौधे जिस प्रक्रिया द्वारा सूर्य के प्रकाश की उपस्थिति में अपना भोजन स्वयं बनाते हैं, उसे क्या कहते हैं?",
  "question_type": "conceptual",
  "grade": "Class 7",
  "subject": "Science",
  "chapter": "पादपों में पोषण",
  "source": "NCERT Class 7 Science Chapter 1, Page 2",
  "verification_status": "verified_human"
}
```

### Field Definitions

| Field Name | Type | Required | Description | Example / Values |
| :--- | :--- | :--- | :--- | :--- |
| `id` | String | Yes | Unique hierarchical identifier: `<source>_<subject_abbr>_g<grade>_ch<chapter>_<seq>` | `"ncert_sci_g07_ch01_001"` |
| `context` | String | Yes | Passage text in clean Devanagari Hindi (50–250 words) | Authentic textbook paragraph |
| `answer` | String | Yes | Target answer string. Must be an exact or near-exact substring of the context. | `"प्रकाश संश्लेषण"` |
| `question` | String | Yes | Well-formed question in natural Hindi asking for the answer. | Natural interrogative sentence |
| `question_type` | String | Yes | Pedagogical taxonomy: `factual`, `conceptual`, `definition`, `causal`, `procedural` | `"conceptual"` |
| `grade` | String | Yes | Target grade level | `"Class 6"`, `"Class 7"`, `"Class 8"`, `"Class 9"`, `"Class 10"` |
| `subject` | String | Yes | Subject name in English or Hindi | `"Science"`, `"History"`, `"Geography"`, `"Civics"` |
| `chapter` | String | Yes | Chapter title in Hindi | `"पादपों में पोषण"` |
| `source` | String | Yes | Exact bibliographic citation | `"NCERT Class 7 Science Chapter 1, Page 2"` |
| `verification_status`| String | Yes | Review status: must be `"verified_human"` after manual review; `"unverified"` otherwise | `"verified_human"` |

---

## 5. Question Type Taxonomy & Distribution

To ensure balanced pedagogical training, target the following distribution across each split:

| Question Type | Share | Typical Hindi Interrogative Forms | Pedagogical Purpose |
| :--- | :--- | :--- | :--- |
| **Factual / Entity** | 30% | `किसने...`, `कहाँ...`, `किस वर्ष / कब...`, `कितने...` | Testing recall of dates, names, venues, and measurements. |
| **Definitional / Terminology** | 25% | `...किसे कहते हैं?`, `...से क्या तात्पर्य है?`, `...की क्या परिभाषा है?` | Testing mastery of core scientific and historical terms. |
| **Conceptual / Mechanism** | 25% | `...का क्या कार्य है?`, `...क्या भूमिका निभाता है?`, `...का क्या महत्व है?` | Testing understanding of biological, physical, or social systems. |
| **Causal / Reasoning** | 20% | `...क्यों होता है?`, `...का क्या कारण था?`, `...किस प्रकार प्रभावित करता है?` | Testing cause-and-effect reasoning and analytical comprehension. |

---

## 6. Passage Extraction Guidelines

1. **Self-Contained Passages:** Each passage must contain all background facts necessary to answer the question. Avoid passages with unresolved pronouns referencing previous sections (e.g., `उसने फिर यह किया...` without introducing the subject).
2. **Length Constraints:** 50 to 250 words (typically 3 to 7 sentences). Passages shorter than 30 words lack sufficient context; passages longer than 300 words exceed standard reading-comprehension units.
3. **Punctuation & Formatting:**
   - Preserve standard Hindi punctuation: *purna viram* (`।`), commas (`,`), quotes (`""`), and hyphens.
   - Retain mathematical and scientific notations where appropriate ($CO_2$, $H_2O$, $1857$).
   - Avoid OCR artifacts, broken ligatures, or encoding corruption (e.g. invalid UTF-8 codepoints).

---

## 7. Quality Verification Rubric (Accept / Reject Rules)

Every candidate example must be reviewed by a human annotator against 5 mandatory pass/fail criteria:

### Check 1: Answer-Context Grounding
- **PASS:** The supplied `answer` string is explicitly present in the `context` (or is a direct morphological variant).
- **FAIL:** The answer requires outside knowledge not present in the passage.

### Check 2: Question-Answer Alignment
- **PASS:** Reading only the question and passage, the target answer is the single most natural and direct answer.
- **FAIL:** The question asks for a different entity in the passage, or is so vague that multiple contradictory answers fit equally well.

### Check 3: Interrogative Form Consistency
- **PASS:** The question word matches the answer entity:
  - Person $\rightarrow$ `किसने / कौन`
  - Place $\rightarrow$ `कहाँ / किस स्थान पर`
  - Time / Year $\rightarrow$ `कब / किस वर्ष`
  - Quantity $\rightarrow$ `कितने / कितना`
  - Term $\rightarrow$ `क्या / किसे कहते हैं`
- **FAIL:** Semantic mismatch (e.g., asking "कब" for a location, or "किसने" for a date).

### Check 4: Natural Devanagari Hindi Phrasing
- **PASS:** Natural Hindi syntax that reads like an authentic school examination paper.
- **FAIL:** Obvious machine-translation artifact, unidiomatic English word order, or severe grammatical discord in subject-verb agreement (*ling-vachan*).

### Check 5: Sentence Termination & Punctuation
- **PASS:** Terminated cleanly with a question mark (`?`).
- **FAIL:** Truncated sentence, trailing ellipses, or terminating with a *purna viram* (`।`) instead of a question mark.

---

## 8. Curation & Annotation Workflow

```
[Authentic Textbook Source]
         │
         ▼
 1. Extract Self-Contained Passage (50–200 words)
         │
         ▼
 2. Identify Salient Answer Entity / Concept
         │
         ▼
 3. Draft Natural Question targeting that Answer
         │
         ▼
 4. Apply 5-Point Quality Verification Rubric
         │
    ┌────┴────┐
    │ Passed? │
    └────┬────┘
     Yes │      No ──> Revise or Discard
         ▼
 5. Assign Complete Metadata (Grade, Subject, Source, Verification Status)
         │
         ▼
 6. Run Automated Dataset Validation Script (`scripts/validate_dataset.py`)
         │
         ▼
 7. Commit to Partition (`ncert_gold_train.jsonl` or `ncert_gold_test.jsonl`)
```
