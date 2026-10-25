# Prashn-Setu: Stage 1 Qualitative Evaluation & Systematic Error Analysis

## 1. Executive Summary

This report presents a systematic qualitative audit of the **Prashn-Setu Stage 1 Baseline** (`checkpoints/stage1_indicnlg_qg`), evaluated across 10,553 held-out test examples from the IndicQuestionGeneration Hindi dataset. 

While the quantitative evaluation reproduced the historical baseline (SacreBLEU: **10.196**, chrF++: **29.397**, Hindi-tokenized ROUGE-L: **27.332**), surface metrics alone fail to explain the true capabilities and failure modes of IndicBART on Hindi question generation. 

Our qualitative analysis demonstrates that:
1. **Low BLEU/ROUGE is partially an artifact of reference variation**: When the model generates a valid question using natural alternative phrasing, n-gram overlap with a single reference is penalized.
2. **Standard ROUGE is severely distorted by English-only tokenization**: The default `rouge_score` implementation discards Devanagari characters, artificially depressing ROUGE-1 from 29.7% down to 4.6%.
3. **Severe semantic failure modes persist**: The Stage 1 model suffers from answer-type confusion (asking "when" for places or persons), factual hallucination (mutating numbers and scores), and answer ignoring (asking questions about other entities in the passage).

---

## 2. Qualitative Evaluation Dimensions & Findings

| Evaluation Dimension | Stage 1 Performance & Behavior | Severity / Prevalence |
| :--- | :--- | :--- |
| **Answer Alignment** | In approximately 55–65% of samples, the generated question correctly targets the answer entity. However, in 35–45% of cases, the model confuses entity types (e.g., asking "when" for people or teams) or shifts focus to another entity in the passage. | **Moderate to High Risk** |
| **Context Faithfulness** | Generally high syntactic grounding—the model extracts vocabulary directly from the passage. However, entity-relation bindings frequently scramble. | **Moderate Risk** |
| **Answerability** | Questions targeting temporal ("कब") or quantitative ("कितने") entities are usually answerable. Abstract entity questions often become unanswerable due to swapped predicates. | **Moderate Risk** |
| **Grammar & Naturalness** | Morphosyntactic fluency in Hindi is generally sound. Subject-verb agreement (*ling-vachan*) and postpositional markers (*karak*) are well preserved. | **Low Risk / Strength** |
| **Question Completeness** | Nearly all outputs (99%+) form grammatically complete sentences terminating in Hindi punctuation or question marks. | **Low Risk / Strength** |
| **Factual Hallucination** | Observed numeric mutation (e.g., mutating football score from 24-10 to 24-25) and spurious dates. | **High Risk** |
| **Token Duplication** | Special token repetitions (`<2hi><2hi>`) are successfully resolved during postprocessing. N-gram repetition is controlled via `no_repeat_ngram_size=3`. | **Low Risk / Resolved** |

---

## 3. Systematic Taxonomy of Failure Modes

### Category A: Wh-Word & Answer-Type Mismatch (Semantic Misalignment)
The model frequently selects an interrogative pronoun (`किसने`, `कब`, `कहाँ`, `कितने`) that conflicts with the semantic type of the supplied answer.

* **Example 1:**
  * **Context:** `खेल 7 फरवरी, 2016 को कैलिफोर्निया के सांता क्लारा में सैन फ्रांसिस्को खाड़ी क्षेत्र में लेवी स्टेडियम में खेला गया था।`
  * **Supplied Answer:** `सांता क्लारा, कैलिफोर्निया में सैन फ्रांसिस्को खाड़ी क्षेत्र में लेवी का स्टेडियम।` *(Location)*
  * **Reference Question:** `सुपर बाउल 50 कहां हुआ?` *(Where)*
  * **Generated Question:** `कब खेला गया था लेवी स्टेडियम?` *(When)*
  * **Diagnostic:** The answer is a geographical venue, but the model generated a temporal question word ("कब").

* **Example 2:**
  * **Context:** `दक्षिण फ्लोरिडा/मियामी क्षेत्र ने पहले 10 बार इस कार्यक्रम की मेजबानी की है...`
  * **Supplied Answer:** `10वां` *(Count / Frequency)*
  * **Reference Question:** `दक्षिण फ्लोरिडा/मियामी क्षेत्र ने कितनी बार सुपर बाउल की मेजबानी की है?` *(How many times)*
  * **Generated Question:** `2010 में सुपर बाउल का आयोजन कब किया गया था?` *(When)*
  * **Diagnostic:** Frequency count answered as a calendar year question.

---

### Category B: Factual Hallucination & Numeric Alteration
The model mutates digits or entities present in the passage, creating false factual assertions.

* **Example 3:**
  * **Context:** `...कैरोलिना पैंथर्स को 24-10 से हराकर अपना तीसरा सुपर बाउल खिताब जीता।`
  * **Supplied Answer:** `डेनवर ब्रोंकोस`
  * **Reference Question:** `सुपर बाउल 50 किसने जीता?`
  * **Generated Question:** `नेशनल फुटबॉल कांफ्रेंस के चैंपियन कैरोलिना पैंथर्स को किसने 24-25 से हराया?`
  * **Diagnostic:** The actual score `24-10` was mutated into `24-25`. This is a hallucination of factual game data.

---

### Category C: Target Inversion & Passage Distraction
When a passage contains multiple named entities, the model frequently constructs a question around an irrelevant distractor entity rather than the supplied target answer.

* **Example 4:**
  * **Context:** `मैनिंग 1998 के एनएफएल मसौदे का #1 चयन था, जबकि न्यूटन को 2011 में पहले चुना गया था।`
  * **Supplied Answer:** `न्यूटन` *(Newton)*
  * **Reference Question:** `2011 में, एनएफएल ड्राफ्ट में चुने जाने वाले पहले खिलाड़ी कौन थे?`
  * **Generated Question:** `1998 के एनएफएल मसौदे के लिए मैनिंग का नाम क्या था?`
  * **Diagnostic:** Despite receiving "न्यूटन" as the target answer, the model generated a question targeting "मैनिंग" (Manning) in the other clause.

---

### Category D: Legitimate Paraphrasing (Surface Metric False Negatives)
In several instances, the model generated an accurate, grammatically sound question that received near-zero SacreBLEU or ROUGE because it differed lexically from the human ground truth.

* **Example 5:**
  * **Context:** `मैनिंग और न्यूटन ने 13 साल और 48 दिनों में सुपर बाउल क्वार्टरबैक का विरोध करने के बीच उम्र के सबसे बड़े अंतर का रिकॉर्ड भी बनाया...`
  * **Supplied Answer:** `13 वर्ष और 48 दिन`
  * **Reference Question:** `सुपर बाउल 50 के दौरान मैनिंग न्यूटन से कितना बड़ा था?`
  * **Generated Question:** `मैनिंग और न्यूटन के बीच उम्र का सबसे बड़ा अंतर क्या था?`
  * **Diagnostic:** The generated question is natural, grammatically correct Hindi, and directly prompts for the target answer. However, because it uses "उम्र का सबसे बड़ा अंतर क्या था" instead of "कितना बड़ा था", n-gram matching metrics assign low scores.

* **Example 6:**
  * **Context:** `लीग ने अंततः बोली को तीन साइटों तक सीमित कर दियाः`
  * **Supplied Answer:** `तीन`
  * **Reference Question:** `एनएफएल ने सुपर बाउल 50 के स्थान को कितने स्थानों तक सीमित कर दिया?`
  * **Generated Question:** `लीग ने कितने साइटों को सीमित कर दिया?`
  * **Diagnostic:** Succinct, natural, and directly answer-aligned.

---

### Category E: Declarative Copying with Punctuation Appended
The model copies a declarative clause from the context and terminates it with a question mark rather than inverting the clause into an interrogative syntax.

* **Example 7:**
  * **Context:** `2012 की शुरुआत में, एनएफएल आयुक्त रोजर गुडेल ने कहा कि लीग ने 50 वें सुपर बाउल को शानदार बनाने की योजना बनाई है...`
  * **Supplied Answer:** `50वां सुपर बाउल`
  * **Generated Question:** `2012 की शुरुआत में, एनएफएल आयुक्त रोजर गुडेल ने कहा कि लीग ने 50 वें सुपर बाउल को "शानदार" "बनाने की योजना बनाई है"?`
  * **Diagnostic:** Incomplete transformation from statement to question.

---

## 4. Key Takeaways for Stage 2 Curriculum Adaptation

1. **Answer-Aware Conditioning Needs Reinforcement:** The model often ignores the answer condition when distractors exist. Stage 2 training must reinforce answer-aware cross-attention.
2. **Textbook Phrasing vs SQuAD Translationese:** Stage 1 data is translated English SQuAD, explaining awkward phrasing (e.g. "सुपर बाउल एल"). Stage 2 NCERT/State Board data will expose the model to authentic Hindi pedagogical question forms (e.g., `...किसे कहते हैं?`, `...का क्या तात्पर्य है?`, `...का वर्णन कीजिए`).
3. **Evaluation Metric Standards:** ROUGE must always be reported with native Hindi tokenization alongside standard ROUGE, and complemented by human rating sheets (`results/human_eval_sample_100.csv`).
