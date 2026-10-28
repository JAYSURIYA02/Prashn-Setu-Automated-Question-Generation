"""
Evidence Highlighting, Grounding, and Answerability Verification Engine.
Provides rigorous sentence-level supporting evidence identification,
character-level offset mapping for UI highlighting, and lexical/semantic
answerability classification for generated Hindi questions.
"""

import re
from typing import List, Dict, Any, Optional, Tuple


# Hindi stop words to filter out during keyword/overlap scoring
HINDI_STOPWORDS = {
    "का", "के", "की", "को", "में", "से", "पर", "ने", "है", "हैं", "था", "थे", "थी",
    "होता", "होती", "होते", "हुआ", "हुई", "हुए", "और", "या", "तथा", "एवं", "भी",
    "एक", "यह", "वह", "इस", "उस", "इन", "उन", "कि", "जो", "कर", "करके", "दिया",
    "गया", "गए", "गई", "जाता", "जाती", "जाते", "रहा", "रही", "रहे", "सकता", "सकती",
    "सकते", "अपना", "अपने", "अपनी", "द्वारा", "लिए", "बारे", "तक", "साथ"
}

# Question interrogatives to omit during passage overlap calculation
HINDI_WH_WORDS = {
    "क्या", "कहाँ", "कब", "कौन", "किस", "किसे", "किसने", "किसको", "क्यों", "कैसे",
    "कितना", "कितनी", "कितने", "किसका", "किसकी", "किसके", "क्यो", "काहे"
}


def split_hindi_sentences(passage: str) -> List[Dict[str, Any]]:
    """
    Splits Hindi textbook passage into sentences with character offset ranges.
    Preserves exact start and end character positions for precision UI highlighting.
    """
    if not passage or not passage.strip():
        return []

    # Delimiters: purna viram (।), question mark (?), exclamation (!), double newlines
    pattern = re.compile(r"([^।\?!\n]+[।\?!]?)")
    sentences = []
    
    for match in pattern.finditer(passage):
        text = match.group().strip()
        if len(text) > 3:  # filter noise
            start = match.start()
            # Adjust start if leading whitespace
            orig_match = match.group()
            leading_spaces = len(orig_match) - len(orig_match.lstrip())
            start += leading_spaces
            end = start + len(text)
            sentences.append({
                "index": len(sentences),
                "text": text,
                "start_char": start,
                "end_char": end
            })

    if not sentences and passage.strip():
        sentences.append({
            "index": 0,
            "text": passage.strip(),
            "start_char": 0,
            "end_char": len(passage.strip())
        })

    return sentences


def tokenize_content_words(text: str) -> List[str]:
    """Tokenizes text into meaningful content words, removing stopwords and interrogatives."""
    words = re.findall(r"[\u0900-\u097F\w]+", text.lower())
    return [w for w in words if w not in HINDI_STOPWORDS and w not in HINDI_WH_WORDS and len(w) > 1]


def compute_evidence_support(
    question: str,
    answer: str,
    passage: str
) -> Dict[str, Any]:
    """
    Identifies the primary supporting sentence and classifies answerability status.
    
    Status Labels:
      - 'evidence_found': Question keywords and answer span strongly overlap a source sentence.
      - 'needs_review': Moderate overlap; answer span is partially inferred or distant.
      - 'unsupported': Question terms lack grounding in the source passage.
    """
    sentences = split_hindi_sentences(passage)
    q_words = tokenize_content_words(question)
    a_words = tokenize_content_words(answer) if answer else []

    if not sentences:
        return {
            "status": "unsupported",
            "status_label": "असमर्थित (Unsupported)",
            "status_description": "प्रसंग में कोई वैध वाक्य नहीं मिला।",
            "supporting_sentence": "",
            "start_char": 0,
            "end_char": 0,
            "overlap_score": 0.0,
            "matched_keywords": [],
            "answerability_score": 1.0
        }

    best_sentence = None
    best_score = -1.0
    best_q_matches = []
    best_a_matches = []

    for s in sentences:
        s_words = set(tokenize_content_words(s["text"]))
        if not s_words:
            continue

        q_matches = [w for w in q_words if w in s_words]
        a_matches = [w for w in a_words if w in s_words]

        # Weighted overlap score: answer matches count double
        q_score = (len(q_matches) / max(len(q_words), 1)) if q_words else 0.0
        a_score = (len(a_matches) / max(len(a_words), 1)) if a_words else 0.5

        total_score = (q_score * 0.6) + (a_score * 0.4)

        # Bonus if exact answer string appears directly in sentence
        if answer and answer.strip() and answer.strip().lower() in s["text"].lower():
            total_score += 0.35

        if total_score > best_score:
            best_score = total_score
            best_sentence = s
            best_q_matches = q_matches
            best_a_matches = a_matches

    if best_sentence is None:
        best_sentence = sentences[0]
        best_score = 0.1

    all_matches = list(set(best_q_matches + best_a_matches))

    # Classification thresholding
    if best_score >= 0.55 or (answer and answer.strip() in best_sentence["text"]):
        status = "evidence_found"
        status_label = "साक्ष्य सत्यापित (Evidence Found)"
        desc = "प्रश्न व उत्तर दिए गए प्रसंग के इस वाक्य द्वारा पूर्णतः समर्थित हैं।"
        ans_score = min(5.0, round(3.5 + best_score * 1.5, 1))
    elif best_score >= 0.28:
        status = "needs_review"
        status_label = "समीक्षा अपेक्षित (Needs Review)"
        desc = "प्रश्न प्रसंग से संबंधित है, परंतु उत्तर प्रत्यक्षतः एक ही वाक्य में स्पष्ट नहीं है।"
        ans_score = min(4.0, round(2.5 + best_score * 1.5, 1))
    else:
        status = "unsupported"
        status_label = "असमर्थित (Possible Unsupported)"
        desc = "प्रश्न में प्रयुक्त मुख्य शब्द गद्यांश के मुख्य तथ्यों से सीधे मेल नहीं खाते।"
        ans_score = round(max(1.0, best_score * 3.0), 1)

    return {
        "status": status,
        "status_label": status_label,
        "status_description": desc,
        "supporting_sentence": best_sentence["text"],
        "start_char": best_sentence["start_char"],
        "end_char": best_sentence["end_char"],
        "sentence_index": best_sentence["index"],
        "overlap_score": round(float(best_score), 3),
        "matched_keywords": all_matches,
        "answerability_score": ans_score
    }


def detect_question_repetition(questions: List[str], threshold: float = 0.70) -> List[Dict[str, Any]]:
    """
    Checks list of generated questions for repetitive or near-duplicate patterns.
    Flags questions that share high word n-gram Jaccard similarity.
    """
    results = []
    tokenized_list = [set(tokenize_content_words(q)) for q in questions]

    for i, q in enumerate(questions):
        tokens_i = tokenized_list[i]
        is_duplicate = False
        duplicate_with = None

        if tokens_i:
            for j in range(i):
                tokens_j = tokenized_list[j]
                if not tokens_j:
                    continue
                intersection = len(tokens_i.intersection(tokens_j))
                union = len(tokens_i.union(tokens_j))
                sim = (intersection / union) if union > 0 else 0.0
                if sim >= threshold:
                    is_duplicate = True
                    duplicate_with = j
                    break

        results.append({
            "index": i,
            "question": q,
            "is_repetitive": is_duplicate,
            "duplicate_of_index": duplicate_with
        })

    return results

