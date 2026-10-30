"""
Textbook and Document Processing Engine for Prashn-Setu.
Handles PDF and Plain-Text textbook uploads, Hindi Unicode normalization,
page-boundary tracking, sentence-aware chunking, and batch question generation.
"""

import os
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader

from src.evaluation.advanced_generator import generate_advanced_questions
from src.evaluation.model_registry import MODEL_STAGE2


MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
TARGET_CHUNK_WORD_COUNT = 250          # ~250 words per chunk for focused QG
MAX_CHUNKS_PER_JOB = 10                # Safety cap to prevent GPU exhaustion


def repair_hindi_devanagari_text(text: str) -> str:
    """
    Repairs broken Devanagari text from PDFs, OCR, or KrutiDev/visual-font conversions.
    Corrects:
      1. Detached combining marks (vowels, anusvara, halant separated by spaces or newlines)
      2. Reorders visual short-i (ि / U+093F) placed before consonants or conjunct clusters
      3. Repaired broken conjunct clusters and spaced halant signs (e.g. िक्  त -> क्ति)
      4. Repaired line-broken Hindi words and syllables across newlines without punctuation
      5. Common OCR / legacy font corruptions (missing 'की' matra, misrecognized letters)
      6. Canonical Unicode NFC normalization
    """
    if not text:
        return ""

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    consonants = r'\u0915-\u0939\u0958-\u095F'
    vowels = r'\u0904-\u0914'
    all_combining = r'\u0901-\u0903\u093A-\u094F\u0951-\u0957\u0962\u0963'

    # 1. Remove isolated lines containing only combining marks
    text = re.sub(rf'\n[ \t]*[{all_combining}]+[ \t]*(?=\n|$)', '', text)

    # 2. Fix spaced halant clusters FIRST (e.g. िक्  त -> क्त, म्  ल -> म्ल) so conjuncts stay intact
    text = re.sub(rf'([{consonants}]\u094D)[ \t]+([{consonants}])', r'\1\2', text)

    # 3. Reorder visual short-i (\u093F) placed BEFORE consonant cluster:
    # Cluster: (consonant + nukta? + halant)* + consonant + nukta?
    # E.g. िसं -> सिं, िक -> कि, िचित्रत -> चित्रित, िक्त -> क्ति
    cluster = rf'((?:[{consonants}]\u093C?\u094D)*[{consonants}]\u093C?)'
    text = re.sub(rf'\u093F{cluster}', lambda m: m.group(1) + '\u093F', text)

    # 4. Handle hyphenated word splits across newlines: e.g. जीवन-\nसंघर्ष -> जीवन-संघर्ष
    text = re.sub(r'-\n\s*', '-', text)

    # 5. Reattach detached combining marks separated from their base consonant/vowel by spaces or newlines:
    # E.g. स ु ं -> सुंदर (स + ु + ं)
    for _ in range(4):
        text = re.sub(rf'([{consonants}{vowels}{all_combining}])[ \t\r\n]+([{all_combining}])', r'\1\2', text)

    # 6. Handle syllable splits across newlines where word is cut at anusvara or halant:
    # E.g. सिं\nह -> सिंह, सं\nघष -> संघष, सुं\nदर -> सुंदर, सुं\nसार -> संसार
    text = re.sub(rf'([{consonants}][{all_combining}]*[\u0902\u094D])\n\s*([{consonants}]+)', r'\1\2', text)

    # 7. Convert remaining single newlines between words to a space
    text = re.sub(r'(?<=[^\n])\n(?=[^\n])', ' ', text)

    # 8. Remove invalid double vowel matras on same consonant (e.g. ु after ी):
    text = re.sub(r'([\u093E-\u094C])[\u093A-\u094C]+', r'\1', text)

    # 9. Remove stray floating combining marks preceded by space and not attached to letters
    text = re.sub(rf'(?<=\s)[{all_combining}]+(?=\s|$)', '', text)

    # 10. Fix common OCR / legacy font corruptions:
    # Missing 'की' matra: 'क  ' before a Devanagari word
    text = re.sub(rf'\bक\s{{2,}}(?=[{consonants}{vowels}])', 'की ', text)

    # Known OCR / extraction artifacts:
    text = re.sub(r'प्रम\s*ं?\s*खता', 'प्रमुखता', text)
    text = re.sub(r'म\s*ं?\s*क्ति', 'मुक्ति', text)
    text = re.sub(r'संघष\b', 'संघर्ष', text)
    text = re.sub(r'स[ंु]\s*दर\b', 'सुंदर', text)
    text = re.sub(r'स[ुं]+सार\b', 'संसार', text)

    # 11. Space normalization & NFC
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = unicodedata.normalize('NFC', text.strip())
    return text


def normalize_hindi_text(text: str) -> str:
    """Comprehensive normalization and repair for Hindi textbook and passage text."""
    return repair_hindi_devanagari_text(text)


def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF textbook.
    Returns list of dicts with page number, raw text, and word count.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size exceeds 25 MB limit ({size / (1024*1024):.1f} MB).")

    reader = PdfReader(file_path)
    pages = []

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        clean_text = normalize_hindi_text(raw_text)
        if clean_text:
            words = clean_text.split()
            pages.append({
                "page_number": idx + 1,
                "text": clean_text,
                "word_count": len(words)
            })

    if not pages:
        raise ValueError("Could not extract any readable text from this PDF. It may be a scanned image or protected.")

    return pages


def extract_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
    """Reads plain text file with UTF-8 encoding."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size exceeds 25 MB limit.")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    clean_text = normalize_hindi_text(content)
    if not clean_text:
        raise ValueError("Text file is empty.")

    words = clean_text.split()
    return [{
        "page_number": 1,
        "text": clean_text,
        "word_count": len(words)
    }]


def chunk_page_passages(
    pages: List[Dict[str, Any]],
    target_words: int = TARGET_CHUNK_WORD_COUNT
) -> List[Dict[str, Any]]:
    """
    Splits multi-page text into coherent passages respecting sentence boundaries.
    Never splits sentences in the middle.
    """
    chunks = []
    chunk_idx = 1

    for p in pages:
        page_num = p["page_number"]
        page_text = p["text"]

        # Split on sentence boundaries: । or ? or !
        sentences = [s.strip() for s in re.split(r"(?<=[।\?!])\s+", page_text) if s.strip()]
        if not sentences:
            sentences = [page_text]

        current_sentences = []
        current_word_count = 0

        for s in sentences:
            s_words = len(s.split())
            if current_word_count + s_words > target_words and current_sentences:
                # Flush current chunk
                chunk_passage = " ".join(current_sentences)
                chunks.append({
                    "chunk_id": f"chunk_{chunk_idx}",
                    "page_number": page_num,
                    "text": chunk_passage,
                    "word_count": current_word_count
                })
                chunk_idx += 1
                current_sentences = [s]
                current_word_count = s_words
            else:
                current_sentences.append(s)
                current_word_count += s_words

        if current_sentences:
            chunk_passage = " ".join(current_sentences)
            chunks.append({
                "chunk_id": f"chunk_{chunk_idx}",
                "page_number": page_num,
                "text": chunk_passage,
                "word_count": current_word_count
            })
            chunk_idx += 1

    return chunks


def process_textbook_and_generate(
    file_path: str,
    questions_per_chunk: int = 2,
    category: str = "factual",
    difficulty: str = "medium",
    max_chunks: int = MAX_CHUNKS_PER_JOB,
    model_id: str = MODEL_STAGE2
) -> Dict[str, Any]:
    """
    Complete end-to-end processing pipeline:
      1. Extract text from uploaded PDF or TXT.
      2. Chunk into coherent passages.
      3. Generate Stage 2 questions for each passage.
      4. Return organized question sets with page references.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        pages = extract_text_from_pdf(file_path)
    elif ext in [".txt", ".text"]:
        pages = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported document format '{ext}'. Please upload a PDF (.pdf) or Text (.txt) file.")

    chunks = chunk_page_passages(pages)
    selected_chunks = chunks[:max_chunks]

    all_generated_questions = []

    for chk in selected_chunks:
        qs = generate_advanced_questions(
            context=chk["text"],
            num_questions=questions_per_chunk,
            category=category,
            difficulty=difficulty,
            model_id=model_id,
            source_file=filename,
            page_number=chk["page_number"]
        )
        all_generated_questions.extend(qs)

    total_words = sum(p["word_count"] for p in pages)

    return {
        "filename": filename,
        "total_pages": len(pages),
        "total_words": total_words,
        "total_chunks": len(chunks),
        "processed_chunks": len(selected_chunks),
        "questions_generated": len(all_generated_questions),
        "questions": all_generated_questions
    }

