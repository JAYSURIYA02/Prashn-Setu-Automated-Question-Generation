"""
Textbook and Document Processing Engine for Prashn-Setu.
Handles PDF and Plain-Text textbook uploads, Hindi Unicode normalization,
page-boundary tracking, sentence-aware chunking, and batch question generation.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader

from src.evaluation.advanced_generator import generate_advanced_questions
from src.evaluation.model_registry import MODEL_STAGE2


MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
TARGET_CHUNK_WORD_COUNT = 250          # ~250 words per chunk for focused QG
MAX_CHUNKS_PER_JOB = 10                # Safety cap to prevent GPU exhaustion


def normalize_hindi_text(text: str) -> str:
    """Cleans up formatting, hyphens, and whitespace while preserving Devanagari Unicode."""
    if not text:
        return ""
    # Normalize unicode whitespaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove soft hyphens and word-break hyphens across newlines
    text = re.sub(r"-\n\s*", "", text)
    # Collapse multiple consecutive spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Clean multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

