"""
Persistent SQLite Question Bank and History Manager for Prashn-Setu.
Provides thread-safe CRUD, parameterized full-text search, multi-field filtering,
quiz history persistence, and export integration.
"""

import os
import sqlite3
import json
import time
import uuid
from typing import List, Dict, Any, Optional

from src.config import PROJECT_ROOT

DB_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DB_DIR, "question_bank.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite Question Bank database with row-factory enabled."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_question_bank_db():
    """Initializes database schema and indexes if not already present."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Questions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id TEXT PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT,
        category TEXT DEFAULT 'factual',
        difficulty TEXT DEFAULT 'medium',
        source_passage TEXT,
        source_sentence TEXT,
        source_file TEXT DEFAULT '',
        page_number INTEGER DEFAULT 1,
        model_id TEXT DEFAULT 'stage2',
        options_json TEXT DEFAULT '[]',
        correct_option TEXT DEFAULT '',
        explanation TEXT DEFAULT '',
        evidence_status TEXT DEFAULT 'evidence_found',
        answerability_score REAL DEFAULT 4.0,
        is_edited INTEGER DEFAULT 0,
        tags TEXT DEFAULT '',
        created_at TEXT NOT NULL
    )
    """)

    # Indexes for fast search and filtering
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_category ON questions(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions(difficulty)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_source_file ON questions(source_file)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_created_at ON questions(created_at)")

    # Quiz history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quiz_history (
        id TEXT PRIMARY KEY,
        title TEXT,
        total_questions INTEGER,
        score INTEGER,
        percentage REAL,
        answers_json TEXT,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def save_question(q_data: Dict[str, Any]) -> Dict[str, Any]:
    """Inserts a new question into the Question Bank."""
    init_question_bank_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    q_id = q_data.get("id") or f"q_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    options = q_data.get("options", [])
    options_json = json.dumps(options, ensure_ascii=False) if isinstance(options, (list, dict)) else str(options)

    cursor.execute("""
    INSERT INTO questions (
        id, question, answer, category, difficulty, source_passage,
        source_sentence, source_file, page_number, model_id, options_json,
        correct_option, explanation, evidence_status, answerability_score,
        is_edited, tags, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        q_id,
        q_data.get("question", "").strip(),
        q_data.get("answer", "").strip(),
        q_data.get("category", "factual"),
        q_data.get("difficulty", "medium"),
        q_data.get("source_passage", "").strip(),
        q_data.get("source_sentence", "").strip(),
        q_data.get("source_file", ""),
        int(q_data.get("page_number", 1)),
        q_data.get("model_id", "stage2"),
        options_json,
        q_data.get("correct_option", ""),
        q_data.get("explanation", ""),
        q_data.get("evidence_status", "evidence_found"),
        float(q_data.get("answerability_score", 4.0)),
        int(q_data.get("is_edited", 0)),
        q_data.get("tags", ""),
        now
    ))

    conn.commit()
    conn.close()

    result = dict(q_data)
    result["id"] = q_id
    result["created_at"] = now
    return result


def save_questions_batch(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Saves a batch of generated questions to the database."""
    saved = []
    for q in questions:
        saved.append(save_question(q))
    return saved


def get_questions(
    query: Optional[str] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    source_file: Optional[str] = None,
    model_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """Queries questions with search, filters, and pagination."""
    init_question_bank_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    conditions = []
    params = []

    if query and query.strip():
        q_str = f"%{query.strip()}%"
        conditions.append("(question LIKE ? OR answer LIKE ? OR source_passage LIKE ?)")
        params.extend([q_str, q_str, q_str])

    if category and category != "all":
        conditions.append("category = ?")
        params.append(category)

    if difficulty and difficulty != "all":
        conditions.append("difficulty = ?")
        params.append(difficulty)

    if source_file and source_file != "all":
        conditions.append("source_file = ?")
        params.append(source_file)

    if model_id and model_id != "all":
        conditions.append("model_id = ?")
        params.append(model_id)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Total count
    count_sql = f"SELECT COUNT(*) as total FROM questions {where_clause}"
    cursor.execute(count_sql, params)
    total = cursor.fetchone()["total"]

    # Paginated rows
    fetch_sql = f"""
    SELECT * FROM questions
    {where_clause}
    ORDER BY created_at DESC
    LIMIT ? OFFSET ?
    """
    cursor.execute(fetch_sql, params + [limit, offset])
    rows = cursor.fetchall()

    results = []
    for r in rows:
        d = dict(r)
        try:
            d["options"] = json.loads(d["options_json"])
        except Exception:
            d["options"] = []
        results.append(d)

    conn.close()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "questions": results
    }


def get_question_by_id(q_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single question by ID."""
    init_question_bank_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM questions WHERE id = ?", (q_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    d = dict(row)
    try:
        d["options"] = json.loads(d["options_json"])
    except Exception:
        d["options"] = []
    return d


def update_question(q_id: str, updates: Dict[str, Any]) -> bool:
    """Updates an existing question and flags is_edited = 1."""
    init_question_bank_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    allowed = [
        "question", "answer", "category", "difficulty", "source_sentence",
        "options_json", "correct_option", "explanation", "tags"
    ]
    set_clauses = ["is_edited = 1"]
    params = []

    for k in allowed:
        if k in updates:
            set_clauses.append(f"{k} = ?")
            val = updates[k]
            if k == "options_json" and isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            params.append(val)

    if len(set_clauses) == 1:
        conn.close()
        return False

    params.append(q_id)
    sql = f"UPDATE questions SET {', '.join(set_clauses)} WHERE id = ?"
    cursor.execute(sql, params)
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def delete_question(q_id: str) -> bool:
    """Deletes a question by ID."""
    init_question_bank_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM questions WHERE id = ?", (q_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def get_question_bank_stats() -> Dict[str, Any]:
    """Calculates aggregate statistics for the Question Bank."""
    init_question_bank_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM questions")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT category, COUNT(*) as cnt FROM questions GROUP BY category")
    by_category = {r["category"]: r["cnt"] for r in cursor.fetchall()}

    cursor.execute("SELECT difficulty, COUNT(*) as cnt FROM questions GROUP BY difficulty")
    by_difficulty = {r["difficulty"]: r["cnt"] for r in cursor.fetchall()}

    cursor.execute("SELECT model_id, COUNT(*) as cnt FROM questions GROUP BY model_id")
    by_model = {r["model_id"]: r["cnt"] for r in cursor.fetchall()}

    cursor.execute("SELECT DISTINCT source_file FROM questions WHERE source_file != ''")
    source_files = [r["source_file"] for r in cursor.fetchall()]

    conn.close()

    return {
        "total_questions": total,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_model": by_model,
        "source_files": source_files
    }


def record_quiz_result(quiz_data: Dict[str, Any]) -> str:
    """Stores completed quiz attempt results."""
    init_question_bank_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    quiz_id = f"quiz_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO quiz_history (id, title, total_questions, score, percentage, answers_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        quiz_id,
        quiz_data.get("title", "Hindi Practice Quiz"),
        int(quiz_data.get("total_questions", 0)),
        int(quiz_data.get("score", 0)),
        float(quiz_data.get("percentage", 0.0)),
        json.dumps(quiz_data.get("answers", []), ensure_ascii=False),
        now
    ))

    conn.commit()
    conn.close()
    return quiz_id

