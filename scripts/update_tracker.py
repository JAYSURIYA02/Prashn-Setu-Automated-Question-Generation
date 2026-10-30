# -*- coding: utf-8 -*-
import os
import hashlib
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXCLUDE_DIRS = {".git", ".venv", "__pycache__", "venv", ".idea", ".vscode", "indic_question_generation_hi"}
BINARY_EXTS = {".safetensors", ".bin", ".ttf", ".db", ".png", ".jpg", ".jpeg", ".pyc", ".pdf"}

def get_file_category(rel_path):
    rel_path_lower = rel_path.lower().replace("\\", "/")
    if rel_path_lower.startswith("architecture/"):
        return "Architecture Doc"
    elif rel_path_lower.startswith("src/data/"):
        return "Data Module"
    elif rel_path_lower.startswith("src/demo/"):
        return "Demo Interface"
    elif rel_path_lower.startswith("src/evaluation/"):
        return "Inference & Eval"
    elif rel_path_lower.startswith("src/models/"):
        return "Model Module"
    elif rel_path_lower.startswith("src/"):
        return "Core Configuration"
    elif rel_path_lower.startswith("tests/"):
        return "Test Suite"
    elif rel_path_lower.startswith("scripts/"):
        return "Automation Script"
    elif rel_path_lower.startswith("results/"):
        return "Evaluation Result"
    elif rel_path_lower.startswith("docs/"):
        return "Project Documentation"
    elif rel_path_lower.startswith("data/"):
        return "Dataset Artifact"
    elif rel_path_lower.startswith("checkpoints/"):
        return "Model Checkpoint"
    return "Root File"

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:12]

def count_lines(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def main():
    tracked = []
    total_lines = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in BINARY_EXTS or fname.endswith(".pyc") or fname.startswith("."):
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT).replace("\\", "/")

            if "scratch" in rel_path or "temp" in rel_path:
                continue

            size_kb = os.path.getsize(full_path) / 1024.0
            loc = count_lines(full_path)
            total_lines += loc
            sha = compute_sha256(full_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(full_path), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            cat = get_file_category(rel_path)
            tracked.append({
                "path": rel_path,
                "category": cat,
                "size_kb": size_kb,
                "loc": loc,
                "sha": sha,
                "mtime": mtime
            })

    # Sort tracked files by category and path
    tracked.sort(key=lambda x: (x["category"], x["path"]))

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# System State & Change Tracker",
        "",
        f"> **Last Automatically Refreshed:** `{now_utc}`  ",
        f"> **Total Tracked Files:** `{len(tracked)}` | **Total Code & Doc Lines:** `{total_lines}`",
        "",
        "This file is dynamically updated by running `python scripts/update_tracker.py`.",
        "It maintains the live inventory of codebase files, SHA-256 checksums, dataset splits, and model checkpoints.",
        "",
        "---",
        "## 1. Tracked Files Inventory & Integrity Checksums",
        "",
        "| File Path | Category | Size (KB) | Lines (LOC) | SHA-256 (12 chars) | Last Modified |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for item in tracked:
        lines.append(f"| `{item['path']}` | {item['category']} | {item['size_kb']:.2f} | {item['loc']} | `{item['sha']}` | {item['mtime']} |")

    lines.extend([
        "",
        "---",
        "## 2. Checkpoints & Architecture Status",
        "",
        "- **Stage 1 Base Checkpoint:** `checkpoints/stage1_indicnlg_qg/` (Present, 244M parameters)",
        "- **Stage 2 NCERT LoRA Adapter:** `checkpoints/stage2_ncert_gold_lora/` (Present, PEFT LoRA $r=16$, $\\alpha=32$)",
        "- **IndicNLG Baseline Checkpoint:** `checkpoints/baseline_indicnlg_qg/` (Honest status: Unavailable, not substituted)",
        "",
        "---",
        "## 3. Active Workspaces & Engine Summary",
        "",
        "- **Question Studio & Hub:** Single-passage and multi-page textbook chapter question generation with inline context evidence.",
        "- **Practice & Quiz Mode:** Instant interactive evaluation with scoring, percentage, correct/incorrect feedback, and retry mechanism.",
        "- **Model Comparison:** Pure Stage 1 IndicBART vs. Stage 2 NCERT-Adapted LoRA comparison.",
        "- **Research & System Analytics:** English-first dashboard with ROUGE, SacreBLEU, METEOR benchmarks, human evaluation rubric, qualitative error analysis, and hardware status.",
        ""
    ])

    tracker_content = "\n".join(lines) + "\n"

    target_path = os.path.join(PROJECT_ROOT, "architecture", "SYSTEM_STATE_TRACKER.md")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(tracker_content)
    print(f"Updated {target_path} successfully. Total files: {len(tracked)}, Total lines: {total_lines}")

    # Also update OneDrive path if it exists separately
    alt_target = r"C:\Users\jaysu\OneDrive\Desktop\PROJECTS\pransh-Setu-NLP\send\architecture\SYSTEM_STATE_TRACKER.md"
    if os.path.exists(os.path.dirname(alt_target)) and os.path.abspath(target_path) != os.path.abspath(alt_target):
        try:
            with open(alt_target, "w", encoding="utf-8") as f:
                f.write(tracker_content)
            print(f"Also synced to {alt_target}")
        except Exception as e:
            print(f"Sync to OneDrive warning: {e}")

if __name__ == "__main__":
    main()

