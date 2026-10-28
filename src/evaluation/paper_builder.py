"""
Question Paper Builder and Multi-Format Exporter for Prashn-Setu.
Supports:
  1. Multi-section exam paper organization (MCQ, Short Answer, HOTS/Long Answer).
  2. Marks allocation and total marks validation.
  3. PDF Question Paper generation with true Devanagari Hindi font rendering.
  4. Separate PDF Answer Key generation.
  5. CSV (with UTF-8 BOM for Excel compatibility) and JSON export.
"""

import os
import csv
import json
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from src.config import PROJECT_ROOT

FONT_DIR = os.path.join(PROJECT_ROOT, "src", "demo", "fonts")
REGULAR_FONT_PATH = os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf")
BOLD_FONT_PATH = os.path.join(FONT_DIR, "NotoSansDevanagari-Bold.ttf")

# Register fonts once
_FONTS_REGISTERED = False

def register_hindi_fonts():
    global _FONTS_REGISTERED
    if not _FONTS_REGISTERED:
        if os.path.exists(REGULAR_FONT_PATH):
            pdfmetrics.registerFont(TTFont("NotoHindi", REGULAR_FONT_PATH))
        if os.path.exists(BOLD_FONT_PATH):
            pdfmetrics.registerFont(TTFont("NotoHindi-Bold", BOLD_FONT_PATH))
        _FONTS_REGISTERED = True


def validate_paper_config(paper_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates marks consistency across sections and questions."""
    warnings = []
    sections = paper_data.get("sections", [])
    
    total_assigned_marks = 0
    for s_idx, sec in enumerate(sections):
        sec_title = sec.get("title", f"Section {s_idx + 1}")
        questions = sec.get("questions", [])
        for q in questions:
            total_assigned_marks += float(q.get("marks", 1))

    target_total = float(paper_data.get("total_marks", total_assigned_marks))
    if abs(total_assigned_marks - target_total) > 0.01:
        warnings.append(
            f"कुल अंकों में अंतर: प्रश्नों के कुल अंक ({total_assigned_marks:.0f}) "
            f"प्रश्नपत्र के निर्धारित कुल अंक ({target_total:.0f}) से मेल नहीं खाते हैं।"
        )

    return (len(warnings) == 0, warnings)


def build_paper_styles():
    register_hindi_fonts()
    styles = getSampleStyleSheet()

    font_reg = "NotoHindi" if os.path.exists(REGULAR_FONT_PATH) else "Helvetica"
    font_bold = "NotoHindi-Bold" if os.path.exists(BOLD_FONT_PATH) else "Helvetica-Bold"

    paper_styles = {
        "SchoolTitle": ParagraphStyle(
            "SchoolTitle",
            parent=styles["Normal"],
            fontName=font_bold,
            fontSize=16,
            leading=20,
            alignment=1,  # Center
            textColor=colors.HexColor("#1e3a8a")
        ),
        "ExamTitle": ParagraphStyle(
            "ExamTitle",
            parent=styles["Normal"],
            fontName=font_bold,
            fontSize=13,
            leading=17,
            alignment=1,
            textColor=colors.HexColor("#0f172a")
        ),
        "MetaText": ParagraphStyle(
            "MetaText",
            parent=styles["Normal"],
            fontName=font_reg,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155")
        ),
        "Instruction": ParagraphStyle(
            "Instruction",
            parent=styles["Normal"],
            fontName=font_reg,
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#475569")
        ),
        "SectionHeader": ParagraphStyle(
            "SectionHeader",
            parent=styles["Normal"],
            fontName=font_bold,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1e40af"),
            spaceBefore=10,
            spaceAfter=6
        ),
        "QuestionText": ParagraphStyle(
            "QuestionText",
            parent=styles["Normal"],
            fontName=font_reg,
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#0f172a")
        ),
        "OptionText": ParagraphStyle(
            "OptionText",
            parent=styles["Normal"],
            fontName=font_reg,
            fontSize=10,
            leading=14,
            leftIndent=18,
            textColor=colors.HexColor("#334155")
        ),
        "AnswerText": ParagraphStyle(
            "AnswerText",
            parent=styles["Normal"],
            fontName=font_reg,
            fontSize=9.5,
            leading=13.5,
            leftIndent=15,
            textColor=colors.HexColor("#166534")
        )
    }
    return paper_styles


def generate_paper_pdf(
    paper_data: Dict[str, Any],
    output_path: Optional[str] = None,
    include_answers: bool = False
) -> bytes:
    """
    Builds printable, classroom-ready Hindi Question Paper PDF.
    NEVER includes answers in student question paper if include_answers is False.
    """
    styles = build_paper_styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []

    school = paper_data.get("school_name", "केन्द्रीय विद्यालय / राष्ट्रीय विद्यालय")
    title = paper_data.get("title", "आवधिक परीक्षा — २०२६")
    subject = paper_data.get("subject", "कक्षा १० — हिन्दी (क्षितिज / स्पर्श)")
    time_allowed = paper_data.get("time_allowed", "२:०० घंटे")
    total_marks = paper_data.get("total_marks", "४०")
    instructions = paper_data.get("instructions", [
        "१. सभी प्रश्न हल करना अनिवार्य है।",
        "२. प्रत्येक प्रश्न के अंक उसके सम्मुख अंकित हैं।",
        "३. उत्तर यथासंभव अपनी भाषा में शुद्ध व स्पष्ट लिखिए।"
    ])

    story.append(Paragraph(school, styles["SchoolTitle"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(title, styles["ExamTitle"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>विषय:</b> {subject}", styles["MetaText"]))
    story.append(Spacer(1, 8))

    # Meta Table (Time & Marks)
    meta_data = [
        [
            Paragraph(f"<b>समय:</b> {time_allowed}", styles["MetaText"]),
            Paragraph(f"<b>पूर्णांक:</b> {total_marks}", styles["MetaText"])
        ]
    ]
    meta_table = Table(meta_data, colWidths=[260, 260])
    meta_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94a3b8"), spaceAfter=8))

    # Instructions
    story.append(Paragraph("<b>सामान्य निर्देश:</b>", styles["Instruction"]))
    for inst in instructions:
        story.append(Paragraph(inst, styles["Instruction"]))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    # Sections & Questions
    sections = paper_data.get("sections", [])
    q_global_num = 1

    for sec in sections:
        sec_title = sec.get("title", "खंड")
        sec_desc = sec.get("description", "")
        story.append(Paragraph(sec_title, styles["SectionHeader"]))
        if sec_desc:
            story.append(Paragraph(f"<i>{sec_desc}</i>", styles["Instruction"]))
            story.append(Spacer(1, 4))

        questions = sec.get("questions", [])
        for q in questions:
            q_text = q.get("question", "").strip()
            marks = q.get("marks", 1)
            q_line = f"<b>प्र. {q_global_num}.</b> {q_text} <font color='#64748b'><b>[{marks} अंक]</b></font>"
            story.append(Paragraph(q_line, styles["QuestionText"]))

            # If MCQ, render options
            options = q.get("options", [])
            if options:
                opt_strs = []
                for opt in options:
                    opt_strs.append(f"({opt['key']}) {opt['text']}")
                # Group in pairs if short
                story.append(Spacer(1, 2))
                for opt_str in opt_strs:
                    story.append(Paragraph(opt_str, styles["OptionText"]))

            # Include answer only if explicitly permitted (e.g. Teacher's master copy)
            if include_answers:
                ans_str = q.get("answer", "")
                expl = q.get("explanation", "")
                story.append(Spacer(1, 2))
                story.append(Paragraph(f"<b>[उत्तर]:</b> {ans_str}", styles["AnswerText"]))
                if expl:
                    story.append(Paragraph(f"<b>[स्पष्टीकरण]:</b> {expl}", styles["AnswerText"]))

            story.append(Spacer(1, 8))
            q_global_num += 1

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


def generate_answer_key_pdf(
    paper_data: Dict[str, Any],
    output_path: Optional[str] = None
) -> bytes:
    """Generates a dedicated Answer Key & Marking Scheme PDF."""
    styles = build_paper_styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    title = paper_data.get("title", "आवधिक परीक्षा — २०२६")
    subject = paper_data.get("subject", "कक्षा १० हिन्दी")

    story.append(Paragraph(f"<b>उत्तर कुंजी व मूल्यांकन योजना (Answer Key & Marking Scheme)</b>", styles["SchoolTitle"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>प्रश्नपत्र:</b> {title} | <b>विषय:</b> {subject}", styles["ExamTitle"]))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1e3a8a"), spaceAfter=12))

    sections = paper_data.get("sections", [])
    q_num = 1

    for sec in sections:
        sec_title = sec.get("title", "खंड")
        story.append(Paragraph(sec_title, styles["SectionHeader"]))
        questions = sec.get("questions", [])

        for q in questions:
            q_text = q.get("question", "")
            ans = q.get("answer", "")
            marks = q.get("marks", 1)
            expl = q.get("explanation", "")
            correct_opt = q.get("correct_option", "")

            ans_display = ans
            if correct_opt:
                ans_display = f"विकल्प ({correct_opt}): {ans}"

            story.append(Paragraph(f"<b>प्र. {q_num}. {q_text}</b> [{marks} अंक]", styles["QuestionText"]))
            story.append(Spacer(1, 2))
            story.append(Paragraph(f"<b>आदर्श उत्तर:</b> {ans_display}", styles["AnswerText"]))
            if expl:
                story.append(Paragraph(f"<b>मूल्यांकन बिंदु / साक्ष्य:</b> {expl}", styles["Instruction"]))
            story.append(Spacer(1, 8))
            q_num += 1

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


def export_paper_csv(paper_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Exports question paper to CSV with UTF-8 BOM encoding for Excel."""
    from io import StringIO
    f = StringIO()
    # Write UTF-8 BOM
    f.write("\ufeff")

    writer = csv.writer(f)
    writer.writerow(["Section", "Q_Num", "Marks", "Question", "Answer", "Category", "Difficulty", "Explanation", "Source"])

    sections = paper_data.get("sections", [])
    q_num = 1
    for sec in sections:
        s_title = sec.get("title", "")
        for q in sec.get("questions", []):
            writer.writerow([
                s_title,
                q_num,
                q.get("marks", 1),
                q.get("question", ""),
                q.get("answer", ""),
                q.get("category", ""),
                q.get("difficulty", ""),
                q.get("explanation", ""),
                q.get("source_sentence", "")
            ])
            q_num += 1

    content = f.getvalue()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8-sig") as out_file:
            out_file.write(content)

    return content


def export_paper_json(paper_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Exports question paper to clean JSON format."""
    content = json.dumps(paper_data, ensure_ascii=False, indent=2)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(content)
    return content

