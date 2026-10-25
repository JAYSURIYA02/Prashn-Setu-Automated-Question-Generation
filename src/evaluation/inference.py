"""
Inference & Question Generation API for Prashn-Setu.
Provides QuestionGenerator class and interactive prompt interface.
"""

import os
import re
import argparse
import json
from typing import Optional, List, Dict, Any, Union
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from src.config import ModelConfig, TrainingConfig

class QuestionGenerator:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        if checkpoint_path is None:
            checkpoint_path = (
                "checkpoints/stage2_ncert_gold_lora"
                if os.path.exists("checkpoints/stage2_ncert_gold_lora")
                else "checkpoints/stage1_indicnlg_qg"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading generator on device: {self.device}")
        
        print(f"Loading tokenizer from: {checkpoint_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            do_lower_case=False,
            use_fast=False,
            keep_accents=True
        )
        
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", "ai4bharat/IndicBART")
            if not os.path.exists(base_model_name):
                # Fallback: search for local base model candidates relative to checkpoint or cwd
                candidates = [
                    os.path.normpath(os.path.join(checkpoint_path, "..", os.path.basename(base_model_name))),
                    os.path.normpath(os.path.join(checkpoint_path, "..", "stage1_indicnlg_qg")),
                    os.path.join("checkpoints", os.path.basename(base_model_name)),
                    os.path.join("checkpoints", "stage1_indicnlg_qg"),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        base_model_name = cand
                        break
            print(f"Detected LoRA adapter checkpoint. Loading base model: {base_model_name}...")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            print(f"Loading full model weights from: {checkpoint_path}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.pad_id = self.tokenizer._convert_token_to_id_with_added_voc("<pad>")
        self.bos_id = self.tokenizer._convert_token_to_id_with_added_voc("<s>")
        self.eos_id = self.tokenizer._convert_token_to_id_with_added_voc("</s>")
        self.target_lang_id = self.tokenizer._convert_token_to_id_with_added_voc("<2hi>")

    def generate(
        self,
        context: str,
        answer: str = "",
        num_beams: int = 4,
        max_length: int = 64,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
        no_repeat_ngram_size: int = 3
    ) -> list:
        """Generates natural Hindi questions from textbook passages."""
        if not context or not str(context).strip():
            raise ValueError("Context passage cannot be empty.")

        ctx_clean = str(context).strip()
        ans_clean = str(answer).strip() if answer else ""

        if ans_clean:
            input_text = f"{ans_clean} </s> {ctx_clean} </s> <2hi>"
        else:
            input_text = f"{ctx_clean} </s> <2hi>"

        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                num_beams=num_beams if not do_sample else 1,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.pad_id,
                bos_token_id=self.bos_id,
                eos_token_id=self.eos_id,
                decoder_start_token_id=self.target_lang_id,
                forced_bos_token_id=self.target_lang_id
            )

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        clean_questions = []
        tags_to_strip = ["<2hi>", "<2en>", "<s>", "</s>", "<pad>", "<unk>"]
        
        for q in decoded_outputs:
            for tag in tags_to_strip:
                q = q.replace(tag, "")
            q = " ".join(q.split()).strip()
            if q and not q.endswith(("?", "।")):
                q += "?"
            clean_questions.append(q)

        return clean_questions

    def generate_batch(
        self,
        contexts: list,
        answers: Optional[list] = None,
        batch_size: int = 32,
        num_beams: int = 4,
        max_length: int = 64,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95
    ) -> list:
        """Generates natural Hindi questions for a batch of context-answer pairs."""
        if answers is None:
            answers = ["" for _ in contexts]

        all_results = []
        tags_to_strip = ["<2hi>", "<2en>", "<s>", "</s>", "<pad>", "<unk>"]

        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i + batch_size]
            batch_answers = answers[i:i + batch_size]

            batch_input_texts = []
            for ctx, ans in zip(batch_contexts, batch_answers):
                if ans and ans.strip():
                    batch_input_texts.append(f"{ans.strip()} </s> {ctx.strip()} </s> <2hi>")
                else:
                    batch_input_texts.append(f"{ctx.strip()} </s> <2hi>")

            inputs = self.tokenizer(
                batch_input_texts,
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    num_beams=num_beams if not do_sample else 1,
                    max_length=max_length,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=self.pad_id,
                    bos_token_id=self.bos_id,
                    eos_token_id=self.eos_id,
                    decoder_start_token_id=self.target_lang_id,
                    forced_bos_token_id=self.target_lang_id
                )

            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

            for q in decoded_outputs:
                for tag in tags_to_strip:
                    q = q.replace(tag, "")
                q = " ".join(q.split()).strip()
                if q and not q.endswith(("?", "।")):
                    q += "?"
                all_results.append(q)

        return all_results

    def generate_multi_questions(
        self,
        context: str,
        answer: str = "",
        num_questions: int = 3,
        num_beams: int = 4,
        max_length: int = 64,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95
    ) -> list:
        """
        Generates multiple distinct, diverse questions from a large textbook passage.
        Automatically discovers salient answer entities across the passage and conditions
        on localized sentence context to prevent question drift.
        When a specific answer is provided and multiple questions are requested, generates
        diverse phrasing variations for that target answer.
        """
        if not context or not str(context).strip():
            raise ValueError("Context passage cannot be empty.")

        ctx_clean = str(context).strip()
        ans_clean = str(answer).strip() if answer else ""

        # If user supplied a specific answer and wants only 1 question
        if ans_clean and num_questions <= 1:
            q = self.generate(
                context=ctx_clean,
                answer=ans_clean,
                num_beams=num_beams,
                max_length=max_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )[0]
            return [{
                "question": q,
                "answer": ans_clean,
                "source_sentence": ctx_clean
            }]

        results: List[Dict[str, str]] = []
        seen_questions = set()

        # Case 1: User specified an answer and requested multiple questions
        if ans_clean:
            nb = max(num_beams, num_questions + 1) if not do_sample else 1
            n_seq = min(num_questions * 2, nb) if not do_sample else num_questions * 2
            seq_qs = self.generate(
                context=ctx_clean,
                answer=ans_clean,
                num_beams=nb,
                max_length=max_length,
                num_return_sequences=n_seq,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )
            for q in seq_qs:
                q_clean = " ".join(q.split()).strip()
                if q_clean and q_clean not in seen_questions:
                    seen_questions.add(q_clean)
                    results.append({
                        "question": q_clean,
                        "answer": ans_clean,
                        "source_sentence": ctx_clean
                    })
                    if len(results) >= num_questions:
                        return results[:num_questions]

        # Case 2: Passage-wide multi-target question generation (or supplemental targets)
        candidates = extract_candidate_answers(ctx_clean, max_candidates=max(num_questions * 2, 8))

        # Filter out existing answer if already present
        filtered_cands = [c for c in candidates if not (ans_clean and (c["answer"] == ans_clean or ans_clean in c["answer"]))]

        # Fallback if no candidate entities were found
        if not filtered_cands and not results:
            sentences = [s.strip() for s in re.split(r"(?<=[।\?!])\s+", ctx_clean) if len(s.strip()) > 10]
            if not sentences:
                sentences = [ctx_clean]
            for s_idx, s in enumerate(sentences[:num_questions]):
                filtered_cands.append({
                    "answer": "",
                    "category": "sentence",
                    "sentence": s,
                    "sentence_index": s_idx
                })

        needed = num_questions - len(results)
        batch_targets = filtered_cands[:needed]

        if batch_targets:
            b_contexts = [ctx_clean for _ in batch_targets]
            b_answers = [c["answer"] for c in batch_targets]

            generated_qs = self.generate_batch(
                contexts=b_contexts,
                answers=b_answers,
                batch_size=len(b_contexts),
                num_beams=num_beams,
                max_length=max_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )

            for cand, q in zip(batch_targets, generated_qs):
                q_norm = " ".join(q.split()).strip()
                if q_norm and q_norm not in seen_questions:
                    seen_questions.add(q_norm)
                    results.append({
                        "question": q_norm,
                        "answer": cand.get("answer", ""),
                        "source_sentence": cand.get("sentence", ctx_clean)
                    })

        # Final guarantee: If still fewer questions than requested, generate sampling variations
        if len(results) < num_questions:
            top_cand = results[0]["answer"] if results else ans_clean
            if do_sample:
                # User chose sampling: try with varied temperatures
                for t_step in [temperature, temperature * 1.1, temperature * 1.3]:
                    if len(results) >= num_questions:
                        break
                    extra_qs = self.generate(
                        context=ctx_clean,
                        answer=top_cand,
                        num_beams=1,
                        max_length=max_length,
                        num_return_sequences=min(num_questions - len(results) + 1, 3),
                        do_sample=True,
                        temperature=t_step,
                        top_p=top_p
                    )
                    for q in extra_qs:
                        q_norm = " ".join(q.split()).strip()
                        if q_norm and q_norm not in seen_questions:
                            seen_questions.add(q_norm)
                            results.append({
                                "question": q_norm,
                                "answer": top_cand,
                                "source_sentence": ctx_clean
                            })
                            if len(results) >= num_questions:
                                break
            else:
                # User chose beam search: try with increasing beam widths
                for extra_beams in [num_beams, num_beams + 2, num_beams + 4]:
                    if len(results) >= num_questions:
                        break
                    extra_qs = self.generate(
                        context=ctx_clean,
                        answer=top_cand,
                        num_beams=extra_beams,
                        max_length=max_length,
                        num_return_sequences=min(num_questions - len(results) + 1, extra_beams),
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0
                    )
                    for q in extra_qs:
                        q_norm = " ".join(q.split()).strip()
                        if q_norm and q_norm not in seen_questions:
                            seen_questions.add(q_norm)
                            results.append({
                                "question": q_norm,
                                "answer": top_cand,
                                "source_sentence": ctx_clean
                            })
                            if len(results) >= num_questions:
                                break

        return results[:num_questions]


# Stopwords and grammatical particles for candidate answer cleansing
STOPWORDS_HINDI = {
    "यह", "वह", "इस", "उस", "इन", "उन", "हम", "आप", "वे", "जो", "जिस",
    "सब", "कुछ", "का", "की", "के", "और", "या", "पर", "नहीं", "भी", "तो",
    "ही", "था", "थे", "थी", "है", "हैं", "होता", "होते", "होती", "दिन",
    "समय", "साल", "वर्ष", "तरह", "रूप", "बात", "लोग", "काम", "बारे", "कारण",
    "कहा", "गया", "गई", "गए", "किया", "किए", "हुए", "हुई", "हुआ", "तहत",
    "दौरान", "बावजूद", "अनुसार", "खिलाफ", "विरुद्ध", "सहित", "समेत", "वजह", "द्वारा",
    "अमुक", "अमुक-अमुक"
}

PARTICLES_PREFIX = ["के ", "की ", "का ", "में ", "से ", "पर ", "को ", "ने ", "और ", "तथा ", "एवं ", "करके ", "वाले ", "दिन "]
PARTICLES_SUFFIX = [" के", " की", " का", " में", " से", " पर", " को", " ने", " और", " तथा", " एवं", " दिन", " वाले"]

CANDIDATE_PATTERNS = [
    # 1. Dates, Years, Eras (e.g. २६ जनवरी १९३१, सन् १९४७, १० मई १८५७, १९५० में)
    (r"(?:^|[\s।,])([०-९0-9]{1,2}\s+(?:जनवरी|फ़रवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+[०-९0-9]{4}|(?:सन्|साल|वर्ष)\s*[०-९0-9]{4}|[०-९0-9]{4}\s*में)(?:[\s।,!?]|$)", "date"),
    # 2. Specific Locations (Unicode-safe non-ASCII boundary)
    (r"(?:^|[\s।,])(कोलकाता|इंदौर|राजस्थान|काशी|अयोध्या|मेरठ|वाराणसी|भोपाल|लखनऊ|धर्मतल्ले|दिल्ली|मध्य प्रदेश)(?:[\s।,!?]|$)", "location"),
    # 3. Quoted terms & titles
    (r"[\"'\u201c\u2018]([^\u201d\u2019\"']+)[\"'\u201d\u2019]", "title"),
    # 4. Quantities and Counts (e.g. ३९५ अनुच्छेद, ८ अनुसूचियां, पाँच साल, तीन दरजे, सैकड़ों लोग)
    (r"(?:^|[\s।,])((?:[०-९0-9]+|(?:एक|दो|तीन|चार|पाँच|छह|सात|आठ|नौ|दस|ग्यारह|बारह|तेरह|चौदह|पंद्रह|बीस|पचास|सौ|हज़ार|लाख|सैकड़ों|करोड़ों))\s+[^\s।,]+(?:\s+[^\s।,]+){0,1})(?:[\s।,!?]|$)", "quantity"),
    # 5. Roles & Definitions (e.g. संविधान का जनक, प्रारूप समिति के अध्यक्ष)
    (r"(?:^|[\s।,])([^\s।,]+(?:\s+[^\s।,]+){0,2})\s+(?:कहलाते|कहलाता|कहलाती|कहा जाता|माना जाता|का जनक|के लेखक|के अध्यक्ष|का निर्माण|की उपस्थिति|के संपादन|का उपयोग)(?:[\s।,!?]|$)", "role_def"),
    # 6. Agents & Subjects (e.g. सुभाष चंद्र बोस के नेतृत्व, पुलिस कमिश्नर ने)
    (r"(?:^|[\s।,])([^\s।,]+(?:\s+[^\s।,]+){0,2})\s+(?:ने|के नेतृत्व में|के अनुसार|द्वारा)(?:[\s।,!?]|$)", "agent"),
    # 7. Salient topical entities with postpositions
    (r"(?:^|[\s।,])([^\s।,]{3,}(?:\s+[^\s।,]{3,}){0,1})\s+(?:में|से|पर|को|के लिए)(?:[\s।,!?]|$)", "topic")
]

def clean_candidate_span(span: str) -> str:
    """Strips leading/trailing grammatical particles and whitespace from entity spans."""
    span = span.strip(" ,।\"'\u2018\u201c\u201d\u2019\\/()[]")
    changed = True
    while changed:
        changed = False
        for p in PARTICLES_PREFIX:
            if span.startswith(p):
                span = span[len(p):].strip(" ,।\"'\u2018\u201c\u201d\u2019\\/()[]")
                changed = True
        for tp in PARTICLES_SUFFIX:
            if span.endswith(tp):
                span = span[:-len(tp)].strip(" ,।\"'\u2018\u201c\u201d\u2019\\/()[]")
                changed = True
    return span.strip()

def extract_candidate_answers(passage: str, max_candidates: int = 8) -> list:
    """
    Extracts high-salience candidate answer entities (dates, numbers, entities, definitions)
    from a Hindi textbook passage for multi-question generation.
    Supports clause-level extraction and avoids Devanagari regex boundary truncation.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[।\?!])\s+", passage) if len(s.strip()) > 8]
    candidates = []
    seen_spans = set()

    # Pass 1: Pattern extraction across all sentences and clauses
    for s_idx, sent in enumerate(sentences):
        sent_cands = []
        for pat, cat in CANDIDATE_PATTERNS:
            for m in re.finditer(pat, sent):
                raw = m.group(1) if m.groups() else m.group(0)
                span = clean_candidate_span(raw)
                if len(span) < 2 or len(span) > 30 or span in STOPWORDS_HINDI or len(span.split()) > 3:
                    continue
                # Substring overlap avoidance
                if any(span in s or s in span for s in seen_spans):
                    continue
                seen_spans.add(span)
                sent_cands.append({
                    "answer": span,
                    "category": cat,
                    "sentence": sent,
                    "sentence_index": s_idx
                })
                if len(sent_cands) >= 2:  # max 2 distinct candidates per sentence
                    break
            if len(sent_cands) >= 2:
                break
        candidates.extend(sent_cands)
        if len(candidates) >= max_candidates:
            return candidates[:max_candidates]

    # Pass 2: If still need more candidates, add sentences without candidates
    if len(candidates) < max_candidates:
        covered = {c["sentence_index"] for c in candidates}
        for s_idx, sent in enumerate(sentences):
            if s_idx not in covered:
                candidates.append({
                    "answer": "",
                    "category": "sentence",
                    "sentence": sent,
                    "sentence_index": s_idx
                })
            if len(candidates) >= max_candidates:
                break

    return candidates[:max_candidates]

def interactive_session(generator: QuestionGenerator):
    print("\n" + "="*60)
    print("Prashn-Setu: Interactive Hindi Textbook Question Generator")
    print("Type 'exit' to quit.")
    print("="*60 + "\n")

    while True:
        try:
            print("-" * 50)
            context = input("Enter Hindi Textbook Passage:\n> ").strip()
            if context.lower() in ["exit", "quit", "q"]:
                break
            if not context:
                continue

            answer = input("\nEnter Target Answer (or press Enter for auto multi-questions):\n> ").strip()

            if answer:
                results = generator.generate_multi_questions(context=context, answer=answer, num_questions=1, num_beams=4)
            else:
                results = generator.generate_multi_questions(context=context, answer="", num_questions=4, num_beams=4)
            
            print(f"\nGenerated Questions ({len(results)} distinct targets):")
            for i, item in enumerate(results, 1):
                ans_tag = f" [🎯 Answer: {item['answer']}]" if item.get('answer') else ""
                print(f" [{i}] {item['question']}{ans_tag}")
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hindi questions from textbook passages")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage1_indicnlg_qg",
                        help="Path to trained model or adapter checkpoint")
    args = parser.parse_args()

    q_gen = QuestionGenerator(checkpoint_path=args.checkpoint)
    interactive_session(q_gen)
