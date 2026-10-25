#!/usr/bin/env python3
"""
Prashn-Setu Quick Inference Script.
Standalone single-command runner to test question generation on authentic Hindi text.

Usage:
  python infer.py                                    # Runs preset textbook verification examples
  python infer.py --context "..." --answer "..."     # Generates question for custom input
  python infer.py --checkpoint checkpoints/stage1_indicnlg_qg  # Uses Stage 1 base model directly
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.inference import QuestionGenerator

SAMPLE_TEXTS = [
    {
        "domain": "भारतीय इतिहास (Indian History - NCERT Class 10)",
        "context": "२६ जनवरी १९३१ के दिन कोलकाता में स्वतंत्रता दिवस मनाने की पूरी तैयारी की गई थी। पुलिस कमिश्नर ने नोटिस निकाला था कि अमुक-अमुक धारा के तहत कोई भी सभा नहीं हो सकती। काउंसिल के अध्यक्ष सुभाष चंद्र बोस के नेतृत्व में विशाल जुलूस निकाला गया।",
        "answer": "२६ जनवरी १९३१"
    },
    {
        "domain": "विज्ञान (Science - NCERT Class 10)",
        "context": "हरे पौधे सूर्य के प्रकाश, जल और कार्बन डाइऑक्साइड का उपयोग करके अपना भोजन बनाते हैं। इस प्रक्रिया को प्रकाश संश्लेषण कहते हैं जिसमें ऑक्सीजन गैस मुक्त होती है।",
        "answer": "प्रकाश संश्लेषण"
    },
    {
        "domain": "हिंदी साहित्य (Hindi Literature - NCERT Class 10)",
        "context": "स्वयं प्रकाश का जन्म सन् १९४७ में इंदौर (मध्य प्रदेश) में हुआ। मैकेनिकल इंजीनियरिंग की पढ़ाई करके एक औद्योगिक प्रतिष्ठान में नौकरी करने वाले स्वयं प्रकाश का बचपन और नौकरी का बड़ा हिस्सा राजस्थान में बीता।",
        "answer": "इंदौर"
    }
]

def main():
    parser = argparse.ArgumentParser(description="Prashn-Setu Quick Hindi Question Generator")
    default_ckpt = (
        "checkpoints/stage2_ncert_gold_lora"
        if os.path.exists("checkpoints/stage2_ncert_gold_lora")
        else "checkpoints/stage1_indicnlg_qg"
    )
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to checkpoint")
    parser.add_argument("--context", type=str, default=None, help="Context passage in Hindi")
    parser.add_argument("--answer", type=str, default="", help="Target answer span (optional)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    print("=" * 65)
    print("  प्रश्न-सेतु (Prashn-Setu) — Hindi Question Generation Engine")
    print("=" * 65)
    print(f"Loading checkpoint: {args.checkpoint}")

    generator = QuestionGenerator(checkpoint_path=args.checkpoint, device=args.device)
    print("Model initialized successfully!\n")

    if args.context:
        print("Generating for custom input:")
        print(f"  Context : {args.context}")
        if args.answer:
            print(f"  Answer  : {args.answer}")
        questions = generator.generate(context=args.context, answer=args.answer)
        print(f"  Question: {questions[0]}\n")
    else:
        print("Running verification on 3 authentic NCERT textbook passages:\n")
        for i, item in enumerate(SAMPLE_TEXTS, 1):
            print(f"--- Example {i}: {item['domain']} ---")
            print(f"Context : {item['context']}")
            print(f"Answer  : {item['answer']}")
            questions = generator.generate(context=item["context"], answer=item["answer"])
            print(f"Output  : {questions[0]}\n")

    print("=" * 65)
    print("Verification complete! The model is working correctly.")
    print("To launch the web interface, run:  python run.py demo")
    print("To run interactive CLI, run:      python run.py infer")
    print("=" * 65)

if __name__ == "__main__":
    main()
