# Prashn-Setu: System Architecture & Design Documentation

Welcome to the architectural specification repository for **Prashn-Setu (प्रश्न-सेतु)**, an automated question generation and pedagogical assessment platform for Hindi educational textbooks based on `ai4bharat/IndicBART` and parameter-efficient curriculum adaptation.

This directory houses the foundational system blueprints, component interaction workflows, alternative solution trade-off analyses, installation prerequisites, automated state trackers, and operational guidelines.

---

## 📑 Architectural Index

1. **[FILES_CATALOG.md](FILES_CATALOG.md)**
   * Complete, audited inventory of every file in the active repository.
   * Explains what each file is, why it exists, its dependencies, and operational contracts.
    
2. **[COMPONENT_INTERACTIONS.md](COMPONENT_INTERACTIONS.md)**
   * Architectural block diagrams and Mermaid sequence/dataflow charts.
   * End-to-end workflows: Document Ingestion $\to$ Sentence Chunking $\to$ Stage 2 LoRA Generation $\to$ Evidence Grounding $\to$ Question Bank $\to$ Practice Quiz & Exam Paper Assembly.

3. **[SOLUTION_ANALYSIS_AND_ALTERNATIVES.md](SOLUTION_ANALYSIS_AND_ALTERNATIVES.md)**
   * Deep technical analysis of IndicBART against alternatives: `IndicBARTSS`, `mT5`, `Sarvam-2B`, `Llama-3-Indic`, and RAG architectures.
   * Trade-offs between Answer-Aware vs. Answer-Agnostic question generation, full fine-tuning vs. LoRA, and Devanagari tokenization realities.

4. **[PREREQUISITES_AND_ENVIRONMENT_SETUP.md](PREREQUISITES_AND_ENVIRONMENT_SETUP.md)**
   * Hardware requirements (VRAM, RAM, CPU/GPU acceleration).
   * Cross-platform setup guides for Windows, Linux (Ubuntu/Debian), and Google Colab.
   * Python dependencies and environment isolation.

5. **[OPERATIONAL_GUIDELINES.md](OPERATIONAL_GUIDELINES.md)**
   * Rules and protocols for developers and AI agents modifying this repository.
   * Standards for Devanagari Unicode preservation, research integrity, and non-fabrication of metrics.

6. **[SYSTEM_STATE_TRACKER.md](SYSTEM_STATE_TRACKER.md)**
   * Live ledger of all codebase files, checksums, sizes, and operational statuses across the application.
