# Prashn-Setu: System Architecture & Design Documentation

Welcome to the architectural specification repository for **Prashn-Setu**, an automated question generation engine for Indic-language (Hindi) school textbooks based on `ai4bharat/IndicBART`.

This directory houses the foundational system blueprints, component interaction graphs, alternative solution trade-off analyses, installation prerequisites, automated state trackers, and operational protocols.

---

## 📑 Architectural Index

1. **[FILES_CATALOG.md](file:///mnt/a/iiits/5/NLP/PROJECT/architecture/FILES_CATALOG.md)**
   * Exhaustive inventory of every file in the codebase.
   * Rationale: Why each file exists, what problem it solves, and its exact input/output contracts.
   
2. **[COMPONENT_INTERACTIONS.md](file:///mnt/a/iiits/5/NLP/PROJECT/architecture/COMPONENT_INTERACTIONS.md)**
   * Architectural block diagrams and Mermaid sequence/dataflow charts.
   * Inter-module dependencies and execution flow from raw data to inference.

3. **[SOLUTION_ANALYSIS_AND_ALTERNATIVES.md](file:///mnt/a/iiits/5/NLP/PROJECT/architecture/SOLUTION_ANALYSIS_AND_ALTERNATIVES.md)**
   * Is this the best solution so far?
   * Benchmarking IndicBART against alternatives: `IndicBARTSS`, `mT5`, `Sarvam-2B`, `Llama-3-Indic`, `Gemma-2`, and RAG pipelines.
   * Trade-offs between Answer-Aware vs. Answer-Agnostic question generation.

4. **[PREREQUISITES_AND_ENVIRONMENT_SETUP.md](file:///mnt/a/iiits/5/NLP/PROJECT/architecture/PREREQUISITES_AND_ENVIRONMENT_SETUP.md)**
   * Hardware requirements (VRAM, RAM, GPU architectures).
   * OS-level system libraries (CUDA, cuDNN, Python, SentencePiece).
   * Setup guides for Local Linux (Ubuntu/WSL) and Google Colab environments.

5. **[OPERATIONAL_GUIDELINES.md](file:///mnt/a/iiits/5/NLP/PROJECT/architecture/OPERATIONAL_GUIDELINES.md)**
   * Rules and guidelines for how AI models, developers, and autonomous agents must read, update, modify, and write code and documentation in this repository.

6. **[SYSTEM_STATE_TRACKER.md](file:///mnt/a/iiits/5/NLP/PROJECT/architecture/SYSTEM_STATE_TRACKER.md)**
   * Live ledger of all observed changes, file checksums, dataset statuses, and training checkpoints.
   * Maintained and refreshed automatically using `update_architecture_state.py`.
