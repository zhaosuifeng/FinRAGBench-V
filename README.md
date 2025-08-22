<img src="assets/logo.png" alt="FinRAGBench-V Logo" width="80" align="left" />

# FinRAGBench-V: A Benchmark for Visual RAG in the Financial Domain

FinRAGBench-V is a **comprehensive benchmark for visual retrieval-augmented generation (RAG) in finance**, addressing the challenge that most existing financial RAG research focuses predominantly on text while overlooking rich visual content in financial documents. By integrating multimodal data and providing **visual citation**, FinRAGBench-V ensures traceability and supports robust evaluation of Multimodal Large Language Models (MLLMs).

🤗 [Dataset](https://huggingface.co/datasets/zsfhhh/FinRAGBench-V) | 📄 [Paper](https://arxiv.org/abs/2505.17471) | 🏠 [Project Page](https://github.com/zhaosuifeng/FinRAGBench-V)

---
<img src="assets/main_fig.png" alt="FinRAGBench-V main figure" width="80" align="center" />
## Overview

- **Multimodal Retrieval Corpus:**  
  - 60,780 Chinese pages and 51,219 English pages.
  - Covers heterogeneous data types including financial reports, charts, and tables.
  
- **High-Quality QA Dataset:**  
  - Human-annotated question-answer pairs.
  - Includes seven question categories spanning reasoning, fact-checking, trend analysis, and more.
  
- **Visual Citation:**  
  - Provides page- and block-level evidence to support generated answers.
  - Enables traceable and interpretable multimodal RAG.

---

## Baseline: RGenCite

We provide **RGenCite**, a multimodal RAG baseline that seamlessly integrates:

1. **Retrieval:** Efficiently retrieves relevant textual and visual information from the dataset.  
2. **Generation:** Produces high-quality answers grounded in retrieved content.  
3. **Fine-Grained Visual Citation:** Provides precise visual evidence (page- and block-level) to support answers.

---

## Automatic Citation Evaluation

We propose an **automatic evaluation method** for visual citation, which:

- Measures **precision and recall** at multiple citation levels.
- Uses **box-bounding** and **image-cropping** techniques to assess the alignment of cited visual evidence.
- Provides a systematic way to benchmark multimodal RAG performance beyond text generation.

---

## Key Insights from Experiments

- Different retrieval strategies significantly affect model performance.  
- Task-dependent variations highlight the challenges of multimodal reasoning.  
- Visual citation remains a challenging aspect, underscoring the need for robust multimodal evaluation benchmarks.  

These findings validate **FinRAGBench-V** as a valuable resource for developing and evaluating visual RAG systems in finance.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/zhaosuifeng/FinRAGBench-V.git
cd FinRAGBench-V

