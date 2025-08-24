<img src="assets/logo.png" alt="FinRAGBench-V Logo" width="80" align="left" />

# FinRAGBench-V: A Benchmark for Visual RAG in the Financial Domain

FinRAGBench-V is a **comprehensive benchmark for visual retrieval-augmented generation (RAG) in finance**, addressing the challenge that most existing financial RAG research focuses predominantly on text while overlooking rich visual content in financial documents. By integrating multimodal data and providing **visual citation**, FinRAGBench-V ensures traceability and supports robust evaluation of Multimodal Large Language Models (MLLMs).

🤗 [Dataset](https://huggingface.co/datasets/zsfhhh/FinRAGBench-V) | 📄 [Paper](https://arxiv.org/abs/2505.17471) | 🏠 [Project Page](https://github.com/zhaosuifeng/FinRAGBench-V)

<p align="center">
  <img src="assets/main_fig.png" alt="FinRAGBench-V main figure" width="100%" />
</p>

## Benchmark: FinRAGBench-V

- 📊 **Multimodal Retrieval Corpus:**  We construct a multimodal financial corpus by collecting documents from various real-world financial sources, including research reports, financial statements, prospectuses, academic papers, financial magazines, and financial news. The corpus contains 60,780 Chinese pages and 51,219 English pages from 1,104 Chinese and 1,105 English documents.
  
- 📝 **High-Quality QA Dataset:**  We construct  a high-quality, human-annotated question-answering (QA) dataset spanning heterogeneous data types (charts, tables, and texts) and seven question categories, including time-sensitive, numerical calculations, comparison and sorting, and multi-page queries.

## Baseline: RGenCite

We provide **RGenCite**, a multimodal RAG baseline that seamlessly integrates:

- 🔍 **Retrieval:** Efficiently retrieves relevant textual and visual information from the dataset.  
- ✍️ **Generation:** Produces high-quality answers grounded in retrieved content.  
- 📌 **Fine-Grained Visual Citation:** Provides precise visual evidence (page- and block-level) to support answers.

## Automatic Citation Evaluation

We propose an **automatic evaluation method** for visual citation, which:

- Measures **precision and recall** at multiple citation levels.
- Uses **box-bounding** and **image-cropping** techniques to assess the alignment of cited visual evidence.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/zhaosuifeng/FinRAGBench-V.git
cd FinRAGBench-V

# 2. Create and activate a Conda environment
conda create --name finragbench python=3.11 -y
conda activate finragbench

#3. Install dependencies
pip install -r requirements.txt
pip install -e .

#4. Create folders for English and Chinese corpus and download the corpus parts from zsfhhh/FinRAGBench-V
Create directories for storing the English and Chinese corpus parts:
mkdir -p ./data/corpus/en
Then, download the corresponding corpus parts from zsfhhh/FinRAGBench-V, and move them into the respective folders.

#5. Concatenate parts and unzip
# Navigate to English corpus folder
cd ./data/corpus/en
# Concatenate tar.gz parts into a single archive (if needed)
cat part_*.tar.gz > corpus_en.tar.gz
# Unzip the combined archive
tar -xzvf corpus_en.tar.gz




