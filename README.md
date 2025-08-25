<img src="assets/logo.png" alt="FinRAGBench-V Logo" width="80" align="left" />

# FinRAGBench-V: A Benchmark for Visual RAG in the Financial Domain

FinRAGBench-V is a **comprehensive benchmark for visual retrieval-augmented generation (RAG) in finance**, addressing the challenge that most existing financial RAG research focuses predominantly on text while overlooking rich visual content in financial documents. By integrating multimodal data and providing **visual citation**, FinRAGBench-V ensures traceability and supports robust evaluation of Multimodal Large Language Models (MLLMs).

                                        🤗 [Dataset](https://huggingface.co/datasets/zsfhhh/FinRAGBench-V)  📄 [Paper](https://arxiv.org/abs/2505.17471) 

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

## Installation

```bash
git clone https://github.com/zhaosuifeng/FinRAGBench-V.git
cd FinRAGBench-V
conda create --name finragbench python=3.11 -y
conda activate finragbench
pip install -r requirements.txt
pip install -e .
```

## Dataset Download and Preprocessing
Download the corpus parts from zsfhhh/FinRAGBench-V, store them into ./data/corpus/en and ./data/corpus/ch respectively, and preprocess it.
```bash
mkdir -p ./data/corpus/en
cd ./data/corpus/en
cat part_*.tar.gz > corpus_en.tar.gz
tar -xzvf corpus_en.tar.gz
cd FinRAGBench-V/prepare_data.
python generate_parquet.py
```

## Baseline
For retrieval and its evaluation:
```bash
cd retrieval
python eval_mm_retriever.py
```
use encode_config.json for encoding and retrieve_config.json for retrieval.

For generating answers with visual citations:
```bash
cd generation
python generate.py
```
For evaluating answers and citations:
```bash
cd eval
python eval_generation.py
python eval_citation.py
```
## Other Related Projects
- [OpenMatch](https://github.com/thunlp/OpenMatch)


## Citation
If you find this work useful, please cite:
```bibtex
@misc{zhao2025finragbenchvbenchmarkmultimodalrag,
      title={FinRAGBench-V: A Benchmark for Multimodal RAG with Visual Citation in the Financial Domain}, 
      author={Suifeng Zhao and Zhuoran Jin and Sujian Li and Jun Gao},
      year={2025},
      eprint={2505.17471},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.17471}, 
}
```


