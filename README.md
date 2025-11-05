# Benchmarking the ASKE Model on Legal Knowledge Extraction and Text Classification

## Overview

This repository presents the **ASKE Benchmark**—an evaluation suite and model for automated legal knowledge extraction and classification across multiple tasks and datasets used in legal NLP. ASKE is benchmarked on the LEXGLUE datasets, and its performance is compared to state-of-the-art transformer-based baselines (BERT, RoBERTa, Legal-BERT, DeBERTa, Longformer, CaseLaw-BERT, and others).

- **Paper:** Benchmarking the ASKE Model on Legal Knowledge Extraction and Text Classification



---

## Supported Tasks and Datasets

| Dataset      | Source                | Sub-domain   | Task Type             | Train/Dev/Test Instances | Classes |
|--------------|----------------------|--------------|-----------------------|--------------------------|---------|
| LEXGLUE      | Chalkidis et al.     | EU/US Law    | Multi-label/class/QA  | Multiple                 | Varies  |
| SCOTUS       | Spaeth et al.        | US Law       | Multi-class           | 5K / 1.4K / 1.4K         | 14      |
| UNFAIR-ToS   | Lippi et al.         | Contracts    | Multi-label           | 5.5K / 2.2K / 1.6K       | 8+1     |
| LEDGAR       | Tuggener et al.      | Contracts    | Multi-class           | 60K / 10K / 10K          | 100     |
| EUR-LEX      | Chalkidis et al.     | EU Law       | Multi-label           | 55K / 5K / 5K            | 100     |
| ECtHR Task A | Chalkidis et al.     | ECHR         | Multi-label           | 9K / 1K / 1K             | 10+1    |
| ECtHR Task B | Chalkidis et al.     | ECHR         | Multi-label           | 9K / 1K / 1K             | 10+1    |

---

## Model Overview

- **ASKE Model:** Context-aware embedding for legal documents, iterative concept extraction, zero-shot chunk classification, terminology enrichment, and conceptual graph refinement.
- **Baselines:** BERT, RoBERTa, DeBERTa, Longformer, Legal-BERT, CaseLaw-BERT.
- **Key Features:** Handles domain-specific vocabulary, long documents, clustering and embeddings for concept extraction and legal tasks.

---

## Results

| Model         | ECtHR -F1 | ECtHR m-F1 | SCOTUS -F1 | SCOTUS m-F1 | EUR-LEX -F1 | EUR-LEX m-F1 | LEDGAR -F1 | LEDGAR m-F1 | UNFAIR-ToS -F1 | UNFAIR-ToS m-F1 |
|---------------|-----------|------------|------------|-------------|-------------|--------------|------------|-------------|----------------|-----------------|
| TFIDF SVM     | 69.6      | 58.4       | 78.2       | 69.5        | 71.3        | 51.4         | 87.2       | 82.4        | 95.4           | 78.8            |
| BERT          | 75.4      | 68.6       | 68.3       | 58.3        | 71.4        | 57.2         | 87.6       | 81.8        | 95.6           | 81.3            |
| RoBERTa       | 73.2      | 63.5       | 71.6       | 62.0        | 71.9        | 57.9         | 87.9       | 82.3        | 95.2           | 79.2            |
| DeBERTa       | 74.0      | 66.9       | 71.1       | 62.7        | 72.1        | 57.4         | 88.2       | 83.1        | 95.5           | 80.3            |
| Longformer    | 74.3      | 68.2       | 72.9       | 64.0        | 71.6        | 57.7         | 88.2       | 83.0        | 95.5           | 80.9            |
| BigBird       | 74.4      | 66.9       | 72.8       | 62.0        | 71.5        | 56.8         | 87.8       | 82.6        | 95.7           | 81.3            |
| Legal-BERT    | 75.2      | 70.5       | 76.4       | 66.5        | 72.1        | 57.4         | 88.2       | 83.0        | 96.0           | 83.0            |
| CaseLaw-BERT  | 74.7      | 66.8       | 76.6       | 65.9        | 70.7        | 56.6         | 88.3       | 83.0        | 96.0           | 82.3            |
| **ASKE**      | **75.7**  | **70.3**   | **77.1**   | **67.8**    | **73.0**    | **58.5**     | **88.7**   | **83.5**    | **96.2**       | **83.5**        |

**Bold** = Best model for the dataset/metric.

---

## Installation & Requirements

Install core dependencies (see also `requirements.txt` if added):

pip install pandas matplotlib seaborn sentence-transformers torch scikit-learn numpy tqdm


---

## Usage & Running Experiments

Main experiments can be launched with:

python run_benchmark.py


This script:
- Loads all datasets
- Benchmarks baseline models (LegalBERT, RoBERTa, SBERT, Longformer, etc.)
- Runs and evaluates the ASKE model
- Saves evaluation metrics in JSON
- Generates tables and summary plots

**Note:**
- Datasets must be available in the paths expected (see code or documentation). Extract `legal_nlp_benchmark.zip` if needed.
- All evaluation and configuration is controlled from `run_benchmark.py` and accompanying support code in `/src`.

---

## Practical Applications

- Automated contract analysis (classification, clause extraction, risk identification)
- Enhanced legal document search and review
- Compliance checking and risk management for contracts and regulations
- Building legal ontologies and conceptual graphs for legal AI

---

## Limitations and Future Work

- Evaluation is currently limited to English datasets.
- Transfer to multilingual and cross-jurisdictional corpora is untested.
- Planned improvements: model interpretability, hybridization with larger SOTA models, and extension to additional legal task domains.

---

## Acknowledgments

Thanks to all contributors, collaborators, and dataset creators, and to the reviewers for their feedback.

---

## Resources

- **Paper/Preprint:** [link-to-paper]
- **Code:** [https://github.com/oziofficial5/Aske](https://github.com/oziofficial5/Aske)


---



