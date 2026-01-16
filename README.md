# Text Summerization  Using Large Language Model(LLM)


##  Project Overview

This project implements an **abstractive text summarization system** using a **pre-trained Large Language Model (LLM)** and the **XSUM news dataset**.  
The goal is to demonstrate an **end-to-end NLP pipeline** including dataset handling, model usage, inference, and deployment using a simple web interface.

>  The objective of this project is to showcase **ML workflow understanding and system design**, not to build a highly optimized production model.

---

##  Dataset

- **Dataset Name:** XSUM.
- **Source:** Hugging Face Datasets
- **Task Type:** Abstractive Text Summarization

### Dataset Fields
| Field | Description |
|------|------------|
| `document` | Full news article text |
| `summary` | Ground truth abstractive summary |

The dataset is loaded using:
```python
from datasets import load_dataset
dataset = load_dataset("xsum")

Model Used: facebook/bart-large-xsum

Architecture: Transformer-based Encoder–Decoder (BART)

Justification

The model is fine-tuned specifically on the XSUM dataset

Suitable for abstractive summarization of news articles

Encoder–decoder structure allows understanding long documents

Pre-trained model enables efficient transfer learning


## Pipeline Explanation

1. Load the XSUM dataset containing news articles and summaries.
2. Preprocess text using the BART tokenizer (tokenization, truncation, padding).
3. Pass the processed text to a pre-trained BART encoder–decoder model.
4. Generate abstractive summaries using beam search decoding.
5. Deploy the inference pipeline using a Streamlit web application.


## Steps to Run the Project

1. Clone the repository
```bash
git clone <repository-url>
cd text-summarization

