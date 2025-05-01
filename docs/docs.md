# DoctorPhiv1 - Full Technical Documentation

## 🔍 Introduction
DoctorPhiv1 is a lightweight, research-focused QA model fine-tuned and evaluated for the medical domain. Using Microsoft’s `phi-2` as a base, this project demonstrates how a relatively small LLM can perform reliably on health-related questions by applying precise post-processing and targeted evaluation methods.

---

## 📂 Dataset: MedQuAD

### What is MedQuAD?
MedQuAD is a curated dataset of medical QA pairs, sourced from authoritative sites like MedlinePlus and NIH.

### Preprocessing
- Cleaned line breaks and HTML tags
- Trimmed whitespace
- Extracted question/answer fields

---

## 🧠 Model: Phi-2
- 2.7B parameter model
- Efficient for Colab GPUs using 4-bit quantization
- Loaded using `transformers` + `bitsandbytes`

---

## 🧾 Prompt Format
```text
Question: What are symptoms of ulcers?

Answer:
```

---

## 🔁 Reranking Logic
```python
import re
from bert_score import score
sents = re.split(r'(?<=[.!?])\s+', raw_output)
P, R, F1 = score(sents, [reference]*len(sents), lang="en")
top_sentence = sorted(zip(sents, F1), key=lambda x: x[1], reverse=True)[0][0]
```

- Filters noisy answers
- Keeps semantically strongest sentence only

---

## 🧪 Evaluation Metrics
- **BERTScore** (Primary): Precision, Recall, F1
- **BLEU**: N-gram overlap
- **Exact Match**: Strict string match
- **Token F1**: Bag-of-words overlap
- **Answer Length**: Word count of generated output

---

## 📉 Visualizations
- Histograms: BLEU, F1, Length
- Line plots: Sample-wise metrics
- Correlation matrix: Inter-metric dependency
- Top/Bottom analysis: Identify best and worst cases

---

## ⚙️ Tools
- Hugging Face Transformers & Datasets
- BERTScore
- Seaborn/Matplotlib
- Google Colab + Drive

---

## 🚀 Reproducibility
1. Clone repo or open notebook
2. Replace/extend sample set
3. Run inference, rerank, evaluate
4. Review plots and metrics

---

## 📈 Final Results
- Precision: 0.8613
- Recall: 0.8962
- F1 Score: 0.8764

---

## ⚠️ Disclaimer
Not for real-world medical use. Research/demo only.

---

## 🎯 Goal
Deliver compact, interpretable QA from a quantized model with explainable post-processing—all in a single notebook.
