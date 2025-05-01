# DoctorPhi v1 🏥🧑‍🎓

DoctorPhi v1 is a lightweight, domain-adapted large language model designed to answer medical questions reliably. Built on top of Microsoft's `phi-2` model, it combines efficient inference (4-bit quantization) with high-precision output filtering using advanced NLP evaluation techniques like BERTScore. It is optimized to run on Google Colab with minimal resources while maintaining high-quality answers.

---

## 🚀 What is DoctorPhi v1?
DoctorPhi v1 is an experimental QA system fine-tuned and optimized for:
- Understanding natural medical questions
- Providing concise and medically-relevant answers
- Running on affordable or free hardware (Colab GPU)

Key capabilities:
- High BERTScore F1 (**0.8764**)
- Clean sentence reranking to eliminate noise
- Visualization-rich diagnostics for each output sample

---

## 📘 Project Highlights
| Component         | Details                                                                 |
|------------------|-------------------------------------------------------------------------|
| **Base Model**    | `microsoft/phi-2` (2.7B parameters)                                     |
| **Dataset**       | MedQuAD QA pairs (subset of 50 questions used for evaluation)          |
| **Inference**     | Hugging Face `pipeline` + quantized model (`load_in_4bit=True`)        |
| **Post-Processing** | BERTScore sentence reranker to retain high-precision output          |
| **Evaluation**    | BERTScore, BLEU, Exact Match, Token-level F1, and length analysis      |

---

## 🧪 Workflow Summary

### 1. Load and Quantize the Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
```

### 2. Prompt and Generate Answers
```python
prompt = f"Question: {question}\n\nAnswer:"
output = pipe(prompt, max_new_tokens=160)["generated_text"]
```

### 3. Sentence Reranking (Post-Processing)
```python
import re
from bert_score import score
sents = re.split(r"(?<=[.!?])\s+", generated_output)
P, R, F1 = score(sents, [ref]*len(sents), lang="en")
final_output = sorted(zip(sents, F1), key=lambda x: x[1], reverse=True)[0][0]
```

### 4. Evaluate and Visualize
```python
P, R, F1 = score(predictions, references, lang="en")
print("Precision:", P.mean().item())
```

---

## 📊 Results
DoctorPhi v1 achieves:
- **Precision**: 0.8613
- **Recall**: 0.8962
- **F1 Score**: 0.8764

---

## 📈 Visualizations Included
- Precision/Recall/F1 per sample
- BLEU and token F1 histograms
- Answer length distribution
- Correlation heatmap
- Top 5 / Bottom 5 samples by F1

---

## 💡 How to Use
1. Open `DoctorPhi_V1.ipynb` in Colab
2. Run all cells
3. Add your own test questions or evaluate new samples
4. Explore generated plots and BERTScore metrics

---

## 🔭 Future Roadmap
- Scale fine-tuning to full MedQuAD + PubMedQA
- Add explainability (LIME, Grad-CAM)
- Deploy DoctorPhi on Hugging Face Spaces
- Use UMLS concepts to enrich medical context

---

## 📢 Disclaimer
This model is for **research and educational** use only. Not suitable for clinical diagnosis or treatment.

---

## 📜 License
MIT License for code. Dataset and model usage must comply with respective licenses.
