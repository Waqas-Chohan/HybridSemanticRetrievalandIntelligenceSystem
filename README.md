# 🎫 HSRIS — Hybrid Semantic Retrieval & Intelligence System

> **Assignment 3 | Data Science | Batch 23F | Roll No. 3041 & 3026**

A multi-stage NLP pipeline that searches ~8,470 customer-support tickets using a smart blend of keyword matching and AI-powered semantic understanding — all built **from scratch** using pure PyTorch and NumPy, without any high-level ML wrappers.

---

## 🧪 Execution Platform

This assignment notebook is executed on **Kaggle**.

- Dataset is attached through Kaggle **Add Input**
- GPU is enabled directly from Kaggle notebook settings
- Notebook runs end-to-end on Kaggle without local dataset setup

---

## 📖 What Is This?

Ever wondered how customer-support systems figure out what kind of problem you're describing? This project builds exactly that — a hybrid search engine that understands both **keywords** and **meaning**.

Imagine you write:
> *"My money never came back after I cancelled the order."*

A pure keyword system might miss it because the ticket database says *"refund request"* — not *"money came back"*. Our semantic layer understands those mean the same thing. The hybrid system gives you the best of both worlds.

### How It Works (Under the Hood)

| Layer | What It Does |
|-------|-------------|
| **Label Encoding** | Maps ticket priority (Low → 0, Medium → 1, High → 2, Critical → 3) |
| **One-Hot Encoding** | Converts ticket channel (Chat, Email, Phone, Social Media) into binary vectors |
| **Custom Tokenizer** | Regex-based cleaning + light stemming + stopword removal |
| **TF-IDF (Sparse)** | Builds a vocabulary of 6,000 tokens, computes IDF weights, stores a sparse inverted index |
| **N-Grams** | Generates bigrams & trigrams to capture phrases like "not working" vs "is working" |
| **GloVe (Dense)** | Loads 100-dimensional GloVe word vectors; uses IDF-weighted averaging for sentence embeddings |
| **Hybrid Scoring** | `FinalScore = α × TF-IDF + (1-α) × GloVe` (default α = 0.4) |

---

## 📁 Project Structure

```
assignment03/
│
├── DS_ASS_03_23F_3041_&_23F_3026.ipynb   ← Main Kaggle notebook
└── README.md
```

---

## 🚀 Running on Kaggle

### 1. Open the notebook on Kaggle
Upload or import `DS_ASS_03_23F_3041_&_23F_3026.ipynb` into Kaggle Notebooks.

### 2. Add dataset input
Use **Add Input** in Kaggle and attach:
[Kaggle — Customer Support Ticket Dataset](https://www.kaggle.com/datasets/waseemalastal/customer-support-ticket-dataset)

### 3. Enable GPU
In notebook settings, set **Accelerator = GPU**.

### 4. Run all cells
Execute the notebook from top to bottom on Kaggle.

---

## 📊 Performance Results

| Method | Mean Precision@5 |
|--------|-----------------|
| TF-IDF | 0.167 |
| GloVe (Semantic) | 0.133 |
| **Hybrid (α=0.4)** | **0.167** |

Batch inference of 100 queries on dual T4 GPUs completes in < 1 ms per batch (GPU-warmed).

---

## 💡 Where Semantic Search Wins

Here are real examples where GloVe understood intent that TF-IDF missed:

| Query | TF-IDF Top Result | GloVe Top Result |
|-------|-------------------|------------------|
| *"money returned not received"* | Delivery problem | ✅ Payment / Refund |
| *"laptop won't start after patch"* | Product compatibility | ✅ Hardware issue |
| *"cannot access my profile"* | Battery life | ✅ Account access |
| *"internet keeps dropping"* | Software bug | ✅ Network/WiFi |
| *"lost files after crash"* | Display issue | ✅ Data loss / Recovery |

---

## 🛠️ Technical Constraints Followed

- ✅ No `sklearn.TfidfVectorizer` or `LabelEncoder` — everything built from scratch
- ✅ Sparse TF-IDF stored as Python dicts (inverted index), not dense matrices
- ✅ OOV tokens handled with a zero-vector fallback
- ✅ IDF-weighted averaging (not plain mean-pooling) to prevent semantic dilution
- ✅ PyTorch tensors used for cosine similarity

---

## 📋 Assignment Checklist

- [x] Kaggle Notebook (dual T4 GPU)
- [x] GitHub Repository

---

## 👥 Authors

| Name | Roll No. |
|------|----------|
| Muhammad Taqi | 23F-3026 |
| Waqas Chohan | 23F-3041 |

*Batch 23F — BS Data Science*

---

## 📄 License

This project is submitted as part of an academic assignment. Please do not copy or plagiarise.
