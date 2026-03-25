# 🎫 HSRIS — Hybrid Semantic Retrieval & Intelligence System

> **Assignment 3 | Data Science | Batch 23F | Roll No. 3041 & 3026**

A multi-stage NLP pipeline that searches ~8,470 customer-support tickets using a smart blend of keyword matching and AI-powered semantic understanding — all built **from scratch** using pure PyTorch and NumPy, without any high-level ML wrappers.

---

## 🌐 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link-here.streamlit.app)

> *Replace the link above with your deployed Streamlit Cloud or HuggingFace Spaces URL.*

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
├── DS_ASS_03_23F_3041_&_23F_3026.ipynb   ← Main Kaggle notebook (do not modify)
├── streamlit_app.py                        ← Interactive web app
├── requirements.txt                        ← Python dependencies
├── customer_support_tickets.csv            ← Dataset (place here before running)
├── glove_cache/                            ← Auto-created on first run (GloVe download)
└── README.md                               ← You are here
```

---

## 🚀 Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/DS_ASS_03_23F.git
cd DS_ASS_03_23F
```

### 2. Get the dataset
Download `customer_support_tickets.csv` from
[Kaggle — Customer Support Ticket Dataset](https://www.kaggle.com/datasets/waseemalastal/customer-support-ticket-dataset)
and place it in the project folder.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the app
```bash
streamlit run streamlit_app.py
```

> ⏳ **First launch only:** The app will automatically download the GloVe embeddings file (~350 MB) from Stanford NLP and cache it in `glove_cache/`. This takes 2–5 minutes. Every run after that starts in ~10–20 seconds.

---

## ☁️ Deploying to Streamlit Cloud (Free)

1. Push this repo (including `customer_support_tickets.csv`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repo → set main file to `streamlit_app.py`.
4. Click **Deploy**. Done!

> ⚠️ Streamlit Cloud has a 1 GB file-size limit per repo. The CSV is ~1.2 MB (fine). GloVe is downloaded at runtime, not stored in the repo.

---

## 🖥️ App Features

| Feature | Description |
|---------|-------------|
| 📝 **Query box** | Describe your issue in plain English |
| 🎚️ **Alpha slider** | Slide from 0.0 (pure AI) to 1.0 (pure keyword) |
| 🔀 **Hybrid results** | Top-3 tickets scored by the blended formula |
| 📊 **TF-IDF column** | Keyword-match perspective |
| 🧠 **GloVe column** | Semantic/meaning perspective |
| 🔮 **Type prediction** | Predicted ticket type shown at the top |

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
- [x] GitHub Repository ← *you're here*
- [ ] [Medium Article](https://medium.com/) ← *add your link*
- [ ] [LinkedIn Post](https://linkedin.com/) ← *add your link*
- [ ] [Live App Link](https://streamlit.io/) ← *add after deploying*

---

## 👥 Authors

| Name | Roll No. |
|------|----------|
| *[Your Name]* | 23F-3041 |
| *[Your Name]* | 23F-3026 |

*Batch 23F — BS Data Science*

---

## 📄 License

This project is submitted as part of an academic assignment. Please do not copy or plagiarise.
