"""
HSRIS — Hybrid Semantic Retrieval & Intelligence System
Streamlit Demo App

Replicates the Kaggle notebook pipeline:
  - Custom tokenizer (regex + light stemmer + stopword removal)
  - From-scratch TF-IDF with inverted index  (sparse retrieval)
  - GloVe 100-d IDF-weighted sentence embeddings (dense retrieval)
  - Hybrid scoring: FinalScore = α · TF-IDF + (1-α) · GloVe

Run:
    streamlit run streamlit_app.py
"""

import re
import math
import time
import os
import urllib.request
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_DIM       = 100
MAX_VOCAB       = 6000
MIN_DF          = 3
SEED            = 42

GLOVE_URL  = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"
GLOVE_DIR  = "glove_cache"

DATASET_FILE = "customer_support_tickets.csv"

STOPWORDS = {
    "a","an","the","and","or","but","if","to","of","for","on","in","with",
    "is","are","was","were","be","been","it","this","that","my","your","our",
    "i","im","me","we","you","they","he","she","as","at","from","by","so",
    "can","could","would","should",
}

# Priority colour mapping (for UI)
PRIORITY_COLORS = {
    "Critical": "#e74c3c",
    "High":     "#e67e22",
    "Medium":   "#f1c40f",
    "Low":      "#2ecc71",
}

# Ticket-type emoji mapping (cosmetic)
TYPE_EMOJI = {
    "Technical issue":       "🔧",
    "Billing inquiry":       "💳",
    "Refund request":        "💰",
    "Product inquiry":       "📦",
    "Cancellation request":  "❌",
}

device = torch.device("cpu")   # Streamlit Cloud is CPU-only


# ---------------------------------------------------------------------------
# Text utilities  (mirror of notebook — must stay identical)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\{[^{}]*\}", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b", " ", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def light_stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str):
    text = normalize_text(text)
    tokens = []
    for tok in text.split():
        if tok in STOPWORDS:
            continue
        if len(tok) < 3:
            continue
        tok = light_stem(tok)
        if tok and tok not in STOPWORDS and len(tok) >= 3:
            tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# GloVe download helper
# ---------------------------------------------------------------------------

def _ensure_glove():
    """Download & extract GloVe if not cached locally."""
    glove_path = os.path.join(GLOVE_DIR, GLOVE_FILE)
    if os.path.exists(glove_path):
        return glove_path

    os.makedirs(GLOVE_DIR, exist_ok=True)
    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")

    placeholder = st.empty()
    placeholder.info("⏳ Downloading GloVe embeddings (~350 MB) — first run only, please wait …")
    urllib.request.urlretrieve(GLOVE_URL, zip_path)

    import zipfile
    placeholder.info("📦 Extracting GloVe …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(GLOVE_FILE, GLOVE_DIR)
    os.remove(zip_path)
    placeholder.empty()
    return glove_path


# ---------------------------------------------------------------------------
# Core pipeline — cached across all Streamlit sessions
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="🔄 Building search index (first launch takes ~1–2 min) …")
def build_pipeline():
    """Load data, build TF-IDF index, load GloVe, build dense embeddings."""

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── 1. Load dataset ────────────────────────────────────────────────────
    if not os.path.exists(DATASET_FILE):
        st.error(
            "## 📂 Dataset not found\n\n"
            f"`{DATASET_FILE}` is missing from the app folder.\n\n"
            "**How to fix:**\n"
            "1. Download the CSV from Kaggle 👉 "
            "[Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)\n"
            "2. Rename it to exactly `customer_support_tickets.csv` if needed\n"
            f"3. Place it in the same directory as `streamlit_app.py`:\n"
            f"   `{os.path.abspath('.')}`\n"
            "4. Refresh the page"
        )
        st.stop()

    df = pd.read_csv(DATASET_FILE)

    # ── 2. Build combined document text ────────────────────────────────────
    df["doc_raw"] = (
        df["Ticket Subject"].fillna("").astype(str) + ". " +
        df["Ticket Description"].fillna("").astype(str) + ". " +
        df["Resolution"].fillna("").astype(str)
    )
    df["tokens"] = df["doc_raw"].apply(tokenize)

    # ── 3. Vocabulary ──────────────────────────────────────────────────────
    doc_freq: Counter = Counter()
    for toks in df["tokens"]:
        doc_freq.update(set(toks))

    candidates = [w for w, d in doc_freq.items() if d >= MIN_DF]
    candidates_sorted = sorted(candidates, key=lambda w: doc_freq[w], reverse=True)[:MAX_VOCAB]
    word2idx = {w: i for i, w in enumerate(candidates_sorted)}
    V = len(word2idx)
    N = len(df)

    # ── 4. IDF weights ─────────────────────────────────────────────────────
    idf = np.zeros(V, dtype=np.float32)
    for w, i in word2idx.items():
        idf[i] = math.log((N + 1) / (doc_freq[w] + 1)) + 1.0

    # ── 5. TF-IDF sparse index ─────────────────────────────────────────────
    doc_tfidf = []
    for toks in df["tokens"]:
        tf = Counter([t for t in toks if t in word2idx])
        if not tf:
            doc_tfidf.append({})
            continue
        total = sum(tf.values())
        vec = {}
        sq = 0.0
        for t, c in tf.items():
            idx = word2idx[t]
            wt = (c / total) * float(idf[idx])
            vec[idx] = wt
            sq += wt * wt
        norm = math.sqrt(sq) if sq > 0 else 1.0
        doc_tfidf.append({k: v / norm for k, v in vec.items()})

    inverted_index: dict = defaultdict(list)
    for d, vec in enumerate(doc_tfidf):
        for t_idx, wt in vec.items():
            inverted_index[t_idx].append((d, wt))

    # ── 6. Load GloVe ──────────────────────────────────────────────────────
    glove_path = _ensure_glove()
    glove: dict = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            glove[parts[0]] = np.asarray(parts[1:], dtype=np.float32)

    # ── 7. Dense sentence embeddings ───────────────────────────────────────
    def sentence_embedding(tokens):
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        weight_sum = 0.0
        for t in tokens:
            g = glove.get(t)
            if g is None:
                continue
            w = float(idf[word2idx[t]]) if t in word2idx else 1.0
            vec += w * g
            weight_sum += w
        if weight_sum > 0:
            vec /= weight_sum
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec /= nrm
        return vec

    doc_semantic = np.vstack([sentence_embedding(toks) for toks in df["tokens"]]).astype(np.float32)
    doc_semantic_t = torch.tensor(doc_semantic, dtype=torch.float32, device=device)

    return df, word2idx, idf, inverted_index, glove, doc_semantic_t, sentence_embedding


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def build_query_tfidf(query, word2idx, idf):
    toks = tokenize(query)
    tf = Counter([t for t in toks if t in word2idx])
    if not tf:
        return {}, toks
    total = sum(tf.values())
    qvec = {}
    sq = 0.0
    for t, c in tf.items():
        idx = word2idx[t]
        wt = (c / total) * float(idf[idx])
        qvec[idx] = wt
        sq += wt * wt
    norm = math.sqrt(sq) if sq > 0 else 1.0
    return {k: v / norm for k, v in qvec.items()}, toks


def sparse_tfidf_search(query, word2idx, idf, inverted_index, n_docs, top_k=5):
    qvec, qtoks = build_query_tfidf(query, word2idx, idf)
    if not qvec:
        return [], qtoks
    scores: dict = defaultdict(float)
    for t_idx, q_wt in qvec.items():
        for doc_id, d_wt in inverted_index.get(t_idx, []):
            scores[doc_id] += q_wt * d_wt
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked, qtoks


def semantic_search(query, glove, word2idx, idf, sentence_embedding_fn, doc_semantic_t, top_k=5):
    q_toks = tokenize(query)
    q_vec = sentence_embedding_fn(q_toks)
    q_t = torch.tensor(q_vec, dtype=torch.float32, device=device).unsqueeze(0)
    sims = torch.mm(q_t, doc_semantic_t.T).squeeze(0)
    vals, idxs = torch.topk(sims, k=top_k)
    return [(int(i), float(v)) for i, v in zip(idxs.cpu().numpy(), vals.cpu().numpy())], q_toks


def hybrid_search(query, alpha, word2idx, idf, inverted_index, glove,
                  sentence_embedding_fn, doc_semantic_t, n_docs, top_k=5):
    tf_ranked, _ = sparse_tfidf_search(query, word2idx, idf, inverted_index, n_docs, top_k=n_docs)
    sm_ranked, _ = semantic_search(query, glove, word2idx, idf, sentence_embedding_fn, doc_semantic_t, top_k=n_docs)

    tfidf_scores = np.zeros(n_docs, dtype=np.float32)
    sem_scores   = np.zeros(n_docs, dtype=np.float32)
    for i, s in tf_ranked:
        tfidf_scores[i] = s
    for i, s in sm_ranked:
        sem_scores[i] = s

    final = alpha * tfidf_scores + (1.0 - alpha) * sem_scores
    top_idx = np.argsort(-final)[:top_k]
    return [(int(i), float(final[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _priority_badge(priority: str) -> str:
    color = PRIORITY_COLORS.get(str(priority), "#95a5a6")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.78em;font-weight:600">{priority}</span>'


def _type_badge(ticket_type: str) -> str:
    emoji = TYPE_EMOJI.get(str(ticket_type), "🎫")
    return f'<span style="background:#2c3e50;color:white;padding:2px 8px;border-radius:10px;font-size:0.78em;font-weight:600">{emoji} {ticket_type}</span>'


def render_result_card(row, rank: int, score: float, method_color: str):
    priority    = row.get("Ticket Priority", "N/A")
    ticket_type = row.get("Ticket Type",     "N/A")
    subject     = row.get("Ticket Subject",  "N/A")
    desc        = str(row.get("Ticket Description", "")).replace("{product_purchased}", "[product]")[:200]
    resolution  = str(row.get("Resolution", "N/A"))[:160]

    st.markdown(
        f"""
        <div style="border:1px solid {method_color};border-radius:10px;padding:12px;margin-bottom:10px;background:#fafafa">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="font-weight:700;font-size:1em;color:{method_color}">#{rank}</span>
            {_type_badge(ticket_type)}
            {_priority_badge(priority)}
            <span style="font-size:0.8em;color:#7f8c8d;margin-left:auto">score {score:.4f}</span>
          </div>
          <div style="font-weight:600;margin-bottom:4px">📋 {subject}</div>
          <div style="font-size:0.85em;color:#555;margin-bottom:4px">{desc}…</div>
          <div style="font-size:0.82em;color:#27ae60"><b>Resolution:</b> {resolution}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_column(title: str, method_color: str, results, df):
    st.markdown(
        f"<h4 style='color:{method_color};text-align:center'>{title}</h4>",
        unsafe_allow_html=True,
    )
    if not results:
        st.info("No results found.")
        return
    for rank, (idx, score) in enumerate(results, 1):
        render_result_card(df.iloc[idx], rank, score, method_color)


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="HSRIS — Hybrid Ticket Search",
        page_icon="🎫",
        layout="wide",
    )

    # ── Header ──
    st.markdown(
        """
        <div style="text-align:center;padding:20px 0 10px">
          <h1 style="font-size:2.2em">🎫 HSRIS — Hybrid Semantic Retrieval System</h1>
          <p style="font-size:1.05em;color:#555">
            Search ~8 470 customer-support tickets using <b>TF-IDF</b>, <b>GloVe embeddings</b>, or a <b>blend of both</b>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load pipeline ──
    (df, word2idx, idf, inverted_index,
     glove, doc_semantic_t, sentence_embedding_fn) = build_pipeline()

    n_docs = len(df)

    # ── Sidebar controls ──
    with st.sidebar:
        st.header("⚙️ Search Settings")
        alpha = st.slider(
            "Alpha  (TF-IDF ← → GloVe)",
            min_value=0.0, max_value=1.0, value=0.4, step=0.05,
            help="α=1.0 → pure keyword search (TF-IDF).\nα=0.0 → pure semantic search (GloVe).\nα=0.4 → assignment default."
        )
        top_k = st.slider("Results to show", min_value=1, max_value=10, value=3)
        st.divider()
        st.markdown(
            "**How the score works:**\n\n"
            "`FinalScore = α · TF-IDF + (1-α) · GloVe`\n\n"
            "Slide α toward 1 for keyword matching, toward 0 for semantic meaning."
        )

    # ── Query input ──
    query = st.text_area(
        "📝 Describe your issue:",
        placeholder="e.g. payment failed and I need a refund …",
        height=100,
    )

    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

    if not search_clicked or not query.strip():
        st.info("👆 Enter your issue description above and click **Search**.")
        return

    # ── Run search ──
    t0 = time.time()

    hy_results = hybrid_search(
        query, alpha, word2idx, idf, inverted_index,
        glove, sentence_embedding_fn, doc_semantic_t, n_docs, top_k=top_k,
    )
    tf_results, _ = sparse_tfidf_search(
        query, word2idx, idf, inverted_index, n_docs, top_k=top_k,
    )
    sm_results, _ = semantic_search(
        query, glove, word2idx, idf, sentence_embedding_fn, doc_semantic_t, top_k=top_k,
    )

    elapsed = time.time() - t0

    # ── Predict ticket type (from hybrid top-1) ──
    if hy_results:
        predicted_type = df.iloc[hy_results[0][0]]["Ticket Type"]
        predicted_priority = df.iloc[hy_results[0][0]]["Ticket Priority"]
        st.markdown(
            f"""
            <div style="background:#eaf4fb;border-left:4px solid #2980b9;
                        padding:10px 16px;border-radius:6px;margin-bottom:16px">
              <b>🔮 Predicted ticket type:</b> {_type_badge(predicted_type)} &nbsp;
              {_priority_badge(predicted_priority)} &nbsp;
              <span style="float:right;font-size:0.82em;color:#7f8c8d">
                Search completed in {elapsed*1000:.1f} ms
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Results columns ──
    col_hy, col_tf, col_gl = st.columns(3)

    with col_hy:
        render_results_column(
            f"🔀 Hybrid  (α={alpha:.2f})", "#8e44ad", hy_results, df,
        )
    with col_tf:
        render_results_column("📊 TF-IDF  (keyword)", "#2980b9", tf_results, df)
    with col_gl:
        render_results_column("🧠 GloVe  (semantic)", "#27ae60", sm_results, df)


if __name__ == "__main__":
    main()
