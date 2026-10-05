"""
Flipkart Reviews Sentiment Analysis – Streamlit Deployment App
Run: streamlit run app/streamlit_app.py
"""

import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# Paths
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODELS = ROOT / "models"

sys.path.insert(0, str(ROOT / "src"))

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK data
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# Optional TF for DL
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Caching ----------
@st.cache_resource
def load_ml_artifacts():
    model = joblib.load(MODELS / "best_ml_model.joblib")
    vectorizer = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    label_map = joblib.load(MODELS / "label_map.joblib")
    return model, vectorizer, label_map


@st.cache_resource
def load_dl_artifacts():
    if not TF_AVAILABLE:
        return None, None, None
    model = load_model(MODELS / "best_dl_model.keras")
    tokenizer = joblib.load(MODELS / "tokenizer.joblib")
    cfg = {}
    with open(MODELS / "dl_config.txt") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                cfg[k] = v
    return model, tokenizer, cfg


def preprocess(text: str) -> str:
    stop = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    t = str(text).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    toks = word_tokenize(t)
    toks = [w for w in toks if w not in stop and len(w) > 2 and not w.isdigit()]
    toks = [lemmatizer.lemmatize(w) for w in toks]
    return " ".join(toks)


def predict_ml(text: str):
    model, vectorizer, label_map = load_ml_artifacts()
    inv = {v: k for k, v in label_map.items()}
    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        decision = model.decision_function(X)[0]
        exp = np.exp(decision - np.max(decision))
        proba = exp / exp.sum()
    pred_idx = int(np.argmax(proba))
    return inv[pred_idx], float(proba[pred_idx]), {inv[i]: float(p) for i, p in enumerate(proba)}


def predict_dl(text: str):
    model, tokenizer, cfg = load_dl_artifacts()
    if model is None:
        return None, None, None
    inv = {0: "Negative", 1: "Neutral", 2: "Positive"}
    cleaned = preprocess(text)
    max_len = int(cfg.get("MAX_LEN", 60))
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    proba = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    return inv[pred_idx], float(proba[pred_idx]), {inv[i]: float(p) for i, p in enumerate(proba)}


# ---------- UI ----------
st.title("📦 Flipkart Reviews Sentiment Analysis")
st.markdown(
    """
    **End-to-end NLP + ML + DL student project**  
    Enter a product review and get **Positive / Neutral / Negative** prediction with confidence.
    """
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio(
        "Model",
        ["Traditional ML (TF-IDF + LR/SVM)", "Deep Learning (BiLSTM)"] if TF_AVAILABLE else ["Traditional ML (TF-IDF + LR/SVM)"],
        index=0,
    )
    st.markdown("---")
    st.subheader("📊 Project Info")
    st.markdown(
        """
        - **Dataset**: 30,000 synthetic Flipkart reviews  
        - **Classes**: Positive / Neutral / Negative  
        - **ML**: TF-IDF → Naive Bayes, Logistic Regression, Linear SVM  
        - **DL**: Embedding + Dense, LSTM, BiLSTM  
        - **Primary metric**: Macro F1  
        """
    )
    st.markdown("---")
    st.caption("Models achieve near-perfect accuracy on this synthetic dataset due to clear lexical cues.")

# Main input
if "review_input" not in st.session_state:
    st.session_state["review_input"] = ""

def set_example(text):
    st.session_state["review_input"] = text

col1, col2 = st.columns([2, 1])
with col1:
    review = st.text_area(
        "Customer Review",
        height=150,
        placeholder="Example: Excellent product, works perfectly and battery life is great!",
        key="review_input"
    )
with col2:
    st.markdown("**Quick examples**")
    st.button("😊 Positive example", on_click=set_example, args=("Loved it! Perfect purchase, picture quality is sharp and delivery was fast.",))
    st.button("😐 Neutral example", on_click=set_example, args=("Decent product but nothing special. Average quality for the price.",))
    st.button("😞 Negative example", on_click=set_example, args=("Waste of money, do not recommend. Received a defective product.",))

predict_btn = st.button("🔍 Analyze Sentiment", type="primary", width="stretch")

if predict_btn and review.strip():
    use_dl = "Deep Learning" in model_choice
    with st.spinner("Analyzing..."):
        if use_dl:
            sentiment, conf, probs = predict_dl(review)
            if sentiment is None:
                st.error("Deep Learning model not available. Falling back to ML.")
                sentiment, conf, probs = predict_ml(review)
                use_dl = False
        else:
            sentiment, conf, probs = predict_ml(review)

    # Display
    color = {"Positive": "green", "Neutral": "orange", "Negative": "red"}[sentiment]
    st.markdown(f"### Prediction: :{color}[**{sentiment}**]  (confidence: **{conf:.1%}**)")

    # Probability bars
    st.subheader("Class probabilities")
    for label in ["Positive", "Neutral", "Negative"]:
        p = probs.get(label, 0)
        st.progress(min(p, 1.0), text=f"{label}: {p:.1%}")

    with st.expander("Preprocessed text"):
        st.code(preprocess(review))

elif predict_btn:
    st.warning("Please enter a review.")

# Footer / model comparison
st.markdown("---")
st.subheader("📈 Model Performance (held-out test set)")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Traditional ML**")
    try:
        import pandas as pd
        ml = pd.read_csv(ROOT / "reports" / "ml_results.csv", index_col=0)
        st.dataframe(ml.style.format("{:.4f}"), width="stretch")
    except Exception:
        st.write("ML results: Accuracy / F1 ≈ 1.00 (synthetic data is highly separable)")

with c2:
    st.markdown("**Deep Learning**")
    try:
        dl = pd.read_csv(ROOT / "reports" / "dl_results.csv", index_col=0)
        st.dataframe(dl.style.format("{:.4f}"), width="stretch")
    except Exception:
        st.write("DL (BiLSTM / Dense) also reach near-perfect Macro-F1 on this dataset.")

st.caption(
    "Project covers full pipeline: data cleaning → EDA → NLP preprocessing → "
    "TF-IDF + classic ML → LSTM/BiLSTM → evaluation → prediction pipeline. "
    "Ready for portfolio / Streamlit Cloud deployment."
)
