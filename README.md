# Flipkart Reviews Sentiment Analysis

**End-to-end NLP + Machine Learning + Deep Learning student project**

30,000 synthetic Flipkart-style reviews | 25 columns | 3-class sentiment (Positive / Neutral / Negative)

## Project Structure

```
flipkart_sentiment_project/
├── data/
│   ├── flipkart_reviews.csv          # Original dataset
│   ├── cleaned_reviews.csv           # After missing/dup/invalid handling
│   └── processed_reviews.csv         # With NLP-cleaned text
├── models/
│   ├── best_ml_model.joblib          # Best traditional ML (Tuned LR / SVM)
│   ├── tfidf_vectorizer.joblib
│   ├── label_map.joblib
│   ├── best_dl_model.keras           # Best DL (BiLSTM)
│   ├── tokenizer.joblib
│   └── dl_config.txt
├── reports/
│   ├── plots/                        # EDA & evaluation charts
│   ├── ml_results.csv
│   ├── dl_results.csv
│   └── final_comparison.csv
├── src/
│   ├── pipeline.py                   # Full pipeline (Q1–Q50)
│   └── train_dl_light.py             # Lightweight DL training
├── app/
│   └── streamlit_app.py              # Deployable Streamlit UI
├── requirements.txt
└── README.md
```

## Pipeline Coverage (Questions 1–50)

| Phase | Topics |
|-------|--------|
| 1 | Load, dtypes, missing, duplicates, dates, stats, unrealistic values, clean dataset |
| 2 | Sentiment dist, rating vs sentiment, categories, brands, prices, discounts, delivery, verified, helpful, word freq |
| 3 | Combine text, lower/punct, tokenize, stopwords, stem vs lemma, numbers/spaces, length stats |
| 4 | BoW, TF-IDF (uni/bi), Naive Bayes, Logistic Regression, Linear SVM, stratified split, metrics, class imbalance, Grid/Random search, feature importance, save model |
| 5 | Keras Tokenizer, padding, Embedding+Dense, LSTM, BiLSTM, hyperparams, training curves, DL metrics, ML vs DL, prediction pipeline |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# (Optional) Re-run full pipeline
python src/pipeline.py
# Lightweight DL (if needed)
python src/train_dl_light.py

# Launch Streamlit app
streamlit run app/streamlit_app.py
```

## Results (held-out test)

On this synthetic dataset, lexical cues are very strong, so both traditional ML (TF-IDF + Logistic Regression / Linear SVM) and BiLSTM reach **Macro-F1 ≈ 1.0**.

- Prefer **Traditional ML** for speed, interpretability, and deployment simplicity.
- Prefer **BiLSTM** if you later move to noisier real-world reviews or add pretrained embeddings / BERT.

## Streamlit App Features

- Paste any review → Positive / Neutral / Negative + confidence
- Switch between ML and DL backends
- Class probability bars
- Preprocessed text view
- Side-by-side model performance tables

## Deployment

- **Local**: `streamlit run app/streamlit_app.py`
- **Streamlit Cloud**: Push repo, set main file to `app/streamlit_app.py`, ensure `models/` and `reports/` are included (or download from releases).
- Models are small (joblib + keras); no GPU required for inference.

## Learning Outcomes

- Full NLP pipeline (cleaning → lemma → TF-IDF / sequences)
- Multiclass metrics with class imbalance awareness
- Classic ML vs sequence DL trade-offs
- Reproducible artifacts + interactive demo for portfolio

---
*Student Project – Machine Learning & Deep Learning*
