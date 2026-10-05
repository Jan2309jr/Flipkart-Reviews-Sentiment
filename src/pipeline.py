"""
Flipkart Reviews Sentiment Analysis - Full End-to-End Pipeline
Covers Phases 1-5 (Questions 1-50) of the student project.
Produces cleaned data, EDA plots, ML models, DL models, and a prediction pipeline.
"""

import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "flipkart_reviews.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
for d in [MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def log(msg):
    print(f"[PIPELINE] {msg}")


# =============================================================================
# PHASE 1 – Dataset Understanding & Data Preparation (Q1-Q8)
# =============================================================================

def load_and_validate():
    log("Q1: Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    log(f"Shape: {df.shape}")
    log(f"Columns: {df.columns.tolist()}")
    log(f"First 10 records:\n{df.head(10).to_string()}")
    return df


def identify_dtypes(df):
    log("Q2: Identifying data types...")
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    date_cols = ["Review_Date"]
    text_cols = ["Review_Summary", "Review_Text"]
    log(f"Numerical ({len(numerical)}): {numerical}")
    log(f"Categorical: {[c for c in categorical if c not in text_cols + date_cols]}")
    log(f"Date: {date_cols}")
    log(f"Text/NLP: {text_cols}")
    return numerical, categorical, date_cols, text_cols


def check_missing(df):
    log("Q3: Missing values...")
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"Missing": missing, "Pct": pct})
    log(f"\n{miss_df[miss_df['Missing'] > 0]}")
    return miss_df


def handle_duplicates(df):
    log("Q4: Detecting duplicates...")
    dup_id = df.duplicated(subset=["Review_ID"]).sum()
    dup_text = df.duplicated(subset=["Review_Text"]).sum()
    dup_combo = df.duplicated(subset=["Review_ID", "Review_Text"]).sum()
    log(f"Duplicate Review_ID: {dup_id}, Duplicate Review_Text: {dup_text}, Combo: {dup_combo}")
    # Keep first occurrence
    df = df.drop_duplicates(subset=["Review_ID"], keep="first")
    log(f"After dedup shape: {df.shape}")
    return df


def process_dates(df):
    log("Q5: Converting Review_Date and extracting features...")
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
    df["Year"] = df["Review_Date"].dt.year
    df["Month"] = df["Review_Date"].dt.month
    df["Day"] = df["Review_Date"].dt.day
    df["DayOfWeek"] = df["Review_Date"].dt.day_name()
    log(df[["Review_Date", "Year", "Month", "Day", "DayOfWeek"]].head())
    return df


def basic_stats(df):
    log("Q6: Basic statistical analysis...")
    cols = ["Product_Price", "Selling_Price", "Rating", "Discount_Percentage", "Delivery_Days", "Helpful_Votes"]
    log(f"\n{df[cols].describe().T}")
    return df[cols].describe()


def check_unrealistic(df):
    log("Q7: Checking unrealistic values...")
    issues = {}
    issues["price_neg"] = (df["Product_Price"] < 0).sum()
    issues["discount_out"] = ((df["Discount_Percentage"] < 0) | (df["Discount_Percentage"] > 100)).sum()
    issues["rating_out"] = ((df["Rating"] < 1) | (df["Rating"] > 5)).sum()
    issues["delivery_neg"] = (df["Delivery_Days"] < 0).sum()
    issues["exp_out"] = ((df["Customer_Experience_Score"] < 1) | (df["Customer_Experience_Score"] > 5)).sum()
    log(f"Unrealistic counts: {issues}")
    return issues


def clean_dataset(df):
    log("Q8: Creating cleaned dataset...")
    # Fill missing
    df["Review_Summary"] = df["Review_Summary"].fillna("")
    df["Customer_City"] = df["Customer_City"].fillna("Unknown")
    df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")
    # Clip unrealistic
    df["Discount_Percentage"] = df["Discount_Percentage"].clip(0, 100)
    df["Rating"] = df["Rating"].clip(1, 5)
    df["Delivery_Days"] = df["Delivery_Days"].clip(0, 30)
    df["Customer_Experience_Score"] = df["Customer_Experience_Score"].clip(1, 5)
    df = df.dropna(subset=["Review_Text", "Sentiment"])
    df = df.reset_index(drop=True)
    log(f"Cleaned shape: {df.shape}")
    df.to_csv(ROOT / "data" / "cleaned_reviews.csv", index=False)
    return df


# =============================================================================
# PHASE 2 – Exploratory Data Analysis (Q9-Q19)
# =============================================================================

def eda_sentiment_dist(df):
    log("Q9: Sentiment distribution...")
    counts = df["Sentiment"].value_counts()
    pct = df["Sentiment"].value_counts(normalize=True) * 100
    log(f"\n{counts}\n{pct.round(2)}")
    fig, ax = plt.subplots()
    colors = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}
    counts.plot(kind="bar", color=[colors.get(x, "gray") for x in counts.index], ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_sentiment_dist.png", dpi=120)
    plt.close()
    return counts


def eda_rating_sentiment(df):
    log("Q10: Rating vs Sentiment...")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Sentiment", y="Rating", order=["Negative", "Neutral", "Positive"],
                palette={"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#2ecc71"}, ax=ax)
    ax.set_title("Rating vs Sentiment")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_rating_sentiment.png", dpi=120)
    plt.close()
    log(df.groupby("Sentiment")["Rating"].mean().round(2))


def eda_categories(df):
    log("Q11: Categories by review count...")
    cat_counts = df["Category"].value_counts()
    log(cat_counts)
    fig, ax = plt.subplots()
    cat_counts.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Reviews per Category")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_category_counts.png", dpi=120)
    plt.close()
    return cat_counts


def eda_top_brands(df):
    log("Q12: Top 10 brands...")
    top = df["Brand"].value_counts().head(10)
    log(top)
    fig, ax = plt.subplots()
    top.plot(kind="bar", ax=ax, color="teal")
    ax.set_title("Top 10 Brands by Reviews")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_top_brands.png", dpi=120)
    plt.close()
    return top


def eda_avg_rating_category(df):
    log("Q13: Average rating per category...")
    avg = df.groupby("Category")["Rating"].mean().sort_values(ascending=False)
    log(avg.round(2))
    fig, ax = plt.subplots()
    avg.plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Avg Rating by Category")
    ax.set_ylim(0, 5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "05_avg_rating_category.png", dpi=120)
    plt.close()
    return avg


def eda_price_category(df):
    log("Q14: Avg selling price by category...")
    avg = df.groupby("Category")["Selling_Price"].mean().sort_values(ascending=False)
    log(avg.round(0))
    fig, ax = plt.subplots()
    avg.plot(kind="bar", ax=ax, color="purple")
    ax.set_title("Avg Selling Price by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "06_avg_price_category.png", dpi=120)
    plt.close()
    return avg


def eda_discount_sentiment(df):
    log("Q15: Discount vs Sentiment...")
    avg_disc = df.groupby("Sentiment")["Discount_Percentage"].mean()
    log(avg_disc.round(2))
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Sentiment", y="Discount_Percentage",
                order=["Negative", "Neutral", "Positive"], ax=ax)
    ax.set_title("Discount % vs Sentiment")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "07_discount_sentiment.png", dpi=120)
    plt.close()


def eda_delivery_experience(df):
    log("Q16: Delivery Days vs Experience Score...")
    corr = df[["Delivery_Days", "Customer_Experience_Score"]].corr().iloc[0, 1]
    log(f"Correlation: {corr:.3f}")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=42),
                    x="Delivery_Days", y="Customer_Experience_Score", alpha=0.3, ax=ax)
    ax.set_title(f"Delivery Days vs Experience (corr={corr:.2f})")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "08_delivery_experience.png", dpi=120)
    plt.close()


def eda_verified(df):
    log("Q17: Verified Purchase vs Sentiment...")
    ct = pd.crosstab(df["Verified_Purchase"], df["Sentiment"], normalize="index") * 100
    log(ct.round(1))
    fig, ax = plt.subplots()
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#e74c3c", "#f39c12", "#2ecc71"])
    ax.set_title("Sentiment % by Verified Purchase")
    ax.set_ylabel("%")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "09_verified_sentiment.png", dpi=120)
    plt.close()


def eda_helpful(df):
    log("Q18: Most helpful reviews sentiment...")
    top = df.nlargest(100, "Helpful_Votes")
    log(top["Sentiment"].value_counts())
    log(f"Avg helpful votes (top 100): {top['Helpful_Votes'].mean():.1f}")


def eda_word_freq(df):
    log("Q19: Word frequency by sentiment...")
    stop = set(stopwords.words("english"))
    for sent in ["Positive", "Neutral", "Negative"]:
        texts = " ".join(df[df["Sentiment"] == sent]["Review_Text"].astype(str).str.lower())
        words = [w for w in re.findall(r"[a-z]+", texts) if w not in stop and len(w) > 2]
        top_words = Counter(words).most_common(15)
        log(f"{sent} top words: {top_words}")
        # Plot
        if top_words:
            fig, ax = plt.subplots()
            ws, cs = zip(*top_words)
            ax.barh(list(ws)[::-1], list(cs)[::-1], color="steelblue")
            ax.set_title(f"Top Words – {sent}")
            plt.tight_layout()
            fig.savefig(PLOTS_DIR / f"10_wordfreq_{sent.lower()}.png", dpi=120)
            plt.close()


# =============================================================================
# PHASE 3 – NLP Text Preprocessing (Q20-Q28)
# =============================================================================

def combine_text(df):
    log("Q20: Combining Review_Summary + Review_Text...")
    df["Combined_Review"] = (
        df["Review_Summary"].fillna("").astype(str) + " " +
        df["Review_Text"].fillna("").astype(str)
    ).str.strip()
    return df


def clean_text_basic(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_nlp(df):
    log("Q21-Q26: NLP preprocessing...")
    stop = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    df["text_clean"] = df["Combined_Review"].apply(clean_text_basic)

    def tokenize(t):
        return word_tokenize(t)

    def remove_stop(tokens):
        return [w for w in tokens if w not in stop and len(w) > 2]

    def stem(tokens):
        return [stemmer.stem(w) for w in tokens]

    def lemma(tokens):
        return [lemmatizer.lemmatize(w) for w in tokens]

    # Sample for comparison
    sample = df["text_clean"].head(20).tolist()
    log("Q22-Q25: Tokenization / stop / stem / lemma samples...")
    for i, s in enumerate(sample[:5]):
        toks = tokenize(s)
        no_stop = remove_stop(toks)
        stemmed = stem(no_stop)
        lemmaed = lemma(no_stop)
        log(f"  Orig: {s[:80]}...")
        log(f"  Stem: {' '.join(stemmed)[:80]}...")
        log(f"  Lemma: {' '.join(lemmaed)[:80]}...")

    # Full pipeline: clean + tokenize + stop + lemma (preferred)
    def full_clean(t):
        t = clean_text_basic(t)
        toks = word_tokenize(t)
        toks = [w for w in toks if w not in stop and len(w) > 2 and not w.isdigit()]
        toks = [lemmatizer.lemmatize(w) for w in toks]
        return " ".join(toks)

    df["text_processed"] = df["Combined_Review"].apply(full_clean)
    log("Q26: Numerical tokens & extra spaces removed via full_clean.")

    # Q27
    log("Q27: Avg words/chars by sentiment...")
    df["word_count"] = df["text_processed"].str.split().str.len()
    df["char_count"] = df["text_processed"].str.len()
    stats = df.groupby("Sentiment")[["word_count", "char_count"]].mean().round(1)
    log(stats)

    # Q28: Use lemmatized as primary
    log("Q28: Using lemmatized text_processed as primary representation.")
    return df


# =============================================================================
# PHASE 4 – Feature Engineering & Traditional ML (Q29-Q40)
# =============================================================================

def build_ml_models(df):
    log("Phase 4: Traditional ML...")
    X = df["text_processed"].astype(str).tolist()
    y = df["Sentiment"].map(LABEL_MAP).values.astype(int)

    # Q34: Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    log(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Q29: Bag of Words
    log("Q29: Bag of Words...")
    bow = CountVectorizer(max_features=5000, ngram_range=(1, 1))
    X_bow = bow.fit_transform(X_train)
    log(f"BoW matrix shape: {X_bow.shape}")

    # Q30: TF-IDF uni / bi / uni+bi
    log("Q30: TF-IDF variants...")
    tfidf_uni = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
    tfidf_bi = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
    tfidf_mix = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X_tfidf = tfidf_mix.fit_transform(X_train)
    X_test_tfidf = tfidf_mix.transform(X_test)
    log(f"TF-IDF (1,2) shape: {X_tfidf.shape}")

    # Class weights for imbalance (Q37)
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, cw))
    log(f"Class weights: {class_weight}")

    models = {}
    results = {}

    # Q31: Multinomial NB
    log("Q31: Multinomial Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_tfidf, y_train)
    models["NaiveBayes"] = nb

    # Q32: Logistic Regression
    log("Q32: Logistic Regression...")
    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(X_tfidf, y_train)
    models["LogisticRegression"] = lr

    # Q33: Linear SVM
    log("Q33: Linear SVM...")
    svm = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000)
    svm.fit(X_tfidf, y_train)
    models["LinearSVM"] = svm

    # Q35-Q36: Evaluate
    log("Q35-Q36: Evaluation...")
    for name, model in models.items():
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
        log(f"{name}: Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
        if name == "LogisticRegression":
            log("\nClassification Report (LR):\n" + classification_report(
                y_test, preds, target_names=["Negative", "Neutral", "Positive"]
            ))
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                y_test, preds, display_labels=["Neg", "Neu", "Pos"], ax=ax, cmap="Blues"
            )
            ax.set_title("Confusion Matrix – Logistic Regression")
            plt.tight_layout()
            fig.savefig(PLOTS_DIR / "11_cm_lr.png", dpi=120)
            plt.close()

    # Q38: Hyperparameter tuning on best (LR often strong)
    log("Q38: RandomizedSearchCV on Logistic Regression...")
    param_dist = {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "solver": ["lbfgs", "saga"],
        "penalty": ["l2"],
    }
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=8,
        scoring="f1_macro",
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X_tfidf, y_train)
    best_lr = search.best_estimator_
    log(f"Best params: {search.best_params_}")
    preds = best_lr.predict(X_test_tfidf)
    f1 = f1_score(y_test, preds, average="macro")
    results["Tuned_LR"] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "Recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "F1": f1,
    }
    log(f"Tuned LR F1-macro: {f1:.4f}")
    models["Tuned_LR"] = best_lr

    # Q39: Important words
    log("Q39: Important features for Positive/Negative...")
    feature_names = np.array(tfidf_mix.get_feature_names_out())
    coef = best_lr.coef_
    # Negative class (0)
    top_neg = feature_names[np.argsort(coef[0])[-15:]][::-1]
    top_pos = feature_names[np.argsort(coef[2])[-15:]][::-1]
    log(f"Top Negative words: {top_neg.tolist()}")
    log(f"Top Positive words: {top_pos.tolist()}")

    # Q40: Save best ML + vectorizer
    best_name = max(results, key=lambda k: results[k]["F1"])
    best_model = models[best_name]
    joblib.dump(best_model, MODELS_DIR / "best_ml_model.joblib")
    joblib.dump(tfidf_mix, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(LABEL_MAP, MODELS_DIR / "label_map.joblib")
    log(f"Saved best ML model: {best_name} (F1={results[best_name]['F1']:.4f})")

    # Results table
    res_df = pd.DataFrame(results).T.round(4)
    res_df.to_csv(REPORTS_DIR / "ml_results.csv")
    log(f"\nML Results:\n{res_df}")
    return X_train, X_test, y_train, y_test, tfidf_mix, best_model, results, X_test_tfidf


# =============================================================================
# PHASE 5 – Deep Learning (Q41-Q50)
# =============================================================================

def build_dl_models(X_train, X_test, y_train, y_test):
    log("Phase 5: Deep Learning...")

    # Q41-Q42: Tokenizer + padding
    MAX_VOCAB = 12000
    MAX_LEN = 80
    EMBED_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    seq_train = tokenizer.texts_to_sequences(X_train)
    seq_test = tokenizer.texts_to_sequences(X_test)
    X_tr = pad_sequences(seq_train, maxlen=MAX_LEN, padding="post", truncating="post")
    X_te = pad_sequences(seq_test, maxlen=MAX_LEN, padding="post", truncating="post")
    y_tr = to_categorical(y_train, num_classes=3)
    y_te = to_categorical(y_test, num_classes=3)
    log(f"Vocab size: {min(MAX_VOCAB, len(tokenizer.word_index)+1)}, Max len: {MAX_LEN}")
    log(f"Seq shapes: {X_tr.shape}, {X_te.shape}")

    # Class weights
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(cw))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    results = {}

    # Q43: Embedding + Dense
    log("Q43: Embedding + Dense network...")
    model_dense = Sequential([
        Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ])
    model_dense.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hist_dense = model_dense.fit(
        X_tr, y_tr, validation_split=0.15, epochs=8, batch_size=64,
        class_weight=class_weight, callbacks=callbacks, verbose=1
    )
    results["Dense"] = evaluate_dl(model_dense, X_te, y_test, "Dense")

    # Q44: LSTM
    log("Q44: LSTM model...")
    model_lstm = Sequential([
        Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
        LSTM(64, dropout=0.2, recurrent_dropout=0.0),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ])
    model_lstm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hist_lstm = model_lstm.fit(
        X_tr, y_tr, validation_split=0.15, epochs=8, batch_size=64,
        class_weight=class_weight, callbacks=callbacks, verbose=1
    )
    results["LSTM"] = evaluate_dl(model_lstm, X_te, y_test, "LSTM")
    plot_history(hist_lstm, "LSTM")

    # Q45: BiLSTM
    log("Q45: Bidirectional LSTM...")
    model_bilstm = Sequential([
        Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, dropout=0.2)),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ])
    model_bilstm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hist_bilstm = model_bilstm.fit(
        X_tr, y_tr, validation_split=0.15, epochs=8, batch_size=64,
        class_weight=class_weight, callbacks=callbacks, verbose=1
    )
    results["BiLSTM"] = evaluate_dl(model_bilstm, X_te, y_test, "BiLSTM")
    plot_history(hist_bilstm, "BiLSTM")

    # Q46 experiments noted in results; best by F1
    best_dl_name = max(results, key=lambda k: results[k]["F1"])
    best_dl = {"Dense": model_dense, "LSTM": model_lstm, "BiLSTM": model_bilstm}[best_dl_name]
    log(f"Best DL model: {best_dl_name}")

    # Save
    best_dl.save(MODELS_DIR / "best_dl_model.keras")
    joblib.dump(tokenizer, MODELS_DIR / "tokenizer.joblib")
    with open(MODELS_DIR / "dl_config.txt", "w") as f:
        f.write(f"MAX_VOCAB={MAX_VOCAB}\nMAX_LEN={MAX_LEN}\nEMBED_DIM={EMBED_DIM}\nBEST={best_dl_name}\n")

    res_df = pd.DataFrame(results).T.round(4)
    res_df.to_csv(REPORTS_DIR / "dl_results.csv")
    log(f"\nDL Results:\n{res_df}")
    return tokenizer, best_dl, results, MAX_LEN


def evaluate_dl(model, X_te, y_test, name):
    preds_prob = model.predict(X_te, verbose=0)
    preds = np.argmax(preds_prob, axis=1)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    log(f"{name}: Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    if name in ("BiLSTM", "LSTM"):
        log("\nClassification Report:\n" + classification_report(
            y_test, preds, target_names=["Negative", "Neutral", "Positive"]
        ))
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_test, preds, display_labels=["Neg", "Neu", "Pos"], ax=ax, cmap="Greens"
        )
        ax.set_title(f"Confusion Matrix – {name}")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / f"12_cm_{name.lower()}.png", dpi=120)
        plt.close()
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


def plot_history(hist, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.history["accuracy"], label="train")
    axes[0].plot(hist.history["val_accuracy"], label="val")
    axes[0].set_title(f"{name} Accuracy")
    axes[0].legend()
    axes[1].plot(hist.history["loss"], label="train")
    axes[1].plot(hist.history["val_loss"], label="val")
    axes[1].set_title(f"{name} Loss")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"13_history_{name.lower()}.png", dpi=120)
    plt.close()


# =============================================================================
# Prediction Pipeline (Q50)
# =============================================================================

def predict_sentiment(text, model_type="ml"):
    """Reusable prediction function for new reviews."""
    # Load artifacts
    label_map = joblib.load(MODELS_DIR / "label_map.joblib")
    inv_map = {v: k for k, v in label_map.items()}

    # Preprocess
    stop = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def full_clean(t):
        t = str(t).lower()
        t = re.sub(r"[^a-z\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        toks = word_tokenize(t)
        toks = [w for w in toks if w not in stop and len(w) > 2 and not w.isdigit()]
        toks = [lemmatizer.lemmatize(w) for w in toks]
        return " ".join(toks)

    cleaned = full_clean(text)

    if model_type == "ml":
        vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        model = joblib.load(MODELS_DIR / "best_ml_model.joblib")
        X = vectorizer.transform([cleaned])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            pred = int(np.argmax(proba))
            conf = float(proba[pred])
        else:
            # LinearSVC decision_function
            decision = model.decision_function(X)[0]
            # Softmax-like
            exp = np.exp(decision - np.max(decision))
            proba = exp / exp.sum()
            pred = int(np.argmax(proba))
            conf = float(proba[pred])
        return inv_map[pred], conf, {inv_map[i]: float(p) for i, p in enumerate(proba)}
    else:
        tokenizer = joblib.load(MODELS_DIR / "tokenizer.joblib")
        model = load_model(MODELS_DIR / "best_dl_model.keras")
        with open(MODELS_DIR / "dl_config.txt") as f:
            cfg = dict(line.strip().split("=") for line in f if "=" in line)
        max_len = int(cfg["MAX_LEN"])
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        proba = model.predict(padded, verbose=0)[0]
        pred = int(np.argmax(proba))
        conf = float(proba[pred])
        return inv_map[pred], conf, {inv_map[i]: float(p) for i, p in enumerate(proba)}


def compare_and_report(ml_results, dl_results):
    log("Q49: ML vs DL comparison...")
    all_res = {**{f"ML_{k}": v for k, v in ml_results.items()},
               **{f"DL_{k}": v for k, v in dl_results.items()}}
    df = pd.DataFrame(all_res).T.round(4)
    df.to_csv(REPORTS_DIR / "final_comparison.csv")
    log(f"\nFinal Comparison:\n{df}")
    best = df["F1"].idxmax()
    log(f"Overall best by Macro-F1: {best} ({df.loc[best, 'F1']:.4f})")
    log("For this dataset, traditional ML (TF-IDF + LR/SVM) is often competitive or better "
        "than lightweight LSTM on short reviews, with far faster training/inference and "
        "easier interpretability. DL can improve with more data, pretrained embeddings, or BERT.")


def main():
    log("=" * 60)
    log("FLIPKART SENTIMENT ANALYSIS – FULL PIPELINE START")
    log("=" * 60)

    df = load_and_validate()
    identify_dtypes(df)
    check_missing(df)
    df = handle_duplicates(df)
    df = process_dates(df)
    basic_stats(df)
    check_unrealistic(df)
    df = clean_dataset(df)

    # EDA
    eda_sentiment_dist(df)
    eda_rating_sentiment(df)
    eda_categories(df)
    eda_top_brands(df)
    eda_avg_rating_category(df)
    eda_price_category(df)
    eda_discount_sentiment(df)
    eda_delivery_experience(df)
    eda_verified(df)
    eda_helpful(df)
    eda_word_freq(df)

    # NLP
    df = combine_text(df)
    df = preprocess_nlp(df)
    df.to_csv(ROOT / "data" / "processed_reviews.csv", index=False)

    # ML
    X_train, X_test, y_train, y_test, vectorizer, best_ml, ml_results, _ = build_ml_models(df)

    # DL
    tokenizer, best_dl, dl_results, max_len = build_dl_models(X_train, X_test, y_train, y_test)

    compare_and_report(ml_results, dl_results)

    # Quick test of prediction pipeline
    log("Q50: Testing prediction pipeline...")
    samples = [
        "Excellent product, works perfectly and battery life is great!",
        "Average quality, nothing special but usable.",
        "Waste of money, defective and poor performance.",
    ]
    for s in samples:
        sent, conf, probs = predict_sentiment(s, "ml")
        log(f"  [{sent} {conf:.2%}] {s[:60]}...")

    log("=" * 60)
    log("PIPELINE COMPLETE. Models saved in models/, plots in reports/plots/")
    log("=" * 60)


if __name__ == "__main__":
    main()
