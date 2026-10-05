"""Lightweight DL training for Flipkart Sentiment (fast CPU-friendly)."""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PLOTS = ROOT / "reports" / "plots"
REPORTS = ROOT / "reports"
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}

def main():
    print("Loading processed data...")
    df = pd.read_csv(ROOT / "data" / "processed_reviews.csv")
    X = df["text_processed"].astype(str).tolist()
    y = df["Sentiment"].map(LABEL_MAP).values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    MAX_VOCAB, MAX_LEN, EMBED_DIM = 8000, 60, 64
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_tr = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post")
    X_te = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding="post")
    y_tr = to_categorical(y_train, 3)
    y_te_cat = to_categorical(y_test, 3)

    cw = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)))
    es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    results = {}

    # Dense
    print("Training Dense...")
    m1 = Sequential([
        Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ])
    m1.compile("adam", "categorical_crossentropy", ["accuracy"])
    h1 = m1.fit(X_tr, y_tr, validation_split=0.1, epochs=5, batch_size=128,
                class_weight=cw, callbacks=[es], verbose=1)
    results["Dense"] = eval_model(m1, X_te, y_test, "Dense")

    # LSTM
    print("Training LSTM...")
    m2 = Sequential([
        Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
        LSTM(32),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ])
    m2.compile("adam", "categorical_crossentropy", ["accuracy"])
    h2 = m2.fit(X_tr, y_tr, validation_split=0.1, epochs=5, batch_size=128,
                class_weight=cw, callbacks=[es], verbose=1)
    results["LSTM"] = eval_model(m2, X_te, y_test, "LSTM")
    plot_hist(h2, "LSTM")

    # BiLSTM
    print("Training BiLSTM...")
    m3 = Sequential([
        Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ])
    m3.compile("adam", "categorical_crossentropy", ["accuracy"])
    h3 = m3.fit(X_tr, y_tr, validation_split=0.1, epochs=5, batch_size=128,
                class_weight=cw, callbacks=[es], verbose=1)
    results["BiLSTM"] = eval_model(m3, X_te, y_test, "BiLSTM")
    plot_hist(h3, "BiLSTM")

    best_name = max(results, key=lambda k: results[k]["F1"])
    best = {"Dense": m1, "LSTM": m2, "BiLSTM": m3}[best_name]
    print(f"Best DL: {best_name}")
    best.save(MODELS / "best_dl_model.keras")
    joblib.dump(tokenizer, MODELS / "tokenizer.joblib")
    with open(MODELS / "dl_config.txt", "w") as f:
        f.write(f"MAX_VOCAB={MAX_VOCAB}\nMAX_LEN={MAX_LEN}\nEMBED_DIM={EMBED_DIM}\nBEST={best_name}\n")

    pd.DataFrame(results).T.round(4).to_csv(REPORTS / "dl_results.csv")
    print(pd.DataFrame(results).T.round(4))

    # Merge comparison
    ml = pd.read_csv(REPORTS / "ml_results.csv", index_col=0)
    dl = pd.DataFrame(results).T
    all_r = pd.concat([ml.add_prefix("ML_"), dl.add_prefix("DL_")], axis=1) if False else None
    # Simple side-by-side
    compare = pd.concat([
        ml.assign(Type="ML"),
        pd.DataFrame(results).T.assign(Type="DL")
    ])
    compare.to_csv(REPORTS / "final_comparison.csv")
    print("Done. Artifacts saved.")


def eval_model(model, X_te, y_test, name):
    preds = np.argmax(model.predict(X_te, verbose=0), axis=1)
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "Recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "F1": f1_score(y_test, preds, average="macro", zero_division=0),
    }
    print(f"{name}: {metrics}")
    print(classification_report(y_test, preds, target_names=["Negative", "Neutral", "Positive"]))
    return metrics


def plot_hist(hist, name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="val")
    ax[0].legend(); ax[0].set_title(f"{name} Acc")
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="val")
    ax[1].legend(); ax[1].set_title(f"{name} Loss")
    plt.tight_layout()
    fig.savefig(PLOTS / f"13_history_{name.lower()}.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    main()
