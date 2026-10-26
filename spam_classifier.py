"""
Spam detector: TF-IDF features + Multinomial Naive Bayes.

Dataset: UCI SMS Spam Collection (data/SMSSpamCollection, tab-separated: label<TAB>message)

Usage:
    python spam_classifier.py train                 # train, evaluate, save model
    python spam_classifier.py predict "your text"   # classify one or more messages
    python spam_classifier.py interactive           # type messages, get predictions
"""

import csv
import re
import string
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_PATH = HERE / "data" / "SMSSpamCollection"
MODEL_PATH = HERE / "spam_model.joblib"


# ---------------------------------------------------------------------------
# 1. Text preprocessing
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"\b\d{5,}\b")          # long digit runs: phone numbers, short codes
MONEY_RE = re.compile(r"[£$€]\s?\d+[\d,.]*")
NUM_RE = re.compile(r"\b\d+\b")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def preprocess(text: str) -> str:
    """Lowercase, replace URLs/emails/phones/money/numbers with tokens,
    strip punctuation and stop words."""
    text = text.lower()
    # Replace patterns that are strong spam signals with placeholder tokens,
    # so "call 08712345678" and "call 09061234567" look the same to the model.
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = text.translate(PUNCT_TABLE)
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return " ".join(words)


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"],
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    df = df.dropna().drop_duplicates(subset="message")
    df["target"] = (df["label"] == "spam").astype(int)   # 1 = spam, 0 = ham
    return df


# ---------------------------------------------------------------------------
# 3. Model: TF-IDF -> Multinomial Naive Bayes
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess, sublinear_tf=True)),
        ("nb", MultinomialNB()),
    ])


def train():
    df = load_data()
    print(f"Loaded {len(df)} unique messages "
          f"({df.target.sum()} spam, {(df.target == 0).sum()} ham)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["target"], test_size=0.2, stratify=df["target"], random_state=42)

    # Small grid search over n-gram range, vocab cutoff and NB smoothing (alpha).
    grid = GridSearchCV(
        build_pipeline(),
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "nb__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
        scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print(f"Best params (5-fold CV): {grid.best_params_}  CV F1={grid.best_score_:.3f}\n")

    y_pred = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=["actual ham", "actual spam"],
                       columns=["pred ham", "pred spam"]).to_string(), "\n")

    show_top_spam_words(model)

    # Refit on all data before saving so the shipped model sees every example.
    model.fit(df["message"], df["target"])
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH.name}")


def show_top_spam_words(model: Pipeline, n: int = 15):
    """Words with the highest P(word|spam) / P(word|ham) ratio under Naive Bayes."""
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    log_probs = model.named_steps["nb"].feature_log_prob_   # shape (2, n_features)
    ratio = log_probs[1] - log_probs[0]
    top = ratio.argsort()[::-1][:n]
    print("Most spam-indicative terms:", ", ".join(vocab[top]), "\n")


# ---------------------------------------------------------------------------
# 4. Prediction
# ---------------------------------------------------------------------------
def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run: python spam_classifier.py train")
    return joblib.load(MODEL_PATH)


def predict(messages, model=None):
    """Return a list of (label, spam_probability) for each message."""
    model = model or load_model()
    probs = model.predict_proba(list(messages))[:, 1]
    return [("SPAM" if p >= 0.5 else "NOT SPAM", p) for p in probs]


def print_predictions(messages, model=None):
    for msg, (label, p) in zip(messages, predict(messages, model)):
        print(f"[{label:8}] P(spam)={p:.3f}  {msg[:80]}")


def interactive():
    model = load_model()
    print("Type a message and press Enter (empty line to quit).")
    while True:
        try:
            msg = input("> ").strip()
        except EOFError:
            break
        if not msg:
            break
        print_predictions([msg], model)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "predict" and len(sys.argv) > 2:
        print_predictions(sys.argv[2:])
    elif cmd == "interactive":
        interactive()
    else:
        print(__doc__)
