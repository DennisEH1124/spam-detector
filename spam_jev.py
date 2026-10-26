"""
Spam detector, version 2: TypeSafe's Jev "System One" decision model.

Instead of training a model locally (version 1: TF-IDF + Naive Bayes), each message
is sent to Jev with a yes/no ("noul") question. Jev returns the probability that the
answer is yes, i.e. P(spam). There is no training step; the dataset is used only to
measure how well Jev does.

Requirements:
    pip install typesafe-sdk
    set TYPESAFE_API_KEY=<your key from the TypeSafe console>

Usage:
    python spam_jev.py predict "your text" ...   # classify one or more messages
    python spam_jev.py evaluate [N]              # score Jev on N random dataset messages (default 100)
    python spam_jev.py interactive               # type messages, get predictions
"""

import csv
import os
import random
import sys
from pathlib import Path

try:
    from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient
except ImportError:
    sys.exit("Missing dependency. Run: pip install typesafe-sdk")

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "SMSSpamCollection"
MODEL = "jev-latest"

# Jev's docs suggest treating >= 0.7 as yes, <= 0.3 as no, and the middle as uncertain.
SPAM_THRESHOLD = 0.7
HAM_THRESHOLD = 0.3

SPAM_QUESTION = Noul(
    instructions="Is this message spam (unsolicited, promotional, scam or phishing) "
                 "rather than a genuine personal or business message?",
    criteria=NoulCriteria(
        true="Prize or lottery claims, unsolicited offers, premium-rate numbers or "
             "short codes to text, urgent account warnings, links to claim something, "
             "or mass marketing",
        false="Ordinary conversation between people who know each other, or a genuine "
              "message about real plans, work, or an existing relationship",
    ),
)


# ---------------------------------------------------------------------------
# 1. Classification with Jev
# ---------------------------------------------------------------------------
def spam_probability(client: TypeSafeClient, message: str) -> float:
    response = client.system_one(
        model=MODEL,
        state=message,
        questions={"is_spam": SPAM_QUESTION},
    )
    return response.answers["is_spam"].noul


def label_for(p: float) -> str:
    if p >= SPAM_THRESHOLD:
        return "SPAM"
    if p <= HAM_THRESHOLD:
        return "NOT SPAM"
    return "UNSURE"


def print_prediction(message: str, p: float):
    print(f"[{label_for(p):8}] P(spam)={p:.3f}  {message[:80]}")


def predict(messages):
    with TypeSafeClient() as client:
        for msg in messages:
            print_prediction(msg, spam_probability(client, msg))


def interactive():
    print("Type a message and press Enter (empty line to quit).")
    with TypeSafeClient() as client:
        while True:
            try:
                msg = input("> ").strip()
            except EOFError:
                break
            if not msg:
                break
            print_prediction(msg, spam_probability(client, msg))


# ---------------------------------------------------------------------------
# 2. Evaluation against the labelled UCI SMS Spam Collection
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH):
    """Return a de-duplicated list of (is_spam, message) pairs."""
    seen, rows = set(), []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) != 2 or row[1] in seen:
                continue
            seen.add(row[1])
            rows.append((row[0] == "spam", row[1]))
    return rows


def evaluate(n: int = 100, seed: int = 42):
    rows = load_data()
    spam = [r for r in rows if r[0]]
    ham = [r for r in rows if not r[0]]
    # Stratified sample so the spam/ham ratio matches the full dataset (~13% spam).
    rng = random.Random(seed)
    n_spam = max(1, round(n * len(spam) / len(rows)))
    sample = rng.sample(spam, n_spam) + rng.sample(ham, n - n_spam)
    rng.shuffle(sample)
    print(f"Evaluating Jev ({MODEL}) on {n} messages ({n_spam} spam, {n - n_spam} ham)...\n")

    tp = fp = tn = fn = unsure = 0
    with TypeSafeClient() as client:
        for i, (is_spam, msg) in enumerate(sample, 1):
            p = spam_probability(client, msg)
            pred_spam = p >= 0.5          # plain 0.5 cut-off for the metrics
            if HAM_THRESHOLD < p < SPAM_THRESHOLD:
                unsure += 1
            if pred_spam and is_spam:
                tp += 1
            elif pred_spam:
                fp += 1
                print(f"  false positive  P={p:.2f}  {msg[:70]}")
            elif is_spam:
                fn += 1
                print(f"  missed spam     P={p:.2f}  {msg[:70]}")
            else:
                tn += 1
            if i % 25 == 0:
                print(f"  ... {i}/{n} done")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\nAccuracy:        {(tp + tn) / n:.4f}")
    print(f"Spam precision:  {precision:.4f}")
    print(f"Spam recall:     {recall:.4f}")
    print(f"Spam F1:         {f1:.4f}")
    print(f"Unsure ({HAM_THRESHOLD}-{SPAM_THRESHOLD}): {unsure}\n")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"{'':12}{'pred ham':>10}{'pred spam':>11}")
    print(f"{'actual ham':12}{tn:>10}{fp:>11}")
    print(f"{'actual spam':12}{fn:>10}{tp:>11}")


if __name__ == "__main__":
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY to your TypeSafe API key first.")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "predict" and len(sys.argv) > 2:
        predict(sys.argv[2:])
    elif cmd == "evaluate":
        evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif cmd == "interactive":
        interactive()
    else:
        print(__doc__)
