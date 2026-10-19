"""
Step 5 — Fine-Tuning.

This is the *workflow* of fine-tuning, on a small model that runs on your
laptop. Real LLM fine-tuning is the same five steps, just bigger:

    1. collect labelled examples        (data where you know the answer)
    2. split into train / test          (so you can measure honestly)
    3. train the model                  (it learns patterns from the examples)
    4. evaluate on the unseen test set  (accuracy vs a baseline)
    5. use it                           (predict the rows you don't know)

Task: predict workplace_type (remote / hybrid / onsite) from the description.
That fills a real gap - 5,352 postings have no workplace_type.

Run:   python finetune.py
"""

import sqlite3

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DB_PATH = "jobs.db"
MODEL_PATH = "workplace_model.joblib"


def load():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT title, workplace_type, description FROM jobs", con)
    con.close()
    df["description"] = df["description"].fillna("")
    return df


def main():
    df = load()
    labelled = df[df["workplace_type"].notna() & (df["workplace_type"] != "")]
    missing = df[df["workplace_type"].isna() | (df["workplace_type"] == "")]

    print(f"labelled examples : {len(labelled):,}   (we know the answer)")
    print(f"missing to fill   : {len(missing):,}   (what we want to predict)")

    # ---- 2. train / test split ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        labelled["description"], labelled["workplace_type"],
        test_size=0.2, random_state=0, stratify=labelled["workplace_type"],
    )

    # ---- 3. train ----
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          stop_words="english")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(X_tr), y_tr)

    # ---- 4. evaluate on data the model has never seen ----
    pred = clf.predict(vec.transform(X_te))
    acc = accuracy_score(y_te, pred)
    baseline = y_te.value_counts(normalize=True).max()

    print(f"\naccuracy on unseen data : {acc:.0%}")
    print(f"baseline (always guess the most common class) : {baseline:.0%}")
    print(f"lift over baseline      : {acc - baseline:+.0%}")
    print("\nper class:")
    print(classification_report(y_te, pred, zero_division=0))

    # ---- 5. retrain on everything, then fill the gaps ----
    vec_all = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                              stop_words="english")
    clf_all = LogisticRegression(max_iter=1000)
    clf_all.fit(vec_all.fit_transform(labelled["description"]),
                labelled["workplace_type"])

    filled = clf_all.predict(vec_all.transform(missing["description"]))
    print("predictions for the rows that had no workplace_type:")
    print(pd.Series(filled).value_counts().to_string())

    print("\nexamples:")
    for title, p in list(zip(missing["title"], filled))[:8]:
        print(f"  {p:<8} {title[:60]}")

    joblib.dump({"vec": vec_all, "clf": clf_all}, MODEL_PATH)
    print(f"\nSaved the trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
