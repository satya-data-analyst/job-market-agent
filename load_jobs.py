"""
Agent — Day 1: load the job postings CSV into SQLite, and look at it.

This is the same first step you did for Superstore: get the data into a
database. But today we also *look* at how messy it is, because that messiness
is what makes the agent's answers realistic.

Run:   python load_jobs.py
"""

import os
import sqlite3

import pandas as pd

CSV_PATH = "data/job_postings_with_salaries_2026_10.csv"
DB_PATH = "jobs.db"
TABLE = "jobs"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at: {CSV_PATH}")
        print("Download it from Kaggle and put it in the data/ folder.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns from {CSV_PATH}\n")

    # Write the whole table into SQLite. This is the database the SQL tool will query.
    con = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE, con, if_exists="replace", index=False)
    con.close()
    print(f"Wrote table '{TABLE}' to {DB_PATH}\n")

    # The 'messy check': how empty is each column?
    print("Empty values per column (this is the messiness):")
    nulls = df.isna().sum().sort_values(ascending=False)
    for col, n in nulls.items():
        pct = 100 * n / len(df)
        print(f"  {col:22s} {n:6,d}   ({pct:4.1f}% empty)")

    # A few real rows, so you can see what the data actually looks like.
    show = [c for c in [
        "title", "company", "workplace_type",
        "salary_min_annual", "salary_max_annual", "country_code",
    ] if c in df.columns]
    print("\nSample rows:")
    print(df[show].head(5).to_string(index=False))

    # A peek at the text we'll use for the RAG side.
    if "description" in df.columns:
        desc = df["description"].dropna()
        if len(desc):
            print("\nSample description (first 300 characters of one posting):")
            print(desc.iloc[0][:300] + " ...")

    print("\nDay 1 done: the data is in jobs.db, and you've seen how messy it is.")


if __name__ == "__main__":
    main()
