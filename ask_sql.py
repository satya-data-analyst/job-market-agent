"""
Agent — Day 2: the SQL tool.

Turns a plain-English question into SQL, runs it against jobs.db, and shows
the result. This is one of the two tools the agent will have.

Note the important detail: the schema we hand the model includes how empty
each column is, so it avoids writing queries against columns that are mostly
null (like seniority, which is 94% empty).

Run:   python ask_sql.py
"""

import os
import re
import sqlite3

from openai import OpenAI

DB_PATH = "jobs.db"
TABLE = "jobs"
MODEL = "openai/gpt-oss-120b"


def get_api_key():
    """Key from the environment, or from a local .env file."""
    key = os.environ.get("GROQ_API_KEY")
    if not key and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GROQ_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit(
            "No API key found. Set GROQ_API_KEY or create a .env file:\n"
            '  GROQ_API_KEY="your_key_here"'
        )
    return key


def get_schema():
    """Return the table schema, with the % of empty values per column."""
    con = sqlite3.connect(DB_PATH)
    total = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    lines = []
    for _, name, ctype, *_ in con.execute(f"PRAGMA table_info({TABLE})"):
        nulls = con.execute(
            f"SELECT COUNT(*) FROM {TABLE} WHERE \"{name}\" IS NULL"
        ).fetchone()[0]
        pct = 100 * nulls / total if total else 0
        note = f"   -- {pct:.0f}% empty" if pct >= 5 else ""
        lines.append(f"  {name} {ctype}{note}")
    con.close()
    return f"Table {TABLE} ({total} rows):\n" + "\n".join(lines)


def write_sql(question):
    client = OpenAI(base_url="https://api.groq.com/openai/v1",
                    api_key=get_api_key())
    prompt = f"""You are a SQLite expert. Write ONE SQLite query to answer the question.

{get_schema()}

Rules:
- Only a single SELECT statement. Never modify data.
- Use the exact column names shown above.
- Columns marked "% empty" are mostly missing — avoid relying on them, or
  filter out the NULLs explicitly.
- If salary_period is not 'year', use salary_min_annual / salary_max_annual
  for yearly comparisons.
- Return ONLY the SQL. No explanation, no markdown fences.

Question: {question}"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def clean_sql(text):
    """Strip markdown fences if the model added them anyway."""
    text = re.sub(r"^```(?:sql)?|```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def run_sql(sql):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(sql).fetchall()
    con.close()
    return rows


def main():
    question = input("Ask a question about the jobs data: ").strip()
    if not question:
        print("No question given.")
        return

    sql = clean_sql(write_sql(question))
    print("\nSQL written by the model:")
    print(sql)

    try:
        rows = run_sql(sql)
    except Exception as e:
        print(f"\nQuery failed: {e}")
        return

    if not rows:
        print("\nNo rows returned.")
        return

    cols = rows[0].keys()
    print(f"\nResult ({len(rows)} rows):")
    print(" | ".join(cols))
    for r in rows[:15]:
        print(" | ".join(str(r[c]) for c in cols))
    if len(rows) > 15:
        print(f"... and {len(rows) - 15} more rows")


if __name__ == "__main__":
    main()
