"""
Agent — Day 4: one assistant, two tools.

The model first decides whether the question needs the DATABASE (SQL) or the
job DESCRIPTIONS (RAG), then uses that tool to answer.

Memory-friendly: it never loads all 10,000 descriptions at once — only the
ones it actually matched.

Run:   python agent.py
"""

import os
import re
import sqlite3

import numpy as np
from fastembed import TextEmbedding
from openai import OpenAI

DB_PATH = "jobs.db"
TABLE = "jobs"
MODEL = "openai/gpt-oss-120b"
CACHE = "desc_embeddings.npz"
TOP_K = 8


def get_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GROQ_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit('No API key. Set GROQ_API_KEY or create a .env file.')
    return key


client = None  # set in main()


def chat(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ---------- tool 1: SQL over the database ----------

def get_schema():
    con = sqlite3.connect(DB_PATH)
    total = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    lines = []
    for _, name, ctype, *_ in con.execute(f"PRAGMA table_info({TABLE})"):
        nulls = con.execute(
            f'SELECT COUNT(*) FROM {TABLE} WHERE "{name}" IS NULL'
        ).fetchone()[0]
        pct = 100 * nulls / total if total else 0
        note = f"   -- {pct:.0f}% empty" if pct >= 5 else ""
        lines.append(f"  {name} {ctype}{note}")
    con.close()
    return f"Table {TABLE} ({total} rows):\n" + "\n".join(lines)


def tool_sql(question):
    sql = chat(f"""You are a SQLite expert. Write ONE SQLite query to answer the question.

{get_schema()}

Rules:
- Only a single SELECT. Never modify data.
- Use the exact column names above.
- Columns marked "% empty" are mostly missing — filter out the NULLs.
- For salary questions use the *_annual columns, but EXCLUDE implausible values,
  e.g. add `AND salary_max_annual < 1000000`. A few rows have mis-parsed pay
  periods and show absurd annual figures that skew results.
- SQLite text comparison is CASE-SENSITIVE. Never use `col = 'Remote'`.
  Compare case-insensitively: LOWER(col) = 'remote' or LOWER(col) LIKE '%remote%'.
  (The values in this table are lower-case.)
- For ANY salary question, restrict to ONE currency, e.g. `salary_currency = 'USD'`,
  and state which currency you used. Mixing currencies gives nonsense medians.
- Return ONLY the SQL, no explanation, no markdown.

Question: {question}""")
    sql = re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.MULTILINE).strip()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(sql).fetchall()
    finally:
        con.close()

    if not rows:
        return "The query returned no rows.", sql
    cols = list(rows[0].keys())
    out = "| " + " | ".join(cols) + " |\n"
    out += "|" + "|".join(["---"] * len(cols)) + "|\n"
    for r in rows[:20]:
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    if len(rows) > 20:
        out += f"\n*... and {len(rows) - 20} more rows*"
    return out, sql


# ---------- tool 2: RAG over the descriptions ----------

def tool_docs(question, model, vecs, ids):
    q = np.asarray(list(model.query_embed([question]))[0], dtype=float)
    q = q / np.linalg.norm(q)
    m = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    scores = m @ q
    top = np.argsort(-scores)[:TOP_K]

    # Fetch only the matched descriptions — keeps memory small.
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    parts, src = [], []
    for i in top:
        row = con.execute(
            f"SELECT title, company, description FROM {TABLE} WHERE job_id = ?",
            (ids[i],),
        ).fetchone()
        if row:
            parts.append(f"[{row['title']} at {row['company']}]\n{row['description'][:1200]}")
            src.append(f"  {scores[i]:.3f}  {row['title']} @ {row['company']}")
    con.close()

    answer = chat("""Answer using ONLY the job postings below.
If the answer is not there, say you don't know. Mention the companies you used.

Postings:
""" + "\n\n".join(parts) + f"\n\nQuestion: {question}")
    return answer, "\n".join(src)


# ---------- the router ----------

def choose_tool(question):
    word = chat(f"""You route questions to one of two tools.

SQL  -> questions about numbers, counts, salaries, locations, companies,
        dates, levels: "how many", "average", "top", "which companies",
        "salary range", "list jobs where ...".
DOCS -> anything about what the job postings SAY: skills, requirements,
        responsibilities, benefits, holidays, travel, policies, security,
        "what does X require", "do any jobs mention ...", "is there any ...".
CHAT -> ONLY a pure greeting or small talk ("hi", "hello", "how are you",
        "thanks"). If the message asks ANYTHING about the data, use SQL or DOCS.

Answer with exactly one word: SQL, DOCS or CHAT.

Question: {question}""").upper()
    if "CHAT" in word:
        return "CHAT"
    return "SQL" if "SQL" in word else "DOCS"


def main():
    global client
    question = input("Ask the agent: ").strip()
    if not question:
        print("No question given.")
        return
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=get_api_key())

    tool = choose_tool(question)
    print(f"\n[agent chose: {tool}]")

    if tool == "CHAT":
        print("\n" + chat(
            f"The user said: {question}\n"
            "You are the Job Market Agent. Reply in ONE friendly sentence. "
            "Do NOT state any facts about jobs, salaries, benefits or the "
            "market - you have no data for this message. Then invite them to "
            "ask about salaries, roles, skills, companies or locations."))
        return

    if tool == "SQL":
        answer, sql = tool_sql(question)
        print("\nSQL:\n" + sql)
        print("\nResult:\n" + answer)
    else:
        data = np.load(CACHE, allow_pickle=True)
        vecs, ids = data["vecs"].astype(float), data["ids"].astype(str)
        model = TextEmbedding()
        answer, src = tool_docs(question, model, vecs, ids)
        print("\nTop postings used:\n" + src)
        print("\nAnswer:\n" + answer)


if __name__ == "__main__":
    main()
