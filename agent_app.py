"""
Job Market Agent — Streamlit app.

One assistant, two tools: it decides whether a question needs the DATABASE
(SQL) or the job DESCRIPTIONS (RAG), then answers with that tool.

Run:   streamlit run agent_app.py
"""

import os
import re
import sqlite3

import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from openai import OpenAI

DB_PATH = "jobs.db"
TABLE = "jobs"
MODEL = "openai/gpt-oss-120b"
CACHE = "desc_embeddings.npz"
TOP_K = 8

st.set_page_config(page_title="Job Market Agent", page_icon="🧭")


def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("GROQ_API_KEY")
    if not key and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GROQ_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def safe_md(text):
    """Escape $ so Streamlit doesn't render it as LaTeX math."""
    return text.replace("$", r"\$")


def chat(prompt):
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=get_api_key())
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


@st.cache_resource(show_spinner="Loading embedding model...")
def get_model():
    return TextEmbedding()


@st.cache_resource(show_spinner=False)
def get_vectors():
    data = np.load(CACHE, allow_pickle=True)
    return data["vecs"].astype(float), data["ids"].astype(str)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def db_stats():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    con.close()
    return rows


def choose_tool(question):
    word = chat(f"""You route questions to one of three tools.

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


def run_sql(question):
    sql = chat(f"""You are a SQLite expert. Write ONE SQLite query to answer the question.

{get_schema()}

Rules:
- Only a single SELECT. Never modify data.
- Use the exact column names above.
- Columns marked "% empty" are mostly missing — filter out the NULLs.
- For salary questions, restrict to `salary_period = 'year'` and use
  salary_min / salary_max directly (they are already annual). Do NOT use the
  *_annual columns: some rows have mis-detected pay periods and are inflated
  ~12x, which skews every "top salaries" list.
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
    except Exception as e:
        con.close()
        return f"Query failed: {e}", sql, ""
    con.close()

    if not rows:
        return "The query returned no rows.", sql, ""
    cols = list(rows[0].keys())
    out = "| " + " | ".join(cols) + " |\n"
    out += "|" + "|".join(["---"] * len(cols)) + "|\n"
    for r in rows[:25]:
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    if len(rows) > 25:
        out += f"\n*... and {len(rows) - 25} more rows*"
    return out, sql, ""


def run_docs(question):
    model = get_model()
    vecs, ids = get_vectors()

    q = np.asarray(list(model.query_embed([question]))[0], dtype=float)
    q = q / np.linalg.norm(q)
    m = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    scores = m @ q
    top = np.argsort(-scores)[:TOP_K]

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
            src += [f"**{row['title']}** @ {row['company']}  —  score {scores[i]:.3f}"]
    con.close()

    answer = chat("""Answer using ONLY the job postings below.
If the answer is not there, say you don't know. Mention the companies you used.

Postings:
""" + "\n\n".join(parts) + f"\n\nQuestion: {question}")
    return answer, "", "\n\n".join(src)


# ---------------- UI ----------------

st.title("Job Market Agent")
st.caption("One assistant, two tools. It decides whether your question needs the "
           "database (SQL) or the job descriptions (RAG) — then answers with that tool.")

if not os.path.exists(DB_PATH) or not os.path.exists(CACHE):
    st.error("Missing `jobs.db` or `desc_embeddings.npz` in the app folder.")
    st.stop()

with st.sidebar:
    st.subheader("What's inside")
    st.write(f"**{db_stats():,}** job postings in the database (SQL tool)")
    st.write("**1,000** descriptions indexed for search (RAG tool)")
    st.divider()
    st.caption("Try:")
    st.caption("• How many jobs are remote?")
    st.caption("• Which 10 companies have the most postings?")
    st.caption("• What is the median annual salary in the US?")
    st.caption("• What skills do data analyst roles require?")
    st.caption("• Do any jobs mention AI agents?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(safe_md(m["content"]))
        if m.get("detail"):
            with st.expander(m.get("detail_label", "Details")):
                st.markdown(safe_md(m["detail"]))

if question := st.chat_input("Ask about the job market..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(safe_md(question))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            tool = choose_tool(question)
            if tool == "CHAT":
                answer = chat(
                    f"Today is {__import__('datetime').date.today().strftime('%A, %d %B %Y')}.\n" f"The user said: {question}\n"
                    "You are a job-market assistant. Reply in one friendly sentence, "
                    "then invite them to ask about the job data (salaries, roles, "
                    "skills, companies, locations).")
                detail, label = "", "Details"
            elif tool == "SQL":
                answer, sql, _ = run_sql(question)
                detail, label = f"```sql\n{sql}\n```", "SQL it ran"
            else:
                answer, _, src = run_docs(question)
                detail, label = src, "Postings it used"

        st.markdown(f"`tool: {tool}`")
        st.markdown(safe_md(answer))
        if detail:
            with st.expander(label):
                st.markdown(safe_md(detail))

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "detail": detail, "detail_label": label,
    })
