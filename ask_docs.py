"""
Agent — Day 3: the RAG tool.

Searches the job *descriptions* by meaning and answers from the best matches.

It uses a cached file (desc_embeddings.npz) if one exists — so you can build
that file anywhere (a Colab notebook, a friend's laptop) and just drop it here.

Run:   python ask_docs.py
"""

import os
import sqlite3

import numpy as np
from fastembed import TextEmbedding
from openai import OpenAI

DB_PATH = "jobs.db"
TABLE = "jobs"
TOP_K = 10
MODEL = "openai/gpt-oss-120b"
CACHE = "desc_embeddings.npz"

# Only used if there is no cache file and we have to embed locally.
MAX_DOCS = 2000     # how many postings to embed locally
MAX_CHARS = 800     # how much of each description to embed


def get_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GROQ_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit('No API key. Set GROQ_API_KEY or create a .env file.')
    return key


def load_docs():
    """Load every posting (used to look a matched id back up)."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        f"SELECT job_id, title, company, description FROM {TABLE} "
        f"WHERE description IS NOT NULL"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def build_index(docs, model):
    """Use the cached vectors if present, else embed a slice locally."""
    if os.path.exists(CACHE):
        data = np.load(CACHE, allow_pickle=True)
        print(f"Loaded {len(data['ids']):,} cached embeddings from {CACHE}")
        return data["vecs"], data["ids"].astype(str)

    subset = docs[:MAX_DOCS]
    texts = [d["description"][:MAX_CHARS] for d in subset]
    print(f"No cache found. Embedding {len(texts):,} descriptions locally...")
    vecs = []
    for i, v in enumerate(model.embed(texts)):
        vecs.append(v)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(texts)} done")

    vecs = np.array(vecs, dtype=float)
    ids = np.array([str(d["job_id"]) for d in subset])
    np.savez(CACHE, vecs=vecs, ids=ids)
    print(f"Cached embeddings to {CACHE}")
    return vecs, ids


def cosine_scores(q, matrix):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return m @ q


def main():
    question = input("Ask about the job descriptions: ").strip()
    if not question:
        print("No question given.")
        return

    docs = load_docs()
    by_id = {str(d["job_id"]): d for d in docs}
    model = TextEmbedding()

    vecs, ids = build_index(docs, model)

    q_vec = list(model.query_embed([question]))[0]
    scores = cosine_scores(q_vec, vecs)
    top = np.argsort(-scores)[:TOP_K]

    matches = []
    for i in top:
        doc = by_id.get(ids[i])
        if doc:                       # skip any id not in the local DB
            matches.append((doc, float(scores[i])))

    print("\nTop matching postings:")
    for d, s in matches:
        print(f"  {s:.3f}  {d['title']}  @  {d['company']}")

    context = "\n\n".join(
        f"[{d['title']} at {d['company']}]\n{d['description'][:1500]}"
        for d, _ in matches
    )
    prompt = f"""Answer the question using ONLY the job postings below.
If the answer is not in them, say you don't know. Mention the company names you used.

Postings:
{context}

Question: {question}"""

    client = OpenAI(base_url="https://api.groq.com/openai/v1",
                    api_key=get_api_key())
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print("\nAnswer:\n" + resp.choices[0].message.content.strip())


if __name__ == "__main__":
    main()
