"""
Step 4 — Evaluation.

Runs a golden set of questions through the agent and scores two things:

  1. Routing accuracy  - did it pick the right tool (SQL / DOCS / CHAT)?
  2. Answer accuracy   - does the answer contain the expected fact?

This is how you turn "it looks good to me" into a number you can improve.

Run:   python evaluate.py
"""

import json
import time

import numpy as np
from fastembed import TextEmbedding
from openai import OpenAI

import agent


def score_answer(answer, expect_any):
    """True if any expected keyword appears. None when there is nothing to check."""
    if not expect_any:
        return None
    low = answer.lower()
    return any(k.lower() in low for k in expect_any)


def run_one(item, model, vecs, ids):
    q = item["question"]
    tool = agent.choose_tool(q)
    if tool == "SQL":
        answer, _ = agent.tool_sql(q)
    elif tool == "DOCS":
        answer, _ = agent.tool_docs(q, model, vecs, ids)
    elif tool == "BOTH":
        answer, _, _ = agent.tool_both(q)
    else:
        answer = "(chat reply)"
    return tool, answer


def main():
    golden = json.load(open("golden_set.json", encoding="utf-8"))

    agent.client = OpenAI(base_url="https://api.groq.com/openai/v1",
                          api_key=agent.get_api_key())

    data = np.load("desc_embeddings.npz", allow_pickle=True)
    vecs, ids = data["vecs"].astype(float), data["ids"].astype(str)
    model = TextEmbedding()

    results = []
    for i, item in enumerate(golden, 1):
        if i > 1:
            time.sleep(6)   # stay under the free-tier tokens-per-minute limit
        tool, answer = run_one(item, model, vecs, ids)
        tool_ok = tool == item["expect_tool"]
        ans_ok = score_answer(answer, item.get("expect_any"))

        results.append({**item, "got_tool": tool, "tool_ok": tool_ok,
                        "answer_ok": ans_ok, "answer": answer[:300]})

        mark = "OK  " if tool_ok else "MISS"
        ans_mark = "" if ans_ok is None else ("  ans:OK" if ans_ok else "  ans:MISS")
        print(f"{i:2d}. [{mark}]{ans_mark}  {item['question'][:58]}")
        if not tool_ok:
            print(f"       expected {item['expect_tool']}, got {tool}")

    n = len(results)
    tool_acc = sum(r["tool_ok"] for r in results) / n
    scored = [r for r in results if r["answer_ok"] is not None]
    ans_acc = (sum(r["answer_ok"] for r in scored) / len(scored)) if scored else 0.0

    print("\n" + "=" * 52)
    print(f"  Routing accuracy : {tool_acc:.0%}   "
          f"({sum(r['tool_ok'] for r in results)}/{n})")
    print(f"  Answer accuracy  : {ans_acc:.0%}   "
          f"({sum(r['answer_ok'] for r in scored)}/{len(scored)})")
    print("=" * 52)

    json.dump(results, open("eval_results.json", "w", encoding="utf-8"), indent=2)
    print("Saved eval_results.json (full answers + scores)")


if __name__ == "__main__":
    main()
