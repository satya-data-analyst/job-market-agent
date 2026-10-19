"""
Automated tests for the Job Market Agent.

Run with:   pytest -q

These tests deliberately need NO API key and NO internet — they check the data,
the golden set and the scoring logic. That's what makes them safe to run in CI
on every single push.
"""

import json
import os
import sqlite3

import numpy as np
import pytest

EXPECTED_COLUMNS = [
    "job_id", "title", "company", "description",
    "salary_min", "salary_max", "salary_period", "salary_currency",
    "workplace_type", "country_code",
]


def test_database_exists():
    assert os.path.exists("jobs.db"), "jobs.db is missing"


def test_database_has_expected_columns():
    con = sqlite3.connect("jobs.db")
    cols = {row[1] for row in con.execute("PRAGMA table_info(jobs)")}
    con.close()
    for c in EXPECTED_COLUMNS:
        assert c in cols, f"jobs.db is missing column: {c}"


def test_database_row_count():
    con = sqlite3.connect("jobs.db")
    n = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    con.close()
    assert n == 10000, f"expected 10000 rows, found {n}"


def test_embeddings_load_and_match():
    data = np.load("desc_embeddings.npz", allow_pickle=True)
    vecs, ids = data["vecs"], data["ids"]
    assert len(ids) == len(vecs), "ids and vectors differ in length"
    assert vecs.shape[1] == 384, f"unexpected embedding size {vecs.shape[1]}"
    assert len(ids) > 100, "too few embeddings to be useful"


def test_embedding_ids_exist_in_database():
    data = np.load("desc_embeddings.npz", allow_pickle=True)
    ids = {str(i) for i in data["ids"]}
    con = sqlite3.connect("jobs.db")
    db_ids = {str(r[0]) for r in con.execute("SELECT job_id FROM jobs")}
    con.close()
    missing = ids - db_ids
    assert not missing, f"{len(missing)} embedding ids are not in the database"


def test_golden_set_is_wellformed():
    golden = json.load(open("golden_set.json", encoding="utf-8"))
    assert len(golden) >= 10, "golden set is too small"
    for item in golden:
        assert item["question"].strip()
        assert item["expect_tool"] in {"SQL", "DOCS", "CHAT"}
        assert "expect_any" in item
        assert "category" in item


def test_golden_set_covers_all_routes():
    golden = json.load(open("golden_set.json", encoding="utf-8"))
    tools = {i["expect_tool"] for i in golden}
    assert tools == {"SQL", "DOCS", "CHAT"}, f"golden set misses routes: {tools}"


def test_scoring_logic():
    evaluate = pytest.importorskip("evaluate")
    assert evaluate.score_answer("there are 2111 remote jobs", ["2111"]) is True
    assert evaluate.score_answer("no idea", ["2111"]) is False
    assert evaluate.score_answer("anything at all", []) is None
