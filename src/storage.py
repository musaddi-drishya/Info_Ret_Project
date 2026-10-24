"""SQLite storage for documents, metadata, queries and relevance judgments."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "ir_corpus.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    corpus      TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    category    TEXT,
    title       TEXT,
    text        TEXT NOT NULL,
    source      TEXT,
    collected   TEXT,
    n_tokens    INTEGER,
    PRIMARY KEY (corpus, doc_id)
);
CREATE TABLE IF NOT EXISTS queries (
    corpus      TEXT NOT NULL,
    query_id    TEXT NOT NULL,
    text        TEXT NOT NULL,
    category    TEXT,
    PRIMARY KEY (corpus, query_id)
);
CREATE TABLE IF NOT EXISTS qrels (
    corpus      TEXT NOT NULL,
    query_id    TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    relevance   INTEGER NOT NULL,
    annotator   TEXT NOT NULL DEFAULT 'category-proxy',
    PRIMARY KEY (corpus, query_id, doc_id, annotator)
);
CREATE INDEX IF NOT EXISTS idx_doc_len ON documents(corpus, n_tokens);
"""


def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def save_corpus(corpus, docs, queries=None, qrels=None, path=DB_PATH):
    """docs/queries/qrels are DataFrames; existing rows for `corpus` are replaced."""
    con = connect(path)
    with con:
        for table in ("documents", "queries", "qrels"):
            con.execute(f"DELETE FROM {table} WHERE corpus = ?", (corpus,))
        docs.assign(corpus=corpus).to_sql("documents", con, if_exists="append", index=False)
        if queries is not None:
            queries.assign(corpus=corpus).to_sql("queries", con, if_exists="append", index=False)
        if qrels is not None:
            qrels.assign(corpus=corpus).to_sql("qrels", con, if_exists="append", index=False)
    con.close()


def load_corpus(corpus, path=DB_PATH):
    con = connect(path)
    docs = pd.read_sql("SELECT * FROM documents WHERE corpus = ? ORDER BY doc_id", con, params=(corpus,))
    queries = pd.read_sql("SELECT * FROM queries WHERE corpus = ? ORDER BY query_id", con, params=(corpus,))
    qrels = pd.read_sql("SELECT * FROM qrels WHERE corpus = ?", con, params=(corpus,))
    con.close()
    return docs, queries, qrels


def docs_in_length_range(corpus, lo, hi, path=DB_PATH):
    """Used when picking source documents for adversarial construction."""
    con = connect(path)
    df = pd.read_sql("SELECT * FROM documents WHERE corpus = ? AND n_tokens BETWEEN ? AND ?",
                     con, params=(corpus, lo, hi))
    con.close()
    return df
