"""Build the relevance-judging pool for the custom campus corpus.

For every query in data/judgments/campus_queries.csv, the top-10 documents of
Cosine, Pivoted and BM25 are pooled (TREC-style pooling) and written to
data/judgments/campus_qrels.csv with empty judge1 / judge2 / final columns.

Two group members fill judge1 and judge2 independently (1 = relevant, 0 = not),
then agree on `final` where they differ. Re-running keeps existing judgments.

Usage:  python scripts/make_judging_pool.py          # build / extend the pool
        python scripts/make_judging_pool.py --kappa  # inter-annotator agreement
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import CAMPUS_QRELS, CAMPUS_QUERIES, build_campus  # noqa: E402
from src.index import InvertedIndex  # noqa: E402
from src.models import BM25, CosineTFIDF, PivotedCosine  # noqa: E402
from src.preprocess import preprocess  # noqa: E402

POOL_DEPTH = 10


def build_pool():
    docs, queries, _ = build_campus()
    if queries is None:
        sys.exit(f"write {CAMPUS_QUERIES} first (columns: query_id,text)")
    idx = InvertedIndex.build([preprocess(t) for t in docs["text"]])
    models = [CosineTFIDF(idx), PivotedCosine(idx, slope=0.7), BM25(idx)]
    rows = []
    for q in queries.itertuples():
        terms = preprocess(q.text)
        pooled = dict.fromkeys(d for m in models for d in m.rank(terms, POOL_DEPTH))
        for d in pooled:
            doc = docs.iloc[d]
            rows.append({"query_id": q.query_id, "query": q.text, "doc_id": doc.doc_id, "title": doc.title,
                         "n_tokens": doc.n_tokens, "url": doc.source,
                         "snippet": doc.text.replace("\n", " ")[:300]})
    pool = pd.DataFrame(rows)
    for c in ("judge1", "judge2", "final"):
        pool[c] = pd.NA
    if CAMPUS_QRELS.exists():   # keep judgments already made
        old = pd.read_csv(CAMPUS_QRELS)[["query_id", "doc_id", "judge1", "judge2", "final"]]
        pool = pool.drop(columns=["judge1", "judge2", "final"]).merge(old, on=["query_id", "doc_id"], how="left")
    CAMPUS_QRELS.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(CAMPUS_QRELS, index=False)
    print(f"{len(pool)} (query, doc) pairs for {queries.shape[0]} queries -> {CAMPUS_QRELS}")


def cohen_kappa():
    j = pd.read_csv(CAMPUS_QRELS).dropna(subset=["judge1", "judge2"])
    if j.empty:
        sys.exit("no pairs judged by both judges yet")
    a, b = j["judge1"].astype(int), j["judge2"].astype(int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"{len(j)} doubly-judged pairs, observed agreement {po:.3f}, Cohen's kappa {kappa:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", action="store_true")
    cohen_kappa() if ap.parse_args().kappa else build_pool()
