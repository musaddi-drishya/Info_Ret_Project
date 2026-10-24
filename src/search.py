"""Interactive side-by-side search over a corpus.

Usage:  python -m src.search campus "phd admission deadline" [--k 5] [--slope 0.7]
"""
import argparse
import sys

from .models import BM25, CosineTFIDF, PivotedCosine, PivotedUnique
from .pipeline import load_collection
from .preprocess import preprocess
from .storage import load_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--slope", type=float, default=0.7)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # titles contain en dashes etc.

    coll = load_collection(args.corpus)
    titles = load_corpus(args.corpus)[0]["title"].fillna("").tolist()
    idx = coll.build_index()
    q = preprocess(args.query)
    print(f"query terms: {q}\n")
    for model in (CosineTFIDF(idx), PivotedCosine(idx, args.slope), PivotedUnique(idx), BM25(idx)):
        scores = model.score(q)
        print(f"== {model.name}")
        for rank, d in enumerate(model.rank(q, args.k), 1):
            print(f"  {rank}. [{scores[d]:.3f}] {coll.doc_ids[d]:<24} len={int(idx.doc_len[d]):4d}  {titles[d][:60]}")
        print()


if __name__ == "__main__":
    main()
