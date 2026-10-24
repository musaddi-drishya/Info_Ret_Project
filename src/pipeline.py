"""Glue: SQLite corpus -> tokens -> inverted index -> models -> evaluation."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .index import InvertedIndex
from .preprocess import preprocess
from .storage import ROOT, load_corpus

CACHE = ROOT / "data" / "cache"


@dataclass
class Collection:
    name: str
    doc_ids: list
    categories: list
    tokens: list                      # list of token lists (preprocessed)
    queries: list                     # list of (query_id, q_terms)
    relevant: dict                    # query_id -> set of doc indices
    query_category: dict = field(default_factory=dict)

    @property
    def doc_len(self):
        return np.array([len(t) for t in self.tokens], dtype=float)

    def build_index(self):
        return InvertedIndex.build(self.tokens)

    def with_docs(self, tokens, doc_ids=None, categories=None, relevant=None):
        """Copy with replaced/extended documents (used by the adversarial experiments)."""
        return Collection(self.name, doc_ids or self.doc_ids, categories or self.categories, tokens,
                          self.queries, relevant or self.relevant, self.query_category)


def load_collection(name, annotator=None):
    docs, queries, qrels = load_corpus(name)
    if annotator is not None:
        qrels = qrels[qrels["annotator"] == annotator]
    pos = {d: i for i, d in enumerate(docs["doc_id"])}
    relevant = {}
    for r in qrels[qrels["relevance"] > 0].itertuples():
        if r.doc_id in pos:
            relevant.setdefault(r.query_id, set()).add(pos[r.doc_id])
    return Collection(
        name=name,
        doc_ids=list(docs["doc_id"]),
        categories=list(docs["category"]),
        tokens=[preprocess(t) for t in docs["text"]],
        queries=[(q.query_id, preprocess(q.text)) for q in queries.itertuples()],
        relevant=relevant,
        query_category=dict(zip(queries["query_id"], queries["category"])),
    )


def load_cached_index(coll):
    """Inverted index cached as .npz (proposal Sec. 6.4); rebuilt if the corpus changed."""
    path = CACHE / f"{coll.name}_index.npz"
    if path.exists():
        idx = InvertedIndex.load(path)
        if idx.N == len(coll.tokens) and np.array_equal(idx.doc_len, coll.doc_len):
            return idx
    idx = coll.build_index()
    idx.save(path)
    return idx


def collection_stats(coll, idx):
    dl = coll.doc_len
    return pd.Series({"docs": idx.N, "vocabulary": len(idx.terms), "postings": int(idx.term_ptr[-1]),
                      "tokens": int(dl.sum()), "len_min": int(dl.min()), "len_median": float(np.median(dl)),
                      "len_mean": float(dl.mean()), "len_max": int(dl.max()), "queries": len(coll.queries),
                      "avg_relevant_per_query": float(np.mean([len(v) for v in coll.relevant.values()]))})
