"""From-scratch inverted index.

Postings are built as a Python dict  term -> {doc: tf}  and then frozen into a
CSR-style layout (term_ptr / doc_ids / tfs NumPy arrays) so that scoring a
query term is a single vectorised slice. The frozen arrays can be cached to an
.npz file and reloaded without re-tokenising the collection.
"""
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


class InvertedIndex:
    def __init__(self):
        self.vocab = {}          # term -> term id
        self.terms = []          # term id -> term
        self.term_ptr = None     # postings of term t live in [term_ptr[t], term_ptr[t+1])
        self.doc_ids = None      # int32 doc indices, sorted within each posting list
        self.tfs = None          # raw term frequencies (float32)
        self.df = None           # document frequency per term
        self.doc_len = None      # number of index tokens per doc
        self.doc_unique = None   # number of unique terms per doc
        self.doc_avg_tf = None   # mean tf over the doc's unique terms
        self.N = 0
        self.build_seconds = 0.0

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, tokenized_docs):
        t0 = time.perf_counter()
        idx = cls()
        postings = defaultdict(dict)
        doc_len, doc_unique = [], []
        for d, toks in enumerate(tokenized_docs):
            counts = Counter(toks)
            doc_len.append(len(toks))
            doc_unique.append(len(counts))
            for term, tf in counts.items():
                postings[term][d] = tf

        idx.terms = sorted(postings)
        idx.vocab = {t: i for i, t in enumerate(idx.terms)}
        sizes = np.fromiter((len(postings[t]) for t in idx.terms), dtype=np.int64, count=len(idx.terms))
        idx.term_ptr = np.zeros(len(idx.terms) + 1, dtype=np.int64)
        np.cumsum(sizes, out=idx.term_ptr[1:])
        idx.doc_ids = np.empty(idx.term_ptr[-1], dtype=np.int32)
        idx.tfs = np.empty(idx.term_ptr[-1], dtype=np.float32)
        for i, t in enumerate(idx.terms):
            plist = postings[t]
            s, e = idx.term_ptr[i], idx.term_ptr[i + 1]
            docs = np.fromiter(plist.keys(), dtype=np.int32, count=len(plist))
            order = np.argsort(docs)
            idx.doc_ids[s:e] = docs[order]
            idx.tfs[s:e] = np.fromiter(plist.values(), dtype=np.float32, count=len(plist))[order]

        idx.N = len(doc_len)
        idx.df = sizes.astype(np.float64)
        idx.doc_len = np.asarray(doc_len, dtype=np.float64)
        idx.doc_unique = np.asarray(doc_unique, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            idx.doc_avg_tf = np.where(idx.doc_unique > 0, idx.doc_len / idx.doc_unique, 1.0)
        idx.build_seconds = time.perf_counter() - t0
        return idx

    # ----------------------------------------------------------------- access
    def postings(self, term):
        """Return (doc_ids, tfs) for a term, or empty arrays if unseen."""
        tid = self.vocab.get(term)
        if tid is None:
            return np.empty(0, np.int32), np.empty(0, np.float32)
        s, e = self.term_ptr[tid], self.term_ptr[tid + 1]
        return self.doc_ids[s:e], self.tfs[s:e]

    def term_slice(self, term):
        tid = self.vocab.get(term)
        if tid is None:
            return None
        return tid, self.term_ptr[tid], self.term_ptr[tid + 1]

    def posting_term_ids(self):
        return np.repeat(np.arange(len(self.terms)), np.diff(self.term_ptr))

    def memory_bytes(self):
        arrays = (self.term_ptr, self.doc_ids, self.tfs, self.df, self.doc_len, self.doc_unique, self.doc_avg_tf)
        vocab_bytes = sum(len(t) + 49 for t in self.terms) + 8 * len(self.terms)  # approx str + dict slot
        return int(sum(a.nbytes for a in arrays) + vocab_bytes)

    # ------------------------------------------------------------------ cache
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, terms=np.array(self.terms, dtype=object), term_ptr=self.term_ptr,
                            doc_ids=self.doc_ids, tfs=self.tfs, df=self.df, doc_len=self.doc_len,
                            doc_unique=self.doc_unique, doc_avg_tf=self.doc_avg_tf)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        idx = cls()
        idx.terms = list(z["terms"])
        idx.vocab = {t: i for i, t in enumerate(idx.terms)}
        for k in ("term_ptr", "doc_ids", "tfs", "df", "doc_len", "doc_unique", "doc_avg_tf"):
            setattr(idx, k, z[k])
        idx.N = len(idx.doc_len)
        return idx
