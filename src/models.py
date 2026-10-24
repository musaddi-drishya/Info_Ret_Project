"""Retrieval models built on top of InvertedIndex.

All scores are computed term-at-a-time with NumPy:

    score(q, d) = sum_{t in q}  w_q(t) * w_d(t, d)  /  norm(d)

The models differ only in the per-posting weight w_d and the per-document
normalisation factor norm(d):

    CosineTFIDF       w_d = (1 + ln tf) * idf          norm = ||d||_2                       (ltc.ltc)
    PivotedCosine     w_d = (1 + ln tf) * idf          norm = (1 - s) * pivot + s * ||d||_2
    PivotedUnique     w_d = (1 + ln tf)/(1 + ln avgtf) * idf
                                                       norm = (1 - s) * pivot + s * |unique(d)|
    BM25              Robertson/Walker tf saturation with length parameter b (no separate norm)

pivot = mean of the old normalisation factor over the collection (Singhal et al., 1996),
so that documents of average length keep (approximately) their cosine weight.
"""
from collections import Counter

import numpy as np


def log_tf(tf):
    return np.where(tf > 0, 1.0 + np.log(np.maximum(tf, 1e-12)), 0.0)


class _TermAtATime:
    name = "base"

    def __init__(self, index):
        self.index = index
        self.idf = self._idf()
        self.weights = self._posting_weights()   # aligned with index.tfs
        self.norm = self._doc_norm()              # per-document normaliser

    def _idf(self):
        return np.log(self.index.N / self.index.df)

    def _posting_weights(self):
        raise NotImplementedError

    def _doc_norm(self):
        return np.ones(self.index.N)

    def query_weights(self, q_terms):
        counts = Counter(t for t in q_terms if t in self.index.vocab)
        return {t: (1.0 + np.log(c)) * self.idf[self.index.vocab[t]] for t, c in counts.items()}

    def score(self, q_terms):
        scores = np.zeros(self.index.N)
        for t, wq in self.query_weights(q_terms).items():
            _, s, e = self.index.term_slice(t)
            scores[self.index.doc_ids[s:e]] += wq * self.weights[s:e]
        return scores / self.norm

    def rank(self, q_terms, k=None):
        scores = self.score(q_terms)
        order = np.argsort(-scores, kind="stable")
        order = order[scores[order] > 0]
        return order if k is None else order[:k]


class CosineTFIDF(_TermAtATime):
    name = "Cosine"

    def _posting_weights(self):
        tids = self.index.posting_term_ids()
        return log_tf(self.index.tfs) * self.idf[tids]

    def _doc_norm(self):
        sq = np.bincount(self.index.doc_ids, weights=self.weights ** 2, minlength=self.index.N)
        return np.sqrt(np.maximum(sq, 1e-12))


class PivotedCosine(CosineTFIDF):
    name = "Pivoted"

    def __init__(self, index, slope=0.7, pivot=None):
        self.slope = slope
        self._pivot_override = pivot
        super().__init__(index)

    def _doc_norm(self):
        cos = super()._doc_norm()
        self.cosine_norm = cos
        self.pivot = cos.mean() if self._pivot_override is None else self._pivot_override
        return (1.0 - self.slope) * self.pivot + self.slope * cos

    def with_slope(self, slope):
        """Cheap re-normalisation for slope sweeps (posting weights are reused)."""
        clone = object.__new__(PivotedCosine)
        clone.__dict__.update(self.__dict__)
        clone.slope = slope
        clone.norm = (1.0 - slope) * self.pivot + slope * self.cosine_norm
        return clone


class PivotedUnique(_TermAtATime):
    """Singhal et al.'s pivoted *unique* normalisation (Lnu weighting)."""
    name = "PivUnique"

    def __init__(self, index, slope=0.2, pivot=None):
        self.slope = slope
        self._pivot_override = pivot
        super().__init__(index)

    def _posting_weights(self):
        tids = self.index.posting_term_ids()
        avg = self.index.doc_avg_tf[self.index.doc_ids]
        return log_tf(self.index.tfs) / (1.0 + np.log(avg)) * self.idf[tids]

    def _doc_norm(self):
        u = self.index.doc_unique
        self.pivot = u.mean() if self._pivot_override is None else self._pivot_override
        return (1.0 - self.slope) * self.pivot + self.slope * u


class BM25(_TermAtATime):
    name = "BM25"

    def __init__(self, index, k1=1.2, b=0.75, avgdl=None):
        self.k1, self.b = k1, b
        self._avgdl = avgdl
        super().__init__(index)

    def _idf(self):
        df, N = self.index.df, self.index.N
        return np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _posting_weights(self):
        idx = self.index
        avgdl = idx.doc_len.mean() if self._avgdl is None else self._avgdl
        dl = idx.doc_len[idx.doc_ids]
        tf = idx.tfs
        sat = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
        return sat * self.idf[idx.posting_term_ids()]

    def query_weights(self, q_terms):
        counts = Counter(t for t in q_terms if t in self.index.vocab)
        return {t: float(c) for t, c in counts.items()}


MODELS = {
    "Cosine": lambda idx: CosineTFIDF(idx),
    "Pivoted": lambda idx: PivotedCosine(idx, slope=0.7),
    "PivUnique": lambda idx: PivotedUnique(idx, slope=0.2),
    "BM25": lambda idx: BM25(idx),
}
