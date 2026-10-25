"""Sanity checks for the from-scratch components.  Run:  python -m pytest -q  (or python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import average_precision, ndcg_at_k  # noqa: E402
from src.index import InvertedIndex  # noqa: E402
from src.models import BM25, CosineTFIDF, PivotedCosine, PivotedUnique  # noqa: E402
from src.preprocess import preprocess, stem  # noqa: E402

DOCS = [preprocess(t) for t in (
    "the space shuttle launch was delayed by nasa",
    "nasa plans a new orbit for the space station, space is big",
    "baseball season opens with a home run",
    "the pitcher threw a perfect game in baseball history " * 3,
    "encryption keys and the clipper chip privacy debate",
)]


def test_porter_reference_words():
    ref = {"caresses": "caress", "ponies": "poni", "relational": "relat", "hopping": "hop",
           "generalization": "gener", "agreed": "agre", "motoring": "motor", "adjustable": "adjust"}
    assert {w: stem(w) for w in ref} == ref


def test_index_postings_roundtrip(tmp_path=Path("data/cache/_test")):
    idx = InvertedIndex.build(DOCS)
    docs, tfs = idx.postings("space")
    assert list(docs) == [0, 1] and list(tfs) == [1, 2]
    idx.save(tmp_path / "idx.npz")
    idx2 = InvertedIndex.load(tmp_path / "idx.npz")
    assert np.array_equal(idx.doc_ids, idx2.doc_ids) and idx2.vocab == idx.vocab


def test_pivoted_slope_one_equals_cosine():
    idx = InvertedIndex.build(DOCS)
    q = preprocess("space nasa baseball")
    assert np.allclose(CosineTFIDF(idx).score(q), PivotedCosine(idx, slope=1.0).score(q))


def test_pivoted_unique_ignores_repetition():
    idx = InvertedIndex.build(DOCS + [DOCS[2] * 5])
    s = PivotedUnique(idx).score(preprocess("baseball home run"))
    assert np.isclose(s[2], s[5])


def test_models_rank_relevant_first():
    idx = InvertedIndex.build(DOCS)
    for m in (CosineTFIDF(idx), PivotedCosine(idx), PivotedUnique(idx), BM25(idx)):
        assert m.rank(preprocess("clipper chip encryption"))[0] == 4


def test_metrics():
    assert average_precision(np.array([1, 0, 1, 0.]), 2) == (1 + 2 / 3) / 2
    assert ndcg_at_k(np.array([1, 1, 0.]), 2, 10) == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
