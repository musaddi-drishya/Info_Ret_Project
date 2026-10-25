"""IR evaluation: P/R/F1@k, MAP, NDCG@k and the Singhal et al. length-bias analysis."""
import time

import numpy as np
import pandas as pd

KS = (5, 10, 20)


def average_precision(ranked_rel, n_relevant):
    if n_relevant == 0:
        return 0.0
    hits = np.cumsum(ranked_rel)
    precisions = hits / np.arange(1, len(ranked_rel) + 1)
    return float((precisions * ranked_rel).sum() / n_relevant)


def ndcg_at_k(ranked_rel, n_relevant, k):
    gains = ranked_rel[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    ideal = (1.0 / np.log2(np.arange(2, min(k, n_relevant) + 2))).sum()
    return float((gains * discounts).sum() / ideal) if ideal > 0 else 0.0


def evaluate_run(model, queries, relevant, ks=KS, depth=None):
    """
    model     : object with .rank(q_terms) -> array of doc indices
    queries   : list of (query_id, q_terms)
    relevant  : dict query_id -> set of doc indices
    Returns (per-query DataFrame, rankings dict, mean query latency in ms).
    """
    rows, rankings, latencies = [], {}, []
    for qid, q_terms in queries:
        t0 = time.perf_counter()
        ranking = model.rank(q_terms)
        latencies.append(time.perf_counter() - t0)
        if depth:
            ranking = ranking[:depth]
        rankings[qid] = ranking
        rel = relevant.get(qid, set())
        ranked_rel = np.fromiter((d in rel for d in ranking), dtype=float, count=len(ranking))
        row = {"query_id": qid, "AP": average_precision(ranked_rel, len(rel)),
               "NDCG@10": ndcg_at_k(ranked_rel, len(rel), 10)}
        for k in ks:
            hits = ranked_rel[:k].sum()
            p, r = hits / k, (hits / len(rel) if rel else 0.0)
            row[f"P@{k}"], row[f"R@{k}"] = p, r
            row[f"F1@{k}"] = 2 * p * r / (p + r) if p + r else 0.0
        rows.append(row)
    return pd.DataFrame(rows), rankings, 1000 * float(np.mean(latencies))


def summarize(per_query):
    s = per_query.drop(columns="query_id").mean()
    return s.rename({"AP": "MAP"})


# ---------------------------------------------------------------- length bias
def length_bins(doc_len, n_bins=10):
    """Equal-frequency bins over document length (bin 0 = shortest docs)."""
    ranks = np.argsort(np.argsort(doc_len, kind="stable"), kind="stable")
    return (ranks * n_bins // len(doc_len)).astype(int)


def length_bias(rankings, relevant, doc_len, top_k=50, n_bins=10):
    """
    Retrieval-vs-relevance analysis (Singhal, Buckley & Mitra 1996, Sec. 3):
    P(retrieved | bin) vs P(relevant | bin), pooled over all queries.

    Returns a DataFrame per bin plus two scalar summaries:
      TVD  = 0.5 * sum |P_ret - P_rel|  (0 = retrieval follows relevance exactly)
      LogLenGap = mean ln(len) of retrieved - mean ln(len) of relevant
                  (< 0 : system prefers shorter docs than are relevant; > 0 : longer)
    """
    bins = length_bins(doc_len, n_bins)
    ret = np.concatenate([r[:top_k] for r in rankings.values()])
    rel = np.concatenate([np.fromiter(relevant[q], dtype=int) for q in rankings if relevant.get(q)])
    p_ret = np.bincount(bins[ret], minlength=n_bins) / len(ret)
    p_rel = np.bincount(bins[rel], minlength=n_bins) / len(rel)
    edges = [int(np.median(doc_len[bins == b])) for b in range(n_bins)]
    table = pd.DataFrame({"bin": range(n_bins), "median_len": edges, "P_retrieved": p_ret, "P_relevant": p_rel})
    loglen = np.log(np.maximum(doc_len, 1))
    return table, {"TVD": 0.5 * float(np.abs(p_ret - p_rel).sum()),
                   "LogLenGap": float(loglen[ret].mean() - loglen[rel].mean())}
