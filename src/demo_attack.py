"""Live verbosity-attack demo for the presentation.

Pads one document 1x/2x/5x/10x by repeating its own text and shows how its rank
for a query moves under each model. Two scenarios are shown:
  * relevant    - a relevant doc sitting just outside the top-10 (does padding "help" it?)
  * attacker    - a NON-relevant doc that mentions the query terms (can padding push spam in?)

Usage:  python -m src.demo_attack                       # Reuters, query RT05
        python -m src.demo_attack --corpus newsgroups --query NG22
        python -m src.demo_attack --doc rt-1234          # pad a specific document
"""
import argparse
import sys

import numpy as np

from .models import BM25, CosineTFIDF, PivotedCosine, PivotedUnique
from .pipeline import load_collection

FACTORS = (1, 2, 5, 10)
MODELS = (("Cosine", lambda i: CosineTFIDF(i)),
          ("Pivoted s=0.7", lambda i: PivotedCosine(i, 0.7)),
          ("Pivoted s=0.3", lambda i: PivotedCosine(i, 0.3)),
          ("Piv-unique", lambda i: PivotedUnique(i)),
          ("BM25", lambda i: BM25(i)))


def rank_of(model, q, d):
    s = model.score(q)
    return 1 + int((s > s[d]).sum())


def pick_target(coll, q, rel, relevant):
    """Relevant: the relevant doc ranked closest to 15 under pivoted s=0.7.
    Attacker: the best-ranked non-relevant doc outside the top-20 that contains a query term."""
    model = PivotedCosine(coll.build_index(), 0.7)
    ranking = list(model.rank(q))
    if relevant:
        cands = [d for d in ranking if d in rel]
        return min(cands, key=lambda d: abs(ranking.index(d) - 14))
    return next(d for d in ranking[20:] if d not in rel)


def run(coll, qid, target, label):
    q = dict(coll.queries)[qid]
    rel = coll.relevant.get(qid, set())
    toks = coll.tokens[target]
    print(f"\n{label}: doc {coll.doc_ids[target]} [{coll.categories[target]}], {len(toks)} tokens, "
          f"{'RELEVANT' if target in rel else 'NOT relevant'} to the query")
    print(f"  {'model':<15}" + "".join(f"{f'{m}x':>8}" for m in FACTORS) + "   change")
    ranks = {name: [] for name, _ in MODELS}
    for m in FACTORS:
        padded = list(coll.tokens)
        padded[target] = toks * m
        idx = coll.with_docs(padded).build_index()
        for name, make in MODELS:
            ranks[name].append(rank_of(make(idx), q, target))
    for name, r in ranks.items():
        delta = r[0] - r[-1]
        verdict = "promoted" if delta > 0 else "demoted" if delta < 0 else "unchanged"
        top10 = "  << enters top-10" if r[0] > 10 >= r[-1] else ""
        print(f"  {name:<15}" + "".join(f"{x:>8}" for x in r) + f"   {verdict} {abs(delta)}{top10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="reuters")
    ap.add_argument("--query", default=None, help="query id, e.g. RT05 (default: first query)")
    ap.add_argument("--doc", default=None, help="doc id to pad (default: auto-pick both scenarios)")
    args = ap.parse_args()

    coll = load_collection(args.corpus)
    qid = args.query or ("RT05" if args.corpus == "reuters" else coll.queries[0][0])
    qtext = dict(coll.queries)[qid]
    rel = coll.relevant.get(qid, set())
    print(f"Corpus: {args.corpus} ({len(coll.tokens)} docs)   Query {qid}: {' '.join(qtext)}   "
          f"({len(rel)} relevant docs)")
    print("Rank of the padded document as it is repeated 1x / 2x / 5x / 10x (lower = higher in results)")

    if args.doc:
        run(coll, qid, coll.doc_ids.index(args.doc), "Chosen document")
    else:
        run(coll, qid, pick_target(coll, qtext, rel, relevant=True), "Scenario A - relevant doc")
        run(coll, qid, pick_target(coll, qtext, rel, relevant=False), "Scenario B - attacker (non-relevant doc)")
    print("\nNo words were added: the padded doc only repeats its own text. An ideal length "
          "normaliser would leave every row unchanged.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
