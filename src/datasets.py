"""Corpus builders: raw files -> stratified samples -> SQLite.

Run:  python -m src.datasets
"""
import html
import json
import re

import numpy as np
import pandas as pd

from .preprocess import preprocess
from .queries import NEWSGROUPS_QUERIES, REUTERS_QUERIES, REUTERS_TOPICS
from .storage import ROOT, save_corpus

RAW = ROOT / "data" / "raw"
SEED = 42
SAMPLE_SIZE = 1000
LENGTH_STRATA = 5


def stratified_sample(df, per_category, rng, strata=LENGTH_STRATA):
    """Sample `per_category` docs from each category, spread evenly over length quantiles."""
    picked = []
    for _, g in df.groupby("category"):
        if len(g) <= per_category:
            picked.append(g)
            continue
        g = g.assign(stratum=pd.qcut(g["n_tokens"].rank(method="first"), strata, labels=False))
        base, extra = divmod(per_category, strata)
        for s, gs in g.groupby("stratum"):
            n = base + (1 if s < extra else 0)
            picked.append(gs.sample(n=min(n, len(gs)), random_state=rng.integers(1 << 31)))
    return pd.concat(picked).drop(columns="stratum", errors="ignore")


def _with_lengths(df):
    return df.assign(n_tokens=[len(preprocess(t)) for t in df["text"]])


def _proxy_qrels(docs, queries):
    rows = [(q.query_id, d, 1) for q in queries.itertuples()
            for d in docs.loc[docs["category"] == q.category, "doc_id"]]
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "relevance"]).assign(annotator="category-proxy")


# ---------------------------------------------------------------- 20 Newsgroups
QUOTE_RE = re.compile(r"^\s*(>|\|>|:|In article|.* writes:$|.* wrote:$)")


def clean_newsgroup_post(raw):
    """Drop headers (they leak the newsgroup label), keep Subject, drop quoted replies and signatures."""
    head, _, body = raw.partition("\n\n")
    subject = next((l[8:].strip() for l in head.splitlines() if l.lower().startswith("subject:")), "")
    subject = re.sub(r"^(re:\s*)+", "", subject, flags=re.I)
    lines = []
    for line in body.splitlines():
        if line.strip() == "--":          # signature delimiter
            break
        if not QUOTE_RE.match(line):
            lines.append(line)
    return subject, "\n".join(lines).strip()


def build_newsgroups():
    rows = []
    for split in ("20news-bydate-train", "20news-bydate-test"):
        for f in sorted((RAW / "20ng" / split).glob("*/*")):
            subject, body = clean_newsgroup_post(f.read_text(encoding="latin-1"))
            rows.append({"doc_id": f"ng-{f.parent.name}-{f.name}", "category": f.parent.name,
                         "title": subject, "text": f"{subject}\n{body}", "source": f"20news/{split}",
                         "collected": "1993"})
    df = _with_lengths(pd.DataFrame(rows))
    df = df[df["n_tokens"] >= 20]
    rng = np.random.default_rng(SEED)
    docs = stratified_sample(df, SAMPLE_SIZE // df["category"].nunique(), rng).reset_index(drop=True)
    queries = pd.DataFrame([(f"NG{i+1:02d}", t, c) for i, (c, t) in enumerate(NEWSGROUPS_QUERIES)],
                           columns=["query_id", "text", "category"])
    return docs, queries, _proxy_qrels(docs, queries)


# --------------------------------------------------------------- Reuters-21578
def build_reuters():
    rows = []
    for f in sorted((RAW / "reuters").glob("reut2-*.sgm")):
        for doc in re.findall(r"<REUTERS(.*?)</REUTERS>", f.read_text(encoding="latin-1"), re.S):
            topics = re.findall(r"<D>(.*?)</D>", re.search(r"<TOPICS>(.*?)</TOPICS>", doc, re.S).group(1))
            body = re.search(r"<BODY>(.*?)</BODY>", doc, re.S)
            if len(topics) != 1 or topics[0] not in REUTERS_TOPICS or not body:
                continue
            title = re.search(r"<TITLE>(.*?)</TITLE>", doc, re.S)
            title = html.unescape(title.group(1)) if title else ""
            text = html.unescape(body.group(1)).replace("\x03", "")
            text = re.sub(r"\s*Reuter\s*$", "", text.strip())
            newid = re.search(r'NEWID="(\d+)"', doc).group(1)
            rows.append({"doc_id": f"rt-{newid}", "category": topics[0], "title": title,
                         "text": f"{title}\n{text}", "source": f"reuters21578/{f.name}", "collected": "1987"})
    df = _with_lengths(pd.DataFrame(rows))
    df = df[df["n_tokens"] >= 20]
    rng = np.random.default_rng(SEED)
    per_cat = SAMPLE_SIZE // len(REUTERS_TOPICS)
    docs = stratified_sample(df, per_cat, rng)
    # Small topics cannot supply `per_cat` docs: top up from the largest topics, length-stratified.
    shortfall = SAMPLE_SIZE - len(docs)
    if shortfall > 0:
        rest = df[~df["doc_id"].isin(docs["doc_id"])]
        big = rest[rest["category"].isin(["earn", "acq", "crude", "trade"])]
        docs = pd.concat([docs, stratified_sample(big, -(-shortfall // 4), rng).head(shortfall)])
    docs = docs.reset_index(drop=True)
    queries = pd.DataFrame([(f"RT{i+1:02d}", t, c) for i, (c, t) in enumerate(REUTERS_QUERIES)],
                           columns=["query_id", "text", "category"])
    return docs, queries, _proxy_qrels(docs, queries)


# --------------------------------------------------------------- Campus corpus
CAMPUS_JSON = RAW / "campus" / "campus_articles.json"
CAMPUS_QUERIES = ROOT / "data" / "judgments" / "campus_queries.csv"
CAMPUS_QRELS = ROOT / "data" / "judgments" / "campus_qrels.csv"


def build_campus():
    """Scraped campus articles (scripts/collect_campus.py) + human judgments, if present."""
    arts = json.loads(CAMPUS_JSON.read_text(encoding="utf-8"))
    docs = pd.DataFrame([{"doc_id": a["doc_id"], "category": a.get("category", ""), "title": a["title"],
                          "text": f"{a['title']}\n{a['text']}", "source": a["url"],
                          "collected": a["collected"]} for a in arts])
    docs = _with_lengths(docs)
    queries = pd.read_csv(CAMPUS_QUERIES) if CAMPUS_QUERIES.exists() else None
    qrels = None
    if CAMPUS_QRELS.exists():
        j = pd.read_csv(CAMPUS_QRELS)
        long = j.melt(id_vars=["query_id", "doc_id"], value_vars=[c for c in j if c.startswith("judge")],
                      var_name="annotator", value_name="relevance").dropna()
        if len(long):
            final = j.dropna(subset=["final"]) if "final" in j else None
            parts = [long.assign(relevance=long["relevance"].astype(int))]
            if final is not None and len(final):
                parts.append(final[["query_id", "doc_id"]].assign(relevance=final["final"].astype(int),
                                                                  annotator="final"))
            qrels = pd.concat(parts)[["query_id", "doc_id", "relevance", "annotator"]]
    return docs, queries, qrels


def main():
    for name, builder in (("newsgroups", build_newsgroups), ("reuters", build_reuters)):
        docs, queries, qrels = builder()
        save_corpus(name, docs, queries, qrels)
        print(f"{name}: {len(docs)} docs, {docs['category'].nunique()} categories, "
              f"{len(queries)} queries, {len(qrels)} relevant pairs, "
              f"length min/median/max = {docs['n_tokens'].min()}/{int(docs['n_tokens'].median())}/"
              f"{docs['n_tokens'].max()}")
    if CAMPUS_JSON.exists():
        docs, queries, qrels = build_campus()
        save_corpus("campus", docs, queries, qrels)
        print(f"campus: {len(docs)} docs, queries={0 if queries is None else len(queries)}, "
              f"judged pairs={0 if qrels is None else len(qrels)}")


if __name__ == "__main__":
    main()
