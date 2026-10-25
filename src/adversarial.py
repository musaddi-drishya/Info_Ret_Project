"""Controlled adversarial document construction (proposal Sec. 8).

Attacks operate on preprocessed token lists so that document length is
controlled exactly. Relevance of every synthetic document is known by
construction (it inherits the categories of its source documents).
"""
import numpy as np


# ---------------------------------------------------------- E3: verbosity padding
def verbosity_attack(coll, n_targets=50, factors=(1, 2, 5, 10), seed=7):
    """
    Pick `n_targets` documents and replace each by its content repeated m times.
    Yields (factor, attacked collection, target doc indices).
    No new information is added, so an ideal normaliser should leave rankings unchanged.
    """
    rng = np.random.default_rng(seed)
    targets = np.sort(rng.choice(len(coll.tokens), size=n_targets, replace=False))
    for m in factors:
        toks = list(coll.tokens)
        for d in targets:
            toks[d] = coll.tokens[d] * m
        yield m, coll.with_docs(toks), targets


# ------------------------------------------------- E4: multi-topical camouflage
def _segment(coll, category, n_tokens, rng, exclude=()):
    """Concatenate distinct docs of one category until `n_tokens` tokens are reached."""
    pool = [i for i, c in enumerate(coll.categories) if c == category and i not in exclude]
    rng.shuffle(pool)
    seg, used = [], []
    for i in pool:
        seg.extend(coll.tokens[i])
        used.append(i)
        if len(seg) >= n_tokens:
            break
    return seg[:n_tokens], used


def multitopic_docs(coll, n_topics, target_len=300, n_docs=20, seed=11):
    """
    Synthetic docs of exactly `target_len` tokens covering `n_topics` distinct categories
    (target_len / n_topics tokens each). n_topics = 1 gives a genuinely single-topic long doc.
    Returns list of dicts {tokens, categories, sources}.
    """
    rng = np.random.default_rng(seed + n_topics)
    cats = sorted(set(coll.query_category.values()))
    out = []
    for _ in range(n_docs):
        chosen = rng.choice(cats, size=n_topics, replace=False)
        per = target_len // n_topics
        toks, sources = [], []
        for c in chosen:
            seg, used = _segment(coll, c, per, rng)
            toks.extend(seg)
            sources.extend(used)
        out.append({"tokens": toks[:target_len], "categories": list(chosen), "sources": sources})
    return out


def verbose_docs(coll, target_len=300, n_docs=20, seed=13):
    """Single-topic docs padded to `target_len` by repeating one short source doc (verbosity control)."""
    rng = np.random.default_rng(seed)
    cats = sorted(set(coll.query_category.values()))
    lens = coll.doc_len
    out = []
    for _ in range(n_docs):
        c = rng.choice(cats)
        pool = [i for i, cc in enumerate(coll.categories) if cc == c and 30 <= lens[i] <= target_len / 3]
        src = int(rng.choice(pool))
        reps = int(np.ceil(target_len / len(coll.tokens[src])))
        out.append({"tokens": (coll.tokens[src] * reps)[:target_len], "categories": [c], "sources": [src]})
    return out


def inject(coll, synthetic, prefix):
    """Append synthetic docs to the collection; they are relevant to every query of their categories."""
    base = len(coll.tokens)
    toks = list(coll.tokens) + [s["tokens"] for s in synthetic]
    ids = list(coll.doc_ids) + [f"{prefix}-{i:03d}" for i in range(len(synthetic))]
    cats = list(coll.categories) + ["+".join(s["categories"]) for s in synthetic]
    relevant = {q: set(v) for q, v in coll.relevant.items()}
    for j, s in enumerate(synthetic):
        for q, c in coll.query_category.items():
            if c in s["categories"]:
                relevant.setdefault(q, set()).add(base + j)
    new_idx = np.arange(base, base + len(synthetic))
    return coll.with_docs(toks, ids, cats, relevant), new_idx


# ------------------------------------------------------------------ diagnostics
def target_ranks(model, coll, targets):
    """For each (query, target doc relevant to it): 1-based rank of the target. Returns array."""
    targets = set(int(t) for t in targets)
    ranks = []
    for qid, q in coll.queries:
        rel_targets = [t for t in coll.relevant.get(qid, ()) if t in targets]
        if not rel_targets:
            continue
        scores = model.score(q)
        for t in rel_targets:
            ranks.append(1 + int((scores > scores[t]).sum()))
    return np.array(ranks)


def intrusions(model, coll, targets, k=10):
    """How often attacked docs enter the top-k of queries they are NOT relevant to."""
    targets = set(int(t) for t in targets)
    n = 0
    for qid, q in coll.queries:
        rel = coll.relevant.get(qid, set())
        n += sum(1 for d in model.rank(q, k) if d in targets and d not in rel)
    return n
