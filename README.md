# Correcting Length Bias - Pivoted Document Length Normalization

**Group 6 · Information Retrieval · Classic Paper Replication & Stress Testing**
Divyansh Jain · Drishya Musaddi · Aditya Goyal · Himabh Gemini

We replicate *Pivoted Document Length Normalization* (Singhal, Buckley & Mitra, SIGIR 1996)
**from scratch** (Python + NumPy + Pandas, no scikit-learn / Lucene / Elasticsearch), measure
document-length bias with the paper's retrieval-vs-relevance analysis, and stress-test the
normaliser with two theory-driven adversarial attacks: **verbosity padding** (length via repetition)
and **multi-topic camouflage** (length via genuinely different topics, after Na et al.).

## Pipeline

```
raw docs ─► preprocess ─► SQLite ─► inverted index (.npz cache) ─► TF-IDF weights
 (20NG,      tokenize,     docs,      dict postings → CSR NumPy        │
  Reuters,   stopwords,    queries,                                    ▼
  campus)    Porter stem   qrels                          normalisation: cosine | pivoted cosine
                                                          | pivoted unique | BM25
                                                                       │
             adversarial generator (verbosity ×m, k-topic docs) ──►  rank ─► evaluation
                                                                  P/R/F1@k, MAP, NDCG, length-bias, time, memory
```

| Module | What it does |
|---|---|
| `src/preprocess.py` | Regex tokenizer, stopword list, **Porter stemmer implemented from scratch** |
| `src/storage.py` | SQLite schema: documents (text, metadata, source, date, length), queries, qrels |
| `src/datasets.py` | 20 Newsgroups / Reuters-21578 parsers, header & quote stripping, **stratified sampling by category × length quintile** (1,000 docs each), campus corpus loader |
| `src/queries.py` | 30 + 25 hand-written topic queries (category membership = relevance proxy) |
| `src/index.py` | Inverted index: dict postings frozen into CSR NumPy arrays, `.npz` caching |
| `src/models.py` | `CosineTFIDF` (ltc), `PivotedCosine` (slope *s*, pivot = mean cosine norm), `PivotedUnique` (Lnu), `BM25` |
| `src/evaluate.py` | P@k, R@k, F1@k, AP/MAP, NDCG@10, retrieval-vs-relevance length bins, TVD and log-length-gap bias scores |
| `src/adversarial.py` | Verbosity padding (×2/×5/×10), k-topic synthetic docs at fixed length, verbose control, target-rank and intrusion diagnostics |
| `src/experiments.py` | Runs E1–E7, writes `results/tables/*.csv` and `results/figures/*.png` |
| `src/search.py` | Side-by-side search: one query, all four models, document lengths shown |
| `src/demo_attack.py` | Live verbosity attack: pad one document 1×/2×/5×/10× and watch its rank move per model |
| `tests/test_core.py` | Sanity checks: Porter reference words, index round-trip, pivoted(s=1) ≡ cosine, metrics |
| `report/architecture.py`, `report/build_report.py` | Architecture diagram and the report PDF, built from `results/tables` |
| `scripts/collect_campus.py` | Polite crawler for ~100 BITS Pilani news/notice pages (Requests + BeautifulSoup) |
| `scripts/make_judging_pool.py` | TREC-style pooled judging sheet for the campus corpus, Cohen's κ |

## Pivoted normalisation (as implemented)

```
score(q,d) = Σ_t  w_q(t) · w_d(t,d) / N_d
w(t,x)     = (1 + ln tf(t,x)) · ln(N / df(t))
N_d        = (1 − s) · p + s · ||d||₂          p = mean_d ||d||₂   (s = 1 → plain cosine)
```
Pivoted unique: `w_d = (1+ln tf)/(1+ln avg_tf(d)) · idf`, `N_d = (1−s)·p + s·|unique terms(d)|`, `s = 0.2`.

## Experiments

| ID | Model | Manipulation | Output |
|---|---|---|---|
| E1 | TF-IDF + cosine | none | `*_E1E2_clean.csv`, `*_length_bias.png` |
| E2 | TF-IDF + pivoted cosine | none | same |
| E3 | all | 50 docs padded ×1/×2/×5/×10 | `*_E3_verbosity.csv`, `*_verbosity.png` |
| E4 | all + pivoted at s ∈ {0.3, 0.5, 0.9} | 20 synthetic docs per condition: verbose, 1/2/3/4-topic, 300 tokens | `*_E4_multitopic.csv` |
| E5 | pivoted cosine | slope sweep s = 0…1, clean and under ×5 padding | `*_E5_slope_*.csv`, `*_slope_sweep.png` |
| E6* | pivoted unique | all of the above | in E1–E4 tables |
| E7* | BM25 | all of the above + *b* sweep | `*_E7_bm25_b_sweep.csv` |

## Reproduce

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
# 1. raw data (≈22 MB) into data/raw/
curl -L -o data/raw/20ng/20news-bydate.tar.gz http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
curl -L -o data/raw/reuters/reuters.zip https://archive.ics.uci.edu/static/public/137/reuters+21578+text+categorization+collection.zip
#    extract both archives in place (tar -xzf / unzip, then tar -xzf reuters21578.tar.gz)
# 2. campus corpus (≈3 min, 1 request/s)
python scripts/collect_campus.py --target 100
# 3. build SQLite corpora (tokenises ~30k docs once, a few minutes)
python -m src.datasets
# 4. run E1–E7 (about 15 s), then the figures and report
python -m src.experiments
python report/architecture.py
python report/build_report.py --repo https://github.com/<user>/<repo>
# 5. demo / tests
python -m src.search campus "phd admission application deadline"
python -m src.demo_attack
python tests/test_core.py
```

## Notes and limitations

- Public-corpus relevance is the **category proxy** (proposal §4.3). Because sampling is stratified by
  length, P(relevant | length) is flat by design - unlike TREC, where long documents are more often relevant.
  This is central to interpreting E1/E2.
- The campus corpus is judged by two group members via `data/judgments/campus_qrels.csv` (pooled top-10 of
  Cosine / Pivoted / BM25); agreement is reported with Cohen's κ.
- Slopes in E5 are swept on the evaluation queries, so the "best slope" is optimistic.
