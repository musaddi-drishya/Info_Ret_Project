"""Build the intermediate status report PDF from results/tables/*.csv.

Usage:  python report/build_report.py --repo https://github.com/<user>/<repo>
Output: report/6_CorrectingLengthBias.pdf
"""
import argparse
import json
from html import escape
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
F = ROOT / "results" / "figures"
OUT = ROOT / "report" / "6_CorrectingLengthBias.pdf"

MEMBERS = ["Divyansh Jain", "Drishya Musaddi", "Aditya Goyal", "Himabh Gemini"]
CONTRIBUTIONS = [
    ("Divyansh Jain", "Inverted index (dict postings → CSR NumPy, .npz cache); scoring models: TF-IDF cosine, "
                      "pivoted cosine, pivoted-unique, BM25; experiment runner and search CLI", "Complete"),
    ("Drishya Musaddi", "Corpus pipeline: 20NG / Reuters parsing, header/quote stripping, stratified sampling; "
                        "SQLite schema; campus crawler; judging pool", "Complete; campus judging in progress"),
    ("Aditya Goyal", "Preprocessing (tokenizer, stopwords, from-scratch Porter stemmer); evaluation metrics; "
                     "retrieval-vs-relevance length-bias analysis; figures", "Complete"),
    ("Himabh Gemini", "Adversarial generator (verbosity padding, multi-topic camouflage, verbose control); "
                      "attack experiments E3, E4 and the stretch comparisons E6/E7; live attack demo; this report",
     "Complete; failure analysis ongoing"),
]

ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=10.6, spaceAfter=3)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.4, leading=9)
CELL = ParagraphStyle("c", parent=BODY, fontSize=7.2, leading=8.6, spaceAfter=0)
H1 = ParagraphStyle("h1", parent=ss["Heading2"], fontSize=10.5, leading=12, spaceBefore=6, spaceAfter=3,
                    textColor=colors.HexColor("#1f3b63"))
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=13.5, leading=16, spaceAfter=2)
SUB = ParagraphStyle("st", parent=BODY, alignment=TA_CENTER, fontSize=8.6)
HEAD = ParagraphStyle("hd", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
CAP = ParagraphStyle("cap", parent=SMALL, alignment=TA_CENTER, textColor=colors.HexColor("#444"))


def table(rows, widths, header_bg="#1f3b63", font=7.2, zebra=True):
    wrap = lambda c, st: Paragraph(str(c), st) if isinstance(c, str) and len(c) > 18 else c
    data = [[wrap(c, HEAD) for c in rows[0]]] + [[wrap(c, CELL) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", font), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if zebra:
        style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f1f4f8")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def f3(x):
    return f"{x:.3f}"


def load(name, corpus):
    return pd.read_csv(T / f"{corpus}_{name}.csv")


def campus_status():
    p = ROOT / "data" / "raw" / "campus" / "campus_articles.json"
    q = ROOT / "data" / "judgments" / "campus_qrels.csv"
    n = len(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else 0
    judged = 0
    pairs = 0
    if q.exists():
        j = pd.read_csv(q)
        pairs, judged = len(j), int(j[["judge1", "judge2"]].notna().all(axis=1).sum())
    return n, pairs, judged


def build(repo):
    story = []
    story += [Paragraph("Correcting Length Bias: Replicating and Adversarially Stress-Testing "
                        "Pivoted Document Length Normalization", TITLE),
              Paragraph("<b>Intermediate Status Update</b> · Information Retrieval · Group 6 · 25 October 2026", SUB),
              Paragraph(" · ".join(MEMBERS), SUB),
              Paragraph(f'GitHub: <link href="{escape(repo)}" color="blue"><u>{escape(repo)}</u></link>', SUB), Spacer(1, 4)]

    # ------------------------------------------------------------ 1. status
    n_campus, pairs, judged = campus_status()
    story.append(Paragraph("1. Implementation Status &amp; Architecture", H1))
    story.append(Paragraph(
        "The full from-scratch retrieval stack (Python + NumPy + Pandas + SQLite; no scikit-learn, Lucene or "
        "Elasticsearch in the core) is implemented and all five core experiments (E1–E5) plus two stretch "
        "experiments (E6 pivoted-unique, E7 BM25) run end-to-end on both public corpora in under 15 s. "
        "The architecture is unchanged from the proposal except for two additions: a TREC-style pooled "
        "judging module for the campus corpus and BM25/pivoted-unique scorers sharing the same index.", BODY))
    status = [["Component", "Status", "Notes"],
              ["Preprocessing (tokenizer, stopwords, Porter stemmer)", "Done", "Stemmer written from scratch; spot-checked against reference outputs"],
              ["SQLite storage + inverted index + .npz cache", "Done", "Dict postings frozen into CSR NumPy arrays"],
              ["TF-IDF + cosine (E1), pivoted cosine (E2)", "Done", "ltc weighting; pivot = mean cosine norm"],
              ["20NG / Reuters samples, 55 queries", "Done", "1,000 docs each, stratified by category × length quintile"],
              ["Metrics, length-bias analysis, time/memory", "Done", "P/R/F1@5,10,20, MAP, NDCG@10, TVD, log-length gap"],
              ["Verbosity (E3) and multi-topic (E4) attacks", "Done", "Fixed-length synthetic docs; relevance by construction"],
              ["Slope sweep (E5), clean and attacked", "Done", "s = 0.0 … 1.0, step 0.1"],
              ["Stretch: pivoted-unique (E6), BM25 + b-sweep (E7)", "Done", "Ahead of proposal schedule (planned week 6)"],
              ["Custom campus corpus", "Collected" if n_campus else "Pending",
               f"{n_campus} BITS Pilani news/notice pages crawled; "
               + (f"{judged}/{pairs} pooled pairs double-judged" if pairs else "queries + judging pool being prepared")],
              ["Failure-mode write-up, OCR-noise stretch, demo", "Pending", "See Section 4"]]
    story.append(table(status, [6.3 * cm, 1.6 * cm, 9.6 * cm]))
    story.append(Spacer(1, 4))
    story.append(Image(str(F / "architecture.png"), width=17.5 * cm, height=17.5 * cm * 3.3 / 9.2))
    story.append(Paragraph("Figure 1. System architecture (green = complete, yellow = in progress).", CAP))

    # ---------------------------------------------------------- 2. pipeline
    ng, rt = load("stats", "newsgroups").set_index("Unnamed: 0")["value"], load("stats", "reuters").set_index("Unnamed: 0")["value"]
    story.append(Paragraph("2. IR Pipeline Details", H1))
    story.append(Paragraph(
        "<b>Data.</b> 20 Newsgroups: headers removed (they leak the label) except <i>Subject</i>; quoted replies "
        "and signatures stripped. Reuters-21578: single-topic documents from 20 topics. Each corpus is sampled to "
        "1,000 documents, stratified by category and by length quintile within category; documents with fewer "
        "than 20 index terms are dropped. 30 (20NG) and 25 (Reuters) short topic queries were hand-written; "
        "relevance = category membership (proxy, as declared in the proposal). The campus corpus is crawled with "
        "Requests + BeautifulSoup at 1 request/s, de-duplicated by slug and content hash; because the portal "
        "posts far fewer than 100 items a week, the 100 pages span Jan 2025 – Oct 2026. 18 campus queries are "
        "drafted and a 205-pair judging pool (pooled top-10 of Cosine, Pivoted and BM25) is ready.", BODY))
    story.append(Paragraph(
        "<b>Preprocessing.</b> Lower-casing, regex tokenisation (<font face='Courier'>[a-z][a-z0-9]+</font>), "
        "~180-word stopword list, Porter stemming (own implementation, steps 1a–5b, memoised).", BODY))
    story.append(Paragraph(
        "<b>Indexing.</b> Term → {doc: tf} dictionary postings are frozen into CSR arrays "
        "(<font face='Courier'>term_ptr, doc_ids, tfs</font>) so that each query term is one vectorised slice "
        "(term-at-a-time accumulation). Document length, unique-term count and mean tf are stored per document; "
        "the index is cached as compressed .npz. "
        f"20NG: {int(ng['vocabulary']):,} terms, {int(ng['postings']):,} postings, {ng['index_MB']:.1f} MB, built in "
        f"{ng['index_build_s']:.2f} s; Reuters: {int(rt['vocabulary']):,} terms, {int(rt['postings']):,} postings, "
        f"{rt['index_MB']:.1f} MB, {rt['index_build_s']:.2f} s.", BODY))
    story.append(Paragraph(
        "<b>Retrieval models.</b> score(q,d) = Σ<sub>t</sub> w<sub>q</sub>(t)·w<sub>d</sub>(t,d) / N<sub>d</sub> with "
        "w = (1+ln tf)·ln(N/df). <b>Cosine</b>: N<sub>d</sub> = ||d||<sub>2</sub>. <b>Pivoted cosine</b>: "
        "N<sub>d</sub> = (1−s)·p + s·||d||<sub>2</sub>, p = mean ||d||<sub>2</sub>, default s = 0.7 (s = 1 recovers "
        "cosine). <b>Pivoted unique</b> (Lnu): w<sub>d</sub> = (1+ln tf)/(1+ln avg-tf<sub>d</sub>)·idf, "
        "N<sub>d</sub> = (1−s)·p + s·|unique(d)|, s = 0.2. <b>BM25</b>: k<sub>1</sub> = 1.2, b = 0.75. "
        "Slope sweeps reuse cached posting weights and only recompute N<sub>d</sub>.", BODY))
    story.append(Paragraph(
        "<b>Evaluation.</b> P@k, R@k, F1@k (k = 5, 10, 20), MAP over full rankings, NDCG@10, query latency and "
        "index memory. Length bias follows Singhal et al.: documents are split into 10 equal-frequency length "
        "bins and P(retrieved | bin) over the top-50 of every query is compared with P(relevant | bin). We "
        "summarise it as the total-variation distance (TVD) between the two distributions and the "
        "<i>log-length gap</i> = mean ln|d| of retrieved − of relevant (negative ⇒ short-document preference). "
        "<b>Attacks.</b> E3 replaces 50 random documents by ×2/×5/×10 repetitions of themselves (no new "
        "information). E4 injects 20 synthetic 300-token documents per condition built from 1, 2, 3 or 4 "
        "categories (equal share each), plus a <i>verbose</i> control (one short document repeated to 300 tokens); "
        "each is relevant to all queries of its source categories.", BODY))

    # ---------------------------------------------------------- 3. results
    story.append(Paragraph("3. Preliminary Evaluation Results", H1))
    rows = [["Corpus", "Model", "P@10", "R@10", "F1@10", "P@20", "MAP", "NDCG@10", "TVD", "LenGap", "ms/q"]]
    for corpus, label in (("newsgroups", "20NG"), ("reuters", "Reuters")):
        for r in load("E1E2_clean", corpus).itertuples():
            name = {"Cosine": "Cosine (baseline)", "Pivoted": "Pivoted s=0.7", "PivUnique": "Piv-unique*",
                    "BM25": "BM25*"}[r.model]
            rows.append([label, name, f3(r._3), f3(r._4), f3(r._5), f3(r._6), f3(r.MAP), f3(r._9), f3(r.TVD),
                         f"{r.LogLenGap:+.3f}", f"{r.ms_per_query:.2f}"])
    story.append(KeepTogether([table(rows, [1.4 * cm, 2.7 * cm] + [1.3 * cm] * 4 + [1.25 * cm, 1.5 * cm, 1.25 * cm, 1.4 * cm, 1.1 * cm]),
                               Paragraph("Table 1. Clean-collection effectiveness, length bias and latency "
                                         "(* stretch models; log-len gap > 0 = long-document preference).", CAP)]))
    story.append(Spacer(1, 3))

    sw = {c: load("E5_slope_sweep", c) for c in ("newsgroups", "reuters")}
    best = {c: sw[c].loc[sw[c]["MAP"].idxmax()] for c in sw}
    c_ng = load("E1E2_clean", "newsgroups").set_index("model")
    c_rt = load("E1E2_clean", "reuters").set_index("model")
    story.append(Paragraph(
        f"<b>RQ1 / RQ2 — length bias.</b> Our cosine baseline does <i>not</i> reproduce the short-document "
        f"preference reported on TREC: on both corpora it already retrieves long documents slightly more often "
        f"than they are relevant (log-length gap {c_ng.loc['Cosine', 'LogLenGap']:+.3f} on 20NG, "
        f"{c_rt.loc['Cosine', 'LogLenGap']:+.3f} on Reuters; Figure 2 left). Pivoting therefore moves the system "
        f"further from the relevance curve (TVD {c_ng.loc['Cosine', 'TVD']:.3f} → {c_ng.loc['Pivoted', 'TVD']:.3f} "
        f"on 20NG) while MAP is essentially flat ({c_ng.loc['Cosine', 'MAP']:.4f} → {c_ng.loc['Pivoted', 'MAP']:.4f}; "
        f"best slope s = {best['newsgroups']['slope']:.1f} / {best['reuters']['slope']:.1f}). The reason is the "
        "evaluation set-up: with category-proxy relevance and length-stratified sampling, P(relevant | length) is "
        "flat, whereas pivoting is designed for collections where it rises with length. This is the main "
        "motivation for the human-judged campus corpus (Section 4).", BODY))

    figs = Table([[Image(str(F / "newsgroups_length_bias.png"), width=8.3 * cm, height=8.3 * cm * 3.4 / 6),
                   Image(str(F / "reuters_slope_sweep.png"), width=9.4 * cm, height=9.4 * cm * 3.2 / 8)]],
                 colWidths=[8.5 * cm, 9.5 * cm])
    figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(KeepTogether([figs, Paragraph(
        "Figure 2. Left: retrieval vs relevance by length bin (20NG). Right: MAP and log-length gap vs pivot slope "
        "on Reuters, clean and under ×5 verbosity padding.", CAP)]))

    vb = load("E3_verbosity", "reuters")
    vrows = [["Model", "P@10 (1×→10×)", "R@10", "F1@10", "MAP", "padded-doc median rank", "intrusions@10"]]
    for m in ("Cosine", "Pivoted", "PivUnique", "BM25"):
        a = vb[(vb.model == m) & (vb.factor == 1)].iloc[0]
        b = vb[(vb.model == m) & (vb.factor == 10)].iloc[0]
        vrows.append([m, f"{a['P@10']:.3f} → {b['P@10']:.3f}", f"{a['R@10']:.3f} → {b['R@10']:.3f}",
                      f"{a['F1@10']:.3f} → {b['F1@10']:.3f}", f"{a['MAP']:.4f} → {b['MAP']:.4f}",
                      f"{a['target_median_rank']:.0f} → {b['target_median_rank']:.0f}",
                      f"{int(a['intrusions@10'])} → {int(b['intrusions@10'])}"])
    vs = load("E5_slope_under_verbosity", "reuters").set_index("slope")
    story.append(KeepTogether([Spacer(1, 3), table(vrows, [2.2 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.9 * cm, 2.6 * cm, 2.2 * cm]),
                               Paragraph("Table 2. Verbosity attack on Reuters, before (1×) → after (10× padding of 50 "
                                         "documents). Intrusions = padded docs entering the top-10 of queries they are "
                                         "not relevant to.", CAP)]))
    story.append(Paragraph(
        f"<b>RQ3 — verbosity.</b> Repetition adds no information, so an ideal normaliser should leave rankings "
        f"unchanged. Cosine slightly <i>demotes</i> padded documents, pivoted cosine and BM25 <i>promote</i> them, and "
        f"pivoted-unique is nearly immune: repetition leaves its denominator (unique-term count) unchanged and "
        f"dividing by 1+ln(avg-tf) largely cancels the inflated tf (exactly when tf equals the document average). Aggregate "
        f"P/R/F1/MAP move by less than 0.01 because only 5% of the collection is attacked, but the effect grows "
        f"sharply as the slope falls (Figure 2 right): at s = 0 the padded documents jump from median rank "
        f"{vs.loc[0.0, 'target_median_rank_clean']:.0f} to {vs.loc[0.0, 'target_median_rank_attacked']:.0f} and "
        f"cause {int(vs.loc[0.0, 'intrusions@10'])} top-10 intrusions, versus "
        f"{int(vs.loc[0.7, 'intrusions@10'])} at s = 0.7 and {int(vs.loc[1.0, 'intrusions@10'])} at s = 1.", BODY))

    mt = load("E4_multitopic", "reuters").set_index(["condition", "model"])
    conds = ["verbose(1 topic, repeated)", "1-topic", "2-topic", "3-topic", "4-topic"]
    mrows = [["Median rank of synthetic doc (300 tokens)", "Cosine", "Pivoted s=0.7", "Pivoted s=0.3", "Piv-unique", "BM25"]]
    for c in conds:
        mrows.append([c.replace("verbose(1 topic, repeated)", "verbose (1 doc repeated)")] +
                     [f"{mt.loc[(c, m), 'median_rank']:.0f}" for m in ("Cosine", "Pivoted", "Pivoted(s=0.3)", "PivUnique", "BM25")])
    story.append(KeepTogether([table(mrows, [5.5 * cm] + [2.4 * cm] * 5),
                               Paragraph("Table 3. Multi-topic camouflage on Reuters: median rank of injected documents "
                                         "for the queries they are relevant to (lower = better).", CAP)]))
    story.append(Paragraph(
        "<b>RQ4 — multi-topicality.</b> At equal length, every model ranks genuinely multi-topic documents far "
        "below single-topic ones (cosine: median rank "
        f"{mt.loc[('1-topic', 'Cosine'), 'median_rank']:.0f} → {mt.loc[('4-topic', 'Cosine'), 'median_rank']:.0f} "
        "from 1 to 4 topics): each topic gets only a quarter of the term mass, so length normalisation alone cannot "
        "tell <i>diluted</i> from <i>verbose</i>. Lower slopes help multi-topic documents, but they reward the "
        f"verbose control by the same mechanism (pivoted s = 0.3: verbose {mt.loc[(conds[0], 'Pivoted(s=0.3)'), 'median_rank']:.0f} "
        f"vs 2-topic {mt.loc[('2-topic', 'Pivoted(s=0.3)'), 'median_rank']:.0f}). "
        "<b>RQ5 — slope.</b> MAP is flat for s ∈ [0.6, 1.0] and drops steeply below 0.5, which is "
        "exactly where verbosity-padded documents begin to intrude: the slope that is safe on clean data is "
        "also the one that limits the attack.", BODY))

    # ---------------------------------------------------------- 4. plan
    story.append(Paragraph("4. Path to Final Submission (remaining ≈ 20–25%)", H1))
    plan = [["Work item", "Owner", "Target"],
            ["Campus corpus: finalise 15–20 queries, double-judge pooled pairs, Cohen's κ, run E1–E5 on it",
             "Drishya, Aditya", "1–3 Nov"],
            ["Held-out slope tuning (split queries) to remove the optimistic bias of E5; paired randomisation "
             "significance tests for Table 1", "Divyansh", "3–5 Nov"],
            ["Failure-mode analysis: per-query case studies of intrusions and multi-topic misses; pivoted-unique "
             "under multi-topic camouflage", "Himabh", "3–7 Nov"],
            ["Stretch: OCR-like noise (character substitutions) × length normalisation", "Himabh, Aditya", "7–10 Nov"],
            ["Final report, slides, live CLI search demo comparing models side by side", "All", "10–14 Nov"]]
    story.append(table(plan, [11.2 * cm, 3.4 * cm, 2.9 * cm]))
    story.append(Paragraph(
        "<b>Stretch objectives.</b> Pivoted-unique normalisation (E6) and BM25 with a b-sweep (E7) are already "
        "implemented and included above, ahead of the planned week-6 slot. OCR-noise robustness has not started.",
        BODY))

    # --------------------------------------------------- 5. contributions
    story.append(Paragraph("5. Contribution Matrix", H1))
    story.append(table([["Member", "Components owned and completed to date", "Status"]] +
                       [list(r) for r in CONTRIBUTIONS], [3.1 * cm, 11.2 * cm, 3.2 * cm]))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.3 * cm, title="Group 6 — Intermediate Status Update",
                            author=", ".join(MEMBERS))
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"wrote {OUT}")


def _footer(canvas, doc):
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666"))
    canvas.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, f"Group 6 · Intermediate status update · page {doc.page}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/<username>/<repository>")
    build(ap.parse_args().repo)
