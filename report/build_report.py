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
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
F = ROOT / "results" / "figures"
OUT = ROOT / "report" / "6_CorrectingLengthBias.pdf"

MEMBERS = ["Divyansh Jain", "Drishya Musaddi", "Aditya Goyal", "Himabh Gemini"]
CONTRIBUTIONS = [
    ("Divyansh Jain", "Inverted index and .npz caching; the four scoring models (TF-IDF cosine, pivoted cosine, "
                      "pivoted unique, BM25); experiment runner and search tool", "Done"),
    ("Drishya Musaddi", "Loading and cleaning 20 Newsgroups and Reuters, stratified sampling, queries; SQLite "
                        "storage; campus crawler and the judging pool", "Done, campus judging in progress"),
    ("Aditya Goyal", "Preprocessing and our own Porter stemmer; evaluation metrics; length-bias analysis; "
                     "plots and tests", "Done"),
    ("Himabh Gemini", "Adversarial document generation (verbosity padding, multi-topic documents); attack "
                      "experiments; attack demo; this report", "Done, failure analysis ongoing"),
]

SERIF, SERIF_B, SERIF_I = "Times-Roman", "Times-Bold", "Times-Italic"
BODY = ParagraphStyle("body", fontName=SERIF, fontSize=10, leading=11.8, alignment=TA_JUSTIFY, spaceAfter=3)
EQ = ParagraphStyle("eq", parent=BODY, alignment=TA_CENTER, spaceBefore=1, spaceAfter=3)
CELL = ParagraphStyle("cell", fontName=SERIF, fontSize=8.6, leading=10)
HEAD = ParagraphStyle("head", parent=CELL, fontName=SERIF_B)
H1 = ParagraphStyle("h1", fontName=SERIF_B, fontSize=11.5, leading=13.5, spaceBefore=5, spaceAfter=2)
H2 = ParagraphStyle("h2", fontName=SERIF_B, fontSize=10, leading=12, spaceBefore=3, spaceAfter=1)
TITLE = ParagraphStyle("title", fontName=SERIF_B, fontSize=15, leading=18.5, alignment=TA_CENTER, spaceAfter=5)
SUB = ParagraphStyle("sub", fontName=SERIF, fontSize=10, leading=12.5, alignment=TA_CENTER)
CAP = ParagraphStyle("cap", fontName=SERIF_I, fontSize=8.8, leading=10.5, alignment=TA_CENTER, spaceBefore=1,
                     spaceAfter=3)
REF = ParagraphStyle("ref", parent=BODY, fontSize=8.8, leading=10.6, leftIndent=12, firstLineIndent=-12,
                     alignment=0, spaceAfter=1)


def table(rows, widths, align_right_from=None):
    """Plain report-style table: thin black rules, light grey header."""
    wrap = lambda c, st: Paragraph(str(c), st) if isinstance(c, str) and len(c) > 16 else c
    data = [[wrap(c, HEAD) for c in rows[0]]] + [[wrap(c, CELL) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("FONT", (0, 0), (-1, -1), SERIF, 8.6), ("FONT", (0, 0), (-1, 0), SERIF_B, 8.6),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if align_right_from is not None:
        style.append(("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"))
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
    story += [Paragraph("Correcting Length Bias: Replicating and Stress-Testing<br/>"
                        "Pivoted Document Length Normalization", TITLE),
              Paragraph("Information Retrieval, Intermediate Status Update", SUB),
              Paragraph("Group 6: " + ", ".join(MEMBERS), SUB),
              Paragraph("25 October 2026", SUB),
              Paragraph(f'Code: <link href="{escape(repo)}" color="blue"><u>{escape(repo)}</u></link>', SUB),
              Spacer(1, 6)]

    # ------------------------------------------------------------ 1. status
    n_campus, pairs, judged = campus_status()
    story.append(Paragraph("1. Implementation Status and Architecture", H1))
    story.append(Paragraph(
        "We have built the complete retrieval system from scratch in Python, using only NumPy, Pandas and "
        "SQLite (no scikit-learn, Lucene or Elasticsearch for the core algorithm). All five planned experiments "
        "(E1 to E5) are running on both public datasets, and we also finished the two stretch experiments "
        "(pivoted unique normalization and BM25) earlier than planned. A full run of every experiment takes "
        "about 15 seconds. The architecture is the same as in our proposal, with two additions: a judging "
        "module for the campus dataset, and the two extra scoring functions, which reuse the same index.", BODY))
    status = [["Component", "Status", "Notes"],
              ["Preprocessing: tokenizer, stopwords, Porter stemmer", "Done", "Stemmer written ourselves and checked against known outputs"],
              ["SQLite storage, inverted index, .npz cache", "Done", "Postings stored as NumPy arrays for fast scoring"],
              ["TF-IDF with cosine (E1) and pivoted cosine (E2)", "Done", "Pivot set to the average cosine norm"],
              ["20 Newsgroups and Reuters samples, 55 queries", "Done", "1,000 documents each, balanced by topic and length"],
              ["Evaluation metrics, length-bias analysis, timing", "Done", "P, R and F1 at 5, 10 and 20; MAP; NDCG@10"],
              ["Verbosity (E3) and multi-topic (E4) attacks", "Done", "Synthetic documents with known relevance"],
              ["Slope sweep (E5), with and without attack", "Done", "Slope from 0 to 1 in steps of 0.1"],
              ["Stretch: pivoted unique (E6), BM25 (E7)", "Done", "Planned for week 6, finished early"],
              ["Campus dataset", "Collected" if n_campus else "Pending",
               f"{n_campus} BITS Pilani news and notice pages; "
               + (f"{judged} of {pairs} pairs judged so far" if pairs and judged else
                  f"{pairs} pairs ready for judging" if pairs else "judging not started")],
              ["Failure analysis, OCR-noise stretch, demo", "Pending", "See Section 4"]]
    story.append(table(status, [6.4 * cm, 1.7 * cm, 8.9 * cm]))
    story.append(Spacer(1, 5))
    story.append(Image(str(F / "architecture.png"), width=15.0 * cm, height=15.0 * cm * 3.3 / 9.2))
    story.append(Paragraph("Figure 1: System architecture. Green parts are complete, yellow parts are in progress.", CAP))

    # ---------------------------------------------------------- 2. pipeline
    ng, rt = load("stats", "newsgroups").set_index("Unnamed: 0")["value"], load("stats", "reuters").set_index("Unnamed: 0")["value"]
    story.append(Paragraph("2. IR Pipeline", H1))
    story.append(Paragraph("2.1 Data", H2))
    story.append(Paragraph(
        "For 20 Newsgroups we removed the message headers, since they contain the newsgroup name and would give "
        "away the answer, but kept the subject line. We also removed quoted replies and signatures. From "
        "Reuters-21578 we used only documents with a single topic, from 20 topics. We sampled 1,000 documents "
        "from each dataset so that every category and every length range is represented, and dropped very short "
        "documents (under 20 terms). We wrote 30 queries for 20 Newsgroups and 25 for Reuters; as stated in our "
        "proposal, a document counts as relevant if it belongs to the query's category.", BODY))
    story.append(Paragraph(
        "The campus dataset was collected with a small crawler (Requests and BeautifulSoup, one request per "
        "second) that removes duplicate notices. The campus portal does not post 100 items in a single week, so "
        "our 100 pages cover January 2025 to October 2026. We have written 18 queries for it and prepared 205 "
        "query-document pairs for two of us to judge.", BODY))
    story.append(Paragraph("2.2 Preprocessing and indexing", H2))
    story.append(Paragraph(
        "Text is lower-cased and split into words, stopwords are removed, and words are stemmed with our own "
        "implementation of the Porter stemmer. The inverted index is first built as a dictionary of posting "
        "lists and then converted to NumPy arrays, so that scoring a query term is a single array operation. "
        "For each document we also store its length, number of unique terms and average term frequency. "
        f"The 20 Newsgroups index has {int(ng['vocabulary']):,} terms and {int(ng['postings']):,} postings, "
        f"takes {ng['index_MB']:.1f} MB and is built in {ng['index_build_s']:.1f} s. The Reuters index has "
        f"{int(rt['vocabulary']):,} terms, takes {rt['index_MB']:.1f} MB and is built in {rt['index_build_s']:.1f} s.",
        BODY))
    story.append(Paragraph("2.3 Retrieval models", H2))
    story.append(Paragraph(
        "All models score a document as the sum, over query terms, of the query weight times the document "
        "weight, divided by a normalization factor N(d). Term weights use w = (1 + ln tf) &#215; ln(N/df). The "
        "models differ only in N(d):", BODY))
    story.append(Paragraph(
        "Cosine: N(d) = ||d|| &nbsp;&nbsp;&nbsp;&nbsp; Pivoted cosine: N(d) = (1 &#8722; s) &#215; p + s &#215; ||d||",
        EQ))
    story.append(Paragraph(
        "Here p, the pivot, is the average value of ||d|| over the collection, and s is the slope (we use "
        "s = 0.7 by default; s = 1 gives plain cosine). For pivoted unique normalization we follow Singhal et al. "
        "and use the number of unique terms in the document instead of ||d||, with s = 0.2. For BM25 we use "
        "k1 = 1.2 and b = 0.75.", BODY))
    story.append(Paragraph("2.4 Evaluation and attacks", H2))
    story.append(Paragraph(
        "We report P@k, R@k and F1@k for k = 5, 10 and 20, MAP, NDCG@10, query time and index size. To measure "
        "length bias we use the method from the original paper: documents are sorted into 10 bins by length, and "
        "for each bin we compare how often it is retrieved (top 50 per query) with how often it is relevant. We "
        "summarize the difference with two numbers: the total variation distance (TVD) between the two "
        "distributions, and the log-length gap, which is positive when retrieved documents are longer than "
        "relevant ones. For the verbosity attack (E3) we replace 50 random documents with copies of themselves "
        "repeated 2, 5 or 10 times. For the multi-topic attack (E4) we add 20 synthetic documents of 300 terms "
        "each, made from 1, 2, 3 or 4 topics, plus a control made by repeating one short document.", BODY))

    # ---------------------------------------------------------- 3. results
    story.append(Paragraph("3. Preliminary Results", H1))
    rows = [["Dataset", "Model", "P@10", "R@10", "F1@10", "P@20", "MAP", "NDCG@10", "TVD", "Len. gap", "ms/query"]]
    for corpus, label in (("newsgroups", "20NG"), ("reuters", "Reuters")):
        for r in load("E1E2_clean", corpus).itertuples():
            name = {"Cosine": "Cosine (baseline)", "Pivoted": "Pivoted, s = 0.7", "PivUnique": "Pivoted unique",
                    "BM25": "BM25"}[r.model]
            rows.append([label, name, f3(r._3), f3(r._4), f3(r._5), f3(r._6), f3(r.MAP), f3(r._9), f3(r.TVD),
                         f"{r.LogLenGap:+.3f}", f"{r.ms_per_query:.2f}"])
    story.append(KeepTogether([table(rows, [1.35 * cm, 2.65 * cm] + [1.15 * cm] * 4 + [1.15 * cm, 1.65 * cm, 1.15 * cm,
                                                                                   1.45 * cm, 1.55 * cm],
                                     align_right_from=2),
                               Paragraph("Table 1: Results on the original (unattacked) datasets. Pivoted unique "
                                         "and BM25 are our stretch models.", CAP)]))

    sw = {c: load("E5_slope_sweep", c) for c in ("newsgroups", "reuters")}
    best = {c: sw[c].loc[sw[c]["MAP"].idxmax()] for c in sw}
    c_ng = load("E1E2_clean", "newsgroups").set_index("model")
    c_rt = load("E1E2_clean", "reuters").set_index("model")
    story.append(Paragraph("3.1 Length bias (RQ1 and RQ2)", H2))
    story.append(Paragraph(
        "We expected cosine normalization to favor short documents, as the original paper found on TREC. "
        "This is not what we see. On both datasets cosine already retrieves long documents slightly more often "
        f"than they are relevant (log-length gap of {c_ng.loc['Cosine', 'LogLenGap']:+.3f} on 20 Newsgroups and "
        f"{c_rt.loc['Cosine', 'LogLenGap']:+.3f} on Reuters, Figure 2 left). Pivoting pushes retrieval further "
        f"towards long documents (TVD goes from {c_ng.loc['Cosine', 'TVD']:.3f} to {c_ng.loc['Pivoted', 'TVD']:.3f} "
        f"on 20 Newsgroups), while MAP hardly changes ({c_ng.loc['Cosine', 'MAP']:.4f} against "
        f"{c_ng.loc['Pivoted', 'MAP']:.4f}). The best slopes were {best['newsgroups']['slope']:.1f} and "
        f"{best['reuters']['slope']:.1f}. We think the reason is our setup: because we sampled evenly across "
        "lengths and use categories as relevance, long documents are not more likely to be relevant here, "
        "whereas pivoting assumes they are. This is one of the main reasons we want results on the "
        "human-judged campus dataset.", BODY))

    figs = Table([[Image(str(F / "newsgroups_length_bias.png"), width=7.4 * cm, height=7.4 * cm * 3.4 / 6),
                   Image(str(F / "reuters_slope_sweep.png"), width=8.8 * cm, height=8.8 * cm * 3.2 / 8)]],
                 colWidths=[7.8 * cm, 9.2 * cm])
    figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(KeepTogether([figs, Paragraph(
        "Figure 2: Left, retrieval against relevance by length bin on 20 Newsgroups. Right, MAP and log-length "
        "gap against the slope on Reuters, with and without 5x verbosity padding.", CAP)]))

    vb = load("E3_verbosity", "reuters")
    vrows = [["Model", "P@10", "R@10", "F1@10", "MAP", "Median rank of padded docs", "Wrong docs in top 10"]]
    for m, label in (("Cosine", "Cosine"), ("Pivoted", "Pivoted, s = 0.7"), ("PivUnique", "Pivoted unique"),
                     ("BM25", "BM25")):
        a = vb[(vb.model == m) & (vb.factor == 1)].iloc[0]
        b = vb[(vb.model == m) & (vb.factor == 10)].iloc[0]
        vrows.append([label, f"{a['P@10']:.3f} / {b['P@10']:.3f}", f"{a['R@10']:.3f} / {b['R@10']:.3f}",
                      f"{a['F1@10']:.3f} / {b['F1@10']:.3f}", f"{a['MAP']:.4f} / {b['MAP']:.4f}",
                      f"{a['target_median_rank']:.0f} / {b['target_median_rank']:.0f}",
                      f"{int(a['intrusions@10'])} / {int(b['intrusions@10'])}"])
    vs = load("E5_slope_under_verbosity", "reuters").set_index("slope")
    story.append(Paragraph("3.2 Verbosity attack (RQ3)", H2))
    story.append(KeepTogether([table(vrows, [2.8 * cm, 2.15 * cm, 2.15 * cm, 2.15 * cm, 2.6 * cm, 2.6 * cm, 2.55 * cm]),
                               Paragraph("Table 2: Verbosity attack on Reuters, before / after padding 50 documents "
                                         "to 10 times their length. The last column counts padded documents that "
                                         "appear in the top 10 of queries they are not relevant to.", CAP)]))
    story.append(Paragraph(
        "Repeating a document adds no new information, so ideally its rank should not change. Cosine slightly "
        "lowers padded documents, while pivoted cosine and BM25 raise them. Pivoted unique normalization barely "
        "reacts, because repeating text does not change the number of unique terms. Precision, recall and MAP "
        "change by less than 0.01, since only 5% of the documents were attacked, but the effect gets much "
        f"stronger as the slope decreases (Figure 2 right). At s = 0 the padded documents move from median rank "
        f"{vs.loc[0.0, 'target_median_rank_clean']:.0f} to {vs.loc[0.0, 'target_median_rank_attacked']:.0f} and "
        f"appear {int(vs.loc[0.0, 'intrusions@10'])} times in the top 10 of unrelated queries, compared with "
        f"{int(vs.loc[0.7, 'intrusions@10'])} at s = 0.7 and {int(vs.loc[1.0, 'intrusions@10'])} at s = 1.", BODY))

    mt = load("E4_multitopic", "reuters").set_index(["condition", "model"])
    conds = [("verbose(1 topic, repeated)", "One short document repeated"), ("1-topic", "1 topic"),
             ("2-topic", "2 topics"), ("3-topic", "3 topics"), ("4-topic", "4 topics")]
    mrows = [["Injected document (300 terms)", "Cosine", "Pivoted, s = 0.7", "Pivoted, s = 0.3", "Pivoted unique", "BM25"]]
    for c, label in conds:
        mrows.append([label] + [f"{mt.loc[(c, m), 'median_rank']:.0f}"
                                for m in ("Cosine", "Pivoted", "Pivoted(s=0.3)", "PivUnique", "BM25")])
    story.append(Paragraph("3.3 Multi-topic documents (RQ4) and slope (RQ5)", H2))
    story.append(KeepTogether([table(mrows, [5.0 * cm] + [2.4 * cm] * 5, align_right_from=1),
                               Paragraph("Table 3: Median rank of the injected documents on Reuters, for the "
                                         "queries they are relevant to (lower is better).", CAP)]))
    story.append(Paragraph(
        "When all documents have the same length, every model ranks documents covering several topics much "
        f"lower than single-topic ones (with cosine, the median rank goes from "
        f"{mt.loc[('1-topic', 'Cosine'), 'median_rank']:.0f} for one topic to "
        f"{mt.loc[('4-topic', 'Cosine'), 'median_rank']:.0f} for four). Each topic only gets a quarter of the "
        "document, and length normalization cannot tell this apart from padding. A lower slope helps the "
        "multi-topic documents, but it helps the repeated control document in the same way (at s = 0.3 it "
        f"reaches rank {mt.loc[(conds[0][0], 'Pivoted(s=0.3)'), 'median_rank']:.0f}, against "
        f"{mt.loc[('2-topic', 'Pivoted(s=0.3)'), 'median_rank']:.0f} for two-topic documents). For the slope "
        "itself, MAP is almost flat between 0.6 and 1.0 and drops quickly below 0.5, which is also where the "
        "padding attack starts to work. So the slope that works best on clean data is also the one that "
        "limits the attack.", BODY))

    # ---------------------------------------------------------- 4. plan
    story.append(Paragraph("4. Plan for the Final Submission", H1))
    story.append(Paragraph(
        "We estimate that about 20 to 25% of the work is left. The table below shows what remains and who is "
        "responsible.", BODY))
    plan = [["Task", "Owner", "Target date"],
            ["Campus dataset: finish the queries, judge the 205 pairs independently, compute Cohen's kappa, "
             "and run E1 to E5 on it", "Drishya, Aditya", "1 to 3 Nov"],
            ["Choose the slope on separate queries (so the result is not tuned on the test set) and add "
             "significance tests for Table 1", "Divyansh", "3 to 5 Nov"],
            ["Failure analysis: look at individual queries where attacks succeed; test pivoted unique on "
             "multi-topic documents", "Himabh", "3 to 7 Nov"],
            ["Stretch: OCR-like noise and its effect on length normalization", "Himabh, Aditya", "7 to 10 Nov"],
            ["Final report, slides and demo", "All", "10 to 14 Nov"]]
    story.append(table(plan, [11.0 * cm, 3.3 * cm, 2.7 * cm]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Stretch objectives.</b> Pivoted unique normalization and BM25 (including a sweep over its b "
        "parameter) are finished and included in the results above. The OCR-noise experiment has not been "
        "started yet.", BODY))

    # --------------------------------------------------- 5. contributions
    story.append(Paragraph("5. Contribution Matrix", H1))
    story.append(table([["Member", "Components owned and completed so far", "Status"]] +
                       [list(r) for r in CONTRIBUTIONS], [3.1 * cm, 10.6 * cm, 3.3 * cm]))

    # ------------------------------------------------------- references
    story.append(Paragraph("References", H1))
    for ref in (
            "[1] A. Singhal, C. Buckley and M. Mitra. Pivoted Document Length Normalization. "
            "In Proceedings of SIGIR, 1996.",
            "[2] S.-H. Na, I.-S. Kang and J.-H. Lee. Improving Term Frequency Normalization for Multi-topical "
            "Documents and Application to Language Modeling Approaches. arXiv:1502.02277.",
            "[3] S. E. Robertson and S. Walker. Some Simple Effective Approximations to the 2-Poisson Model for "
            "Probabilistic Weighted Retrieval. In Proceedings of SIGIR, 1994."):
        story.append(Paragraph(ref, REF))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm, title="Group 6 Intermediate Status Update",
                            author=", ".join(MEMBERS))
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"wrote {OUT}")


def _footer(canvas, doc):
    canvas.setFont(SERIF, 9)
    canvas.drawCentredString(A4[0] / 2, 1.0 * cm, str(doc.page))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/<username>/<repository>")
    build(ap.parse_args().repo)
