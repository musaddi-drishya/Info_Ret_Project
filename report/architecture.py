"""Draw the system architecture diagram -> results/figures/architecture.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "architecture.png"

DONE, PART, NEW = "#d8ecd8", "#fff1c9", "#dbe7f7"


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=6.6, color="#333")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1, color="#444"))


def main():
    fig, ax = plt.subplots(figsize=(9.2, 3.3))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    W, H = 1.55, 0.85
    top, bot = 2.45, 0.55
    # top row: main pipeline
    box(ax, 0.05, top, W, H, "Corpora", "20NG · Reuters (1k each)\ncampus ~100 docs", DONE)
    box(ax, 2.05, top, W, H, "Preprocessing", "tokenize · stopwords\nPorter stemmer (own)", DONE)
    box(ax, 4.05, top, W, H, "SQLite store", "docs · metadata · length\nqueries · qrels", DONE)
    box(ax, 6.05, top, W, H, "Inverted index", "dict postings → CSR\nNumPy · .npz cache", DONE)
    box(ax, 8.05, top, 2.0, H, "Weighting + norm", "TF-IDF · cosine · pivoted(s)\npivoted-unique* · BM25*", DONE)
    # bottom row
    box(ax, 0.05, bot, W, H, "Campus crawler", "Requests + BS4\n1 req/s", DONE)
    box(ax, 2.05, bot, W, H, "Relevance judging", "pooled top-10 · 2 judges\nCohen's κ", PART)
    box(ax, 4.05, bot, W, H, "Adversarial gen.", "verbosity ×2/5/10\nk-topic camouflage", DONE)
    box(ax, 6.05, bot, W, H, "Rank + evaluate", "P/R/F1@k · MAP · NDCG\nlength-bias · time/mem", DONE)
    box(ax, 8.05, bot, 2.0, H, "Analysis / report", "slope sweep · failure modes\nOCR-noise* · demo", PART)
    for x in (1.6, 3.6, 5.6, 7.6):
        arrow(ax, (x, top + H / 2), (x + 0.45, top + H / 2))
    arrow(ax, (0.8, bot + H), (0.8, top))
    arrow(ax, (2.8, top), (2.8, bot + H))
    arrow(ax, (3.6, bot + H / 2), (4.05, top + 0.1))
    arrow(ax, (4.8, top), (4.8, bot + H))
    arrow(ax, (5.6, bot + H / 2), (6.05, top + 0.1))
    arrow(ax, (9.05, top), (7.6, bot + H))
    arrow(ax, (7.6, bot + H / 2), (8.05, bot + H / 2))
    for i, (c, lbl) in enumerate(((DONE, "complete"), (PART, "in progress"), )):
        ax.add_patch(FancyBboxPatch((0.1 + i * 1.5, 3.42), 0.22, 0.13, boxstyle="round,pad=0.01", fc=c, ec="#444"))
        ax.text(0.4 + i * 1.5, 3.485, lbl, fontsize=7, va="center")
    ax.text(3.2, 3.485, "* stretch objectives", fontsize=7, va="center")
    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)


if __name__ == "__main__":
    main()
