"""Run experiments E1-E7 and write tables (results/tables) and figures (results/figures).

Usage:  python -m src.experiments [--corpora newsgroups reuters]
"""
import argparse
import json
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adversarial import inject, intrusions, multitopic_docs, target_ranks, verbose_docs, verbosity_attack
from .evaluate import evaluate_run, length_bias, summarize
from .models import BM25, CosineTFIDF, PivotedCosine, PivotedUnique
from .pipeline import collection_stats, load_cached_index, load_collection
from .storage import ROOT

TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
SLOPES = np.round(np.arange(0.0, 1.01, 0.1), 2)
HEADLINE = ["P@5", "P@10", "R@10", "F1@10", "P@20", "R@20", "MAP", "NDCG@10"]


def make_models(idx, slope=0.7):
    return {"Cosine": CosineTFIDF(idx), "Pivoted": PivotedCosine(idx, slope=slope),
            "PivUnique": PivotedUnique(idx, slope=0.2), "BM25": BM25(idx)}


def run_clean(coll, idx):
    """E1, E2 (+E6/E7 baselines on clean data): effectiveness, cost and length bias."""
    rows, bias_tables, rankings = [], {}, {}
    for name, model in make_models(idx).items():
        pq, ranks, ms = evaluate_run(model, coll.queries, coll.relevant)
        bt, bias = length_bias(ranks, coll.relevant, coll.doc_len)
        bias_tables[name], rankings[name] = bt, ranks
        rows.append({"model": name, **summarize(pq)[HEADLINE].to_dict(), **bias, "ms_per_query": ms})
    return pd.DataFrame(rows), bias_tables


def run_slope_sweep(coll, idx):
    """E5 (clean): MAP and length bias as a function of the pivot slope."""
    base = PivotedCosine(idx, slope=1.0)
    rows = []
    for s in SLOPES:
        m = base.with_slope(s)
        pq, ranks, _ = evaluate_run(m, coll.queries, coll.relevant)
        _, bias = length_bias(ranks, coll.relevant, coll.doc_len)
        rows.append({"slope": s, **summarize(pq)[["MAP", "P@10", "NDCG@10"]].to_dict(), **bias})
    return pd.DataFrame(rows)


def run_verbosity(coll):
    """E3: replace 50 docs by m-times repeated copies; measure effectiveness, target ranks, intrusions."""
    rows = []
    for m, attacked, targets in verbosity_attack(coll):
        idx = attacked.build_index()
        for name, model in make_models(idx).items():
            pq, _, _ = evaluate_run(model, attacked.queries, attacked.relevant)
            r = target_ranks(model, attacked, targets)
            rows.append({"factor": m, "model": name, **summarize(pq)[["P@10", "R@10", "F1@10", "MAP"]].to_dict(),
                         "target_median_rank": float(np.median(r)), "target_in_top10": float((r <= 10).mean()),
                         "intrusions@10": intrusions(model, attacked, targets)})
    return pd.DataFrame(rows)


def run_verbosity_slope(coll, factor=5):
    """E5 (attacked): slope sweep under the verbosity attack."""
    clean_idx = coll.build_index()
    _, attacked, targets = next((x for x in verbosity_attack(coll, factors=(factor,))))
    idx = attacked.build_index()
    base_clean, base_att = PivotedCosine(clean_idx, 1.0), PivotedCosine(idx, 1.0)
    rows = []
    for s in SLOPES:
        mc, ma = base_clean.with_slope(s), base_att.with_slope(s)
        r_clean, r_att = target_ranks(mc, coll, targets), target_ranks(ma, attacked, targets)
        map_clean = summarize(evaluate_run(mc, coll.queries, coll.relevant)[0])["MAP"]
        map_att = summarize(evaluate_run(ma, attacked.queries, attacked.relevant)[0])["MAP"]
        rows.append({"slope": s, "MAP_clean": map_clean, "MAP_attacked": map_att,
                     "target_median_rank_clean": float(np.median(r_clean)),
                     "target_median_rank_attacked": float(np.median(r_att)),
                     "intrusions@10": intrusions(ma, attacked, targets)})
    return pd.DataFrame(rows)


def run_multitopic(coll, target_len=300):
    """E4: inject equal-length docs with 1..4 topics (and a verbose control); where do they rank?"""
    conditions = {"verbose(1 topic, repeated)": verbose_docs(coll, target_len)}
    for k in (1, 2, 3, 4):
        conditions[f"{k}-topic"] = multitopic_docs(coll, k, target_len)
    rows = []
    for cond, synth in conditions.items():
        attacked, new = inject(coll, synth, "adv")
        idx = attacked.build_index()
        models = make_models(idx)
        models.update({f"Pivoted(s={s})": PivotedCosine(idx, slope=s) for s in (0.3, 0.5, 0.9)})
        for name, model in models.items():
            r = target_ranks(model, attacked, new)
            pq, _, _ = evaluate_run(model, attacked.queries, attacked.relevant)
            rows.append({"condition": cond, "model": name, "median_rank": float(np.median(r)),
                         "MRR": float(np.mean(1.0 / r)), "in_top10": float((r <= 10).mean()),
                         "MAP": summarize(pq)["MAP"]})
    return pd.DataFrame(rows)


def run_bm25_b_sweep(coll):
    """E7 extra: BM25 length parameter b under the verbosity attack (m=5)."""
    _, attacked, targets = next(verbosity_attack(coll, factors=(5,)))
    idx_c, idx_a = coll.build_index(), attacked.build_index()
    rows = []
    for b in SLOPES:
        mc, ma = BM25(idx_c, b=b), BM25(idx_a, b=b)
        rows.append({"b": b, "MAP_clean": summarize(evaluate_run(mc, coll.queries, coll.relevant)[0])["MAP"],
                     "MAP_attacked": summarize(evaluate_run(ma, attacked.queries, attacked.relevant)[0])["MAP"],
                     "target_median_rank_clean": float(np.median(target_ranks(mc, coll, targets))),
                     "target_median_rank_attacked": float(np.median(target_ranks(ma, attacked, targets)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- plots
def plot_length_bias(bias_tables, name):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    rel = bias_tables["Cosine"]
    x = np.arange(len(rel))
    ax.plot(x, rel["P_relevant"], "k-o", lw=2, label="P(relevant)")
    for m, style in (("Cosine", "C0--s"), ("Pivoted", "C1-^"), ("BM25", "C2:d")):
        ax.plot(x, bias_tables[m]["P_retrieved"], style, label=f"P(retrieved) {m}")
    ax.set_xticks(x, [str(v) for v in rel["median_len"]], fontsize=8)
    ax.set_xlabel("document length bin (median tokens, short → long)")
    ax.set_ylabel("probability")
    ax.set_title(f"Retrieval vs relevance by length — {name}", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}_length_bias.png", dpi=160)
    plt.close(fig)


def plot_slope(sweep, vslope, name):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].plot(sweep["slope"], sweep["MAP"], "C1-o", label="MAP")
    ax2 = ax[0].twinx()
    ax2.plot(sweep["slope"], sweep["LogLenGap"], "C0--s", label="LogLenGap")
    ax2.axhline(0, color="grey", lw=0.8)
    ax[0].set_xlabel("slope s (1.0 = cosine)")
    ax[0].set_ylabel("MAP", color="C1")
    ax2.set_ylabel("log-length gap (ret − rel)", color="C0")
    ax[0].set_title("Clean collection", fontsize=10)
    ax[1].plot(vslope["slope"], vslope["MAP_clean"], "k-o", label="clean")
    ax[1].plot(vslope["slope"], vslope["MAP_attacked"], "C3-^", label="5× verbosity")
    ax[1].set_xlabel("slope s")
    ax[1].set_ylabel("MAP")
    ax[1].set_title("Under verbosity attack", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Slope sensitivity — {name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}_slope_sweep.png", dpi=160)
    plt.close(fig)


def plot_verbosity(verb, name):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m in ("Cosine", "Pivoted", "PivUnique", "BM25"):
        d = verb[verb["model"] == m]
        ax.plot(d["factor"], d["target_median_rank"], "-o", label=m)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10], ["1×", "2×", "5×", "10×"])
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("padding factor")
    ax.set_ylabel("median rank of padded doc (log)")
    ax.set_title(f"Verbosity attack — {name}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}_verbosity.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------- main
def run_corpus(name):
    t0 = time.perf_counter()
    coll = load_collection(name)
    prep_s = time.perf_counter() - t0
    tracemalloc.start()
    idx = coll.build_index()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    load_cached_index(coll)
    stats = collection_stats(coll, idx)
    stats["preprocess_s"], stats["index_build_s"] = prep_s, idx.build_seconds
    stats["index_MB"], stats["build_peak_MB"] = idx.memory_bytes() / 2**20, peak / 2**20

    clean, bias_tables = run_clean(coll, idx)
    sweep = run_slope_sweep(coll, idx)
    verb = run_verbosity(coll)
    vslope = run_verbosity_slope(coll)
    multi = run_multitopic(coll)
    bsweep = run_bm25_b_sweep(coll)

    stats.to_frame("value").to_csv(TABLES / f"{name}_stats.csv")
    clean.to_csv(TABLES / f"{name}_E1E2_clean.csv", index=False)
    pd.concat({m: t for m, t in bias_tables.items()}).to_csv(TABLES / f"{name}_length_bias_bins.csv")
    sweep.to_csv(TABLES / f"{name}_E5_slope_sweep.csv", index=False)
    verb.to_csv(TABLES / f"{name}_E3_verbosity.csv", index=False)
    vslope.to_csv(TABLES / f"{name}_E5_slope_under_verbosity.csv", index=False)
    multi.to_csv(TABLES / f"{name}_E4_multitopic.csv", index=False)
    bsweep.to_csv(TABLES / f"{name}_E7_bm25_b_sweep.csv", index=False)
    plot_length_bias(bias_tables, name)
    plot_slope(sweep, vslope, name)
    plot_verbosity(verb, name)
    return {"stats": stats.to_dict(), "clean": clean.to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["newsgroups", "reuters"])
    args = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 4)
    summary = {}
    for name in args.corpora:
        t0 = time.perf_counter()
        summary[name] = run_corpus(name)
        print(f"\n=== {name} ({time.perf_counter() - t0:.0f}s) ===")
        for f in ("stats", "E1E2_clean", "E5_slope_sweep", "E3_verbosity", "E5_slope_under_verbosity",
                  "E4_multitopic", "E7_bm25_b_sweep"):
            print(f"\n-- {f}\n{pd.read_csv(TABLES / f'{name}_{f}.csv').to_string(index=False)}")
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
