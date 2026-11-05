"""
STEP 6: Validation statistics and the Chapter 5 figures.

WHAT THIS DOES
--------------
Reads the outputs of 03_compute_metrics.py and 05_framework_calibration.py and
produces the remaining Chapter 5 results:

  Table 13  Aave and ENS tier shares against the framework bands, and the
            distance from the bands (percentage points outside the bands,
            summed across the three tiers)
  Table 14  Spearman correlation, within each DAO, between distance from the
            bands at each snapshot and outcomes over the following quarter
            (runway: same quarter), with Bonferroni-adjusted p-values
  Figure 8  Simulated tail loss by native-token share and stablecoin share,
            against the observed archetype medians
  Figure 9  Aave and ENS tier allocations at each snapshot, against the bands

INPUTS   output/framework_grid.csv, output/validation_quarters.csv,
         output/metrics_main_period.csv
OUTPUTS  output/validation_summary.txt, output/validation_distance.csv,
         output/figures/fig5_1_frontier.png, output/figures/fig5_2_validation.png

HOW TO RUN
----------
    python 06_validation_and_figures.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(SCRIPT_DIR, "..", "output")
FIGURES = os.path.join(OUTPUT, "figures")

BANDS = {"tier1_stable": (30, 50), "tier2_crypto": (30, 50), "tier3_native": (10, 30)}
OUTCOMES = [  # column, label, sign expected if the framework helps
    ("next_q_abs_es95", "|ES 95%|", "Positive"),
    ("next_q_abs_es90", "|ES 90%|", "Positive"),
    ("next_q_vol", "Volatility", "Positive"),
    ("next_q_mdd", "Maximum drawdown", "Positive"),
    ("next_q_sharpe", "Sharpe ratio", "Negative"),
    ("runway_months", "Runway (same quarter)", "Negative"),
]

ORANGE, BLUE, GREEN, GREY = "#eb6834", "#2a78d6", "#1baf7a", "#666666"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": "#222222", "xtick.color": GREY,
    "ytick.color": GREY, "axes.grid": True, "grid.color": "#e6e6e6",
    "legend.frameon": False, "savefig.dpi": 300,
})


def distance(row):
    """Percentage points by which the three tier shares lie outside their bands."""
    d = 0.0
    for col, (lo, hi) in BANDS.items():
        x = row[col]
        d += max(0.0, lo - x, x - hi)
    return d


def table_13(val):
    lines = ["TABLE 13 - TIER SHARES AGAINST THE FRAMEWORK BANDS", ""]
    for p, g in val.groupby("protocol"):
        lines.append(p)
        for col, (lo, hi) in BANDS.items():
            lines.append(f"  {col:14s} mean {g[col].mean():5.1f}  range {g[col].min():5.1f}-{g[col].max():5.1f}"
                         f"  (band {lo}-{hi})")
        inside = (g["distance_pp"] == 0).sum()
        lines.append(f"  snapshots within all bands: {inside} of {len(g)}")
        lines.append(f"  distance from bands (pp): mean {g['distance_pp'].mean():.1f}, "
                     f"range {g['distance_pp'].min():.1f}-{g['distance_pp'].max():.1f}")
        lines.append("")
    return lines


def table_14(val):
    lines = ["TABLE 14 - SPEARMAN CORRELATION: DISTANCE FROM BANDS vs NEXT-QUARTER OUTCOMES",
             f"{'Outcome':24s}{'DAO':6s}{'n':>4s}{'rho':>8s}{'p':>8s}{'adj p':>8s}  if framework helps", ""]
    k = len(OUTCOMES)
    for p, g in val.groupby("protocol"):
        for col, label, sign in OUTCOMES:
            if col not in g or g[col].notna().sum() < 5:
                lines.append(f"{label:24s}{p:6s}{'-':>4s}{'-':>8s}{'-':>8s}{'-':>8s}  {sign}")
                continue
            sub = g[["distance_pp", col]].dropna()
            rho, pval = stats.spearmanr(sub["distance_pp"], sub[col])
            lines.append(f"{label:24s}{p:6s}{len(sub):4d}{rho:8.2f}{pval:8.3f}{min(1.0, pval * k):8.3f}  {sign}")
        lines.append("")
    lines.append(f"adj p = Bonferroni across the {k} outcomes within each DAO.")
    return lines


def figure_8(grid, observed):
    es = {t: observed.loc[observed["archetype"] == t, "abs_es_95"].median() * 100 for t in "ABC"}
    budget = es["A"] / 2          # budget rule: half the tail loss of the median concentrated treasury
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axvspan(10, 30, color=ORANGE, alpha=0.07, lw=0)
    ax.text(20, 10.5, "Tier 3 band\n10–30%", ha="center", va="center", color="#b5481f", fontsize=9)
    tiers = [10, 20, 30, 40, 50, 60]
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(tiers)))
    for t1, c in zip(tiers, shades):
        s = grid[(grid["tier1_stable"] == t1) & (grid["tier3_native"] <= 50)].sort_values("tier3_native")
        if s.empty:
            continue
        ax.plot(s["tier3_native"], s["abs_es_95"] * 100, color=c, lw=2.2, marker="o", ms=4)
        ax.annotate(f"Tier 1 = {t1}%", (s["tier3_native"].iloc[-1], s["abs_es_95"].iloc[-1] * 100),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color="#222222")
    for t, ls in (("A", "--"), ("B", "--"), ("C", ":")):
        ax.axhline(es[t], color=GREY, ls=ls, lw=1.1)
        ax.text(0.5, es[t] + 0.15, f"Type {t} observed median ({es[t]:.1f}%)", color=GREY, fontsize=8.5)
    ax.axhline(budget, color="#111111", lw=1.4)
    ax.text(60.5, budget - 0.15, f"Tail-risk budget:\nhalf the Type A\nmedian ({budget:.1f}%)", va="top", fontsize=8.5, color="#111111")
    ax.set_xlim(-1, 72)
    ax.set_ylim(2, 12.6)
    ax.set_xlabel("Tier 3 (native token) share of the simulated treasury (%)")
    ax.set_ylabel("|ES 95%| (%), median across native tokens")
    path = os.path.join(FIGURES, "fig5_1_frontier.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_9(val):
    prot = sorted(val["protocol"].unique())
    fig, axes = plt.subplots(1, len(prot), figsize=(8, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    series = [("tier3_native", "Tier 3: native token", ORANGE, "o"),
              ("tier1_stable", "Tier 1: stablecoins", GREEN, "^"),
              ("tier2_crypto", "Tier 2: ETH/BTC and other crypto", BLUE, "s")]
    for ax, p in zip(axes, prot):
        g = val[val["protocol"] == p].sort_values("date")
        x = np.arange(len(g))
        ax.axhspan(30, 50, color=GREEN, alpha=0.10, lw=0)
        ax.axhspan(10, 30, color=ORANGE, alpha=0.08, lw=0)
        for col, label, colour, mk in series:
            ax.plot(x, g[col], color=colour, marker=mk, ms=4, lw=2, label=label)
        labels = [s.replace("-20", "\n") for s in g["snapshot"]]
        ax.set_xticks(x[::2]); ax.set_xticklabels(labels[::2], fontsize=8)
        ax.set_ylim(0, 100); ax.set_title(p, fontsize=10)
    axes[0].set_ylabel("Share of treasury (%)")
    axes[-1].text(len(g) - 1, 48, "Tier 1 and Tier 2 bands (30–50%)", ha="right", va="top", fontsize=8, color="#127a55")
    axes[-1].text(len(g) - 1, 28, "Tier 3 band (10–30%)", ha="right", va="top", fontsize=8, color="#b5481f")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    path = os.path.join(FIGURES, "fig5_2_validation.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    os.makedirs(FIGURES, exist_ok=True)
    grid = pd.read_csv(os.path.join(OUTPUT, "framework_grid.csv"))
    val = pd.read_csv(os.path.join(OUTPUT, "validation_quarters.csv"), parse_dates=["date"])
    observed = pd.read_csv(os.path.join(OUTPUT, "metrics_main_period.csv"))

    val["distance_pp"] = val.apply(distance, axis=1).round(1)
    val.to_csv(os.path.join(OUTPUT, "validation_distance.csv"), index=False)

    lines = table_13(val) + table_14(val)
    lines += ["", "Figures written to:", "  " + figure_8(grid, observed), "  " + figure_9(val)]
    report = "\n".join(lines)
    with open(os.path.join(OUTPUT, "validation_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
