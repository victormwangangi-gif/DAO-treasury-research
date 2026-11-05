"""
STEP 4: Robustness checks and the figures for Chapter 4.

WHAT THIS DOES
--------------
ROBUSTNESS (does H1 survive reasonable changes to the design?)
  1. Threshold sensitivity: re-classifies every protocol with the Type A line
     at 65 / 70 / 75% native and the Type C line at 35 / 40 / 45% stablecoin
     (9 combinations) and re-runs the primary H1 test each time.
  2. Leave-one-out: drops each protocol in turn and re-runs H1, so no single
     treasury drives the result.
  3. Without Olympus: H1 and Kruskal-Wallis with the Olympus outlier removed.
  4. Other tail levels: H1 with Expected Shortfall at 90%, 97.5% and 99%.
  5. Sub-periods: H1 separately for the first and second half of the period.

FIGURES (saved as 300-dpi PNG files, ready for Word)
  Figure 4.1  Average treasury composition by protocol
  Figure 4.2  Treasury value index by archetype (median, log scale)
  Figure 4.3  Tail loss |ES 95%| by protocol
  Figure 4.4  Native-token share vs tail loss
  Figure 4.5  August 2024 sell-off: value path by archetype
  Figure 4.6  Native-bluechip correlation, before vs during each event

INPUTS   data/processed/* (from 02) and output/* (from 03)
OUTPUTS  output/robustness.txt, output/robustness_*.csv, output/figures/*.png

HOW TO RUN
----------
    python 04_robustness_and_figures.py
"""

import os
import math
import importlib.util
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(SCRIPT_DIR, "..", "data", "processed")
OUTPUT = os.path.join(SCRIPT_DIR, "..", "output")
FIG_DIR = os.path.join(OUTPUT, "figures")

# reuse the exact metric and test functions from step 3
_spec = importlib.util.spec_from_file_location("m3", os.path.join(SCRIPT_DIR, "03_compute_metrics.py"))
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)

MAIN_START, MAIN_END = m3.MAIN_START, m3.MAIN_END
EVENTS = {"usdc_depeg_2023": ("USDC depeg", pd.Timestamp("2023-03-08"), pd.Timestamp("2023-03-20")),
          "aug_2024_selloff": ("August 2024 sell-off", pd.Timestamp("2024-08-01"), pd.Timestamp("2024-09-10"))}

# Colours: validated categorical palette (dataviz reference slots 2, 1, 3).
# Type A = orange (native concentration), B = blue, C = aqua (stablecoins).
# Asset classes reuse the matching hue so the figures read as one system.
COL = {"A": "#eb6834", "B": "#2a78d6", "C": "#1baf7a"}
MARK = {"A": "o", "B": "s", "C": "^"}          # shape = second channel (print / colour-blind)
CLASS_COL = {"native": "#eb6834", "bluechip": "#2a78d6", "stablecoin": "#1baf7a", "other": "#a3a29b"}
LABEL = {"A": "Type A: native-concentrated", "B": "Type B: partially diversified", "C": "Type C: stablecoin-heavy"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300,
})


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def classify(native, stable, a_thr=0.70, c_thr=0.40):
    return "A" if native >= a_thr else ("C" if stable >= c_thr else "B")


def h1(df, col="abs_es"):
    """One-sided MWU: Type A > Type B+C. Returns n_A, n_BC, p, delta."""
    a = df.loc[df["archetype"] == "A", col].dropna().values
    bc = df.loc[df["archetype"] != "A", col].dropna().values
    if len(a) < 2 or len(bc) < 2:
        return len(a), len(bc), np.nan, np.nan
    _, p, _ = m3.mwu(a, bc, "greater")
    return len(a), len(bc), p, m3.cliffs_delta(a, bc)


def kw(df, col="abs_es"):
    g = [df.loc[df["archetype"] == k, col].dropna().values for k in "ABC"]
    if min(len(x) for x in g) < 3:
        return np.nan
    return stats.kruskal(*g).pvalue


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


def legend_handles(keys):
    return [plt.Line2D([], [], color=COL[k], marker=MARK[k], linestyle="-", lw=2, ms=6, label=LABEL[k])
            for k in keys]


# ----------------------------------------------------------------------------
# robustness
# ----------------------------------------------------------------------------

def robustness(nav, arch, metrics):
    lines = ["ROBUSTNESS OF THE PRIMARY TEST (H1: |ES| Type A > Type B+C, one-sided MWU)", ""]
    base = metrics[["protocol", "abs_es_95"]].rename(columns={"abs_es_95": "abs_es"}).merge(
        arch[["protocol", "native_pct", "stablecoin_pct", "archetype"]], on="protocol")

    # 1. thresholds
    rows = []
    for a_thr in (0.65, 0.70, 0.75):
        for c_thr in (0.35, 0.40, 0.45):
            d = base.copy()
            d["archetype"] = [classify(n, s, a_thr, c_thr) for n, s in zip(d["native_pct"], d["stablecoin_pct"])]
            nA, nBC, p, dl = h1(d)
            cnt = d["archetype"].value_counts()
            rows.append({"type_A_line": f"{a_thr:.0%}", "type_C_line": f"{c_thr:.0%}",
                         "n_A": cnt.get("A", 0), "n_B": cnt.get("B", 0), "n_C": cnt.get("C", 0),
                         "p_H1": p, "cliffs_delta": dl, "p_kruskal": kw(d),
                         "type_A_members": ", ".join(sorted(d.loc[d["archetype"] == "A", "protocol"]))})
    thr = pd.DataFrame(rows)
    thr.to_csv(os.path.join(OUTPUT, "robustness_thresholds.csv"), index=False)
    lines.append("1. THRESHOLD SENSITIVITY")
    lines.append(thr.drop(columns="type_A_members").to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    for _, r in thr.drop_duplicates("type_A_line").iterrows():
        lines.append(f"   Type A at {r['type_A_line']}: {r['type_A_members']}")
    lines.append(f"   H1 significant (p < 0.05) in {int((thr['p_H1'] < 0.05).sum())} of {len(thr)} combinations")
    lines.append("")

    # 2. leave-one-out
    loo = []
    for p_ in base["protocol"]:
        nA, nBC, p, dl = h1(base[base["protocol"] != p_])
        loo.append({"dropped": p_, "n_A": nA, "n_BC": nBC, "p_H1": p, "cliffs_delta": dl})
    loo = pd.DataFrame(loo)
    loo.to_csv(os.path.join(OUTPUT, "robustness_leave_one_out.csv"), index=False)
    lines.append("2. LEAVE-ONE-OUT")
    lines.append(loo.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append(f"   largest p = {loo['p_H1'].max():.4f} (dropping {loo.loc[loo['p_H1'].idxmax(), 'dropped']})")
    lines.append("")

    # 3. without Olympus
    d = base[base["protocol"] != "Olympus DAO"]
    nA, nBC, p, dl = h1(d)
    lines.append("3. WITHOUT OLYMPUS DAO")
    lines.append(f"   H1: n = {nA} vs {nBC}, p = {p:.4f}, delta = {dl:.3f};  Kruskal-Wallis p = "
                 + ("not run (a group < 3)" if np.isnan(kw(d)) else f"{kw(d):.4f}"))
    lines.append("")

    # 4 & 5. other tail levels and sub-periods
    r = nav[(nav["date"] >= MAIN_START) & (nav["date"] <= MAIN_END)]
    mid = MAIN_START + (MAIN_END - MAIN_START) / 2
    alt = []
    for level in (0.90, 0.95, 0.975, 0.99):
        es = r.groupby("protocol")["log_return"].apply(lambda s: abs(m3.expected_shortfall(s, level)[0]))
        d = base[["protocol", "archetype"]].merge(es.rename("abs_es"), on="protocol")
        nA, nBC, p, dl = h1(d)
        k = int(math.floor(r.groupby("protocol")["log_return"].count().median() * (1 - level)))
        alt.append({"test": f"ES {level:.1%}, full period", "tail_obs": k, "p_H1": p, "cliffs_delta": dl})
    for name, sub in (("first half", r[r["date"] < mid]), ("second half", r[r["date"] >= mid])):
        es = sub.groupby("protocol")["log_return"].apply(lambda s: abs(m3.expected_shortfall(s, 0.95)[0]))
        d = base[["protocol", "archetype"]].merge(es.rename("abs_es"), on="protocol")
        nA, nBC, p, dl = h1(d)
        k = int(math.floor(sub.groupby("protocol")["log_return"].count().median() * 0.05))
        alt.append({"test": f"ES 95%, {name} ({sub['date'].min().date()} to {sub['date'].max().date()})",
                    "tail_obs": k, "p_H1": p, "cliffs_delta": dl})
    alt = pd.DataFrame(alt)
    alt.to_csv(os.path.join(OUTPUT, "robustness_tail_and_subperiods.csv"), index=False)
    lines.append("4-5. OTHER TAIL LEVELS AND SUB-PERIODS")
    lines.append(alt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def fig_composition(arch):
    d = arch.copy()
    d["order"] = d["archetype"].map({"A": 0, "B": 1, "C": 2})
    d = d.sort_values(["order", "native_pct"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    left = np.zeros(len(d))
    y = np.arange(len(d))
    for c, lab in (("native", "Native token"), ("bluechip", "ETH / BTC (incl. staked)"),
                   ("stablecoin", "Stablecoins"), ("other", "Other crypto")):
        v = d[f"{c}_pct"].values * 100
        ax.barh(y, v, left=left, color=CLASS_COL[c], height=0.7, label=lab,
                edgecolor="white", linewidth=1)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}  ({a})" for p, a in zip(d["protocol"], d["archetype"])])
    ax.axvline(70, color=INK, lw=0.8, ls="--")
    ax.text(70.5, len(d) - 0.4, "70% Type A line", color=MUTED, fontsize=7.5, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of treasury value (%), mean of 14 quarterly snapshots")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.13), fontsize=7.5)
    save(fig, "fig4_1_composition.png")


def fig_index(nav, arch):
    d = nav[(nav["date"] >= MAIN_START - pd.Timedelta(days=1)) & (nav["date"] <= MAIN_END)]
    d = d.merge(arch[["protocol", "archetype"]], on="protocol")
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for _, g in d.groupby("protocol"):
        ax.plot(g["date"], g["index"], color=COL[g["archetype"].iloc[0]], lw=0.6, alpha=0.25)
    for k in "ABC":
        med = d[d["archetype"] == k].groupby("date")["index"].median()
        ax.plot(med.index, med.values, color=COL[k], lw=2, label=LABEL[k] + " (median)")
    name, s, e = EVENTS["aug_2024_selloff"]
    ax.axvspan(s, e, color=GRID, alpha=0.8, lw=0)
    ax.set_yscale("log")
    ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5) if d["index"].min() * 0.8 <= t <= d["index"].max() * 1.25]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_ylabel("Value index (30 Jun 2023 = 1, log scale)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=7.5, loc="lower left")
    ax.annotate("Aug 2024\nsell-off", xy=(s, 1), xytext=(s, 1.6), fontsize=7, color=MUTED, ha="center")
    save(fig, "fig4_2_value_index.png")


def fig_es(metrics):
    d = metrics.sort_values("abs_es_95")
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    y = np.arange(len(d))
    ax.barh(y, d["abs_es_95"] * 100, color=[COL[a] for a in d["archetype"]], height=0.6)
    for yi, v in zip(y, d["abs_es_95"] * 100):
        ax.text(v + 0.15, yi, f"{v:.1f}%", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(d["protocol"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("|ES 95%|: average loss on the worst 5% of days (%)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[k], label=LABEL[k]) for k in "ABC"],
              fontsize=7.5, loc="lower right")
    save(fig, "fig4_3_tail_loss.png")


def fig_scatter(metrics):
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    for k in "ABC":
        g = metrics[metrics["archetype"] == k]
        ax.scatter(g["native_pct"] * 100, g["abs_es_95"] * 100, color=COL[k], marker=MARK[k], s=46,
                   edgecolor="white", linewidth=1.5, label=LABEL[k], zorder=3)
    for _, r in metrics.iterrows():
        ax.annotate(r["protocol"], (r["native_pct"] * 100, r["abs_es_95"] * 100), xytext=(4, 3),
                    textcoords="offset points", fontsize=7, color=MUTED)
    rho, p = stats.spearmanr(metrics["native_pct"], metrics["abs_es_95"])
    ax.axvline(70, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("Native-token share of treasury (%), period mean")
    ax.set_ylabel("|ES 95%| (%)")
    ax.text(0.02, 0.96, f"Spearman ρ = {rho:.2f} (p = {p:.4f})", transform=ax.transAxes, fontsize=8,
            va="top", color=INK)
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "fig4_4_concentration_vs_es.png")


def fig_event_path(enav, earch, event="aug_2024_selloff"):
    name, s, e = EVENTS[event]
    d = enav[(enav["event"] == event) & ((enav["phase"] == "event") | enav["is_pre_event_day"].astype(bool))]
    d = d.merge(earch[earch["event"] == event][["protocol", "archetype"]], on="protocol")
    v0 = d[d["is_pre_event_day"].astype(bool)].set_index("protocol")["nav_usd"]
    d = d.assign(rel=d["nav_usd"] / d["protocol"].map(v0) - 1)
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    for _, g in d.groupby("protocol"):
        ax.plot(g["date"], g["rel"] * 100, color=COL[g["archetype"].iloc[0]], lw=0.6, alpha=0.25)
    for k in "ABC":
        med = d[d["archetype"] == k].groupby("date")["rel"].median()
        if len(med):
            n = d.loc[d["archetype"] == k, "protocol"].nunique()
            ax.plot(med.index, med.values * 100, color=COL[k], lw=2, marker=MARK[k], ms=3,
                    markevery=7, label=f"{LABEL[k]} (median, n = {n})")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Change from 31 Jul 2024 value (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7.5, loc="lower left")
    save(fig, "fig4_5_aug2024_path.png")


def fig_correlation(corr):
    d = corr[corr["pair"] == "native-bluechip"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    span = max(d["mean"].max() - d["mean"].min(), 0.2)

    def spread(vals, gap):
        """Nudge label positions apart so close values don't overprint."""
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        pos = list(vals)
        for a, b in zip(order, order[1:]):
            if pos[b] - pos[a] < gap:
                pos[b] = pos[a] + gap
        return pos

    for ax, (ev, (name, _, _)) in zip(axes, EVENTS.items()):
        pts = []
        for k in "ABC":
            b = d[(d["period"] == f"{ev}_baseline") & (d["archetype"] == k)]["mean"]
            w = d[(d["period"] == f"{ev}_window") & (d["archetype"] == k)]["mean"]
            if len(b) and len(w):
                ax.plot([0, 1], [b.iloc[0], w.iloc[0]], color=COL[k], marker=MARK[k], lw=2, ms=6, label=LABEL[k])
                pts.append((b.iloc[0], w.iloc[0]))
        if pts:
            for x, ha, vals in ((-0.06, "right", [p[0] for p in pts]), (1.06, "left", [p[1] for p in pts])):
                for v, yv in zip(vals, spread(vals, 0.06 * span)):
                    ax.text(x, yv, f"{v:.2f}", va="center", ha=ha, fontsize=7.5, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre-event baseline", "Event window"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(name, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean correlation, native vs ETH/BTC")
    axes[0].legend(fontsize=7, loc="lower center", bbox_to_anchor=(1.05, -0.42), ncol=3)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig4_6_correlation.png")


# ----------------------------------------------------------------------------

def main():
    need = [os.path.join(PROCESSED, "daily_nav.csv"), os.path.join(PROCESSED, "archetype_summary.csv"),
            os.path.join(OUTPUT, "metrics_main_period.csv")]
    for f in need:
        if not os.path.exists(f):
            print(f"ERROR: missing {f}. Run 02 and 03 first.")
            return
    nav = pd.read_csv(need[0], parse_dates=["date"])
    arch = pd.read_csv(need[1])
    metrics = pd.read_csv(need[2])
    enav = pd.read_csv(os.path.join(PROCESSED, "event_nav.csv"), parse_dates=["date"])
    earch = pd.read_csv(os.path.join(PROCESSED, "event_archetypes.csv"))
    corr = pd.read_csv(os.path.join(OUTPUT, "correlations_by_archetype.csv"))

    report = robustness(nav, arch, metrics)
    with open(os.path.join(OUTPUT, "robustness.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)

    print("Drawing figures...")
    fig_composition(arch)
    fig_index(nav, arch)
    fig_es(metrics)
    fig_scatter(metrics)
    fig_event_path(enav, earch)
    fig_correlation(corr)
    print(f"\nFigures saved in {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
