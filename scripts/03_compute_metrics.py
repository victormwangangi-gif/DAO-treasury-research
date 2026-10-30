"""
STEP 3: Compute the risk metrics and run the hypothesis tests.

WHAT THIS DOES (Sections 3.5 - 3.7 of your paper)
-------------------------------------------------
MAIN ANALYSIS PERIOD (1 Jul 2023 - 30 Sep 2026), per protocol:
  Sharpe ratio (annualised, 365 days, risk-free = 3-month T-bill from FRED),
  95% Expected Shortfall, maximum drawdown, volatility, total return.

EVENT STUDIES, per protocol and event (archetype from the pre-event snapshot):
  cumulative log return over the window, maximum drawdown within the window
  (from the pre-event value), operational runway at the trough, and for the
  March 2023 event the USDC share before the event.

TESTS
  PRIMARY (confirmatory, alpha = 0.05):
    H1  one-sided Mann-Whitney U: |ES| Type A > Type B+C, with Cliff's delta
  SECONDARY (Bonferroni-adjusted within each outcome family):
    Kruskal-Wallis on |ES| across A/B/C (only if every group has >= 3),
      with pairwise post hoc tests if significant
    H2  maximum drawdown (main period and each event)
    H3  operational runway at each event trough
    Event-window returns (each event), and
    H5  two-sided: Type C vs all others, USDC event return (only if C >= 3)
    Sharpe ratio (main period)
  ROBUSTNESS:
    OLS |ES| ~ TypeA + ln(peak treasury size), HC3 standard errors
    Spearman correlation: native share vs |ES|
  DIVERSIFICATION:
    Correlations between asset-class returns inside each treasury, for the
    main period and each event (baseline vs window), averaged by archetype.

INPUTS NEEDED
  data/processed/*  from 02_build_nav.py
  data/raw/operating_expenses.csv  (for runway; a blank template is created
      the first time you run this - fill in monthly_opex_usd per protocol)

OUTPUTS (output/)
  metrics_main_period.csv, archetype_means_main.csv, event_metrics.csv,
  correlations_by_protocol.csv, correlations_by_archetype.csv,
  hypothesis_tests.txt

HOW TO RUN
----------
    python -m pip install statsmodels     (once)
    python 03_compute_metrics.py
"""

import os
import math
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "output")

NAV_FILE       = os.path.join(PROCESSED_DIR, "daily_nav.csv")
ARCHETYPE_FILE = os.path.join(PROCESSED_DIR, "archetype_summary.csv")
EVENT_NAV_FILE = os.path.join(PROCESSED_DIR, "event_nav.csv")
EVENT_ARCH_FILE = os.path.join(PROCESSED_DIR, "event_archetypes.csv")
OPEX_FILE      = os.path.join(RAW_DIR, "operating_expenses.csv")
RF_FILE        = os.path.join(RAW_DIR, "DGS3MO.csv")
RF_URL         = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"

MAIN_START = pd.Timestamp("2023-07-01")
MAIN_END   = pd.Timestamp("2026-09-30")
EVENT_NAMES = {"usdc_depeg_2023": "USDC depeg, 8-20 Mar 2023",
               "aug_2024_selloff": "August 2024 sell-off, 1 Aug-10 Sep 2024"}

RF_FALLBACK = 0.045        # used only if the FRED series cannot be loaded
DAYS = 365                 # crypto trades every day
ES_LEVEL = 0.95
ALPHA = 0.05
MIN_GROUP = 2              # minimum protocols per group to run a two-group test
CLASSES = ["native", "stablecoin", "bluechip", "other"]


# ----------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------

def load_risk_free():
    """Daily 3-month T-bill yield (decimal, annual) from FRED, cached in data/raw."""
    if not os.path.exists(RF_FILE):
        try:
            pd.read_csv(RF_URL).to_csv(RF_FILE, index=False)
            print("Downloaded 3-month T-bill series from FRED.")
        except Exception:
            print(f"WARNING: could not download {RF_URL}")
            print(f"         Using a flat {RF_FALLBACK:.1%} risk-free rate instead.")
            print("         To fix: open that link in your browser and save the file as")
            print(f"         {os.path.abspath(RF_FILE)}")
            return None
    rf = pd.read_csv(RF_FILE)
    rf.columns = ["date", "rate"]
    rf["date"] = pd.to_datetime(rf["date"])
    rf["rate"] = pd.to_numeric(rf["rate"], errors="coerce") / 100
    rf = rf.set_index("date")["rate"]
    return rf.reindex(pd.date_range("2022-12-01", MAIN_END, freq="D")).ffill()


def daily_rf(dates, rf):
    if rf is None:
        return pd.Series(np.log1p(RF_FALLBACK) / DAYS, index=dates)
    return np.log1p(rf.reindex(dates).ffill().bfill()) / DAYS


def sharpe_ratio(r, rf_d):
    ex = (r - rf_d).dropna()
    if len(ex) < 2 or ex.std() == 0:
        return np.nan
    return ex.mean() / ex.std() * np.sqrt(DAYS)


def expected_shortfall(r, level=ES_LEVEL):
    """
    Historical 95% Expected Shortfall (Acerbi & Tasche, 2002): the average of
    the worst 5% of daily log returns, with fractional weighting of the
    boundary observation. Returned as a NEGATIVE number (a loss).
    """
    x = np.sort(np.asarray(r.dropna()))
    n = len(x)
    if n < 20:
        return np.nan, 0
    a = 1 - level
    k = int(math.floor(n * a))
    tail_sum = x[:k].sum() + (n * a - k) * x[k]
    return tail_sum / (n * a), k


def max_drawdown(v):
    v = v.dropna()
    if len(v) < 2:
        return np.nan
    return (v / v.cummax() - 1).min()


# ----------------------------------------------------------------------------
# STATISTICS
# ----------------------------------------------------------------------------

def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    diff = x[:, None] - y[None, :]
    return ((diff > 0).sum() - (diff < 0).sum()) / diff.size


def size_word(d):
    if np.isnan(d):
        return "n/a"
    a = abs(d)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def mwu(x, y, alternative):
    """Mann-Whitney U; exact p-value when samples are small and there are no ties."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ties = len(np.unique(np.concatenate([x, y]))) < len(x) + len(y)
    method = "exact" if (not ties and len(x) + len(y) <= 40) else "asymptotic"
    res = stats.mannwhitneyu(x, y, alternative=alternative, method=method)
    return res.statistic, res.pvalue, method


def two_group_test(df, col, group_a, group_b, alternative, label, transform=None):
    """Returns a dict describing the test, or None if a group is too small."""
    d = df[[col, "archetype"]].dropna()
    x = d.loc[d["archetype"].isin(group_a), col].values
    y = d.loc[d["archetype"].isin(group_b), col].values
    if transform is not None:
        x, y = transform(x), transform(y)
    res = {"label": label, "n1": len(x), "n2": len(y),
           "med1": np.median(x) if len(x) else np.nan,
           "med2": np.median(y) if len(y) else np.nan}
    if len(x) < MIN_GROUP or len(y) < MIN_GROUP:
        res.update(U=np.nan, p=np.nan, method="not run (group too small)", delta=np.nan)
        return res
    U, p, method = mwu(x, y, alternative)
    res.update(U=U, p=p, method=method, delta=cliffs_delta(x, y), alternative=alternative)
    return res


def fmt_test(t, lines, g1="Type A", g2="Type B+C", m=None):
    lines.append(f"  {t['label']}")
    lines.append(f"    {g1}: n = {t['n1']}, median = {t['med1']:.4f}   "
                 f"{g2}: n = {t['n2']}, median = {t['med2']:.4f}")
    if np.isnan(t["p"]):
        lines.append(f"    {t['method']} - report descriptively")
        return
    s = f"    U = {t['U']:.1f}, p = {t['p']:.4f} ({t['method']}, {t['alternative']})"
    if m:
        s += f", Bonferroni p = {min(t['p'] * m, 1):.4f} (m = {m})"
    lines.append(s)
    lines.append(f"    Cliff's delta = {t['delta']:.3f} ({size_word(t['delta'])})")


def run_family(name, tests, lines, g1="Type A", g2="Type B+C"):
    lines.append("-" * 78)
    lines.append(f"FAMILY: {name}")
    lines.append("-" * 78)
    m = sum(1 for t in tests if not np.isnan(t["p"]))
    for t in tests:
        fmt_test(t, lines, t.get("g1", g1), t.get("g2", g2), m=m if m > 0 else None)
        if not np.isnan(t["p"]):
            adj = min(t["p"] * m, 1)
            lines.append(f"    Decision (adjusted, alpha = {ALPHA}): "
                         f"{'REJECT H0' if adj < ALPHA else 'fail to reject H0'}")
        lines.append("")


# ----------------------------------------------------------------------------
# MAIN PERIOD
# ----------------------------------------------------------------------------

def main_period_metrics(nav, arch, rf):
    nav = nav[(nav["date"] >= MAIN_START - pd.Timedelta(days=1)) & (nav["date"] <= MAIN_END)]
    rows = []
    for protocol, p in nav.groupby("protocol"):
        p = p.sort_values("date").set_index("date")
        r = p["log_return"].loc[MAIN_START:]
        es, k = expected_shortfall(r)
        a = arch[arch["protocol"] == protocol].iloc[0]
        rows.append({
            "protocol": protocol, "archetype": a["archetype"],
            "native_pct": a["native_pct"], "stablecoin_pct": a["stablecoin_pct"],
            "peak_nav_usd": a["peak_nav_usd"],
            "n_returns": int(r.notna().sum()), "n_tail": k,
            "first_return": r.dropna().index.min().date() if r.notna().any() else None,
            "total_return": p["index"].iloc[-1] / p["index"].iloc[0] - 1,
            "ann_return": r.mean() * DAYS,
            "ann_volatility": r.std() * np.sqrt(DAYS),
            "sharpe_ratio": sharpe_ratio(r, daily_rf(r.index, rf)),
            "es_95": es, "abs_es_95": abs(es),
            "max_drawdown": max_drawdown(p["index"]),
            "max_drawdown_usd_value": max_drawdown(p["nav_usd"]),
        })
    return pd.DataFrame(rows).sort_values(["archetype", "protocol"])


# ----------------------------------------------------------------------------
# EVENTS
# ----------------------------------------------------------------------------

def ensure_opex_template(protocols):
    if os.path.exists(OPEX_FILE):
        return
    rows = [{"protocol": p, "event": ev, "monthly_opex_usd": "", "source": ""}
            for ev in EVENT_NAMES for p in sorted(protocols)]
    pd.DataFrame(rows).to_csv(OPEX_FILE, index=False)
    print(f"Created a blank {os.path.basename(OPEX_FILE)} in data/raw - fill in")
    print("monthly operating expenses (USD) to enable the runway test (H3).\n")


def event_metrics(enav, earch):
    opex = pd.read_csv(OPEX_FILE) if os.path.exists(OPEX_FILE) else pd.DataFrame(
        columns=["protocol", "event", "monthly_opex_usd"])
    opex["monthly_opex_usd"] = pd.to_numeric(opex["monthly_opex_usd"], errors="coerce")
    rows = []
    for (ev, protocol), g in enav.groupby(["event", "protocol"]):
        g = g.sort_values("date").set_index("date")
        pre = g[g["is_pre_event_day"].astype(bool)]
        win = g[g["phase"] == "event"]
        if pre.empty or win.empty or pd.isna(pre["nav_usd"].iloc[0]):
            continue
        v0 = pre["nav_usd"].iloc[0]
        path = pd.concat([pre["nav_usd"], win["nav_usd"]])
        trough = win["nav_usd"].idxmin()
        liquid = win.loc[trough, "value_stablecoin"] + win.loc[trough, "value_bluechip"]
        ox = opex[(opex["protocol"] == protocol) & (opex["event"] == ev)]["monthly_opex_usd"]
        ox = ox.iloc[0] if len(ox) else np.nan
        ea = earch[(earch["event"] == ev) & (earch["protocol"] == protocol)]
        base = g[g["phase"] == "baseline"]["log_return"]
        rows.append({
            "event": ev, "protocol": protocol,
            "archetype": ea["archetype"].iloc[0] if len(ea) else "?",
            "pre_event_value_usd": v0,
            "cum_log_return": np.log(win["nav_usd"].iloc[-1] / v0),
            "max_drawdown": max_drawdown(path),
            "trough_date": trough.date(),
            "liquid_reserves_at_trough_usd": liquid,
            "monthly_opex_usd": ox,
            "runway_months": liquid / ox if ox and ox > 0 else np.nan,
            "usdc_pct_pre": ea["usdc_pct"].iloc[0] if len(ea) else np.nan,
            "baseline_ann_vol": base.std() * np.sqrt(DAYS),
        })
    return pd.DataFrame(rows).sort_values(["event", "archetype", "protocol"])


# ----------------------------------------------------------------------------
# CORRELATIONS BETWEEN ASSET CLASSES
# ----------------------------------------------------------------------------

PAIRS = [("native", "bluechip"), ("native", "stablecoin"), ("bluechip", "stablecoin"),
         ("native", "other")]


def class_correlations(nav, enav, arch, earch):
    rows = []

    def add(protocol, archetype, period, df):
        for a, b in PAIRS:
            x = df[[f"r_{a}", f"r_{b}"]].dropna()
            if len(x) >= 10 and x.std().min() > 0:
                rows.append({"protocol": protocol, "archetype": archetype, "period": period,
                             "pair": f"{a}-{b}", "rho": x.corr().iloc[0, 1], "n_days": len(x)})

    main = nav[nav["date"] >= MAIN_START]
    for protocol, g in main.groupby("protocol"):
        add(protocol, arch.set_index("protocol").loc[protocol, "archetype"], "main_period", g)
    for (ev, protocol), g in enav.groupby(["event", "protocol"]):
        ea = earch[(earch["event"] == ev) & (earch["protocol"] == protocol)]
        at = ea["archetype"].iloc[0] if len(ea) else "?"
        add(protocol, at, f"{ev}_baseline", g[g["phase"] == "baseline"])
        add(protocol, at, f"{ev}_window", g[g["phase"] == "event"])
    by_p = pd.DataFrame(rows)
    by_a = (by_p.groupby(["period", "pair", "archetype"])["rho"]
                .agg(["mean", "count"]).round(3).reset_index()) if len(by_p) else by_p
    return by_p, by_a


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    for f in (NAV_FILE, ARCHETYPE_FILE):
        if not os.path.exists(f):
            print(f"ERROR: missing {f}. Run 02_build_nav.py first.")
            return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.set_option("display.width", 200)

    nav = pd.read_csv(NAV_FILE, parse_dates=["date"])
    arch = pd.read_csv(ARCHETYPE_FILE)
    enav = pd.read_csv(EVENT_NAV_FILE, parse_dates=["date"]) if os.path.exists(EVENT_NAV_FILE) else pd.DataFrame()
    earch = pd.read_csv(EVENT_ARCH_FILE) if os.path.exists(EVENT_ARCH_FILE) else pd.DataFrame()

    rf = load_risk_free()
    ensure_opex_template(arch["protocol"])

    # ---------------- metrics ----------------
    main_m = main_period_metrics(nav, arch, rf)
    main_m.to_csv(os.path.join(OUTPUT_DIR, "metrics_main_period.csv"), index=False)
    means = (main_m.groupby("archetype")
             .agg(n=("protocol", "count"), median_abs_es_95=("abs_es_95", "median"),
                  median_max_drawdown=("max_drawdown", "median"),
                  median_sharpe=("sharpe_ratio", "median"),
                  median_ann_vol=("ann_volatility", "median"),
                  median_total_return=("total_return", "median"))
             .round(4).reset_index())
    means.to_csv(os.path.join(OUTPUT_DIR, "archetype_means_main.csv"), index=False)

    ev_m = event_metrics(enav, earch) if len(enav) else pd.DataFrame()
    if len(ev_m):
        ev_m.to_csv(os.path.join(OUTPUT_DIR, "event_metrics.csv"), index=False)

    corr_p, corr_a = class_correlations(nav, enav, arch, earch) if len(enav) else (pd.DataFrame(), pd.DataFrame())
    corr_p.to_csv(os.path.join(OUTPUT_DIR, "correlations_by_protocol.csv"), index=False)
    corr_a.to_csv(os.path.join(OUTPUT_DIR, "correlations_by_archetype.csv"), index=False)

    # ---------------- tests ----------------
    L = []
    L.append("HYPOTHESIS TEST RESULTS")
    L.append(f"Main analysis period: {MAIN_START.date()} to {MAIN_END.date()}")
    L.append(f"Risk-free rate: {'FRED DGS3MO (daily)' if rf is not None else f'flat {RF_FALLBACK:.1%} (FRED not loaded)'}")
    L.append("Type B+C = diversified and stablecoin-heavy treasuries combined")
    L.append("")
    counts = main_m["archetype"].value_counts().to_dict()
    L.append(f"Main-period archetype counts: {dict(sorted(counts.items()))}")
    L.append("")

    A, BC = ["A"], ["B", "C"]

    # PRIMARY
    L.append("=" * 78)
    L.append("PRIMARY (CONFIRMATORY) TEST - H1")
    L.append("=" * 78)
    h1 = two_group_test(main_m, "abs_es_95", A, BC, "greater",
                        "H1: |ES 95%| Type A > Type B+C, main period (one-sided MWU)")
    fmt_test(h1, L)
    if not np.isnan(h1["p"]):
        n_comb = math.comb(h1["n1"] + h1["n2"], h1["n1"])
        L.append(f"    Smallest attainable p with these group sizes: 1/{n_comb} = {1 / n_comb:.4f}")
        L.append(f"    Decision at alpha = {ALPHA}: "
                 f"{'REJECT H0 - concentrated treasuries had larger tail losses' if h1['p'] < ALPHA else 'fail to reject H0'}")
    L.append("")

    # SECONDARY
    L.append("=" * 78)
    L.append("SECONDARY TESTS (Bonferroni-adjusted within each family)")
    L.append("=" * 78)

    # Kruskal-Wallis
    L.append("-" * 78)
    L.append("FAMILY: |ES 95%| across all three archetypes")
    L.append("-" * 78)
    groups = {g: main_m.loc[main_m["archetype"] == g, "abs_es_95"].dropna().values for g in "ABC"}
    sizes = {g: len(v) for g, v in groups.items()}
    if min(sizes.values()) >= 3:
        H, p = stats.kruskal(*groups.values())
        L.append(f"  Kruskal-Wallis: H = {H:.3f}, p = {p:.4f}  (group sizes {sizes})")
        if p < ALPHA:
            for g1, g2 in [("A", "B"), ("A", "C"), ("B", "C")]:
                U, pp, meth = mwu(groups[g1], groups[g2], "two-sided")
                d = cliffs_delta(groups[g1], groups[g2])
                L.append(f"    {g1} vs {g2}: U = {U:.1f}, p = {pp:.4f}, Bonferroni p = {min(3 * pp, 1):.4f}, "
                         f"delta = {d:.3f} ({size_word(d)})")
        else:
            L.append("  Not significant - no post hoc comparisons.")
    else:
        L.append(f"  Not run: needs >= 3 protocols in every archetype (have {sizes}).")
    L.append("")

    # Drawdown family (H2)
    dd = [two_group_test(main_m, "max_drawdown", A, BC, "greater",
                         "H2: |max drawdown| Type A > Type B+C, main period", np.abs)]
    for ev, nm in EVENT_NAMES.items():
        if len(ev_m):
            dd.append(two_group_test(ev_m[ev_m["event"] == ev], "max_drawdown", A, BC, "greater",
                                     f"|max drawdown| Type A > Type B+C, {nm}", np.abs))
    run_family("Maximum drawdown (H2)", dd, L)

    if len(ev_m):
        # Runway family (H3)
        rw = [two_group_test(ev_m[ev_m["event"] == ev], "runway_months", A, BC, "less",
                             f"H3: runway at trough Type A < Type B+C, {nm}")
              for ev, nm in EVENT_NAMES.items()]
        run_family("Operational runway at event trough (H3)", rw, L)
        if ev_m["runway_months"].notna().sum() == 0:
            L.append("  (No runway values yet - fill in data/raw/operating_expenses.csv)")
            L.append("")

        # Event return family (incl. H5)
        er = [two_group_test(ev_m[ev_m["event"] == ev], "cum_log_return", A, BC, "less",
                             f"Event return Type A < Type B+C, {nm}")
              for ev, nm in EVENT_NAMES.items()]
        usdc = ev_m[ev_m["event"] == "usdc_depeg_2023"]
        n_c = int((usdc["archetype"] == "C").sum())
        if n_c >= 3:
            h5 = two_group_test(usdc, "cum_log_return", ["C"], ["A", "B"], "two-sided",
                                "H5: event return Type C vs all others, USDC depeg (two-sided)")
            h5.update(g1="Type C", g2="Type A+B")
            er.append(h5)
        run_family("Event-window return (incl. H5)", er, L)
        if n_c < 3:
            L.append(f"  H5 not run: only {n_c} Type C protocol(s) before the USDC event "
                     f"(needs >= 3). Report descriptively:")
            for _, r in usdc.sort_values("cum_log_return").iterrows():
                L.append(f"    {r['protocol']:<18} Type {r['archetype']}  return {r['cum_log_return']:+.4f}  "
                         f"USDC share {100 * r['usdc_pct_pre']:.1f}%")
            L.append("")

    run_family("Sharpe ratio", [two_group_test(main_m, "sharpe_ratio", A, BC, "less",
                                               "Sharpe Type A < Type B+C, main period")], L)

    # ROBUSTNESS
    L.append("=" * 78)
    L.append("ROBUSTNESS")
    L.append("=" * 78)
    rob = main_m.dropna(subset=["abs_es_95", "peak_nav_usd"]).copy()
    rob = rob[rob["peak_nav_usd"] > 0]
    try:
        import statsmodels.api as sm
        X = pd.DataFrame({"const": 1.0,
                          "type_A": (rob["archetype"] == "A").astype(float),
                          "ln_peak_size": np.log(rob["peak_nav_usd"])})
        ols = sm.OLS(rob["abs_es_95"].values, X).fit(cov_type="HC3")
        L.append(f"  OLS: |ES 95%| = b0 + b1*TypeA + b2*ln(peak size)   (HC3 errors, n = {len(rob)})")
        for name in X.columns:
            L.append(f"    {name:<13} coef = {ols.params[name]:+.5f}   se = {ols.bse[name]:.5f}   "
                     f"z = {ols.tvalues[name]:+.2f}   p = {ols.pvalues[name]:.4f}")
        L.append(f"    R-squared = {ols.rsquared:.3f}")
    except ImportError:
        L.append("  OLS not run: statsmodels is not installed.")
        L.append("  Install it with:  python -m pip install statsmodels   and re-run.")
    rho, p = stats.spearmanr(rob["native_pct"], rob["abs_es_95"])
    L.append(f"  Spearman rho (native share vs |ES 95%|) = {rho:.3f}, p = {p:.4f}, n = {len(rob)}")
    L.append("")

    report = "\n".join(L)
    with open(os.path.join(OUTPUT_DIR, "hypothesis_tests.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)

    # ---------------- screen output ----------------
    print("=" * 78)
    print("MAIN PERIOD METRICS PER PROTOCOL")
    print("=" * 78)
    cols = ["protocol", "archetype", "n_returns", "n_tail", "total_return", "ann_volatility",
            "sharpe_ratio", "es_95", "max_drawdown"]
    print(main_m[cols].round(4).to_string(index=False))
    print("\nMedians by archetype:")
    print(means.to_string(index=False))
    if len(ev_m):
        print("\n" + "=" * 78)
        print("EVENT METRICS")
        print("=" * 78)
        print(ev_m[["event", "protocol", "archetype", "cum_log_return", "max_drawdown",
                    "runway_months", "usdc_pct_pre"]].round(4).to_string(index=False))
    if len(corr_a):
        print("\n" + "=" * 78)
        print("MEAN NATIVE-BLUECHIP CORRELATION BY ARCHETYPE (diversification check)")
        print("=" * 78)
        nb = corr_a[corr_a["pair"] == "native-bluechip"]
        print(nb.pivot(index="period", columns="archetype", values="mean").to_string())
    print("\n" + report)
    print(f"All outputs saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
