"""
STEP 7: Additional analyses requested at supervisor review (November 2026).

WHAT THIS DOES
--------------
  1. Stand-alone tail risk of the asset classes (why H1 separates the groups):
     |ES 95%| and volatility of each native token, ETH, BTC, the 75/25 ETH/BTC
     mix and the stablecoin mix, plus the share of portfolio variance that a
     70% native position carries (justification of the Type A threshold).
  2. Uncertainty in |ES 95%|: stationary-bootstrap 95% confidence intervals for
     each treasury (Politis & Romano, 1994), and the share of joint bootstrap
     samples in which the primary H1 test stays significant.
  3. A risk-adjusted return measure that ranks correctly when excess returns
     are negative: the modified Sharpe ratio of Israelsen (2005), tested as H3.
  4. Event-window returns relative to an ETH/BTC benchmark (75/25).
  5. Runway ranks and log scale (no formal test).
  6. Hold-out test of the framework: calibrate on Jul 2023 - Dec 2024 and
     evaluate on Jan 2025 - Sep 2026.
  7. Trading-cost sensitivity: the framework mixes re-simulated with a cost on
     every quarterly rebalancing trade, larger for the native token.

INPUTS   data/processed/*, output/metrics_main_period.csv, output/event_metrics.csv
OUTPUTS  output/revision_analyses.txt and output/revision_*.csv

HOW TO RUN
----------
    python 07_revision_analyses.py
"""

import os
import importlib.util
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m3", os.path.join(SCRIPT_DIR, "03_compute_metrics.py"))
m3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m3)
spec5 = importlib.util.spec_from_file_location("m5", os.path.join(SCRIPT_DIR, "05_framework_calibration.py"))
m5 = importlib.util.module_from_spec(spec5); spec5.loader.exec_module(m5)

PROCESSED = os.path.join(SCRIPT_DIR, "..", "data", "processed")
OUTPUT = os.path.join(SCRIPT_DIR, "..", "output")
MAIN_START, MAIN_END = m3.MAIN_START, m3.MAIN_END
CAL_END = pd.Timestamp("2024-12-31")          # hold-out split
B_BOOT = 2000                                  # bootstrap replications
MEAN_BLOCK = 10                                # mean block length (days)
SEED = 20261105
STABLE_IDS, CRYPTO_MIX = m5.STABLE_IDS, m5.CRYPTO_MIX
EVENTS = {"usdc_depeg_2023": (pd.Timestamp("2023-03-07"), pd.Timestamp("2023-03-20")),
          "aug_2024_selloff": (pd.Timestamp("2024-07-31"), pd.Timestamp("2024-09-10"))}
BANDS = {"t1": (30, 50), "t2": (30, 50), "t3": (10, 30)}
MIXES = {"Framework target (40/40/20)": (40, 40, 20),
         "Most exposed permitted (30/40/30)": (30, 40, 30),
         "Most defensive permitted (50/40/10)": (50, 40, 10)}
NATIVE_COSTS = [0.0, 0.01, 0.02, 0.05]         # cost per unit of native token traded
OTHER_COST = 0.001                             # cost per unit of ETH/BTC or stablecoin traded


def es(r, level=0.95):
    return abs(m3.expected_shortfall(pd.Series(r), level)[0])


def modified_sharpe(r, rf_d):
    """Israelsen (2005): ER / sigma if ER > 0, else ER * sigma (annualised)."""
    ex = (r - rf_d).dropna()
    er, sd = ex.mean() * 365, ex.std() * np.sqrt(365)
    return er / sd if er > 0 else er * sd


def stationary_indices(n, rng):
    """One stationary-bootstrap index path (Politis & Romano, 1994)."""
    p = 1.0 / MEAN_BLOCK
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def mwu_greater(a, b):
    return stats.mannwhitneyu(a, b, alternative="greater", method="exact").pvalue


def sleeve_es(prices, ids, weights=None):
    r = m5.simple_returns(prices, ids, weights, start=MAIN_START - pd.Timedelta(days=1), end=MAIN_END)
    lr = np.log1p(r.loc[MAIN_START:MAIN_END])
    return lr


def simulate_cost(sleeves, w, qstarts, c_native):
    """Quarterly reset with trading costs charged on the reset day."""
    w = np.asarray(w, float)
    costs = np.array([OTHER_COST, OTHER_COST, c_native])
    out = []
    drift = None
    for i, qs in enumerate(qstarts):
        qe = qstarts[i + 1] if i + 1 < len(qstarts) else sleeves.index[-1] + pd.Timedelta(days=1)
        blk = sleeves[(sleeves.index >= qs) & (sleeves.index < qe)]
        if blk.empty:
            continue
        cost = 0.0 if drift is None else float(np.abs(w - drift) @ costs)
        growth = (1 + blk).cumprod()
        value = (growth.values @ w) * (1 - cost)
        prev = np.concatenate([[1.0], value[:-1]])
        out.append(pd.Series(np.log(value / prev), index=blk.index))
        end_vals = growth.values[-1] * w
        drift = end_vals / end_vals.sum()
    return pd.concat(out)


def main():
    rng = np.random.default_rng(SEED)
    L = []
    nav = pd.read_csv(os.path.join(PROCESSED, "daily_nav.csv"), parse_dates=["date"])
    prices = pd.read_csv(os.path.join(PROCESSED, "prices.csv"), parse_dates=["date"])
    shares = pd.read_csv(os.path.join(PROCESSED, "quarterly_shares.csv"), parse_dates=["snapshot_date"])
    met = pd.read_csv(os.path.join(OUTPUT, "metrics_main_period.csv"))
    evm = pd.read_csv(os.path.join(OUTPUT, "event_metrics.csv"))
    enav = pd.read_csv(os.path.join(PROCESSED, "event_nav.csv"), parse_dates=["date"])
    rf = m3.load_risk_free()
    arch = met.set_index("protocol")["archetype"]
    main = nav[(nav["date"] >= MAIN_START) & (nav["date"] <= MAIN_END)]
    R = main.pivot(index="date", columns="protocol", values="log_return").sort_index()
    RN = main.pivot(index="date", columns="protocol", values="r_native").sort_index()

    # 1 ---------------- stand-alone tail risk -----------------------------
    rows = []
    for p in RN.columns:
        r = RN[p].dropna()
        rows.append({"asset": f"{p} native token", "group": "Native token", "type": arch[p],
                     "ann_vol": r.std() * np.sqrt(365), "abs_es_95": es(r)})
    eth = sleeve_es(prices, ["ethereum"]); btc = sleeve_es(prices, ["bitcoin"])
    mix = sleeve_es(prices, list(CRYPTO_MIX), CRYPTO_MIX); stab = sleeve_es(prices, STABLE_IDS)
    for name, r in [("ETH", eth), ("BTC", btc), ("ETH/BTC 75/25", mix), ("Stablecoin mix (USDC/USDT/DAI)", stab)]:
        rows.append({"asset": name, "group": name, "type": "", "ann_vol": r.std() * np.sqrt(365), "abs_es_95": es(r)})
    sa = pd.DataFrame(rows)
    sa.to_csv(os.path.join(OUTPUT, "revision_standalone_risk.csv"), index=False)
    nat = sa[sa["group"] == "Native token"]
    corr = [RN[p].corr(mix.reindex(RN.index)) for p in RN.columns]
    L += ["1. STAND-ALONE TAIL RISK, 1 Jul 2023 - 30 Sep 2026",
          f"  Native tokens (13): |ES 95%| median {nat.abs_es_95.median():.4f}, range "
          f"{nat.abs_es_95.min():.4f}-{nat.abs_es_95.max():.4f}; vol median {nat.ann_vol.median():.3f}, range "
          f"{nat.ann_vol.min():.3f}-{nat.ann_vol.max():.3f}"]
    for _, r in sa[sa["group"] != "Native token"].iterrows():
        L.append(f"  {r.asset:32s} |ES 95%| {r.abs_es_95:.4f}  vol {r.ann_vol:.3f}")
    L.append(f"  Native vs ETH/BTC daily correlation: median {np.nanmedian(corr):.2f}")
    sn, so, rho = nat.ann_vol.median(), sa.loc[sa.asset == "ETH/BTC 75/25", "ann_vol"].iloc[0], np.nanmedian(corr)
    for w in (0.5, 0.6, 0.7, 0.8):
        var = (w * sn) ** 2 + ((1 - w) * so) ** 2 + 2 * w * (1 - w) * rho * sn * so
        contrib = (w * sn) ** 2 + w * (1 - w) * rho * sn * so
        L.append(f"  Native weight {w:.0%} (rest ETH/BTC): native share of portfolio variance {contrib / var:.1%}")
    for s in (0.3, 0.4, 0.5):
        L.append(f"  Stablecoin weight {s:.0%} (rest median native/ETH-BTC 50/50): portfolio vol reduced by {s:.0%}")
    L.append("")

    # 2 ---------------- bootstrap CIs and H1 stability --------------------
    X = R.dropna().values
    cols = list(R.columns)
    n = len(X)
    boot = np.empty((B_BOOT, len(cols)))
    for b in range(B_BOOT):
        idx = stationary_indices(n, rng)
        xs = np.sort(X[idx], axis=0)
        k = int(np.floor(n * 0.05)); frac = n * 0.05 - k
        boot[b] = -(xs[:k].sum(axis=0) + frac * xs[k]) / (n * 0.05)
    ci = pd.DataFrame({"protocol": cols, "archetype": [arch[c] for c in cols],
                       "abs_es_95": [es(R[c].dropna()) for c in cols],
                       "ci_low": np.percentile(boot, 2.5, axis=0), "ci_high": np.percentile(boot, 97.5, axis=0)})
    ci.to_csv(os.path.join(OUTPUT, "revision_es_bootstrap.csv"), index=False)
    A = [i for i, c in enumerate(cols) if arch[c] == "A"]; BC = [i for i, c in enumerate(cols) if arch[c] != "A"]
    ps = np.array([mwu_greater(boot[b, A], boot[b, BC]) for b in range(B_BOOT)])
    sep = np.mean([boot[b, A].min() > boot[b, BC].max() for b in range(B_BOOT)])
    L += [f"2. BOOTSTRAP (stationary, mean block {MEAN_BLOCK} days, {B_BOOT} joint resamples)"]
    for _, r in ci.sort_values(["archetype", "protocol"]).iterrows():
        L.append(f"  {r.protocol:16s} {r.archetype}  |ES| {r.abs_es_95:.4f}  95% CI {r.ci_low:.4f}-{r.ci_high:.4f}")
    L += [f"  H1 one-sided exact MWU across resamples: p <= 0.05 in {np.mean(ps <= 0.05):.1%}; "
          f"median p {np.median(ps):.4f}; 95th percentile p {np.percentile(ps, 95):.4f}",
          f"  Complete separation (every A above every B/C) in {sep:.1%} of resamples", ""]

    # 3 ---------------- modified Sharpe (H3) --------------------------------
    rows = []
    for c in cols:
        r = R[c].dropna()
        rows.append({"protocol": c, "archetype": arch[c],
                     "sharpe": m3.sharpe_ratio(r, m3.daily_rf(r.index, rf)),
                     "modified_sharpe": modified_sharpe(r, m3.daily_rf(r.index, rf))})
    ms = pd.DataFrame(rows)
    ms.to_csv(os.path.join(OUTPUT, "revision_modified_sharpe.csv"), index=False)
    a = ms[ms.archetype == "A"].modified_sharpe; bc = ms[ms.archetype != "A"].modified_sharpe
    u = stats.mannwhitneyu(a, bc, alternative="less", method="exact")
    delta = m3.cliffs_delta(a.values, bc.values)
    L += ["3. MODIFIED SHARPE RATIO (Israelsen, 2005)"]
    for _, r in ms.sort_values(["archetype", "protocol"]).iterrows():
        L.append(f"  {r.protocol:16s} {r.archetype}  Sharpe {r.sharpe:+.2f}  modified {r.modified_sharpe:+.3f}")
    L += [f"  H3 one-sided MWU A < B+C: medians {a.median():+.3f} vs {bc.median():+.3f}, U = {u.statistic:.1f}, "
          f"p = {u.pvalue:.4f}, Cliff's delta = {delta:+.2f}", ""]

    # 4 ---------------- benchmark-adjusted event returns -------------------
    px = prices.pivot_table(index="date", columns="coingecko_id", values="price", aggfunc="last").ffill()
    rows = []
    L += ["4. EVENT RETURNS RELATIVE TO A 75/25 ETH/BTC BENCHMARK (buy and hold from the pre-event day)"]
    for ev, (d0, d1) in EVENTS.items():
        rel = sum(w * px.loc[d1, k] / px.loc[d0, k] for k, w in CRYPTO_MIX.items())
        bench = np.log(rel)
        g = evm[evm.event == ev].copy()
        g["benchmark_log_return"] = bench
        g["excess_log_return"] = g["cum_log_return"] - bench
        rows.append(g)
        ga, gbc = g[g.archetype == "A"], g[g.archetype != "A"]
        u = stats.mannwhitneyu(ga.excess_log_return, gbc.excess_log_return, alternative="less", method="exact")
        d = m3.cliffs_delta(ga.excess_log_return.values, gbc.excess_log_return.values)
        L.append(f"  {ev}: benchmark log return {bench:+.4f}; median excess A {ga.excess_log_return.median():+.4f} "
                 f"(n={len(ga)}) vs B+C {gbc.excess_log_return.median():+.4f} (n={len(gbc)}); U = {u.statistic:.1f}, "
                 f"p = {u.pvalue:.4f}, Bonferroni (m=2) p = {min(1, 2 * u.pvalue):.4f}, delta = {d:+.2f}")
    ev_all = pd.concat(rows)
    ev_all.to_csv(os.path.join(OUTPUT, "revision_event_excess.csv"), index=False)
    for _, r in ev_all.sort_values(["event", "archetype", "protocol"]).iterrows():
        L.append(f"    {r.event:17s} {r.protocol:16s} {r.archetype}  raw {r.cum_log_return:+.3f}  excess {r.excess_log_return:+.3f}")
    L.append("")

    # 5 ---------------- runway ranks -----------------------------------------
    rw = evm.dropna(subset=["runway_months"]).copy()
    rw = rw[~((rw.protocol == "Index Coop") & (rw.event == "usdc_depeg_2023"))]
    rw["log10_runway"] = np.log10(rw["runway_months"])
    rw["rank_within_event"] = rw.groupby("event")["runway_months"].rank(ascending=False).astype(int)
    rw.to_csv(os.path.join(OUTPUT, "revision_runway.csv"), index=False)
    L += ["5. RUNWAY (Index Coop March 2023 artefact dropped)"]
    for _, r in rw.sort_values(["event", "rank_within_event"]).iterrows():
        L.append(f"  {r.event:17s} {r.protocol:16s} {r.archetype}  {r.runway_months:9.1f} months  "
                 f"log10 {r.log10_runway:5.2f}  rank {r.rank_within_event}")
    L.append("")

    # 6/7 -------------- hold-out test and trading costs ---------------------
    start = MAIN_START - pd.Timedelta(days=1)
    t1 = m5.simple_returns(prices, STABLE_IDS, start=start, end=MAIN_END)
    t2 = m5.simple_returns(prices, list(CRYPTO_MIX), CRYPTO_MIX, start=start, end=MAIN_END)
    natives = {}
    for p, g in nav.groupby("protocol"):
        r = g.set_index("date")["r_native"].reindex(t1.index)
        natives[p] = np.expm1(r.fillna(0.0))
    qstarts_all = list(pd.date_range(MAIN_START, MAIN_END, freq="QS"))
    if qstarts_all[0] != MAIN_START:
        qstarts_all = [MAIN_START] + qstarts_all
    periods = {"Calibration (Jul 2023-Dec 2024)": (MAIN_START, CAL_END),
               "Hold-out (Jan 2025-Sep 2026)": (CAL_END + pd.Timedelta(days=1), MAIN_END)}
    grid = [(a_, b_, 100 - a_ - b_) for a_ in range(0, 101, 5) for b_ in range(0, 101 - a_, 5)]
    in_band = [(w1, w2, w3) for w1, w2, w3 in grid
               if BANDS["t1"][0] <= w1 <= BANDS["t1"][1] and BANDS["t2"][0] <= w2 <= BANDS["t2"][1]
               and BANDS["t3"][0] <= w3 <= BANDS["t3"][1]]
    hold_rows = []
    L += ["6. HOLD-OUT TEST (bands fixed; budget = half the Type A median |ES 95%| of the calibration period)"]
    obs = {}
    for name, (s, e) in periods.items():
        rr = R.loc[s:e]
        o = pd.DataFrame({"protocol": cols, "archetype": [arch[c] for c in cols],
                          "abs_es": [es(rr[c].dropna()) for c in cols]})
        obs[name] = o.groupby("archetype")["abs_es"].median()
    cal_budget = obs["Calibration (Jul 2023-Dec 2024)"]["A"] / 2
    full_budget = met[met.archetype == "A"].abs_es_95.median() / 2
    L.append(f"  Budget rule: half the median Type A |ES 95%|. Full period: {full_budget:.4f}; calibration period: {cal_budget:.4f}")
    for name, (s, e) in periods.items():
        qs = [q for q in qstarts_all if s <= q <= e]
        if qs[0] != s:
            qs = [s] + qs
        res = {}
        for mix_ in in_band:
            vals = []
            for p, t3 in natives.items():
                sl = pd.DataFrame({"t1": t1, "t2": t2, "t3": t3}).loc[s:e]
                lr = m5.simulate(sl, [x / 100 for x in mix_], qs)
                vals.append(es(lr))
            res[mix_] = np.median(vals)
        worst = max(res, key=res.get)
        o = obs[name]
        L.append(f"  {name}: observed medians A {o['A']:.4f} B {o['B']:.4f} C {o['C']:.4f}; "
                 f"in-band |ES| range {min(res.values()):.4f}-{max(res.values()):.4f} (worst {worst}); "
                 f"target 40/40/20 {res[(40, 40, 20)]:.4f}; all within calibration budget {cal_budget:.4f}: "
                 f"{all(v <= cal_budget for v in res.values())}")
        for lab, mix_ in MIXES.items():
            hold_rows.append({"period": name, "allocation": lab, "abs_es_95": res[mix_]})
        for t in "ABC":
            hold_rows.append({"period": name, "allocation": f"Observed Type {t} median", "abs_es_95": o[t]})
        hold_rows.append({"period": name, "allocation": "Budget (half calibration Type A median)", "abs_es_95": cal_budget})
    pd.DataFrame(hold_rows).to_csv(os.path.join(OUTPUT, "revision_holdout.csv"), index=False)
    L.append("")

    L += ["7. TRADING-COST SENSITIVITY (full period; cost per unit traded at each quarterly reset;",
          f"   ETH/BTC and stablecoins {OTHER_COST:.1%}; native token as shown)"]
    cost_rows = []
    for lab, mix_ in MIXES.items():
        for c in NATIVE_COSTS:
            es_v, ret_v, mdd_v = [], [], []
            for p, t3 in natives.items():
                sl = pd.DataFrame({"t1": t1, "t2": t2, "t3": t3}).loc[MAIN_START:MAIN_END]
                lr = simulate_cost(sl, [x / 100 for x in mix_], qstarts_all, c)
                v = np.exp(lr.cumsum())
                es_v.append(es(lr)); ret_v.append(v.iloc[-1] - 1); mdd_v.append(abs(m3.max_drawdown(v)))
            cost_rows.append({"allocation": lab, "native_cost": c, "abs_es_95": np.median(es_v),
                              "total_return": np.median(ret_v), "max_drawdown": np.median(mdd_v)})
            L.append(f"  {lab:38s} native cost {c:4.0%}: |ES| {np.median(es_v):.4f}  "
                     f"total return {np.median(ret_v):+.3f}  max DD {np.median(mdd_v):.3f}")
    pd.DataFrame(cost_rows).to_csv(os.path.join(OUTPUT, "revision_trading_costs.csv"), index=False)
    L.append("  One-off transition cost of moving a treasury from 80% to 20% native at the native cost rates:")
    for c in NATIVE_COSTS[1:]:
        L.append(f"    {c:.0%} impact: {0.60 * c:.1%} of treasury value")

    report = "\n".join(L)
    with open(os.path.join(OUTPUT, "revision_analyses.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
