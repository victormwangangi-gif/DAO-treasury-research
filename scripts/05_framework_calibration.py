"""
STEP 5: Calibrate the Three-Tier Framework and collect the Aave / ENS
validation data (Objectives 4 and 5).

WHAT THIS DOES
--------------
PART 1 - CALIBRATION BY SIMULATION (Objective 4)
  Builds three "sleeves" of daily returns over 1 Jul 2023 - 30 Sep 2026:
    Tier 1  stablecoins     equal mix of USDC, USDT and DAI
    Tier 2  strategic       75% ETH / 25% BTC
    Tier 3  native token    each sample protocol's OWN native-token holdings
  For every mix of the three tiers in 5% steps (231 mixes) and for each of the
  13 native tokens, it simulates a treasury that is reset to the target mix at
  every quarter-end (as the framework's rebalancing rule would do) and
  measures |ES 95%|, maximum drawdown, volatility, Sharpe ratio and the loss
  in the August 2024 sell-off. Results are summarised as the median across
  the 13 native tokens, so the bands do not depend on one token.

  It also tests the framework's own weak point - the stablecoin reserve - by
  comparing a USDC-only reserve with a USDC/USDT/DAI mix during the March
  2023 depeg.

PART 2 - VALIDATION DATA FOR AAVE AND ENS (Objective 5)
  For every quarterly snapshot: the share of the treasury in each tier, and
  the outcome over the FOLLOWING quarter (Sharpe, |ES 95%|, |ES 90%|,
  volatility, drawdown) plus, for ENS, operational runway in months.

OUTPUTS  output/framework_grid.csv, output/framework_frontier.csv,
         output/validation_quarters.csv, output/framework_calibration.txt
HOW TO RUN
    python 05_framework_calibration.py
"""

import os
import importlib.util
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(SCRIPT_DIR, "..", "data", "processed")
OUTPUT = os.path.join(SCRIPT_DIR, "..", "output")

_spec = importlib.util.spec_from_file_location("m3", os.path.join(SCRIPT_DIR, "03_compute_metrics.py"))
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)

MAIN_START, MAIN_END = m3.MAIN_START, m3.MAIN_END
STABLE_IDS = ["usd-coin", "tether", "dai"]
CRYPTO_MIX = {"ethereum": 0.75, "bitcoin": 0.25}
STEP = 5                                   # grid step in percentage points
AUG = (pd.Timestamp("2024-07-31"), pd.Timestamp("2024-09-10"))
USDC_EVENT = (pd.Timestamp("2023-03-07"), pd.Timestamp("2023-03-20"))
VALIDATION_PROTOCOLS = ["Aave", "ENS"]

# ENS monthly operating expenditure (USD) for the runway series. Sources are
# listed in Appendix A of Chapter 4; rates after Nov 2024 are approximate.
ENS_OPEX = [
    (pd.Timestamp("2023-01-01"), 616_458),   # ENS Labs $11.5k/day + working groups (Jan-Jun 2023)
    (pd.Timestamp("2024-02-01"), 816_792),   # + $3.6m/yr service-provider streams (EP 5.2)
    (pd.Timestamp("2024-12-01"), 1_275_333), # ENS Labs stream raised to $9.7m/yr (EP 5.22) - approximate
]


# ----------------------------------------------------------------------------
# sleeves
# ----------------------------------------------------------------------------

def simple_returns(prices, ids, weights=None, start=None, end=None):
    wide = (prices[prices["coingecko_id"].isin(ids)]
            .pivot_table(index="date", columns="coingecko_id", values="price", aggfunc="last"))
    wide = wide.reindex(pd.date_range(start, end, freq="D")).ffill()
    r = wide.pct_change().fillna(0.0)
    w = pd.Series(weights if weights else {i: 1 / len(ids) for i in ids})
    return (r[w.index] * w).sum(axis=1)       # daily-rebalanced sleeve


def simulate(sleeves, w, quarter_starts):
    """
    sleeves: DataFrame of daily SIMPLE returns, columns t1, t2, t3.
    Weights reset to w at each quarter start; buy-and-hold in between.
    Returns daily log returns of the portfolio.
    """
    out = []
    for i, qs in enumerate(quarter_starts):
        qe = quarter_starts[i + 1] if i + 1 < len(quarter_starts) else sleeves.index[-1] + pd.Timedelta(days=1)
        blk = sleeves[(sleeves.index >= qs) & (sleeves.index < qe)]
        if blk.empty:
            continue
        growth = (1 + blk).cumprod()
        value = growth.values @ np.asarray(w)
        prev = np.concatenate([[1.0], value[:-1]])
        out.append(pd.Series(np.log(value / prev), index=blk.index))
    return pd.concat(out)


def metrics(logr, rf):
    v = np.exp(logr.cumsum())
    es, _ = m3.expected_shortfall(logr, 0.95)
    aug = v[(v.index > AUG[0]) & (v.index <= AUG[1])]
    v0 = v[v.index <= AUG[0]].iloc[-1] if (v.index <= AUG[0]).any() else 1.0
    return {"abs_es_95": abs(es), "max_drawdown": abs(m3.max_drawdown(v)),
            "ann_vol": logr.std() * np.sqrt(365), "sharpe": m3.sharpe_ratio(logr, m3.daily_rf(logr.index, rf)),
            "total_return": v.iloc[-1] - 1,
            "aug2024_drawdown": abs(min(0.0, aug.min() / v0 - 1)) if len(aug) else np.nan}


# ----------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT, exist_ok=True)
    prices = pd.read_csv(os.path.join(PROCESSED, "prices.csv"), parse_dates=["date"])
    nav = pd.read_csv(os.path.join(PROCESSED, "daily_nav.csv"), parse_dates=["date"])
    arch = pd.read_csv(os.path.join(PROCESSED, "archetype_summary.csv"))
    shares = pd.read_csv(os.path.join(PROCESSED, "quarterly_shares.csv"), parse_dates=["snapshot_date"])
    obs = pd.read_csv(os.path.join(OUTPUT, "metrics_main_period.csv"))
    rf = m3.load_risk_free()
    L = []

    # ---------------- PART 1: calibration ----------------
    start, end = MAIN_START - pd.Timedelta(days=1), MAIN_END
    t1 = simple_returns(prices, STABLE_IDS, start=start, end=end)
    t2 = simple_returns(prices, list(CRYPTO_MIX), CRYPTO_MIX, start=start, end=end)
    qstarts = list(pd.date_range(MAIN_START, MAIN_END, freq="QS"))
    if qstarts[0] != MAIN_START:
        qstarts = [MAIN_START] + qstarts

    natives = {}
    for p, g in nav.groupby("protocol"):
        r = g.set_index("date")["r_native"].reindex(t1.index)
        natives[p] = np.expm1(r.fillna(0.0))      # log -> simple

    grid = [(a, b, 100 - a - b) for a in range(0, 101, STEP) for b in range(0, 101 - a, STEP)]
    rows = []
    for p, t3 in natives.items():
        sl = pd.DataFrame({"t1": t1, "t2": t2, "t3": t3}).loc[MAIN_START:MAIN_END]
        for w1, w2, w3 in grid:
            lr = simulate(sl, [w1 / 100, w2 / 100, w3 / 100], qstarts)
            rows.append({"protocol": p, "tier1_stable": w1, "tier2_crypto": w2, "tier3_native": w3,
                         **metrics(lr, rf)})
    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(OUTPUT, "framework_grid_by_protocol.csv"), index=False)
    med = (full.groupby(["tier1_stable", "tier2_crypto", "tier3_native"])
               [["abs_es_95", "max_drawdown", "ann_vol", "sharpe", "total_return", "aug2024_drawdown"]]
               .median().reset_index())
    med.to_csv(os.path.join(OUTPUT, "framework_grid.csv"), index=False)

    es_b = obs.loc[obs["archetype"] == "B", "abs_es_95"].median()
    es_c = obs.loc[obs["archetype"] == "C", "abs_es_95"].median()
    es_a = obs.loc[obs["archetype"] == "A", "abs_es_95"].median()
    L.append("PART 1 - FRAMEWORK CALIBRATION (median across the 13 native tokens)")
    L.append(f"Observed median |ES 95%|: Type A {es_a:.4f}, Type B {es_b:.4f}, Type C {es_c:.4f}")
    L.append(f"Tier 2 sleeve = {CRYPTO_MIX}; Tier 1 sleeve = equal USDC/USDT/DAI; quarterly reset")
    L.append("")

    # frontier: for each native share, the smallest stablecoin share that meets each ES budget
    budgets = {"ES <= Type B median": es_b, "ES <= midpoint B/C": (es_b + es_c) / 2, "ES <= Type C median": es_c}
    fr = []
    for w3 in range(0, 101, 10):
        sub = med[med["tier3_native"] == w3].sort_values("tier1_stable")
        row = {"tier3_native": w3}
        for name, b in budgets.items():
            ok = sub[sub["abs_es_95"] <= b]
            row[f"min_tier1_for_{name}"] = ok["tier1_stable"].min() if len(ok) else np.nan
        bal = sub.iloc[(sub["tier1_stable"] - sub["tier2_crypto"]).abs().argsort()[:1]]
        row.update({"balanced_t1_t2": f"{int(bal['tier1_stable'].iloc[0])}/{int(bal['tier2_crypto'].iloc[0])}",
                    "balanced_es": bal["abs_es_95"].iloc[0], "balanced_mdd": bal["max_drawdown"].iloc[0],
                    "balanced_aug24": bal["aug2024_drawdown"].iloc[0], "balanced_sharpe": bal["sharpe"].iloc[0]})
        fr.append(row)
    fr = pd.DataFrame(fr)
    fr.to_csv(os.path.join(OUTPUT, "framework_frontier.csv"), index=False)
    L.append("FRONTIER: smallest Tier 1 (stablecoin %) needed to meet each tail-risk budget,")
    L.append("and the metrics of a 'balanced' mix (rest split evenly between Tier 1 and Tier 2)")
    L.append(fr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    L.append("")

    show = med[med["tier3_native"].isin([0, 10, 20, 30, 40, 50]) & med["tier1_stable"].isin([10, 20, 30, 40, 50, 60])]
    L.append("SELECTED MIXES (Tier1 / Tier2 / Tier3 %)")
    L.append(show.sort_values(["tier3_native", "tier1_stable"]).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    L.append("")

    # Stablecoin reserve in the USDC depeg: single issuer vs diversified
    s0, s1 = USDC_EVENT
    r_usdc = simple_returns(prices, ["usd-coin"], start=s0, end=s1)
    r_mix = simple_returns(prices, STABLE_IDS, start=s0, end=s1)
    worst = lambda r: (1 + r).cumprod().min() - 1
    L.append("STABLECOIN RESERVE IN THE MARCH 2023 USDC DEPEG (worst point vs 7 Mar 2023)")
    L.append(f"  USDC only:            {worst(r_usdc):+.4%}")
    L.append(f"  USDC/USDT/DAI mix:    {worst(r_mix):+.4%}")
    L.append(f"  ETH/BTC sleeve, same window: {worst(simple_returns(prices, list(CRYPTO_MIX), CRYPTO_MIX, start=s0, end=s1)):+.4%}")
    L.append("")

    # ---------------- PART 2: validation data ----------------
    L.append("PART 2 - VALIDATION DATA: tier shares at each snapshot and outcomes over the FOLLOWING quarter")
    vrows = []
    q = shares[(shares["snapshot_type"] == "quarterly") & shares["protocol"].isin(VALIDATION_PROTOCOLS)]
    for p, g in q.groupby("protocol"):
        g = g.sort_values("snapshot_date")
        n = nav[nav["protocol"] == p].set_index("date")
        for _, s in g.iterrows():
            d0 = s["snapshot_date"]
            d1 = d0 + pd.offsets.QuarterEnd(1)
            r = n["log_return"][(n.index > d0) & (n.index <= d1)].dropna()
            row = {"protocol": p, "snapshot": s["snapshot_label"], "date": d0.date(),
                   "tier1_stable": round(100 * s["stablecoin_pct"], 1),
                   "tier2_crypto": round(100 * (s["bluechip_pct"] + s["other_pct"]), 1),
                   "tier3_native": round(100 * s["native_pct"], 1),
                   "treasury_usd_m": round(s["total_usd"] / 1e6, 1)}
            if len(r) >= 60:
                v = np.exp(r.cumsum())
                row.update({"next_q_sharpe": round(m3.sharpe_ratio(r, m3.daily_rf(r.index, rf)), 3),
                            "next_q_abs_es95": round(abs(m3.expected_shortfall(r, 0.95)[0]), 4),
                            "next_q_abs_es90": round(abs(m3.expected_shortfall(r, 0.90)[0]), 4),
                            "next_q_vol": round(r.std() * np.sqrt(365), 3),
                            "next_q_mdd": round(abs(m3.max_drawdown(pd.concat([pd.Series([1.0]), v]))), 4),
                            "next_q_return": round(v.iloc[-1] - 1, 4)})
            if p == "ENS":
                opex = [x for d, x in ENS_OPEX if d <= d0][-1]
                liquid = s["total_usd"] * (s["stablecoin_pct"] + s["bluechip_pct"])
                row["runway_months"] = round(liquid / opex, 1)
            vrows.append(row)
    val = pd.DataFrame(vrows)
    val.to_csv(os.path.join(OUTPUT, "validation_quarters.csv"), index=False)
    pd.set_option("display.width", 220)
    L.append(val.to_string(index=False))
    L.append("")
    L.append("Notes: Tier 2 = ETH/BTC (incl. staked) + other non-native crypto. Outcomes use the daily")
    L.append("time-weighted returns of the quarter after each snapshot (about 91 days; |ES 95%| rests on")
    L.append("4-5 tail days, |ES 90%| on 9). ENS runway = stablecoin + ETH/BTC value / monthly opex.")

    report = "\n".join(L)
    with open(os.path.join(OUTPUT, "framework_calibration.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
