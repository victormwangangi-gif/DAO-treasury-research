"""
STEP 2: Build the daily treasury value series and classify the archetypes.

WHAT THIS DOES (Sections 3.4, 3.5 and 3.7 of your paper)
--------------------------------------------------------
MAIN ANALYSIS PERIOD (1 Jul 2023 - 30 Sep 2026)
  You have 14 quarterly snapshots of how many units of each token a treasury
  holds, and a daily price for each token. The quantities at each quarter-end
  are held constant until the next quarter-end and revalued at daily prices.

  Each day's return uses the holdings in place at the START of that day, so a
  quarter-end change in holdings (spending, income, a token sale) never shows
  up as a gain or loss. The return reflects the price performance of the
  allocation only - a time-weighted return, as Section 3.5 describes.

EVENT STUDIES
  For each event, the holdings on the pre-event snapshot (7 Mar 2023 and
  31 Jul 2024) are held constant through the baseline and the event window.

ARCHETYPES
  Main analysis: mean composition across the quarterly snapshots.
  Events: composition on the pre-event snapshot.
  Rule: native >= 70% -> A; else stablecoin >= 40% -> C; else B.

DATA CLEANING
  - Receipt and LP tokens (price_method = calibrated in token_map.csv) are
    valued at DefiLlama's USD value on each snapshot date and then move day by
    day with the price of the asset they track (e.g. aEthUSDC with USDC,
    cUSDC with USDC, the OHM half of an OHM/DAI pool with OHM).
  - 1% materiality cut-off: a token WITHOUT a price is dropped if it never
    reaches 1% of a protocol's treasury (DefiLlama USD value) on any snapshot.
    Priced tokens are always kept.
  - Material tokens with no price are reported, since they cannot be valued.
  - Optional manual fixes: put rows in data/raw/manual_corrections.csv (same
    columns as treasury_composition.csv). A row replaces the matching
    protocol + snapshot_label + asset_symbol row; quantity 0 removes it.

OUTPUTS (data/processed/)
  daily_nav.csv            daily value, index and log return per protocol
  event_nav.csv            daily values for each event (baseline + window)
  quarterly_shares.csv     asset-class shares at every quarterly snapshot
  archetype_summary.csv    main-period archetype per protocol
  event_archetypes.csv     pre-event archetype per protocol and event
  data_quality.csv         dropped / unpriced value per snapshot

HOW TO RUN
----------
    python 02_build_nav.py
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")

COMPOSITION_FILE = os.path.join(RAW_DIR, "treasury_composition.csv")
EVENT_FILE       = os.path.join(RAW_DIR, "event_snapshots.csv")
CORRECTIONS_FILE = os.path.join(RAW_DIR, "manual_corrections.csv")
PRICES_FILE      = os.path.join(PROCESSED_DIR, "prices.csv")

MAIN_START = pd.Timestamp("2023-07-01")   # first daily return
MAIN_END   = pd.Timestamp("2026-09-30")   # last daily return

EVENTS = {
    "usdc_depeg_2023": {"baseline_days": 21,
                        "start": pd.Timestamp("2023-03-08"),
                        "end":   pd.Timestamp("2023-03-20")},
    "aug_2024_selloff": {"baseline_days": 120,
                         "start": pd.Timestamp("2024-08-01"),
                         "end":   pd.Timestamp("2024-09-10")},
}

TYPE_A_NATIVE_THRESHOLD = 0.70
TYPE_C_STABLE_THRESHOLD = 0.40
MATERIALITY = 0.01            # 1% cut-off
MAX_PRICE_GAP_DAYS = 3        # a token is "dead" if its prices stop more than 3 days
                              # before the date (e.g. MATIC after its migration to POL)
CLASSES = ["native", "stablecoin", "bluechip", "other"]
LABELS = {"A": "Native-token-concentrated", "B": "Diversified", "C": "Stablecoin-heavy"}


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def classify(native, stable):
    if native >= TYPE_A_NATIVE_THRESHOLD:
        return "A"
    if stable >= TYPE_C_STABLE_THRESHOLD:
        return "C"
    return "B"


def is_usdc(symbol, cg_id):
    """USDC itself or a USDC receipt token (aEthUSDC, cUSDC, ...) - judged by symbol,
    so dollar tokens merely priced against USDC (USDGLO, USTBL) are not counted."""
    return "USDC" in str(symbol).upper()


def load_snapshots():
    comp = pd.read_csv(COMPOSITION_FILE)
    events = pd.read_csv(EVENT_FILE) if os.path.exists(EVENT_FILE) else comp.iloc[0:0]
    snaps = pd.concat([comp, events], ignore_index=True)
    if "usd_value_defillama" not in snaps.columns:
        snaps["usd_value_defillama"] = np.nan

    if os.path.exists(CORRECTIONS_FILE):
        fix = pd.read_csv(CORRECTIONS_FILE)
        fix = fix[~fix["notes"].astype(str).str.startswith("EXAMPLE")] if "notes" in fix else fix
        # 'check' rows are on-chain quantities for a date DefiLlama DOES cover;
        # they are compared with DefiLlama to validate the wallet list, then dropped.
        chk = fix[fix["snapshot_type"] == "check"]
        fix = fix[fix["snapshot_type"] != "check"]
        if len(chk):
            dl = (snaps[snaps["snapshot_type"] == "quarterly"]
                  .groupby(["protocol", "snapshot_label", "asset_symbol"])["quantity"].sum())
            print("OVERLAP CHECK: on-chain backfill vs DefiLlama on a date both cover")
            for _, r in chk.iterrows():
                d = dl.get((r["protocol"], r["snapshot_label"], r["asset_symbol"]), np.nan)
                ratio = r["quantity"] / d if d and d > 0 else np.nan
                verdict = ("OK" if 0.95 <= ratio <= 1.05 else
                           "MISMATCH - wallet list differs, do not use backfill" if np.isfinite(ratio)
                           else "no DefiLlama value to compare")
                print(f"  {r['protocol']:<12} {r['snapshot_label']:<8} {r['asset_symbol']:<6} "
                      f"on-chain {r['quantity']:>16,.0f}   DefiLlama {d:>16,.0f}   {verdict}")
            print()
        key = ["protocol", "snapshot_label", "asset_symbol"]
        snaps = snaps.merge(fix[key].drop_duplicates(), on=key, how="left", indicator=True)
        snaps = snaps[snaps["_merge"] == "left_only"].drop(columns="_merge")
        snaps = pd.concat([snaps, fix], ignore_index=True)
        print(f"Applied {len(fix)} manual correction rows from manual_corrections.csv")

    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"])
    snaps["quantity"] = pd.to_numeric(snaps["quantity"], errors="coerce")
    snaps = snaps[snaps["quantity"].fillna(0) > 0].copy()
    snaps["coingecko_id"] = snaps["coingecko_id"].where(
        snaps["coingecko_id"].notna() & (snaps["coingecko_id"].astype(str) != ""), None)
    if "price_method" not in snaps.columns:
        snaps["price_method"] = "market"
    snaps["price_method"] = snaps["price_method"].fillna("market").replace("", "market")
    return snaps


def price_on(px_by_id, cg_id, date):
    """
    Price of cg_id on date. Inside a token's price history a missing day takes
    the last known price (marking an illiquid token at its last trade); before
    its first price or after its prices stop, it has no price.
    """
    hist = px_by_id.get(cg_id)
    if hist is None or not len(hist):
        return np.nan
    if date < hist.index[0] or date > hist.index[-1] + pd.Timedelta(days=MAX_PRICE_GAP_DAYS):
        return np.nan
    return hist[hist.index <= date].iloc[-1]


def calibrate(snaps, prices):
    """
    For calibrated tokens, convert the quantity into units of the tracked asset:
        quantity = DefiLlama USD value on the snapshot date / tracked-asset price
    so that the token is worth exactly DefiLlama's value on the snapshot date and
    then moves with the tracked asset. Tokens that cannot be calibrated (no
    DefiLlama value or no price) are marked unpriced.
    """
    px_by_id = {k: g.set_index("date")["price"].sort_index()
                for k, g in prices.groupby("coingecko_id")}
    s = snaps.copy()
    s["quantity_raw"] = s["quantity"]
    s["valuation_flag"] = ""
    s["our_value"] = np.nan

    # LOOK-ALIKE CHECK for directly priced tokens. Scam airdrops often reuse a
    # real symbol (a fake "WETH" or "USDC"), so quantity x real price can be
    # huge while DefiLlama, which checks the contract, values it near zero.
    # If our value and DefiLlama's differ by more than 2x, DefiLlama's value is
    # used instead (calibrated), or the token is dropped if DefiLlama gives ~0.
    mkt = (s["price_method"] == "market") & s["coingecko_id"].notna() & s["usd_value_defillama"].notna()
    for i in s.index[mkt]:
        p = price_on(px_by_id, s.at[i, "coingecko_id"], s.at[i, "snapshot_date"])
        if not (np.isfinite(p) and p > 0):
            continue
        ours, v = s.at[i, "quantity"] * p, s.at[i, "usd_value_defillama"]
        s.at[i, "our_value"] = ours
        if ours > 1000 and not (0.5 <= (v / ours) <= 2.0):
            s.at[i, "valuation_flag"] = "lookalike"
            s.at[i, "price_method"] = "calibrated"     # handled just below

    cal = s["price_method"] == "calibrated"
    for i in s.index[cal]:
        cg, d = s.at[i, "coingecko_id"], s.at[i, "snapshot_date"]
        p = price_on(px_by_id, cg, d) if cg else np.nan
        v = s.at[i, "usd_value_defillama"]
        ok = np.isfinite(p) and p > 0 and pd.notna(v) and v > 0
        if ok and s.at[i, "valuation_flag"] == "lookalike":
            ok = v > 0.01 * s.at[i, "our_value"]     # DefiLlama says ~worthless: drop
        if ok:
            s.at[i, "quantity"] = v / p
        else:
            s.at[i, "coingecko_id"] = None
    return s


def apply_materiality(snaps):
    """Drop tokens that never reach 1% of the protocol's treasury on any snapshot."""
    s = snaps.copy()
    tot = s.groupby(["protocol", "snapshot_label"])["usd_value_defillama"].transform("sum")
    s["dl_share"] = s["usd_value_defillama"] / tot
    # the two halves of an LP token ("X[OHM]", "X[DAI]") are judged together
    s["base_symbol"] = s["asset_symbol"].astype(str).str.replace(r"\[.*\]$", "", regex=True)
    base_share = s.groupby(["protocol", "snapshot_label", "base_symbol"])["dl_share"].transform("sum")
    max_share = base_share.groupby([s["protocol"], s["base_symbol"]]).transform("max")
    # rows without a DefiLlama value (e.g. manual corrections) are always kept
    s["material"] = max_share.isna() | (max_share >= MATERIALITY)
    return s


# Positions are keyed by "coingecko_id|asset_class", so the same price series can
# sit in two classes (e.g. BADGER held directly and inside an LP are both native,
# while ETH inside the same LP is bluechip).

def price_matrix(prices, keys, start, end):
    """
    Daily prices per position key, plus a matching True/False table of 'stale'
    days: days inside a token's price history with no new price, which carry
    the last known price forward.
    """
    dates = pd.date_range(start, end, freq="D")
    ids = sorted({k.split("|")[0] for k in keys})
    raw = (prices[prices["coingecko_id"].isin(ids)]
           .pivot_table(index="date", columns="coingecko_id", values="price", aggfunc="last"))
    full = raw.reindex(pd.date_range(min(raw.index.min(), dates[0]), max(raw.index.max(), dates[-1]),
                                     freq="D")) if len(raw) else raw.reindex(dates)
    first = full.apply(lambda c: c.first_valid_index())
    last = full.apply(lambda c: c.last_valid_index())
    live = pd.DataFrame({c: (full.index >= first[c]) &
                             (full.index <= last[c] + pd.Timedelta(days=MAX_PRICE_GAP_DAYS))
                         if pd.notna(first[c]) else np.zeros(len(full), bool)
                         for c in full.columns}, index=full.index)
    stale = full.isna() & live
    filled = full.ffill().where(live)
    filled = filled.reindex(dates).reindex(columns=ids)
    stale = stale.reindex(dates).reindex(columns=ids).fillna(False).astype(bool)
    P = pd.DataFrame({k: filled[k.split("|")[0]] for k in keys}, index=dates)
    S = pd.DataFrame({k: stale[k.split("|")[0]] for k in keys}, index=dates)
    return P, S


def holdings_vector(rows, keys):
    """Quantity per position key (summing the same asset across chains/tokens)."""
    return rows.groupby("pkey")["quantity"].sum().reindex(keys).fillna(0.0)


def class_of(rows, keys):
    return pd.Series({k: k.split("|", 1)[1] for k in keys})


# ----------------------------------------------------------------------------
# MAIN-PERIOD NAV
# ----------------------------------------------------------------------------

def build_main_nav(qsnaps, prices):
    out = []
    for protocol, p in qsnaps.groupby("protocol"):
        p = p[p["coingecko_id"].notna()]
        ids = sorted(p["pkey"].unique())
        snap_dates = sorted(p["snapshot_date"].unique())
        if not ids or not snap_dates:
            continue
        first = max(pd.Timestamp(snap_dates[0]), MAIN_START - pd.Timedelta(days=1))
        P, S = price_matrix(prices, ids, first, MAIN_END)
        cls = class_of(p, ids)

        # H[d] = holdings in force at the close of day d (latest snapshot <= d)
        H = pd.DataFrame(np.nan, index=P.index, columns=ids)
        for sd in snap_dates:
            sd = pd.Timestamp(sd)
            if sd in H.index or sd < H.index[0]:
                H.loc[max(sd, H.index[0])] = holdings_vector(p[p["snapshot_date"] == sd], ids).values
        H = H.ffill()

        # Mark-to-market USD value (includes quarter-end changes in holdings)
        nav_usd = (H * P).sum(axis=1, min_count=1)

        # Time-weighted return: yesterday's holdings at yesterday's vs today's prices
        Hy = H.shift(1)
        Py = P.shift(1)
        ok = Hy.notna() & P.notna() & Py.notna()
        val_today = (Hy * P).where(ok, 0.0)
        val_yday = (Hy * Py).where(ok, 0.0)

        df = pd.DataFrame({"date": P.index, "protocol": protocol,
                           "nav_usd": nav_usd.values})
        a, b = val_today.sum(axis=1), val_yday.sum(axis=1)
        df["log_return"] = np.where(b > 0, np.log(a.where(a > 0) / b.where(b > 0)), np.nan)
        # share of the treasury priced at a carried-forward (stale) price that day
        df["stale_pct"] = np.where(b > 0, val_yday.where(S, 0.0).sum(axis=1) / b.where(b > 0), np.nan)
        for c in CLASSES:
            cols = [i for i in ids if cls[i] == c]
            ca, cb = val_today[cols].sum(axis=1), val_yday[cols].sum(axis=1)
            df[f"r_{c}"] = np.where(cb > 0, np.log(ca.where(ca > 0) / cb.where(cb > 0)), np.nan)
        df.loc[df["date"] < MAIN_START, [c for c in df.columns if c.startswith("r_")] + ["log_return"]] = np.nan
        df["index"] = np.exp(df["log_return"].fillna(0).cumsum())
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------
# SNAPSHOT SHARES AND ARCHETYPES
# ----------------------------------------------------------------------------

def snapshot_shares(snaps, prices):
    """Asset-class shares at each snapshot, valued at the snapshot-date price."""
    rows = []
    px_by_id = {k: g.set_index("date")["price"].sort_index()
                for k, g in prices.groupby("coingecko_id")}
    for (protocol, label, stype), p in snaps.groupby(["protocol", "snapshot_label", "snapshot_type"]):
        d = pd.Timestamp(p["snapshot_date"].iloc[0])
        vals = [r["quantity"] * price_on(px_by_id, r["coingecko_id"], d) if r["coingecko_id"] else np.nan
                for _, r in p.iterrows()]
        p = p.assign(value=vals)
        priced = p[p["value"].notna()]
        total = priced["value"].sum()
        unpriced_dl = p.loc[p["value"].isna(), "usd_value_defillama"].sum()
        dl_total = p["usd_value_defillama"].sum()
        row = {"protocol": protocol, "snapshot_type": stype, "snapshot_label": label,
               "snapshot_date": d.date(), "total_usd": total}
        for c in CLASSES:
            row[f"{c}_pct"] = priced.loc[priced["asset_class"] == c, "value"].sum() / total if total > 0 else np.nan
        row["usdc_pct"] = (priced.loc[[is_usdc(s, i) for s, i in zip(priced["asset_symbol"], priced["coingecko_id"])],
                                      "value"].sum() / total) if total > 0 else np.nan
        row["unpriced_material_pct"] = unpriced_dl / dl_total if dl_total > 0 else np.nan
        dl_native = p.loc[p["asset_class"] == "native", "usd_value_defillama"].sum()
        row["native_pct_defillama"] = dl_native / dl_total if dl_total > 0 else np.nan
        row["archetype"] = classify(row["native_pct"], row["stablecoin_pct"]) if total > 0 else "?"
        rows.append(row)
    return pd.DataFrame(rows)


def main_archetypes(qshares, nav):
    rows = []
    for protocol, q in qshares.groupby("protocol"):
        q = q.sort_values("snapshot_date")
        mean = {c: q[f"{c}_pct"].mean() for c in CLASSES}
        arch = classify(mean["native"], mean["stablecoin"])
        changes = [f"{prev}->{cur} at {lab}" for prev, cur, lab in
                   zip(q["archetype"].shift(1), q["archetype"], q["snapshot_label"])
                   if isinstance(prev, str) and prev != cur]
        main_nav = nav[(nav["protocol"] == protocol) & (nav["date"] >= MAIN_START - pd.Timedelta(days=1))]
        rows.append({"protocol": protocol, "archetype": arch, "archetype_label": LABELS[arch],
                     **{f"{c}_pct": round(mean[c], 4) for c in CLASSES},
                     "n_snapshots": len(q),
                     "peak_nav_usd": main_nav["nav_usd"].max(),
                     "archetype_changes": "; ".join(changes),
                     "stable_archetype": "yes" if not changes else "NO"})
    return pd.DataFrame(rows).sort_values(["archetype", "protocol"])


# ----------------------------------------------------------------------------
# EVENT NAV
# ----------------------------------------------------------------------------

def build_event_nav(esnaps, prices):
    out = []
    for ev, spec in EVENTS.items():
        e = esnaps[(esnaps["snapshot_label"] == ev) & esnaps["coingecko_id"].notna()]
        pre = spec["start"] - pd.Timedelta(days=1)
        base_start = spec["start"] - pd.Timedelta(days=spec["baseline_days"])
        for protocol, p in e.groupby("protocol"):
            ids = sorted(p["pkey"].unique())
            P, _S = price_matrix(prices, ids, base_start - pd.Timedelta(days=1), spec["end"])
            q = holdings_vector(p, ids)
            cls = class_of(p, ids)
            V = P * q.values
            df = pd.DataFrame({"event": ev, "protocol": protocol, "date": P.index})
            df["phase"] = np.where(df["date"] < spec["start"], "baseline", "event")
            df.loc[df["date"] < base_start, "phase"] = "baseline_anchor"
            df["is_pre_event_day"] = df["date"] == pre
            ok = P.notna() & P.shift(1).notna()
            vt, vy = V.where(ok, 0.0), V.shift(1).where(ok, 0.0)
            df["nav_usd"] = V.sum(axis=1, min_count=1).values
            a, b = vt.sum(axis=1), vy.sum(axis=1)
            df["log_return"] = np.where(b > 0, np.log(a.where(a > 0) / b.where(b > 0)), np.nan)
            for c in CLASSES:
                cols = [i for i in ids if cls[i] == c]
                df[f"value_{c}"] = V[cols].sum(axis=1).values
                ca, cb = vt[cols].sum(axis=1), vy[cols].sum(axis=1)
                df[f"r_{c}"] = np.where(cb > 0, np.log(ca.where(ca > 0) / cb.where(cb > 0)), np.nan)
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    for f, msg in [(COMPOSITION_FILE, "Run 01b_defillama_extract.py first."),
                   (PRICES_FILE, "Run 01_fetch_prices.py first.")]:
        if not os.path.exists(f):
            print(f"ERROR: missing {f}\n{msg}")
            return
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    prices = pd.read_csv(PRICES_FILE, parse_dates=["date"])
    priced_ids = set(prices["coingecko_id"].unique())
    snaps = load_snapshots()
    # Debt positions (e.g. Aave variableDebt tokens) carry a NEGATIVE DefiLlama
    # value. The study measures gross treasury assets, so they are left out and
    # reported below.
    neg = snaps["usd_value_defillama"] < 0
    gross = (snaps[~neg].groupby(["protocol", "snapshot_label"])["usd_value_defillama"]
             .sum().rename("gross"))
    liab = (snaps[neg].groupby(["protocol", "snapshot_label"])["usd_value_defillama"].sum()
            .rename("debt").to_frame().join(gross))
    liab["debt_pct_of_assets"] = -100 * liab["debt"] / liab["gross"]
    snaps = snaps[~neg]
    snaps = apply_materiality(calibrate(snaps, prices))
    look = snaps[snaps["valuation_flag"] == "lookalike"]

    # ---- data-quality report ----
    snaps["has_price"] = snaps["coingecko_id"].isin(priced_ids)
    # The 1% cut-off is only for tokens we CANNOT value. A token with a price
    # is kept however small it is - dropping hundreds of small priced positions
    # (e.g. Aave's many aTokens) would understate the non-native share.
    snaps["material"] = snaps["material"] | snaps["has_price"]
    dq = (snaps.assign(v=snaps["usd_value_defillama"].fillna(0))
               .groupby(["protocol", "snapshot_label"])
               .apply(lambda g: pd.Series({
                   "dropped_immaterial_pct": g.loc[~g["material"], "v"].sum() / g["v"].sum() if g["v"].sum() else 0,
                   "unpriced_material_pct": g.loc[g["material"] & ~g["has_price"], "v"].sum() / g["v"].sum() if g["v"].sum() else 0,
               }), include_groups=False)
               .reset_index())
    dq.to_csv(os.path.join(PROCESSED_DIR, "data_quality.csv"), index=False)

    unpriced = (snaps[snaps["material"] & ~snaps["has_price"]]
                .groupby("base_symbol")
                .agg(max_share_pct=("dl_share", lambda s: round(100 * s.max(), 2)),
                     protocols=("protocol", lambda s: ", ".join(sorted(set(s)))))
                .sort_values("max_share_pct", ascending=False))

    snaps = snaps[snaps["material"]].copy()
    snaps.loc[~snaps["has_price"], "coingecko_id"] = None
    snaps["pkey"] = snaps["coingecko_id"].astype(str) + "|" + snaps["asset_class"].astype(str)
    qsnaps = snaps[snaps["snapshot_type"] == "quarterly"]
    esnaps = snaps[snaps["snapshot_type"] == "event"]

    print(f"Loaded {qsnaps['protocol'].nunique()} protocols, "
          f"{qsnaps['snapshot_label'].nunique()} quarterly snapshots, "
          f"{esnaps['snapshot_label'].nunique()} event snapshots.\n")

    # ---- main period ----
    print("Building daily NAV for the main analysis period...")
    nav = build_main_nav(qsnaps, prices)
    nav.to_csv(os.path.join(PROCESSED_DIR, "daily_nav.csv"), index=False)
    main_obs = nav[(nav["date"] >= MAIN_START) & nav["log_return"].notna()]
    print(f"  {len(main_obs):,} daily returns "
          f"(about {len(main_obs) // max(nav['protocol'].nunique(), 1):,} per protocol)\n")

    shares = snapshot_shares(snaps, prices)
    shares.to_csv(os.path.join(PROCESSED_DIR, "quarterly_shares.csv"), index=False)
    qshares = shares[shares["snapshot_type"] == "quarterly"]

    arch = main_archetypes(qshares, nav)
    arch.to_csv(os.path.join(PROCESSED_DIR, "archetype_summary.csv"), index=False)

    # ---- events ----
    print("Building event-study NAV series...")
    enav = build_event_nav(esnaps, prices)
    enav.to_csv(os.path.join(PROCESSED_DIR, "event_nav.csv"), index=False)
    ea = shares[shares["snapshot_type"] == "event"].rename(columns={"snapshot_label": "event"})
    ea = ea.merge(arch[["protocol", "archetype"]].rename(columns={"archetype": "main_archetype"}),
                  on="protocol", how="left")
    ea["differs_from_main"] = np.where(ea["archetype"] != ea["main_archetype"], "YES", "")
    ea.drop(columns="snapshot_type").to_csv(os.path.join(PROCESSED_DIR, "event_archetypes.csv"), index=False)

    # ---- report ----
    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("MAIN-PERIOD ARCHETYPES (mean of quarterly snapshots)")
    print("=" * 78)
    show = arch[["protocol", "archetype", "native_pct", "stablecoin_pct", "bluechip_pct",
                 "other_pct", "n_snapshots", "stable_archetype"]].copy()
    for c in CLASSES:
        show[f"{c}_pct"] = (100 * show[f"{c}_pct"]).round(1)
    print(show.to_string(index=False))
    print("\nArchetype counts:", arch["archetype"].value_counts().sort_index().to_dict())
    moved = arch[arch["stable_archetype"] == "NO"]
    if len(moved):
        print("\nProtocols that changed archetype during the period (report descriptively):")
        for _, r in moved.iterrows():
            print(f"  {r['protocol']}: {r['archetype_changes']}")

    print("\n" + "=" * 78)
    print("EVENT ARCHETYPES (pre-event snapshot)")
    print("=" * 78)
    for ev in EVENTS:
        sub = ea[ea["event"] == ev]
        if sub.empty:
            print(f"  {ev}: no snapshots")
            continue
        print(f"  {ev}: {len(sub)} protocols, counts {sub['archetype'].value_counts().sort_index().to_dict()}")
        for _, r in sub[sub["differs_from_main"] == "YES"].iterrows():
            print(f"     {r['protocol']}: {r['archetype']} before the event, {r['main_archetype']} in the main period")

    print("\n" + "=" * 78)
    print("DATA QUALITY")
    print("=" * 78)
    worst = dq.groupby("protocol")[["dropped_immaterial_pct", "unpriced_material_pct"]].max()
    print((100 * worst).round(2).rename(columns=lambda c: "max_" + c).to_string())
    if len(unpriced):
        print("\nMaterial tokens WITHOUT a price (left out of the NAV - fix these):")
        print(unpriced.to_string())
    st = nav[nav["date"] >= MAIN_START].groupby("protocol")["stale_pct"]
    stale_rep = pd.DataFrame({"days_over_5pct": st.apply(lambda x: int((x > 0.05).sum())),
                              "max_stale_pct": (100 * st.max()).round(1)})
    stale_rep = stale_rep[stale_rep["days_over_5pct"] > 0]
    if len(stale_rep):
        print("\nDays when more than 5% of a treasury had no fresh price (last price carried forward):")
        print(stale_rep.to_string())
    if len(look):
        lk = (look.groupby(["protocol", "asset_symbol"])
                  .agg(snapshots=("snapshot_label", "nunique"),
                       max_value_at_market_price=("our_value", "max"),
                       max_value_defillama=("usd_value_defillama", "max"))
                  .sort_values("max_value_at_market_price", ascending=False))
        print("\nTokens whose market value disagrees with DefiLlama by >2x (likely fake")
        print("look-alike tokens) - valued at DefiLlama's figure instead (largest 15):")
        print(lk.head(15).to_string(float_format=lambda x: f"{x:,.0f}"))
    if len(liab):
        print("\nDebt positions left out (largest, as % of gross treasury assets):")
        print(liab.groupby(level="protocol")["debt_pct_of_assets"].max().round(2)
              .rename("max_debt_pct").to_string())
    gap = shares[(shares["snapshot_type"] == "quarterly") &
                 ((shares["native_pct"] - shares["native_pct_defillama"]).abs() > 0.10)]
    if len(gap):
        print("\nSnapshots where our native share differs from DefiLlama's by >10 points:")
        print(gap[["protocol", "snapshot_label", "native_pct", "native_pct_defillama"]]
              .round(3).to_string(index=False))
    print(f"\nAll outputs saved to {os.path.abspath(PROCESSED_DIR)}")


if __name__ == "__main__":
    main()
