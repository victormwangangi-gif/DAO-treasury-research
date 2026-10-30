"""
STEP 1b: Pull treasury composition automatically from DefiLlama's FREE API.

This replaces most of the manual data entry. Run 00_explore_defillama.py first
to confirm the API structure matches what this script expects.

WHAT THIS DOES
--------------
For each of the 13 sample protocols:
  - fetches the full treasury history from api.llama.fi/treasury/{slug}
  - INCLUDING the 'own tokens' bucket, which DefiLlama separates out by default
  - takes 14 quarterly snapshots (30 Jun 2023 - 30 Sep 2026) for the main analysis
  - takes 2 pre-event snapshots (7 Mar 2023, 31 Jul 2024) for the event studies
  - writes data/raw/treasury_composition.csv   (quarterly)
         data/raw/event_snapshots.csv          (pre-event)
    in the format 02_build_nav.py expects. You do NOT fill in the template
    by hand unless a protocol needs a manual correction.
  - applies data/raw/token_map.csv, which classifies receipt tokens (aTokens,
    cTokens, savings tokens), splits liquidity-pool tokens into their two
    halves, and gives each token a price source (see that file's 'note' column)

CRITICAL NOTE
-------------
DefiLlama's headline "Treasury" figure EXCLUDES the protocol's own governance
token. Your study is ABOUT native token concentration, so this script
deliberately merges the own-tokens bucket back in. Without this, every Type A
protocol would be misclassified as Type C.

HOW TO RUN
----------
    python 01b_defillama_extract.py

THEN
----
Open the output CSV and check it by hand before using it. Automated extraction
is a starting point, not a substitute for verification. Spot-check at least
3 protocol-quarters against the DefiLlama website.
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime

BASE = "https://api.llama.fi"

# Paths are relative to THIS script's folder, so it works whichever folder
# you run it from (this is what caused the earlier "non-existent directory" crash).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RAW_DIR, "treasury_composition.csv")    # quarterly - main analysis
EVENT_FILE = os.path.join(RAW_DIR, "event_snapshots.csv")          # pre-event - event studies
UNMAPPED_LOG = os.path.join(RAW_DIR, "unmapped_tokens.csv")
TOKEN_MAP_FILE = os.path.join(RAW_DIR, "token_map.csv")

# --------------------------------------------------------------------------
# PROTOCOLS TO FETCH
# Verify each slug using 00_explore_defillama.py before adding it here.
# Start with 3 for your pilot, then expand.
# --------------------------------------------------------------------------
PROTOCOLS = {
    # Final 14-protocol sample (paper Section 3.2). Slugs confirmed by 00c run.
    "aave":             "Aave",
    "arbitrum-dao":     "Arbitrum DAO",
    "badger-dao":       "Badger DAO",
    # "bancor": "Bancor",   EXCLUDED: DefiLlama has no Bancor treasury data
    #                        between 25 Jun 2024 and 16 Jun 2026 (7 of 14 quarters
    #                        missing) and none before May 2023 (fails the
    #                        coverage criterion, Section 3.2).
    "beefy":            "Beefy",
    "compound-finance": "Compound",
    "convex-finance":   "Convex Finance",
    "ens":              "ENS",
    "gitcoin":          "Gitcoin",
    "index-coop":       "Index Coop",
    "lido":             "Lido",
    "olympus-dao":      "Olympus DAO",
    "ribbon-finance":   "Ribbon Finance",
    "shapeshift":       "ShapeShift",
}

# Quarterly snapshots for the MAIN ANALYSIS PERIOD (paper Sections 3.2 and 3.4):
# 30 Jun 2023 to 30 Sep 2026 = 14 snapshots.
QUARTER_ENDS = pd.date_range("2023-06-30", "2026-09-30", freq="QE")

# Pre-event snapshots for the EVENT STUDIES (paper Section 3.7):
# taken the day before each event window opens.
EVENT_SNAPSHOTS = {
    "usdc_depeg_2023":  pd.Timestamp("2023-03-07"),   # window 8-20 Mar 2023
    "aug_2024_selloff": pd.Timestamp("2024-07-31"),   # window 1 Aug - 10 Sep 2024
}
EVENT_MAX_GAP_DAYS = 2      # event snapshots must be (almost) exactly on the day

# Max days between a quarter-end and the nearest available observation.
# If the gap is larger, the snapshot is skipped rather than silently wrong.
MAX_GAP_DAYS = 10

# --------------------------------------------------------------------------
# TOKEN CLASSIFICATION
# asset_class drives your Type A/B/C archetype assignment, so get this right.
# --------------------------------------------------------------------------
STABLECOINS = {
    "USDC", "USDT", "DAI", "BUSD", "TUSD", "FRAX", "LUSD", "SUSD", "GUSD",
    "USDP", "MIM", "USDD", "FEI", "USDE", "PYUSD", "CRVUSD", "USDS",
    "SUSDS", "GHO",
}

BLUECHIP = {
    "ETH", "WETH", "BTC", "WBTC", "STETH", "WSTETH", "RETH", "CBETH",
    "SETH2", "ANKRETH", "FRXETH", "SFRXETH", "WEETH",
}

# Map each protocol slug to its own governance token symbol.
# Used as a safety net in case the API bucket name is ambiguous.
NATIVE_TOKENS = {
    "aave": "AAVE",            "arbitrum-dao": "ARB",
    "badger-dao": "BADGER",    "bancor": "BNT",
    "beefy": "BIFI",           "compound-finance": "COMP",
    "convex-finance": "CVX",   "ens": "ENS",
    "gitcoin": "GTC",          "index-coop": "INDEX",
    "lido": "LDO",             "olympus-dao": "OHM",
    "ribbon-finance": "RBN",   "shapeshift": "FOX",
}

# CoinGecko ids for price lookup in script 01.
# Anything missing here is written to unmapped_tokens.csv for you to fill in.
COINGECKO_IDS = {
    "UNI": "uniswap",             "MKR": "maker",
    "AAVE": "aave",               "COMP": "compound-governance-token",
    "LDO": "lido-dao",            "CRV": "curve-dao-token",
    "OHM": "olympus",             "GTC": "gitcoin",
    "SNX": "havven",              "ENS": "ethereum-name-service",
    "FXS": "frax-share",          "YFI": "yearn-finance",
    "BAL": "balancer",            "ARB": "arbitrum",
    "USDC": "usd-coin",           "USDT": "tether",
    "DAI": "dai",                 "FRAX": "frax",
    "LUSD": "liquity-usd",        "TUSD": "true-usd",
    "BUSD": "binance-usd",        "SUSD": "nusd",
    "ETH": "ethereum",            "WETH": "weth",
    "WBTC": "wrapped-bitcoin",    "BTC": "bitcoin",
    "STETH": "staked-ether",      "WSTETH": "wrapped-steth",
    "RETH": "rocket-pool-eth",    "CBETH": "coinbase-wrapped-staked-eth",
    "LINK": "chainlink",          "MATIC": "matic-network",
    # native tokens of the final sample
    "CVX": "convex-finance",      "BADGER": "badger-dao",
    "BNT": "bancor",              "BIFI": "beefy-finance",
    "INDEX": "index-cooperative", "RBN": "ribbon-finance",
    "FOX": "shapeshift-fox-token", "GOHM": "governance-ohm",
    # other stablecoins seen in DAO treasuries
    "GHO": "gho",                 "USDS": "usds",
}


def classify(symbol, protocol_slug, from_own_bucket):
    """Assign asset_class. Own-token bucket always wins."""
    s = symbol.upper().strip()
    if from_own_bucket:
        return "native"
    if s == NATIVE_TOKENS.get(protocol_slug, "___NONE___"):
        return "native"
    if s in STABLECOINS:
        return "stablecoin"
    if s in BLUECHIP:
        return "bluechip"
    return "other"


def fetch_treasury(slug):
    url = f"{BASE}/treasury/{slug}"
    try:
        r = requests.get(url, timeout=45)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code} - protocol may not have treasury data")
            return None
        return r.json()
    except Exception as e:
        print(f"    Request failed: {e}")
        return None


def extract_token_series(data, slug):
    """
    Walk the chainTvls buckets and pull out (date, symbol, quantity, is_own).

    DefiLlama nests token quantities under:
        data['chainTvls'][<bucket>]['tokens'] = [{'date': unix, 'tokens': {...}}]

    Buckets whose name contains 'own' hold the native governance token, which
    DefiLlama excludes from headline treasury figures.
    """
    if not data or "chainTvls" not in data:
        return []

    records = []
    for bucket, content in data["chainTvls"].items():
        if not isinstance(content, dict):
            continue

        bucket_lower = bucket.lower()

        # Skip buckets that are not treasury holdings
        if any(x in bucket_lower for x in ("borrowed", "staking", "pool2", "vesting")):
            continue

        normalised = bucket_lower.replace("-", "").replace("_", "")
        is_own = "owntokens" in normalised

        # CRITICAL: DefiLlama returns BOTH a chain-prefixed own-tokens bucket
        # ('Ethereum-OwnTokens') AND a bare aggregate ('OwnTokens') holding the
        # SAME values. Including both double-counts every native token holding.
        # We keep the chain-prefixed versions and skip the bare aggregate.
        if is_own and normalised == "owntokens":
            continue

        series = content.get("tokens")
        if not isinstance(series, list):
            continue

        # DefiLlama's own USD valuation of each token, used ONLY for screening
        # (which tokens matter). Prices for the analysis still come from CoinGecko.
        usd_lookup = {}
        for upoint in content.get("tokensInUsd") or []:
            if isinstance(upoint, dict) and upoint.get("date") is not None:
                udate = pd.to_datetime(int(upoint["date"]), unit="s").normalize()
                for usym, uval in (upoint.get("tokens") or {}).items():
                    try:
                        usd_lookup[(udate, usym.upper().strip())] = float(uval)
                    except (TypeError, ValueError):
                        pass

        for point in series:
            if not isinstance(point, dict):
                continue
            ts = point.get("date")
            tokens = point.get("tokens", {})
            if ts is None or not isinstance(tokens, dict):
                continue
            date = pd.to_datetime(int(ts), unit="s").normalize()
            for symbol, qty in tokens.items():
                try:
                    qty = float(qty)
                except (TypeError, ValueError):
                    continue
                if qty <= 0:
                    continue
                sym = symbol.upper().strip()
                records.append({
                    "date": date,
                    "symbol": sym,
                    "quantity": qty,
                    "usd_value_defillama": usd_lookup.get((date, sym), 0.0),
                    "is_own": is_own,
                    "bucket": bucket,
                })

    return records


def find_token_map():
    """Look in data/raw first, then the scripts folder and Downloads, also
    accepting browser-renamed copies such as 'token_map (1).csv'."""
    import glob
    if os.path.exists(TOKEN_MAP_FILE):
        return TOKEN_MAP_FILE
    for folder in (RAW_DIR, SCRIPT_DIR, os.path.join(os.path.expanduser("~"), "Downloads")):
        hits = sorted(glob.glob(os.path.join(folder, "token_map*.csv")), key=os.path.getmtime)
        if hits:
            return hits[-1]          # newest copy
    return None


def load_token_map():
    """
    token_map.csv: one row per token (or per half of a liquidity-pool token).
      price_method = market      -> priced directly at its CoinGecko id
                     calibrated  -> valued at DefiLlama's USD value on each
                                    snapshot date, then moved day by day with
                                    the price of the coingecko_id given
                     exclude     -> dropped (spam tokens)
    A row with a protocol name applies only to that protocol and wins over a
    row with the protocol left blank.
    """
    path = find_token_map()
    if path is None:
        print("WARNING: token_map.csv not found - receipt and LP tokens stay unmapped.")
        print(f"         Put it here: {os.path.abspath(TOKEN_MAP_FILE)}")
        return {}
    print(f"Using token map: {path}")
    m = pd.read_csv(path, keep_default_na=False)
    m["asset_symbol"] = m["asset_symbol"].str.upper().str.strip()
    m["weight"] = pd.to_numeric(m["weight"]).fillna(1.0)
    out = {}
    for (sym, prot), g in m.groupby(["asset_symbol", "protocol"]):
        out[(sym, prot)] = g.to_dict("records")
    return out


TOKEN_MAP = load_token_map()


def map_token(symbol, display_name):
    return TOKEN_MAP.get((symbol, display_name)) or TOKEN_MAP.get((symbol, ""))


# --------------------------------------------------------------------------
# RULE-BASED MAPPING for lending receipt tokens that token_map.csv does not
# list one by one: Aave aTokens (aEthUSDC, aArbWETH, aPolDAI, aUSDT, ...),
# Compound cTokens (cUSDC, cUSDCv3, cETH), Spark spTokens, Yearn yvTokens and
# Fluid fTokens. Each is classed like the asset it is a receipt for, and
# valued at DefiLlama's USD value on the snapshot date (calibrated), moving
# with that asset's price in between.
# --------------------------------------------------------------------------
UNDERLYING = {
    # stablecoins -> their own price, so a USDC depeg only hits USDC receipts
    "USDC": ("stablecoin", "usd-coin"), "USDCN": ("stablecoin", "usd-coin"),
    "USDCE": ("stablecoin", "usd-coin"), "USDBC": ("stablecoin", "usd-coin"),
    "USDT": ("stablecoin", "tether"), "USDT0": ("stablecoin", "tether"),
    "DAI": ("stablecoin", "dai"), "SDAI": ("stablecoin", "dai"),
    "LUSD": ("stablecoin", "liquity-usd"), "FRAX": ("stablecoin", "frax"),
    "GHO": ("stablecoin", "gho"), "USDE": ("stablecoin", "ethena-usde"),
    "SUSDE": ("stablecoin", "ethena-usde"), "USDS": ("stablecoin", "usds"),
    "SUSDS": ("stablecoin", "usds"), "TUSD": ("stablecoin", "true-usd"),
    "BUSD": ("stablecoin", "binance-usd"), "SUSD": ("stablecoin", "nusd"),
    "PYUSD": ("stablecoin", "usd-coin"), "RLUSD": ("stablecoin", "usd-coin"),
    "CRVUSD": ("stablecoin", "usd-coin"), "GUSD": ("stablecoin", "usd-coin"),
    "USDP": ("stablecoin", "usd-coin"), "EURC": ("stablecoin", "euro-coin"),
    "EUROC": ("stablecoin", "euro-coin"), "EURS": ("stablecoin", "euro-coin"),
    # ETH and liquid staking tokens
    **{k: ("bluechip", "ethereum") for k in
       ("ETH", "WETH", "STETH", "WSTETH", "RETH", "CBETH", "WEETH", "OSETH", "ETHX",
        "RSETH", "EZETH", "FRXETH", "SFRXETH", "METH")},
    # BTC wrappers
    **{k: ("bluechip", "bitcoin") for k in ("BTC", "WBTC", "CBBTC", "TBTC", "LBTC", "BTCB")},
    # other crypto assets
    "AAVE": ("other", "aave"), "LINK": ("other", "chainlink"), "CRV": ("other", "curve-dao-token"),
    "BAL": ("other", "balancer"), "UNI": ("other", "uniswap"), "MKR": ("other", "maker"),
    "SNX": ("other", "havven"), "LDO": ("other", "lido-dao"), "ARB": ("other", "arbitrum"),
    "OP": ("other", "optimism"), "AVAX": ("other", "avalanche-2"), "WAVAX": ("other", "avalanche-2"),
    "MATIC": ("other", "matic-network"), "WMATIC": ("other", "matic-network"),
    "GNO": ("other", "gnosis"), "ENS": ("other", "ethereum-name-service"),
    "COMP": ("other", "compound-governance-token"), "CVX": ("other", "convex-finance"),
    "YFI": ("other", "yearn-finance"), "BNB": ("other", "binancecoin"),
    "WBNB": ("other", "binancecoin"), "FTM": ("other", "fantom"), "WFTM": ("other", "fantom"),
    "ENJ": ("other", "enjincoin"), "AMPL": ("other", "ampleforth"),
}
# longest prefixes first, so 'AETHLIDO' is tried before 'AETH' and 'A'
RECEIPT_PREFIXES = sorted(
    ["AETHLIDO", "AETHETHERFI", "AETH", "AARB", "AOPT", "ABAS", "APOL", "AAVA", "AGNO",
     "ASCR", "ABNB", "AZKS", "ALIN", "AMET", "ASON", "ACEL", "AM", "AV", "A",
     "C", "SP", "YV", "F"], key=len, reverse=True)


def rule_map(symbol, slug, from_own_bucket):
    """Return a token_map-style row for a recognisable receipt token, else None."""
    if symbol in COINGECKO_IDS:
        return None                      # priced directly already
    base = symbol[:-2] if symbol.endswith("V3") else symbol
    under = None
    if base in UNDERLYING:              # e.g. weETH, cbBTC, sDAI held directly
        under = base
    else:
        for pre in RECEIPT_PREFIXES:
            if base.startswith(pre) and base[len(pre):] in UNDERLYING:
                under = base[len(pre):]
                break
    if under is None:
        return None
    cls, cg = UNDERLYING[under]
    if from_own_bucket or under == NATIVE_TOKENS.get(slug):
        cls = "native"
    return [{"component": "", "weight": 1.0, "asset_class": cls,
             "coingecko_id": cg, "price_method": "calibrated"}]


def nearest_date(dates, target):
    if len(dates) == 0:
        return None, None
    gaps = (dates - target).abs()
    nearest = dates.iloc[gaps.values.argmin()]
    return nearest, abs((nearest - target).days)


def take_snapshots(records, slug, display_name, targets, max_gap, snapshot_type):
    """Pick the observation nearest each target date.

    The native-token buckets and the other buckets are matched to the target
    date SEPARATELY, because DefiLlama does not always record them on the same
    days (this is what left ENS and Olympus without a native token in some
    quarters). Snapshots further than max_gap days from any available
    observation are skipped loudly, never filled silently.
    """
    if not records:
        return []

    df = pd.DataFrame(records)
    # If a symbol appears in multiple buckets on the same date, sum it
    df = (df.groupby(["date", "symbol", "is_own"], as_index=False)
            [["quantity", "usd_value_defillama"]].sum())

    rows = []
    groups = {flag: pd.Series(sorted(df.loc[df["is_own"] == flag, "date"].unique()))
              for flag in (False, True)}

    for label, target in targets:
        main_date, main_gap = nearest_date(groups[False], target)
        own_date, own_gap = nearest_date(groups[True], target)
        main_ok = main_date is not None and main_gap <= max_gap
        own_ok = own_date is not None and own_gap <= max_gap
        if not main_ok and not own_ok:
            near = main_date.date() if main_date is not None else "none"
            print(f"    {label} ({target.date()}): no data within {max_gap} days "
                  f"(nearest {near}) - SKIPPED")
            continue
        picks = []
        if main_ok:
            picks.append((False, main_date, main_gap))
        elif len(groups[False]):
            near = main_date.date() if main_date is not None else "none"
            print(f"    {label}: non-native buckets have no data within {max_gap} days "
                  f"(nearest {near}) - snapshot uses native holdings only")
        if own_ok:
            picks.append((True, own_date, own_gap))
        elif len(groups[True]):
            near = own_date.date() if own_date is not None else "none"
            print(f"    {label}: native-token bucket has no data within {max_gap} days "
                  f"(nearest {near}) - add it via manual_corrections.csv")

        for flag, d, gap_days in picks:
            snap = df[(df["date"] == d) & (df["is_own"] == flag)]
            for _, r in snap.iterrows():
                symbol = r["symbol"]
                base = {
                    "protocol": display_name,
                    "snapshot_type": snapshot_type,
                    "snapshot_label": label,
                    "snapshot_date": target.date(),
                    "source": "DefiLlama API",
                    "notes": f"obs {d.date()} (gap {gap_days}d)",
                }
                mapped = (map_token(symbol, display_name)
                          or rule_map(symbol, slug, r["is_own"]))
                if mapped:
                    if mapped[0]["price_method"] == "exclude":
                        continue
                    for m in mapped:
                        comp = m["component"]
                        rows.append({**base,
                            "asset_symbol": f"{symbol}[{comp}]" if comp else symbol,
                            "coingecko_id": m["coingecko_id"],
                            "price_method": m["price_method"],
                            "quantity": r["quantity"],
                            "usd_value_defillama": round(r["usd_value_defillama"] * m["weight"], 2),
                            "asset_class": m["asset_class"],
                        })
                else:
                    cg = COINGECKO_IDS.get(symbol, "")
                    rows.append({**base,
                        "asset_symbol": symbol,
                        "coingecko_id": cg,
                        "price_method": "market" if cg else "",
                        "quantity": r["quantity"],
                        "usd_value_defillama": round(r["usd_value_defillama"], 2),
                        "asset_class": classify(symbol, slug, r["is_own"]),
                    })

    cols = ["protocol", "snapshot_type", "snapshot_label", "snapshot_date", "asset_symbol",
            "coingecko_id", "price_method", "quantity", "usd_value_defillama",
            "asset_class", "source", "notes"]
    return [{c: r[c] for c in cols} for r in rows]


def main():
    all_rows, event_rows = [], []
    quarter_targets = [(f"Q{qe.quarter}-{qe.year}", qe) for qe in QUARTER_ENDS]
    event_targets = list(EVENT_SNAPSHOTS.items())

    for slug, display_name in PROTOCOLS.items():
        print(f"\n{display_name} ({slug})")
        data = fetch_treasury(slug)

        if data is None:
            print("    FAILED - collect this protocol manually")
            continue

        records = extract_token_series(data, slug)
        if not records:
            print("    No token-level data found in response.")
            print("    Run 00_explore_defillama.py on this slug to inspect it.")
            continue

        buckets = sorted({r["bucket"] for r in records})
        own_buckets = [b for b in buckets if "own" in b.lower()]
        print(f"    Buckets used: {buckets}")
        if own_buckets:
            print(f"    Native-token bucket found: {own_buckets}")
        else:
            print("    WARNING: no 'own tokens' bucket found.")
            print(f"    Falling back to symbol match on "
                  f"'{NATIVE_TOKENS.get(slug, '?')}'. VERIFY THIS MANUALLY.")

        rows = take_snapshots(records, slug, display_name, quarter_targets,
                              MAX_GAP_DAYS, "quarterly")
        all_rows.extend(rows)
        n_quarters = len({r['snapshot_date'] for r in rows})
        flag = "" if n_quarters == len(QUARTER_ENDS) else "   <-- INCOMPLETE, check"
        print(f"    Quarterly: {n_quarters}/{len(QUARTER_ENDS)} snapshots{flag}")

        ev = take_snapshots(records, slug, display_name, event_targets,
                            EVENT_MAX_GAP_DAYS, "event")
        event_rows.extend(ev)
        got = sorted({r['snapshot_label'] for r in ev})
        print(f"    Event snapshots: {got if got else 'none'}")

        time.sleep(2)

    if not all_rows:
        print("\nNo data extracted. Run 00_explore_defillama.py to diagnose.")
        return

    out = pd.DataFrame(all_rows)
    out.to_csv(OUTPUT_FILE, index=False)
    if event_rows:
        pd.DataFrame(event_rows).to_csv(EVENT_FILE, index=False)
        print(f"\nWrote {len(event_rows)} event-snapshot rows to {EVENT_FILE}")

    print("\n" + "=" * 70)
    print(f"Wrote {len(out)} rows to {OUTPUT_FILE}")
    print("=" * 70)

    print("\nRows per protocol:")
    print(out.groupby("protocol").size().to_string())

    print("\nAsset class distribution:")
    print(out["asset_class"].value_counts().to_string())

    # ------------------------------------------------------------------
    # USD-WEIGHTED CHECKS (row counts alone are misleading: one dust token
    # = one row, same as a $500m position)
    # ------------------------------------------------------------------
    out["snap_total"] = out.groupby(["protocol", "snapshot_date"])["usd_value_defillama"].transform("sum")
    out["share"] = (out["usd_value_defillama"] / out["snap_total"]).fillna(0)

    print("\n" + "=" * 70)
    print("SNAPSHOTS AND NATIVE COVERAGE PER PROTOCOL")
    print("=" * 70)
    cov = pd.DataFrame({
        "snapshots": out.groupby("protocol")["snapshot_date"].nunique(),
        "snapshots_with_native": out[out["asset_class"] == "native"]
                                   .groupby("protocol")["snapshot_date"].nunique(),
    }).fillna(0).astype(int)
    cov["flag"] = ["" if (a == len(QUARTER_ENDS) and b == a) else "<-- CHECK"
                   for a, b in zip(cov["snapshots"], cov["snapshots_with_native"])]
    print(cov.to_string())

    print("\n" + "=" * 70)
    print("MEAN USD SHARE BY ASSET CLASS (DefiLlama valuation, % of treasury)")
    print("=" * 70)
    shares = (out.groupby(["protocol", "snapshot_date", "asset_class"])["share"].sum()
                 .unstack(fill_value=0)            # absent class = 0% that quarter
                 .groupby("protocol").mean() * 100)
    for c in ("native", "stablecoin", "bluechip", "other"):
        if c not in shares.columns:
            shares[c] = 0.0
    shares = shares[["native", "stablecoin", "bluechip", "other"]]
    shares["prelim_type"] = ["A" if n >= 70 else "C" if st >= 40 else "B"
                             for n, st in zip(shares["native"], shares["stablecoin"])]
    print(shares.round(1).to_string())
    print("\nprelim_type uses DefiLlama's own prices; the final archetypes come")
    print("from 02_build_nav.py. 'other' should now be small for most protocols.")

    # Material unmapped / 'other' tokens: >= 1% of a treasury at any snapshot
    # (the same 1% cut-off Schellinger et al. 2023 used).
    out["unmapped"] = out["coingecko_id"].fillna("") == ""
    review = out[out["unmapped"]]
    material = (review.groupby(["asset_symbol"])
                      .agg(max_share_pct=("share", lambda x: round(x.max() * 100, 2)),
                           protocols=("protocol", lambda x: ", ".join(sorted(set(x)))),
                           has_coingecko_id=("unmapped", lambda x: "no" if x.all() else "yes"),
                           current_class=("asset_class", "first"))
                      .reset_index())
    material = material[material["max_share_pct"] >= 1.0].sort_values("max_share_pct", ascending=False)
    material.to_csv(UNMAPPED_LOG, index=False)
    n_immaterial = review["asset_symbol"].nunique() - len(material)

    print("\n" + "=" * 70)
    print("TOKENS STILL WITHOUT A PRICE SOURCE (>= 1% of a treasury)")
    print("=" * 70)
    if material.empty:
        print("None - every material token has a price source.")
    else:
        print(material.to_string(index=False))
    print(f"\n{n_immaterial} unpriced tokens are each below 1% of every treasury and")
    print("can be left out (document this as a 1% materiality cut-off).")
    print(f"Saved to {UNMAPPED_LOG}")

    out.drop(columns=["snap_total", "share", "unmapped"]).to_csv(OUTPUT_FILE, index=False)

    print("\nNEXT STEPS")
    print("-" * 70)
    print("1. Paste the three tables above back so the mapping can be checked.")
    print("2. Spot-check 3 protocol-quarters against defillama.com by hand.")
    print("3. Then run 01_fetch_prices.py, 02_build_nav.py, 03_compute_metrics.py.")


if __name__ == "__main__":
    main()
