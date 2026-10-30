"""
STEP 1: Fetch daily price history for every asset in your treasury data.

WHAT THIS DOES
--------------
Reads the two files written by 01b_defillama_extract.py
(treasury_composition.csv and event_snapshots.csv), finds every unique
CoinGecko id, downloads a daily USD price for each one from Jan 2023 to
Sep 2026, and saves them to data/processed/prices.csv

HOW TO RUN (VS Code terminal)
-----------------------------
    python 01_fetch_prices.py

WHERE THE PRICES COME FROM
--------------------------
CoinGecko's free (Demo) API only returns the LAST 365 DAYS of history, which
is not enough for a 2023-2026 study. So this script uses, in order:

  1. DefiLlama's free coins API (coins.llama.fi). No key needed. It looks
     tokens up by their CoinGecko id, so the ids in your data still work.
  2. Yahoo Finance (yfinance) as a fallback for major tokens.
     Install with: python -m pip install yfinance
  3. For stablecoins only, and only if both fail: an assumed $1.00 peg,
     with a loud warning (it would hide the March 2023 USDC depeg).

At the end it prints a COVERAGE table showing, for every token, the first and
last date it has a price and how many days are missing.
"""

import os
import json
import time
import requests
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

START_DATE = "2023-01-01"     # covers the 21-day baseline before the Mar 2023 USDC event
END_DATE   = "2026-09-30"     # last day of the main analysis period

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
INPUT_FILES = [os.path.join(RAW_DIR, "treasury_composition.csv"),
               os.path.join(RAW_DIR, "event_snapshots.csv")]
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "prices.csv")
COVERAGE_FILE = os.path.join(PROCESSED_DIR, "price_coverage.csv")


# CoinGecko id -> Yahoo Finance ticker, used only if DefiLlama fails.
YAHOO_FALLBACK = {
    "ethereum": "ETH-USD", "weth": "ETH-USD", "bitcoin": "BTC-USD",
    "wrapped-bitcoin": "BTC-USD", "aave": "AAVE-USD",
    "maker": "MKR-USD",
    "curve-dao-token": "CRV-USD", "lido-dao": "LDO-USD", "havven": "SNX-USD",
    "chainlink": "LINK-USD", "arbitrum": "ARB-USD",
    "ethereum-name-service": "ENS-USD",
    "convex-finance": "CVX-USD",
    "uniswap": "UNI7083-USD", "compound-governance-token": "COMP5692-USD",
    "gitcoin": "GTC10052-USD", "optimism": "OP-USD", "solana": "SOL-USD",
    "avalanche-2": "AVAX-USD", "binancecoin": "BNB-USD", "staked-ether": "STETH-USD",
    "bancor": "BNT-USD", "badger-dao": "BADGER-USD",
    "usd-coin": "USDC-USD", "tether": "USDT-USD", "dai": "DAI-USD",
}

# Stablecoins are fetched at MARKET prices like every other asset. A flat $1
# series is used only as a last resort, because it would erase the March 2023
# USDC depeg (USDC traded near $0.86), one of the paper's two event studies.
STABLECOINS = {"usd-coin", "tether", "dai", "frax", "true-usd", "liquity-usd",
               "usds", "gho", "paypal-usd", "first-digital-usd", "crvusd"}


# ----------------------------------------------------------------------------
# FETCHING FUNCTIONS
# ----------------------------------------------------------------------------

class BadRequest(Exception):
    pass


_SHOWN_400 = []


def get_with_retry(url, params=None, tries=4):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:                 # rate limited - wait and retry
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 400:
                if not _SHOWN_400:                   # show the server's reason once
                    _SHOWN_400.append(1)
                    print(f"    HTTP 400 - server says: {r.text[:200]}")
                raise BadRequest()
            print(f"    HTTP {r.status_code} from {url[:70]}")
            return None
        except BadRequest:
            raise
        except Exception as e:
            print(f"    Request error: {e}")
            time.sleep(5)
    return None


LLAMA = "https://coins.llama.fi"
DAY = 86400


def _to_frame(pts, start_date, end_date):
    if not pts:
        return None
    df = pd.DataFrame(pts, columns=["ts", "price"])
    # DefiLlama returns the price recorded NEAREST to each requested midnight,
    # which is often a few minutes before it (e.g. 23:55 the previous day).
    # Rounding to the nearest midnight puts each price on the day requested;
    # cutting the time off instead would shift it a day back and leave gaps.
    df["date"] = pd.to_datetime((df["ts"].astype("int64") + DAY // 2) // DAY * DAY, unit="s")
    df = df.groupby("date", as_index=False)["price"].last()
    df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
    return df[["date", "price"]] if len(df) else None


def llama_chart(ids, start_date, end_date, span):
    """/chart endpoint: one request per chunk of `span` days."""
    start = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    stop = int(pd.Timestamp(end_date, tz="UTC").timestamp())
    pts = {cid: [] for cid in ids}
    t = start
    while t <= stop:
        n = min(span, (stop - t) // DAY + 1)
        keys = ",".join(f"coingecko:{c}" for c in ids)
        data = get_with_retry(f"{LLAMA}/chart/{keys}",
                              {"start": t, "span": n, "period": "1d", "searchWidth": "12h"})
        for key, info in (data or {}).get("coins", {}).items():
            pts.setdefault(key.split(":", 1)[1], []).extend(
                (p["timestamp"], p["price"]) for p in info.get("prices", []))
        t += n * DAY
        time.sleep(0.5)
    return pts


def llama_batch_historical(ids, start_date, end_date, chunk=60):
    """/batchHistorical endpoint: a list of daily timestamps per token."""
    days = pd.date_range(start_date, end_date, freq="D", tz="UTC")
    pts = {cid: [] for cid in ids}
    for i in range(0, len(days), chunk):
        stamps = [int(d.timestamp()) for d in days[i:i + chunk]]
        coins = json.dumps({f"coingecko:{c}": stamps for c in ids}, separators=(",", ":"))
        data = get_with_retry(f"{LLAMA}/batchHistorical", {"coins": coins, "searchWidth": "12h"})
        for key, info in (data or {}).get("coins", {}).items():
            pts.setdefault(key.split(":", 1)[1], []).extend(
                (p["timestamp"], p["price"]) for p in info.get("prices", []))
        time.sleep(0.5)
    return pts


# Ways of asking DefiLlama, tried in order until one works:
# (description, function, tokens per request)
# DefiLlama's limit: at most 500 data points (tokens x days) per request.
LLAMA_METHODS = [
    ("chart, 5 tokens x 100 days",  lambda ids, a, b: llama_chart(ids, a, b, 100), 5),
    ("chart, 1 token x 180 days",   lambda ids, a, b: llama_chart(ids, a, b, 180), 1),
    ("chart, 1 token x 30 days",    lambda ids, a, b: llama_chart(ids, a, b, 30), 1),
    ("batchHistorical, 5 tokens",   lambda ids, a, b: llama_batch_historical(ids, a, b), 5),
    ("batchHistorical, 1 token",    lambda ids, a, b: llama_batch_historical(ids, a, b, 30), 1),
]


def pick_llama_method():
    """Try each method on a 200-day sample of up to 5 tokens; return the index of the
    first that works. 200 days is longer than any request chunk, so a method
    that passes the test also works for the real downloads."""
    a, b = "2024-01-01", "2024-07-18"
    for i, (desc, fn, batch) in enumerate(LLAMA_METHODS):
        try:
            pts = fn(["ethereum", "usd-coin", "tether", "dai", "bitcoin"][:batch], a, b)
            if any(len(v) >= 150 for v in pts.values()):
                print(f"  DefiLlama works with: {desc}")
                return i
        except BadRequest:
            continue
    return None


def fetch_from_defillama(coin_ids, start_date, end_date):
    """
    Daily prices for many CoinGecko ids from DefiLlama's free coins API.
    Returns {coin_id: DataFrame(date, price)}. Ids it cannot find are left out.
    """
    first = pick_llama_method()
    if first is None:
        print("  DefiLlama refused every request format - skipping to Yahoo.")
        return {}
    _, fn, batch = LLAMA_METHODS[first]
    out = {}
    for i in range(0, len(coin_ids), batch):
        group = coin_ids[i:i + batch]
        print(f"  {', '.join(group)}")
        try:
            pts = fn(group, start_date, end_date)
        except BadRequest:
            pts = {}
            for cid in group:               # retry this token alone, with each
                for desc, f2, _b in LLAMA_METHODS[first:]:   # remaining method
                    try:
                        pts.update(f2([cid], start_date, end_date))
                        break
                    except BadRequest:
                        continue
                else:
                    print(f"    {cid}: not recognised by DefiLlama")
        for cid in group:
            df = _to_frame(pts.get(cid, []), start_date, end_date)
            if df is not None:
                out[cid] = df
    return out


def fetch_from_yahoo(coin_id, start_date, end_date):
    """Fallback: fetch from Yahoo Finance. Returns DataFrame or None."""
    ticker = YAHOO_FALLBACK.get(coin_id)
    if ticker is None:
        return None
    try:
        import yfinance as yf
    except ImportError:
        print("    yfinance not installed. Run: python -m pip install yfinance")
        return None
    try:
        end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start_date, end=end_plus,
                           progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        close = data["Close"]
        if hasattr(close, "columns"):        # newer yfinance returns a DataFrame
            close = close.iloc[:, 0]
        df = close.reset_index()
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        return df[["date", "price"]]
    except Exception as e:
        print(f"    Yahoo error: {e}")
        return None


def make_stablecoin_series(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return pd.DataFrame({"date": dates, "price": 1.0})


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    frames_in = [pd.read_csv(f) for f in INPUT_FILES if os.path.exists(f)]
    if not frames_in:
        print(f"ERROR: cannot find {INPUT_FILES[0]}")
        print("Run 01b_defillama_extract.py first.")
        return
    comp = pd.concat(frames_in, ignore_index=True)
    coin_ids = sorted(comp["coingecko_id"].dropna().astype(str).unique())
    print(f"Found {len(coin_ids)} unique assets to fetch.\n")

    results, sources = {}, {}

    print("Source 1: DefiLlama coins API")
    for cid, df in fetch_from_defillama(coin_ids, START_DATE, END_DATE).items():
        results[cid], sources[cid] = df, "defillama"

    missing = [c for c in coin_ids if c not in results]
    if missing:
        print(f"\nSource 2: Yahoo Finance for {len(missing)} missing assets")
    for cid in missing:
        df = fetch_from_yahoo(cid, START_DATE, END_DATE)
        if df is not None:
            results[cid], sources[cid] = df, "yahoo"
            print(f"  {cid}: OK ({len(df)} days)")
        elif cid in STABLECOINS:
            results[cid], sources[cid] = make_stablecoin_series(START_DATE, END_DATE), "ASSUMED_1USD"
            print(f"  *** {cid}: no market prices found - using an assumed $1.00 peg.")
            print("      This hides any depeg. Do NOT rely on it for the March 2023 event.")
        else:
            print(f"  *** {cid}: FAILED - no price source found")

    if not results:
        print("No price data retrieved. Check your internet connection.")
        return

    prices = pd.concat([df.assign(coingecko_id=cid) for cid, df in results.items()],
                       ignore_index=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    prices.to_csv(OUTPUT_FILE, index=False)

    # Coverage report
    full_days = len(pd.date_range(START_DATE, END_DATE, freq="D"))
    cov = (prices.groupby("coingecko_id")
                 .agg(first_date=("date", "min"), last_date=("date", "max"),
                      days=("date", "nunique"))
                 .reindex(coin_ids))
    cov["missing_days"] = full_days - cov["days"].fillna(0).astype(int)
    cov["source"] = [sources.get(c, "NONE") for c in cov.index]
    # gaps INSIDE the token's own history (late launches are not gaps)
    span_days = (pd.to_datetime(cov["last_date"]) - pd.to_datetime(cov["first_date"])).dt.days + 1
    cov["gaps_inside"] = (span_days - cov["days"]).fillna(-1).astype(int)
    cov.to_csv(COVERAGE_FILE)

    print("\n" + "=" * 70)
    print(f"Saved {len(prices):,} price rows to {OUTPUT_FILE}")
    print(f"Assets with prices: {len(results)} of {len(coin_ids)}")
    print("=" * 70)
    print("\nPRICE COVERAGE (only assets with gaps or problems are listed)")
    late = (pd.to_datetime(cov["first_date"]) > pd.Timestamp(START_DATE) + pd.Timedelta(days=3)) | \
           (pd.to_datetime(cov["last_date"]) < pd.Timestamp(END_DATE) - pd.Timedelta(days=3))
    problems = cov[(cov["gaps_inside"] > 5) | late | (cov["source"] != "defillama")]
    if problems.empty:
        print("  None - every asset has a near-complete daily series.")
    else:
        print(problems.to_string())
        print("\n  A token that starts late (e.g. launched in 2024) is fine if the")
        print("  treasury did not hold it before then. FAILED/NONE rows need fixing.")
    print(f"\nFull table saved to {COVERAGE_FILE}")


if __name__ == "__main__":
    main()
