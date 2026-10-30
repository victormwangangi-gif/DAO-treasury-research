"""
STEP 0b (CRITICAL DIAGNOSTIC): Check data coverage before you commit.

WHY THIS EXISTS
---------------
The Uniswap treasury history on DefiLlama starts 2023-02-04. The study's
stress window is May-November 2022. If that gap applies across the board,
the empirical core of the paper cannot be answered from this data source.

This script checks EVERY candidate protocol and reports:
  - the first and last date of treasury history
  - whether the 2022 stress window is covered
  - whether the 2021 study start is covered
  - the size of the non-native treasury (to spot empty/dust buckets)
  - how many distinct tokens are actually tracked

Run this BEFORE collecting anything. It takes a few minutes and will tell you
whether your study period is feasible as designed.

HOW TO RUN
----------
    python 00b_check_coverage.py

Output: printed table plus data/raw/coverage_report.csv
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone

BASE = "https://api.llama.fi"
OUTPUT = "../data/raw/coverage_report.csv"

STUDY_START  = pd.Timestamp("2021-01-01")
STRESS_START = pd.Timestamp("2022-05-01")
STRESS_END   = pd.Timestamp("2022-11-30")

# Candidate protocols. Slugs verified against the /protocols endpoint.
# Add or remove as you like - this is a screening exercise.
CANDIDATES = [
    "uniswap", "uniswap-v3", "uniswap-labs",
    "aave", "aave-v2", "aave-v3",
    "makerdao", "sky",
    "compound-finance", "compound-v2",
    "lido",
    "curve-dex", "curve-finance",
    "olympus-dao",
    "gitcoin",
    "ens",
    "arbitrum-dao",
    "synthetix-v1+v2", "synthetix-v3",
    "yearn-finance",
    "balancer-v2",
    "frax",
    "rocket-pool",
    "gnosis",
    "1inch-network",
    "sushiswap",
    "dydx",
    "instadapp",
    "bancor",
]


def analyse(slug):
    """Fetch one protocol and summarise its treasury data coverage."""
    try:
        r = requests.get(f"{BASE}/treasury/{slug}", timeout=45)
    except Exception as e:
        return {"slug": slug, "status": f"request failed: {e}"}

    if r.status_code != 200:
        return {"slug": slug, "status": f"HTTP {r.status_code}"}

    try:
        data = r.json()
    except Exception:
        return {"slug": slug, "status": "bad JSON"}

    if "chainTvls" not in data:
        return {"slug": slug, "status": "no chainTvls key"}

    own_dates, other_dates = [], []
    own_usd_latest, other_usd_latest = 0.0, 0.0
    other_tokens = set()
    own_tokens = set()

    for bucket, content in data["chainTvls"].items():
        if not isinstance(content, dict):
            continue
        bl = bucket.lower()
        if any(x in bl for x in ("borrowed", "staking", "pool2", "vesting")):
            continue

        is_own = "owntokens" in bl.replace("-", "").replace("_", "")

        # NOTE: DefiLlama returns BOTH 'OwnTokens' and 'Ethereum-OwnTokens'
        # which duplicate each other. Count the chain-prefixed one only,
        # to avoid double counting.
        if is_own and bl == "owntokens":
            continue

        series = content.get("tokensInUsd")
        if not isinstance(series, list) or not series:
            continue

        dates = [pd.to_datetime(int(p["date"]), unit="s")
                 for p in series if isinstance(p, dict) and "date" in p]

        last = series[-1]
        toks = last.get("tokens", {}) if isinstance(last, dict) else {}
        total = sum(v for v in toks.values() if isinstance(v, (int, float)))

        if is_own:
            own_dates += dates
            own_usd_latest += total
            own_tokens |= set(toks.keys())
        else:
            other_dates += dates
            other_usd_latest += total
            other_tokens |= set(toks.keys())

    all_dates = own_dates + other_dates
    if not all_dates:
        return {"slug": slug, "status": "no time series data"}

    first, last = min(all_dates), max(all_dates)

    covers_stress = first <= STRESS_START and last >= STRESS_END
    covers_2021   = first <= STUDY_START

    total_usd = own_usd_latest + other_usd_latest
    native_pct = (own_usd_latest / total_usd) if total_usd > 0 else float("nan")

    return {
        "slug": slug,
        "status": "OK",
        "first_date": first.date(),
        "last_date": last.date(),
        "covers_2021_start": covers_2021,
        "covers_2022_stress": covers_stress,
        "n_nonnative_tokens": len(other_tokens),
        "latest_native_usd": round(own_usd_latest, 0),
        "latest_nonnative_usd": round(other_usd_latest, 0),
        "latest_native_pct": round(native_pct, 3) if total_usd > 0 else None,
    }


def main():
    print(f"Checking {len(CANDIDATES)} protocols...\n")
    rows = []
    for i, slug in enumerate(CANDIDATES, 1):
        print(f"[{i}/{len(CANDIDATES)}] {slug}", end=" ... ", flush=True)
        res = analyse(slug)
        rows.append(res)
        if res.get("status") == "OK":
            flag = ""
            if not res["covers_2022_stress"]:
                flag += " [NO 2022 STRESS DATA]"
            if res["latest_nonnative_usd"] < 100_000:
                flag += " [NON-NATIVE TREASURY IS DUST]"
            print(f"{res['first_date']} to {res['last_date']}{flag}")
        else:
            print(res["status"])
        time.sleep(1.5)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)

    ok = df[df["status"] == "OK"].copy()

    print("\n" + "=" * 78)
    print("COVERAGE REPORT")
    print("=" * 78)
    if ok.empty:
        print("No protocols returned usable data.")
        return

    show = ok[["slug", "first_date", "last_date", "covers_2022_stress",
               "n_nonnative_tokens", "latest_native_pct"]]
    print(show.to_string(index=False))

    n_stress = int(ok["covers_2022_stress"].sum())
    n_2021   = int(ok["covers_2021_start"].sum())
    n_dust   = int((ok["latest_nonnative_usd"] < 100_000).sum())

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"Protocols returning data:              {len(ok)} / {len(CANDIDATES)}")
    print(f"Covering the 2022 stress window:       {n_stress}")
    print(f"Covering the 2021 study start:         {n_2021}")
    print(f"With dust-only non-native treasury:    {n_dust}")

    print()
    if n_stress >= 12:
        print("GOOD: enough protocols cover the 2022 stress window.")
        print("Proceed with the study period as designed. Restrict your sample")
        print("to protocols where covers_2022_stress is True.")
    elif n_stress >= 6:
        print("MARGINAL: some coverage of the 2022 window, but likely too few")
        print("protocols to split into three archetypes with adequate group")
        print("sizes. Read the SAMPLE SIZE WARNING in START_HERE.md.")
        print("Consider supplementing 2021-22 from Dune Analytics.")
    else:
        print("PROBLEM: DefiLlama does not cover the 2022 stress window for")
        print("enough protocols. Your study period cannot be answered from")
        print("this source alone. See 'IF COVERAGE IS BAD' in START_HERE.md")
        print("for the three options: switch stress event, source 2021-22 from")
        print("Dune, or narrow the study period.")

    if n_dust > 0:
        print()
        print(f"WARNING: {n_dust} protocols show under $100k in non-native")
        print("assets. For those, DefiLlama is likely tracking an incomplete")
        print("or wrong treasury address. Verify against the protocol's own")
        print("governance forum before including them.")
        print(ok[ok["latest_nonnative_usd"] < 100_000]
              [["slug", "latest_native_usd", "latest_nonnative_usd"]]
              .to_string(index=False))

    print(f"\nFull report saved to {OUTPUT}")


if __name__ == "__main__":
    main()
