"""
STEP 0c (DECISION TOOL): Which study period is actually feasible?

WHY THIS EXISTS
---------------
The coverage check found that DefiLlama's treasury history begins 2023-02-04
for most protocols. Only 2 of 29 cover the May-November 2022 stress window.
That window is therefore dead on this data source.

This script tests several CANDIDATE stress windows and reports how many
protocols cover each one, so you can choose a study period on evidence rather
than guesswork - and justify that choice to your supervisor.

It also fixes the two problems from the last run:
  - creates output directories automatically (no more path crash)
  - uses only slugs that actually return data

HOW TO RUN
----------
    python 00c_feasible_window.py

You can run it from anywhere. It writes to a 'dao_output' folder next to
the script.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

BASE = "https://api.llama.fi"

# Write next to this script, whatever folder you run it from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "dao_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CANDIDATE STRESS WINDOWS
# Each needs a pre-event baseline, so we also require coverage starting
# `baseline_days` before the window opens.
# ---------------------------------------------------------------------------
CANDIDATE_WINDOWS = {
    "2022 Terra/Luna + FTX (original plan)": {
        "start": "2022-05-01", "end": "2022-11-30", "baseline_days": 120,
        "note": "Original design. General crypto bear market."
    },
    "Mar 2023 USDC depeg (SVB collapse)": {
        "start": "2023-03-08", "end": "2023-03-20", "baseline_days": 21,
        "note": "USDC fell to ~$0.87. Directly tests the Tier 1 stablecoin thesis."
    },
    "Aug 2024 yen carry unwind": {
        "start": "2024-08-01", "end": "2024-09-10", "baseline_days": 120,
        "note": "Sharp global risk-off. Broad crypto drawdown."
    },
    "Apr 2025 drawdown": {
        "start": "2025-04-01", "end": "2025-05-15", "baseline_days": 120,
        "note": "Macro-driven risk-off episode."
    },
}

# Slugs confirmed to return data, plus extras worth testing.
# The ones that returned HTTP 400 last run have been removed or corrected.
CANDIDATES = [
    # confirmed working
    "uniswap", "aave", "sky", "compound-finance", "lido",
    "olympus-dao", "gitcoin", "ens", "arbitrum-dao", "bancor",
    "dydx", "curve-finance",
    # worth testing - alternative slug spellings
    "optimism", "polygon", "safe", "lens-protocol", "radiant-capital",
    "stargate", "pancakeswap", "velodrome", "aerodrome", "jupiter-exchange",
    "ethena", "pendle", "eigenlayer", "morpho", "spark", "across",
    "connext", "hop-protocol", "beefy", "convex-finance", "frax-ether",
    "rocket-pool-eth", "threshold-network", "nouns-dao", "shapeshift",
    "badger-dao", "alchemix", "tokemak", "ribbon-finance", "index-coop",
]

DUST_THRESHOLD_USD = 100_000


def analyse(slug):
    try:
        r = requests.get(f"{BASE}/treasury/{slug}", timeout=45)
    except Exception as e:
        return {"slug": slug, "status": f"error: {type(e).__name__}"}

    if r.status_code != 200:
        return {"slug": slug, "status": f"HTTP {r.status_code}"}

    try:
        data = r.json()
    except Exception:
        return {"slug": slug, "status": "bad JSON"}

    if not isinstance(data, dict) or "chainTvls" not in data:
        return {"slug": slug, "status": "no chainTvls"}

    own_dates, other_dates = [], []
    own_usd, other_usd = 0.0, 0.0
    other_tokens = set()

    for bucket, content in data["chainTvls"].items():
        if not isinstance(content, dict):
            continue
        bl = bucket.lower()
        if any(x in bl for x in ("borrowed", "staking", "pool2", "vesting")):
            continue

        norm = bl.replace("-", "").replace("_", "")
        is_own = "owntokens" in norm
        # skip the bare aggregate that duplicates chain-prefixed own buckets
        if is_own and norm == "owntokens":
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
            own_usd += total
        else:
            other_dates += dates
            other_usd += total
            other_tokens |= set(toks.keys())

    all_dates = own_dates + other_dates
    if not all_dates:
        return {"slug": slug, "status": "no series data"}

    first, last = min(all_dates), max(all_dates)
    total_usd = own_usd + other_usd

    row = {
        "slug": slug,
        "status": "OK",
        "first_date": first,
        "last_date": last,
        "n_nonnative_tokens": len(other_tokens),
        "native_usd": round(own_usd),
        "nonnative_usd": round(other_usd),
        "native_pct": round(own_usd / total_usd, 3) if total_usd > 0 else None,
        "is_dust": other_usd < DUST_THRESHOLD_USD,
    }

    for name, w in CANDIDATE_WINDOWS.items():
        need_from = pd.Timestamp(w["start"]) - pd.Timedelta(days=w["baseline_days"])
        need_to = pd.Timestamp(w["end"])
        row[name] = bool(first <= need_from and last >= need_to)

    return row


def main():
    print(f"Testing {len(CANDIDATES)} protocols against "
          f"{len(CANDIDATE_WINDOWS)} candidate stress windows.\n")

    rows = []
    for i, slug in enumerate(CANDIDATES, 1):
        print(f"[{i:>2}/{len(CANDIDATES)}] {slug:<24}", end="", flush=True)
        res = analyse(slug)
        rows.append(res)
        if res.get("status") == "OK":
            dust = "  [DUST]" if res["is_dust"] else ""
            print(f"{res['first_date'].date()} -> {res['last_date'].date()}{dust}")
        else:
            print(res["status"])
        time.sleep(1.2)

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "OK"].copy()

    path = os.path.join(OUT_DIR, "feasible_window_report.csv")
    df.to_csv(path, index=False)

    if ok.empty:
        print("\nNo protocols returned data.")
        return

    # Failed rows leave NaN in the boolean columns, which makes the whole
    # column object-dtype. In pandas 3.x, `~` on an object column does
    # INTEGER bitwise negation (~True -> -2), which then breaks indexing.
    # Cast explicitly before any boolean operation.
    for col in ["is_dust", *CANDIDATE_WINDOWS.keys()]:
        if col in ok.columns:
            ok[col] = ok[col].fillna(False).astype(bool)

    usable = ok[~ok["is_dust"]].copy()

    print("\n" + "=" * 78)
    print("HOW MANY PROTOCOLS COVER EACH CANDIDATE WINDOW")
    print("=" * 78)
    print(f"(excluding {len(ok) - len(usable)} protocols whose non-native "
          f"treasury is dust)\n")

    summary = []
    for name, w in CANDIDATE_WINDOWS.items():
        n = int(usable[name].sum()) if name in usable.columns else 0
        summary.append({"window": name, "n_protocols": n, "note": w["note"]})
        verdict = ("VIABLE" if n >= 15 else
                   "MARGINAL" if n >= 9 else
                   "NOT VIABLE")
        print(f"  {n:>3} protocols   {verdict:<11}  {name}")
        print(f"                              {w['note']}")
        print()

    pd.DataFrame(summary).to_csv(
        os.path.join(OUT_DIR, "window_summary.csv"), index=False)

    print("=" * 78)
    print("PROTOCOLS BY WINDOW")
    print("=" * 78)
    for name in CANDIDATE_WINDOWS:
        if name not in usable.columns:
            continue
        names = sorted(usable[usable[name]]["slug"].tolist())
        print(f"\n{name}  ({len(names)})")
        print("  " + (", ".join(names) if names else "none"))

    # Archetype feasibility preview using current native_pct
    print("\n" + "=" * 78)
    print("ARCHETYPE PREVIEW (based on CURRENT composition, not stress-window)")
    print("=" * 78)
    print("Rough indication only - final classification happens in 02_build_nav.py\n")

    def arch(p):
        if p is None:
            return "?"
        if p >= 0.70:
            return "A"
        return "B or C"

    usable["likely_archetype"] = usable["native_pct"].apply(arch)
    print(usable[["slug", "native_pct", "likely_archetype", "nonnative_usd"]]
          .sort_values("native_pct", ascending=False)
          .to_string(index=False))

    print("\n" + usable["likely_archetype"].value_counts().to_string())

    print("\n" + "=" * 78)
    print("WHAT THIS MEANS")
    print("=" * 78)
    print("""
Pick the window with the most protocols AND a defensible rationale.

If 'Mar 2023 USDC depeg' has the most coverage, that is arguably the
STRONGEST choice for this specific paper, not merely the available one:
it directly tests whether the stablecoin allocation your Tier 1 prescribes
was actually safe. A stress event that could have falsified your framework
is better evidence than one that could only confirm it.

Whatever you choose, take the reasoning to your supervisor BEFORE
collecting data. Changing the stress window changes Sections 1.1, 1.4,
3.4 and 3.7 of the proposal, and that needs sign-off.
""")
    print(f"Reports written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
