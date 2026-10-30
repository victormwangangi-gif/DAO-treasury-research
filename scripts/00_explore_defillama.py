"""
STEP 0 (DIAGNOSTIC): Explore what DefiLlama's treasury API actually returns.

RUN THIS FIRST, before 01b_defillama_extract.py

WHY
---
DefiLlama's API is free (no key needed) but its response structure is not
formally documented. This script fetches one protocol and prints the structure
so we know exactly what we are working with.

IMPORTANT: DefiLlama's "Treasury" metric EXCLUDES the protocol's own governance
token by default. Own tokens are tracked in a separate bucket. Since your entire
study is about native token concentration, you MUST capture that separate bucket
or your Type A protocols will be misclassified.

HOW TO RUN
----------
    python 00_explore_defillama.py

Then read the output. If it looks different from what the extractor expects,
send the output to your supervisor/assistant and the extractor can be adapted.
"""

import json
import requests

BASE = "https://api.llama.fi"
TEST_PROTOCOL = "uniswap"      # change to test others


def show(label, value, indent=0):
    pad = "  " * indent
    print(f"{pad}{label}: {value}")


def explore_treasury(slug):
    url = f"{BASE}/treasury/{slug}"
    print("=" * 70)
    print(f"FETCHING: {url}")
    print("=" * 70)

    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        print(f"REQUEST FAILED: {e}")
        return None

    print(f"HTTP status: {r.status_code}")
    if r.status_code != 200:
        print(f"Response body (first 500 chars):\n{r.text[:500]}")
        return None

    data = r.json()

    print(f"\nTop-level keys ({len(data)}):")
    for k in sorted(data.keys()):
        v = data[k]
        if isinstance(v, dict):
            desc = f"dict with {len(v)} keys"
        elif isinstance(v, list):
            desc = f"list with {len(v)} items"
        else:
            desc = f"{type(v).__name__} = {str(v)[:60]}"
        show(k, desc, indent=1)

    # The important part: chainTvls contains the per-bucket breakdown
    if "chainTvls" in data:
        print(f"\n{'=' * 70}")
        print("chainTvls BUCKETS  <-- this is where treasury composition lives")
        print("=" * 70)
        for bucket, content in data["chainTvls"].items():
            keys = list(content.keys()) if isinstance(content, dict) else "n/a"
            print(f"\n  BUCKET: '{bucket}'")
            print(f"    sub-keys: {keys}")

            if isinstance(content, dict):
                # 'tokens' = token QUANTITIES over time  (what we want)
                # 'tokensInUsd' = USD values over time
                for field in ("tokens", "tokensInUsd", "tvl"):
                    if field in content and isinstance(content[field], list):
                        series = content[field]
                        print(f"    '{field}': {len(series)} time points")
                        if series:
                            first, last = series[0], series[-1]
                            print(f"      first entry: {json.dumps(first)[:300]}")
                            print(f"      last  entry: {json.dumps(last)[:300]}")

        print(f"\n{'=' * 70}")
        print("BUCKET NAMES FOUND:")
        print("=" * 70)
        for b in data["chainTvls"].keys():
            note = ""
            if "own" in b.lower():
                note = "  <-- NATIVE GOVERNANCE TOKEN (critical for your study)"
            elif "borrow" in b.lower() or "staking" in b.lower():
                note = "  <-- probably exclude from treasury"
            print(f"  {b}{note}")

    return data


def explore_protocols_list():
    """Show how to find the right slug for a protocol."""
    print(f"\n{'=' * 70}")
    print("FINDING PROTOCOL SLUGS")
    print("=" * 70)
    try:
        r = requests.get(f"{BASE}/protocols", timeout=60)
        if r.status_code != 200:
            print(f"Failed: HTTP {r.status_code}")
            return
        protocols = r.json()
        print(f"DefiLlama tracks {len(protocols)} protocols.\n")
        targets = ["uniswap", "aave", "maker", "compound", "lido",
                   "curve", "synthetix", "arbitrum", "gitcoin", "ens",
                   "olympus", "frax", "yearn", "balancer"]
        print(f"{'Name':<28} {'Slug':<28} {'Category'}")
        print("-" * 75)
        for p in protocols:
            name = (p.get("name") or "").lower()
            slug = (p.get("slug") or "")
            if any(t in name for t in targets):
                print(f"{p.get('name','')[:27]:<28} {slug[:27]:<28} {p.get('category','')}")
    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    data = explore_treasury(TEST_PROTOCOL)
    explore_protocols_list()

    print(f"\n{'=' * 70}")
    print("WHAT TO DO NEXT")
    print("=" * 70)
    print("""
1. Look at the BUCKET NAMES above. You are looking for one that holds the
   protocol's OWN governance token - usually called 'OwnTokens' or similar.

2. Check that the 'tokens' field contains QUANTITIES (token counts), not USD.
   Quantities are what script 02_build_nav.py needs.

3. If the structure matches expectations, run 01b_defillama_extract.py

4. If the response looks different, copy this entire output and ask for the
   extractor to be adapted. Do not guess at field names.
""")
