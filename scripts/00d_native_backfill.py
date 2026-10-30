"""
STEP 0d: Rebuild ENS and Olympus native-token balances before mid-2024.

WHY
---
DefiLlama only started recording these DAOs' own-token holdings on
15 Apr 2024 (ENS) and 13 Jul 2024 (Olympus). Before that their native share
would wrongly show as 0%. This script rebuilds the balances from the
blockchain, for the SAME wallets DefiLlama tracks.

HOW
---
Like a bank statement: a wallet's balance on a date = every token transfer
INTO it minus every transfer OUT of it, up to the end of that date. The
transfers come from Etherscan's free API.

The 'check' rows are for a date DefiLlama DOES cover. 02_build_nav.py compares
them with DefiLlama and prints OK or MISMATCH, so you know whether to trust
the rebuilt numbers.

BEFORE YOU RUN
--------------
1. Make a free account at https://etherscan.io/register
2. Go to https://etherscan.io/myapikey, click "Add", copy the key
3. Save it in a file called etherscan_key.txt next to this script
   (that file is listed in .gitignore, so it is never uploaded), or set the
   ETHERSCAN_API_KEY environment variable

HOW TO RUN
----------
    python 00d_native_backfill.py

OUTPUT
------
    data/raw/manual_corrections.csv   (read automatically by 02_build_nav.py)
"""

import os
import time
import itertools
import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_key():
    """Read the Etherscan key from the environment or from etherscan_key.txt (never hard-code it)."""
    key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    path = os.path.join(SCRIPT_DIR, "etherscan_key.txt")
    if not key and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            key = fh.read().strip()
    return key


ETHERSCAN_API_KEY = _load_key()
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DIR, "manual_corrections.csv")
COMP_FILE = os.path.join(RAW_DIR, "treasury_composition.csv")
WALLET_FILE = os.path.join(RAW_DIR, "backfill_by_wallet.csv")
CACHE_DIR = os.path.join(RAW_DIR, "etherscan_cache")
API = "https://api.etherscan.io/v2/api"
END = pd.Timestamp("2024-10-01")          # nothing after this is needed

SNAPSHOTS = [
    # protocol, snapshot_type, snapshot_label, date
    ("ENS", "event",     "usdc_depeg_2023", "2023-03-07"),
    ("ENS", "quarterly", "Q2-2023", "2023-06-30"),
    ("ENS", "quarterly", "Q3-2023", "2023-09-30"),
    ("ENS", "quarterly", "Q4-2023", "2023-12-31"),
    ("ENS", "quarterly", "Q1-2024", "2024-03-31"),
    ("ENS", "check",     "Q2-2024", "2024-06-30"),
    ("Olympus DAO", "event",     "usdc_depeg_2023", "2023-03-07"),
    ("Olympus DAO", "quarterly", "Q2-2023", "2023-06-30"),
    ("Olympus DAO", "quarterly", "Q3-2023", "2023-09-30"),
    ("Olympus DAO", "quarterly", "Q4-2023", "2023-12-31"),
    ("Olympus DAO", "quarterly", "Q1-2024", "2024-03-31"),
    ("Olympus DAO", "quarterly", "Q2-2024", "2024-06-30"),
    ("Olympus DAO", "check",     "Q3-2024", "2024-09-30"),
]

TOKENS = {
    # protocol: [(contract, symbol, coingecko_id)]
    "ENS": [("0xc18360217d8f7ab5e7c516566761ea12ce7f9d72", "ENS", "ethereum-name-service")],
    "Olympus DAO": [("0x64aa3364f17a4d01c6f1751fd97c2bd3d7e7f1d5", "OHM", "olympus"),
                    ("0x0ab87046fbb341d058f17cbc4c1133f25a20a52f", "GOHM", "governance-ohm")],
}

# Wallets DefiLlama tracks (DefiLlama-Adapters: registries/treasury.js
# 'treasury/ens' and projects/olympus/index.js)
WALLETS = {
    "ENS": [
        "0xfe89cc7abb2c4183683ab71653c4cdc9b02d44b7", "0xd7a029db2585553978190db5e85ec724aa4df23f",
        "0x690f0581ececcf8389c223170778cd9d029606f2", "0xcd42b4c4d102cc22864e3a1341bb0529c17fd87d",
        "0x91c32893216de3ea0a55abb9851f581d4503d39b", "0xb162bf7a7fd64ef32b787719335d06b2780e31d1",
        "0x536013c57daf01d78e8a70cad1b1abada9411819", "0x9b9c249be04dd433c7e8fbbf5e61e6741b89966d",
        "0x283af0b28c62c092c9727f1ee09c02ca627eb7f5", "0x253553366da8546fc250f225fe3d25d0c782303b",
        "0x4f2083f5fbede34c2714affb3105539775f7fe64",
    ],
    "Olympus DAO": [
        "0x0d33c811d0fcc711bcb388dfb3a152de445be66f", "0x0e1177e47151be72e5992e0975000e73ab5fd9d4",
        "0x8caf91a6bb38d55fb530dec0fab535fa78d98fad", "0x872ebdd8129aa328c89f6bf032bbd77a4c4bac7e",
        "0xa9b52a2d0ffdbabdb2cb23ebb7cd879cac6618a6", "0x9025046c6fb25fb39e720d97a8fd881ed69a1ef6",
        "0xba42be149e5260eba4b82418a6306f55d532ea47", "0xf7deb867e65306be0cb33918ac1b8f89a72109db",
        "0x3df5a355457db3a4b5c744b8623a7721bf56df78", "0x408a9a09d97103022f53300a3a14ca6c3ff867e8",
        "0xdbf0683fc4fc8ac11e64a6817d3285ec4f2fc42d", "0xdfc95aaf0a107daae2b350458ded4b7906e7f728",
        "0x2d643df5de4e9ba063760d475beaa62821c71681", "0x943c1dfa7da96e54242bd2c78dd3ef5c7b24b18c",
        "0x75e7f7d871f4b5db0fa9b0f01b7422352ec9618f", "0x245cc372c84b3645bf0ffe6538620b04a217988b",
        "0xf65a665d650b5de224f46d729e2bd0885eea9da5", "0x97b3ef4c558ec456d59cb95c65bfb79046e31fca",
        "0x5db0761487e26b555f5bfd5e40f4cbc3e1a7d11e", "0x0ea26319836ff05b8c5c5afd83b8ab17dd46d063",
        "0xe3312c3f1ab30878d9686452f7205ebe11e965eb", "0x061c8610a784b8a1599de5b1157631e35180d818",
        "0x886ce997aa9ee4f8c2282e182ab72a705762399d", "0x31f8cc382c9898b273eff4e0b7626a6987c846e8",
        "0x9a315bdf513367c0377fb36545857d12e85813ef", "0xa8687a15d4be32cc8f0a8a7b9704a4c3993d9613",
        "0xde7b85f52577b113181921a7aa8fc0c22e309475", "0x2075e3b46470cfce124daaf52b46dcf965727dd1",
    ],
}

PAGE = 1000       # rows per request (the free tier returns at most 1,000)


def call(params, tries=5):
    params = {**params, "chainid": 1, "apikey": ETHERSCAN_API_KEY}
    for attempt in range(tries):
        try:
            r = requests.get(API, params=params, timeout=60)
            j = r.json()
        except Exception as e:
            print(f"      request error ({e}), retrying...")
            time.sleep(3 * (attempt + 1))
            continue
        if j.get("status") == "1":
            return j["result"]
        msg = str(j.get("message", "")) + " " + str(j.get("result", ""))
        if "No transactions found" in msg or "No records found" in msg:
            return []
        if "rate limit" in msg.lower() or "Max calls" in msg:
            time.sleep(2 * (attempt + 1))
            continue
        raise RuntimeError(f"Etherscan error: {msg.strip()}")
    raise RuntimeError("Etherscan did not respond after several tries")


def end_block():
    return int(call({"module": "block", "action": "getblocknobytime",
                     "timestamp": int(END.timestamp()), "closest": "before"}))


def transfers(token, wallet, last_block):
    """All transfers of `token` into or out of `wallet`, paging through blocks."""
    rows, start = [], 0
    while True:
        res = call({"module": "account", "action": "tokentx", "contractaddress": token,
                    "address": wallet, "startblock": start, "endblock": last_block,
                    "page": 1, "offset": PAGE, "sort": "asc"})
        time.sleep(0.25)                       # free tier: max 5 calls per second
        if len(res) < PAGE:
            rows.extend(res)
            return rows
        # Page is full: keep everything before the last block, restart AT that
        # block so no transfer in it is lost or counted twice.
        top = int(res[-1]["blockNumber"])
        rows.extend(r for r in res if int(r["blockNumber"]) < top)
        start = top


def cached_transfers(contract, symbol, wallet, last_block):
    """Download once, then reuse the saved copy on later runs."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_{wallet}_{last_block}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        return df.to_dict("records")
    tx = transfers(contract, wallet, last_block)
    pd.DataFrame(tx, columns=["blockNumber", "timeStamp", "from", "to", "value", "tokenDecimal"]
                 ).to_csv(path, index=False)
    return [{k: t[k] for k in ("blockNumber", "timeStamp", "from", "to", "value", "tokenDecimal")}
            for t in tx]


def main():
    if not ETHERSCAN_API_KEY:
        print("ERROR: no Etherscan API key found. Save it in scripts/etherscan_key.txt")
        print("       (get one at https://etherscan.io/myapikey)")
        return

    last_block = end_block()
    print(f"Using transfers up to block {last_block} (before {END.date()})\n")

    comp = pd.read_csv(COMP_FILE) if os.path.exists(COMP_FILE) else None
    out, by_wallet = [], []
    for protocol, toks in TOKENS.items():
        wallets = WALLETS[protocol]
        snaps = [x for x in SNAPSHOTS if x[0] == protocol]
        for contract, symbol, cg in toks:
            print(f"{protocol} - {symbol}: fetching transfers for {len(wallets)} wallets")
            # balance of every wallet on every snapshot date
            bal = pd.DataFrame(0.0, index=wallets, columns=[x[2] for x in snaps])
            for w in wallets:
                tx = cached_transfers(contract, symbol, w, last_block)
                if tx:
                    print(f"    {w[:10]}...  {len(tx):>6} transfers")
                if len(tx) in (1000, 10000):
                    print("      WARNING: suspiciously round count - results may be cut off")
                for t in tx:
                    amt = ((int(t["value"]) if t["to"].lower() == w else 0)
                           - (int(t["value"]) if t["from"].lower() == w else 0)) / 10 ** int(t["tokenDecimal"])
                    ts = int(t["timeStamp"])
                    for _, _, label, date in snaps:
                        if ts < (pd.Timestamp(date) + pd.Timedelta(days=1)).timestamp():
                            bal.at[w, label] += amt
            bal = bal.where(bal.abs() > 1e-9, 0.0)
            for w in wallets:
                for _, stype, label, date in snaps:
                    if bal.at[w, label] != 0:
                        by_wallet.append({"protocol": protocol, "asset_symbol": symbol, "wallet": w,
                                          "snapshot_label": label, "quantity": bal.at[w, label]})

            # Compare with DefiLlama on the check date; if they differ, look for
            # the wallet(s) DefiLlama does not count and leave them out everywhere.
            excluded = []
            chk = [x for x in snaps if x[1] == "check"]
            if chk and comp is not None:
                label = chk[0][2]
                dl = comp[(comp["protocol"] == protocol) & (comp["snapshot_label"] == label) &
                          (comp["asset_symbol"] == symbol)]["quantity"].sum()
                ours = bal[label].sum()
                print(f"    check {label}: on-chain {ours:,.2f} vs DefiLlama {dl:,.2f}")
                if dl > 0 and not (0.95 <= ours / dl <= 1.05):
                    holders = [w for w in wallets if bal.at[w, label] != 0]
                    print("    Balances by wallet on the check date:")
                    for w in holders:
                        print(f"      {w}  {bal.at[w, label]:>16,.2f}")
                    fits = []
                    for k in (1, 2):
                        for combo in itertools.combinations(holders, k):
                            rest = ours - bal.loc[list(combo), label].sum()
                            if 0.98 <= rest / dl <= 1.02:
                                fits.append(combo)
                        if fits:
                            break
                    if len(fits) == 1:
                        excluded = list(fits[0])
                        print(f"    -> leaving out {', '.join(excluded)}: the rest matches DefiLlama")
                    else:
                        print(f"    -> no single explanation found ({len(fits)} candidate sets);"
                              f" {symbol} backfill NOT written")
                        continue
            keep = [w for w in wallets if w not in excluded]
            for _, stype, label, date in snaps:
                qty = bal.loc[keep, label].sum()
                if qty > 1e-6:
                    out.append({
                        "protocol": protocol, "snapshot_type": stype, "snapshot_label": label,
                        "snapshot_date": date, "asset_symbol": symbol, "coingecko_id": cg,
                        "price_method": "market", "quantity": qty, "usd_value_defillama": "",
                        "asset_class": "native", "source": "Etherscan on-chain backfill",
                        "notes": ("transfers summed to end of snapshot date, DefiLlama wallet list"
                                  + (f"; excluded {', '.join(excluded)}" if excluded else "")),
                    })
            print()

    pd.DataFrame(by_wallet).to_csv(WALLET_FILE, index=False)
    res = pd.DataFrame(out)
    if res.empty:
        print("No balances found - check the API key and paste the output to Claude.")
        return
    if os.path.exists(OUTPUT_FILE):
        backup = OUTPUT_FILE.replace(".csv", "_backup.csv")
        os.replace(OUTPUT_FILE, backup)
        print(f"(Your previous manual_corrections.csv was saved as {os.path.basename(backup)})")
    res.to_csv(OUTPUT_FILE, index=False)

    print("=" * 70)
    print("REBUILT NATIVE-TOKEN BALANCES")
    print("=" * 70)
    print(res[["protocol", "snapshot_type", "snapshot_label", "asset_symbol", "quantity"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nSaved to {os.path.abspath(OUTPUT_FILE)}")
    print("Next: run 02_build_nav.py - its OVERLAP CHECK says whether these can be trusted.")


if __name__ == "__main__":
    main()
