# Optimising Treasury Management in Decentralised Autonomous Organisations

Code and data for the research project of the same name, by Victor Kilonzo Mwangangi
(BBS Financial Engineering, Strathmore University, 2026; supervisor: Mr Kevin Otieno).

The study builds a daily value series for the treasuries of 13 DAOs from July 2023 to
September 2026. It classifies each treasury as native-token-concentrated (Type A),
diversified (Type B) or stablecoin-heavy (Type C), and tests whether concentration was
associated with greater downside risk. It then calibrates a Three-Tier Treasury Framework
by simulation and validates it on Aave and ENS.

**Main result:** every native-token-concentrated treasury had a larger 95% Expected
Shortfall than every diversified or stablecoin-heavy treasury (one-sided Mann-Whitney U,
p = 0.0014, Cliff's δ = 1.00).

## Repository structure

```
dao-treasury-research/
├── scripts/            Python scripts, run in the order below
├── data/
│   ├── raw/            hand-built inputs (token map, operating expenditure)
│   └── processed/      created by the scripts
└── output/             tables, test results and figures created by the scripts
    └── figures/
```

## Setup

Python 3.10 or later. From the repository folder:

```
python -m pip install -r requirements.txt
```

`00d_native_backfill.py` needs a free Etherscan API key
(https://etherscan.io/myapikey). Save the key in a file called
`scripts/etherscan_key.txt`. That file is listed in `.gitignore`, so it is never
uploaded.

## Running the analysis

Run the scripts from inside the `scripts` folder, in this order:

| Step | Script | What it does | Paper section |
|---|---|---|---|
| 0 | `00_explore_defillama.py` | Inspects the DefiLlama treasury API response (diagnostic) | 3.3 |
| 0b | `00b_check_coverage.py` | Checks each candidate protocol's history and holdings (sample screening) | 3.2 |
| 0c | `00c_feasible_window.py` | Tests which stress windows the data can cover | 3.7 |
| 1b | `01b_defillama_extract.py` | Extracts the 14 quarterly and 2 pre-event treasury snapshots, including the native-token bucket | 3.3, 3.4 |
| 0d | `00d_native_backfill.py` | Rebuilds ENS and Olympus native balances before mid-2024 from Etherscan transfer records | 4.1 |
| 1 | `01_fetch_prices.py` | Downloads daily prices (DefiLlama coins API, Yahoo Finance fallback) | 3.3 |
| 2 | `02_build_nav.py` | Builds the daily time-weighted treasury value series and classifies the archetypes | 3.4, 3.5 |
| 3 | `03_compute_metrics.py` | Risk metrics, event studies, runway and all hypothesis tests | 3.5–3.7, Ch. 4 |
| 4 | `04_robustness_and_figures.py` | Robustness checks and Figures 2–7 | 4.7 |
| 5 | `05_framework_calibration.py` | Simulates 231 tier mixes to calibrate the framework; builds the Aave/ENS validation data | 5.2, 5.3 |
| 6 | `06_validation_and_figures.py` | Distance from the bands, validation correlations, and Figures 8–9 | 5.3 |

Note that `01b` runs before `00d`, because the backfill reads the extracted composition.

## Data sources

- **Treasury composition:** DefiLlama treasury API (`api.llama.fi/treasury/{protocol}`),
  with the native-token bucket recombined.
- **Native balances before mid-2024 (ENS, Olympus DAO):** Etherscan V2 API token transfers.
- **Prices:** DefiLlama coins API, with Yahoo Finance as a fallback.
- **Risk-free rate:** FRED series DGS3MO.
- **Operating expenditure:** published DAO budgets and financial reports. Each figure and
  its source are listed in `data/raw/operating_expenses.csv` and Appendix A of the paper.

The data were retrieved in October 2026. Because the APIs are live, re-running the
extraction later may give slightly different holdings. The processed files in
`data/processed/` and the results in `output/` are the exact versions used in the paper.

## Figure numbering

| File | Figure in the paper |
|---|---|
| `fig4_1_composition.png` | Figure 2 |
| `fig4_2_value_index.png` | Figure 3 |
| `fig4_3_tail_loss.png` | Figure 4 |
| `fig4_5_aug2024_path.png` | Figure 5 |
| `fig4_6_correlation.png` | Figure 6 |
| `fig4_4_concentration_vs_es.png` | Figure 7 |
| `fig5_1_frontier.png` | Figure 8 |
| `fig5_2_validation.png` | Figure 9 |

## Licence and citation

The code is released under the MIT licence (see `LICENSE`). If you use it, please cite:

> Mwangangi, V. K. (2026). *Optimising treasury management in decentralised autonomous
> organisations* [Undergraduate research project]. Strathmore University.

This repository is for academic research only and does not constitute investment advice.
