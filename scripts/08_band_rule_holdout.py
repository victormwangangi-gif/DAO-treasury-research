"""Step 8: derive the framework bands by an explicit rule and test the rule on a hold-out period.

The rule (Section 5.2.2 of the paper):
  * Judgement limits, fixed before simulation: Tier 1 (stablecoins) at most 50%,
    Tier 2 (ETH and BTC) between 30% and 50%, Tier 3 (native token) at least 10%.
  * Band limits are set in 10-point steps; allocations inside a band are checked in 5-point steps.
  * Tier 3 ceiling = the highest native share for which some Tier 1 floor keeps EVERY allocation
    inside the bands within the tail-risk budget; Tier 1 floor = the lowest floor that does so.
  * Budget = half the median |ES 95%| of the observed Type A treasuries in the same period.

The rule is applied to the calibration period (Jul 2023-Dec 2024) alone, and the bands it
produces are then evaluated on the hold-out period (Jan 2025-Sep 2026).
Input: output/revision_grids_by_period.csv (created by grids code below if missing).
Output: output/revision_band_rule.txt
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')
GRID = os.path.join(OUT, 'revision_grids_by_period.csv')

# Half the median Type A |ES 95%| in each period (from 07_revision_analyses.py, section 6)
TYPE_A = {'full': 0.11638, 'cal': 0.11198, 'hold': 0.12110}


def build_grids():
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location('r7', os.path.join(HERE, '07_revision_analyses.py'))
    r7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r7)
    m5 = r7.m5; MS, ME = r7.MAIN_START, r7.MAIN_END
    proc = os.path.join(HERE, '..', 'data', 'processed')
    nav = pd.read_csv(os.path.join(proc, 'daily_nav.csv'), parse_dates=['date'])
    prices = pd.read_csv(os.path.join(proc, 'prices.csv'), parse_dates=['date'])
    start = MS - pd.Timedelta(days=1)
    t1 = m5.simple_returns(prices, m5.STABLE_IDS, start=start, end=ME)
    t2 = m5.simple_returns(prices, list(m5.CRYPTO_MIX), m5.CRYPTO_MIX, start=start, end=ME)
    nat = {p: np.expm1(g.set_index('date')['r_native'].reindex(t1.index).fillna(0))
           for p, g in nav.groupby('protocol')}
    qall = list(pd.date_range(MS, ME, freq='QS'))
    if qall[0] != MS:
        qall = [MS] + qall
    periods = {'full': (MS, ME), 'cal': (MS, pd.Timestamp('2024-12-31')),
               'hold': (pd.Timestamp('2025-01-01'), ME)}
    grid = [(a, b, 100 - a - b) for a in range(0, 101, 5) for b in range(0, 101 - a, 5)]
    rows = []
    for name, (s, e) in periods.items():
        qs = [q for q in qall if s <= q <= e]
        if qs[0] != s:
            qs = [s] + qs
        for mix in grid:
            v = [r7.es(m5.simulate(pd.DataFrame({'t1': t1, 't2': t2, 't3': t3}).loc[s:e],
                                   [x / 100 for x in mix], qs)) for t3 in nat.values()]
            rows.append({'period': name, 't1': mix[0], 't2': mix[1], 't3': mix[2],
                         'abs_es_95': float(np.median(v))})
    pd.DataFrame(rows).to_csv(GRID, index=False)


def in_box(g, f, c):
    return g[(g.t1 >= f) & (g.t1 <= 50) & (g.t2 >= 30) & (g.t2 <= 50) & (g.t3 >= 10) & (g.t3 <= c)]


def derive(g, budget, step=10):
    for c in range(50, 9, -step):
        for f in range(10, 51, step):
            box = in_box(g, f, c)
            if box.empty or box.t3.max() < c:
                continue
            if (box.abs_es_95 <= budget + 1e-12).all():
                return f, c, box
    return None


if __name__ == '__main__':
    if not os.path.exists(GRID):
        build_grids()
    grids = pd.read_csv(GRID)
    lines = []
    for p in ['full', 'cal']:
        g = grids[grids.period == p]
        b = TYPE_A[p] / 2
        f, c, box = derive(g, b)
        lines.append(f'{p}: budget {b:.4%}; bands Tier 1 {f}-50, Tier 2 30-50, Tier 3 10-{c}; '
                     f'{len(box)} allocations; max |ES| {box.abs_es_95.max():.4%}')
        r5 = derive(g, b, step=5)
        lines.append(f'   with 5-point band limits: Tier 1 {r5[0]}-50, Tier 3 10-{r5[1]}')
    # Hold-out evaluation of the calibration-derived bands
    f, c, _ = derive(grids[grids.period == 'cal'], TYPE_A['cal'] / 2)
    budget = TYPE_A['cal'] / 2
    lines.append(f'\nHold-out test of the calibration bands (budget {budget:.4%}):')
    for p in ['cal', 'hold']:
        box = in_box(grids[grids.period == p], f, c).copy()
        n_ok = int((box.abs_es_95.round(4) <= round(budget, 4)).sum())
        lines.append(f' {p}: {n_ok} of {len(box)} within budget')
        for _, r in box.sort_values('abs_es_95', ascending=False).head(3).iterrows():
            lines.append(f'   {int(r.t1)}/{int(r.t2)}/{int(r.t3)}: {r.abs_es_95:.4%}')
        for mix in [(40, 40, 20), (30, 40, 30), (30, 45, 25), (50, 40, 10)]:
            r = box[(box.t1 == mix[0]) & (box.t2 == mix[1]) & (box.t3 == mix[2])].iloc[0]
            lines.append(f'   {mix}: {r.abs_es_95:.4%}')
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(OUT, 'revision_band_rule.txt'), 'w') as fh:
        fh.write(txt + '\n')
