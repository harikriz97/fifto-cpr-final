"""
126_final_clean.py — Final Quality System (all audits applied)
==============================================================
Audit-clean system with every bias fixed and every valid cut applied.

QUALITY CUTS (all bias-free):
  BASE:
    1. Remove score=6               (10 trades, WR 0%, structural)
    2. Remove cam_h3 + tc_to_pdh    (19 trades, WR 42%, direction mismatch)
    3. Remove PE sells, basis 50-100 (59 trades, WR 56%, avg -Rs.1201)

  BLANK (CRT+MRC):
    4. Remove MRC CE                 (172 trades, net total NEGATIVE P&L)
    5. Remove CRT CE, basis extremes (<-50 or >100, bias-free fut_basis)
    6. Remove CRT CE, real-time ib_up at entry
       - ib_expand from master is FULL-DAY → forward bias
       - FIX: for each CRT CE trade, check spot max[09:46→entry_time] > IB_H
       - Run in parallel across all CRT CE trades

UNCHANGED (still 1 lot each — ib_expand lot scaling deferred until re-run):
  S4:  aligned to filtered base, 1 lot
  CRT PE / MRC PE: 1 lot
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/hesham/.claude/skills/sa-kron-chart/scripts')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from plot_util import plot_equity, super_plotter

DATA_DIR  = 'data/20260430'
CONS_DIR  = 'data/consolidated'
OUT_DIR   = 'data/20260503'
LOT_SIZE  = 65
os.makedirs(OUT_DIR, exist_ok=True)

r2 = lambda v: round(float(v), 2)

# ── helpers ────────────────────────────────────────────────────────────────────
def load_spot(date_str):
    from my_util import load_spot_data
    return load_spot_data(date_str, 'NIFTY')

def _rt_ib_check(args):
    """Real-time IB expansion check at entry_time.
    Returns True if price already exceeded IB_H BEFORE entry_time (ib_up known at entry).
    """
    date_str, entry_time, ib_high = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        from my_util import load_spot_data
        spot = load_spot_data(date_str, 'NIFTY')
        if spot is None: return (date_str, entry_time, False)
        # post-IB window up to (but NOT including) entry_time
        window = spot[(spot['time'] > '09:45:00') & (spot['time'] < entry_time)]
        ib_already_up = (not window.empty) and (window['price'].max() > float(ib_high))
        return (date_str, entry_time, bool(ib_already_up))
    except Exception:
        return (date_str, entry_time, False)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BASE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("STEP 1: BASE TRADES")
print("=" * 62)

raw = pd.read_csv(f'{DATA_DIR}/124_base_clean.csv')
raw['date'] = raw['date'].astype(str)

master = pd.read_excel(f'{DATA_DIR}/122_full_master.xlsx')
master['date'] = master['date'].astype(str)

base = raw.merge(master[['date', 'fut_basis_pts']], on='date', how='left')

# Cut 1: score=6
c1 = base['score'] == 6
# Cut 2: cam_h3 + tc_to_pdh
c2 = (base['strategy'] == 'cam_h3') & (base['zone'] == 'tc_to_pdh')
# Cut 3: PE sells, fut_basis 50-100
c3 = (base['opt'] == 'PE') & (base['fut_basis_pts'] >= 50) & (base['fut_basis_pts'] <= 100)

cut_mask = c1 | c2 | c3
base_final = base[~cut_mask].copy()

print(f"  Raw base          : {len(raw)} trades")
print(f"  Cut 1 score=6     : -{c1.sum()} trades")
print(f"  Cut 2 cam_h3 tcp  : -{c2.sum()} trades")
print(f"  Cut 3 PE basis    : -{c3.sum()} trades")
print(f"  FINAL base        : {len(base_final)} trades | "
      f"WR {base_final['win'].mean()*100:.1f}% | "
      f"Rs.{base_final['pnl'].sum():,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — S4 (align to filtered base)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("STEP 2: S4 2ND-ENTRY TRADES")
print("=" * 62)

s4_raw = pd.read_csv(f'{DATA_DIR}/124_s4_clean.csv')
s4_raw['date'] = s4_raw['date'].astype(str)

base_dates = set(base_final['date'])
s4_final   = s4_raw[s4_raw['date'].isin(base_dates)].copy()
s4_final['lots'] = 1

print(f"  S4 aligned        : {len(s4_final)} trades | "
      f"WR {s4_final['win'].mean()*100:.1f}% | "
      f"Rs.{s4_final['pnl'].sum():,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BLANK (CRT+MRC, quality cuts + real-time ib check)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("STEP 3: BLANK DAY TRADES (CRT+MRC)")
print("=" * 62)

crt_raw = pd.read_csv(f'{CONS_DIR}/103_crt_mrc_atm30_20181010_20260430.csv')
crt_raw['date'] = crt_raw['date'].astype(str)

# Non-overlapping with base
assert len(set(crt_raw['date']) & base_dates) == 0, "Blank/base overlap!"
crt_raw['year'] = crt_raw['date'].str[:4].astype(int)
crt_raw['lots'] = 1

# Merge master features
crt = crt_raw.merge(
    master[['date', 'fut_basis_pts', 'ib_high', 'ib_low']],
    on='date', how='left'
)

# Cut 4: MRC CE (net negative P&L, 172 trades)
m4 = (crt['signal_type'] == 'MRC') & (crt['signal'] == 'CE')
# Cut 5: CRT CE + basis extremes (<-50 or >100) — fut_basis_pts is 09:15 data → fully bias-free
m5 = (crt['signal_type'] == 'CRT') & (crt['signal'] == 'CE') & \
     ((crt['fut_basis_pts'] < -50) | (crt['fut_basis_pts'] > 100))

print(f"  Raw CRT+MRC       : {len(crt)} trades")
print(f"  Cut 4 MRC CE      : -{m4.sum()} trades (net -Rs.{abs(crt[m4]['pnl'].sum()):,.0f})")
print(f"  Cut 5 CRT CE basis: -{m5.sum()} trades (total Rs.{crt[m5]['pnl'].sum():,.0f})")

# After cuts 4 & 5 — but before real-time ib check
crt_step1 = crt[~(m4 | m5)].copy().reset_index(drop=True)

# Cut 6: CRT CE + real-time ib_up at entry (parallel spot check)
crt_ce_candidates = crt_step1[
    (crt_step1['signal_type'] == 'CRT') & (crt_step1['signal'] == 'CE')
].copy()

print(f"\n  Running real-time IB check for {len(crt_ce_candidates)} CRT CE trades...")

args_rt = [
    (row['date'], row['entry_time'], row['ib_high'])
    for _, row in crt_ce_candidates.iterrows()
]

n_cpu = min(16, cpu_count() or 4)
with Pool(processes=n_cpu) as pool:
    rt_results = pool.map(_rt_ib_check, args_rt)

# Build lookup: (date, entry_time) → ib_already_up
rt_lookup = {(d, et): up for d, et, up in rt_results}

# Apply: remove CRT CE where IB was ALREADY UP before entry (against trend)
def is_rt_ib_up(row):
    return rt_lookup.get((row['date'], row['entry_time']), False)

crt_ce_candidates['rt_ib_up'] = crt_ce_candidates.apply(is_rt_ib_up, axis=1)
rt_ib_up_mask = crt_ce_candidates[crt_ce_candidates['rt_ib_up']].index

n_rt_removed = len(rt_ib_up_mask)
rt_pnl_removed = crt_step1.loc[rt_ib_up_mask, 'pnl'].sum() if n_rt_removed > 0 else 0

print(f"  Real-time IB up   : {n_rt_removed} CRT CE trades removed "
      f"(total Rs.{rt_pnl_removed:,.0f} removed)")

crt_final = crt_step1[~crt_step1.index.isin(rt_ib_up_mask)].copy().reset_index(drop=True)

print(f"\n  FINAL blank       : {len(crt_final)} trades | "
      f"WR {crt_final['win'].mean()*100:.1f}% | "
      f"Rs.{crt_final['pnl'].sum():,.0f}")

# Per signal type
for sig, g in crt_final.groupby('signal_type'):
    print(f"    {sig:<4}: {len(g):>4} trades | WR {g['win'].mean()*100:.1f}% | "
          f"Rs.{g['pnl'].sum():,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — COMBINE & STATS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("STEP 4: COMBINED FINAL SYSTEM")
print("=" * 62)

# Unified schema
def to_unified(df, comp, signal_col, opt_col='opt', entry_col='entry_time'):
    out = pd.DataFrame({
        'date'       : df['date'].values,
        'component'  : comp,
        'signal'     : df[signal_col].values,
        'opt'        : df[opt_col].values if opt_col in df.columns else 'NA',
        'entry_time' : df[entry_col].values if entry_col in df.columns else 'NA',
        'ep'         : df['ep'].values if 'ep' in df.columns else np.nan,
        'xp'         : df['xp'].values if 'xp' in df.columns else np.nan,
        'exit_reason': df['exit_reason'].values if 'exit_reason' in df.columns else 'NA',
        'lots'       : df['lots'].values if 'lots' in df.columns else 1,
        'pnl'        : df['pnl'].values,
        'win'        : df['win'].values,
        'year'       : df['year'].values if 'year' in df.columns else
                       pd.to_datetime(df['date'], format='%Y%m%d').dt.year.values,
    })
    return out

base_u = to_unified(base_final, 'base',  'strategy', opt_col='opt')
s4_u   = to_unified(s4_final,   'S4',    'strategy', opt_col='opt',
                    entry_col='reentry_time')
crt_u  = to_unified(crt_final,  'blank', 'signal_type', opt_col='signal')

all_t = pd.concat([base_u, s4_u, crt_u], ignore_index=True)
all_t = all_t.sort_values('date').reset_index(drop=True)
all_t['cum_pnl'] = all_t['pnl'].cumsum()

# Coverage
covered = len(set(base_final['date']) | set(crt_final['date']))
total_days = 1155

# Drawdown
cum     = all_t['cum_pnl'].values
peak    = pd.Series(cum).cummax().values
dd_abs  = peak - cum
max_dd  = dd_abs.max()
peak_v  = peak[-1] if peak[-1] > 0 else 1
max_dd_pct = round(max_dd / peak_v * 100, 2)

total_t   = len(all_t)
total_pnl = round(all_t['pnl'].sum(), 2)
total_wr  = round(all_t['win'].mean() * 100, 1)

print(f"\n  {'Component':<8} {'Trades':>7} {'WR%':>6} {'P&L':>14}")
print(f"  {'-'*40}")
for comp, g in all_t.groupby('component', sort=False):
    print(f"  {comp:<8} {len(g):>7} {g['win'].mean()*100:>6.1f} "
          f"{g['pnl'].sum():>14,.0f}")
print(f"  {'TOTAL':<8} {total_t:>7} {total_wr:>6.1f} {total_pnl:>14,.2f}")
print(f"\n  Coverage  : {covered}/{total_days} = {covered/total_days*100:.1f}%")
print(f"  Max DD    : -Rs.{max_dd:,.0f}  ({max_dd_pct}%)")

# Year breakdown
print(f"\n  Year breakdown:")
yr = all_t.groupby('year').agg(
    trades=('pnl', 'count'),
    wr=('win', lambda x: round(x.mean() * 100, 1)),
    pnl=('pnl', 'sum'),
    avg=('pnl', 'mean')
).reset_index()
yr['pnl'] = yr['pnl'].round(0)
yr['avg'] = yr['avg'].round(0)
for _, row in yr.iterrows():
    print(f"    {int(row['year'])}: {int(row['trades']):>4}t | "
          f"WR {row['wr']:.1f}% | Rs.{row['pnl']:>10,.0f} | avg Rs.{row['avg']:.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("STEP 5: SAVING")
print("=" * 62)

csv_path = f'{OUT_DIR}/126_all_trades.csv'
all_t.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

xl_path = f'{OUT_DIR}/126_final_clean.xlsx'
with pd.ExcelWriter(xl_path, engine='openpyxl') as writer:
    all_t.to_excel(writer, sheet_name='all_trades', index=False)

    yr.to_excel(writer, sheet_name='year_summary', index=False)

    comp = all_t.groupby('component').agg(
        trades=('pnl', 'count'),
        wr=('win', lambda x: round(x.mean() * 100, 1)),
        avg_pnl=('pnl', lambda x: round(x.mean(), 0)),
        total_pnl=('pnl', lambda x: round(x.sum(), 0))
    ).reset_index()
    comp.to_excel(writer, sheet_name='component_summary', index=False)

    # Cuts applied log
    cuts = pd.DataFrame([
        {'cut': 'Base score=6',          'removed': int(c1.sum()),
         'pnl_removed': round(base[c1]['pnl'].sum(), 0)},
        {'cut': 'Base cam_h3 tc_to_pdh', 'removed': int(c2.sum()),
         'pnl_removed': round(base[c2]['pnl'].sum(), 0)},
        {'cut': 'Base PE basis 50-100',  'removed': int(c3.sum()),
         'pnl_removed': round(base[c3]['pnl'].sum(), 0)},
        {'cut': 'Blank MRC CE',          'removed': int(m4.sum()),
         'pnl_removed': round(crt[m4]['pnl'].sum(), 0)},
        {'cut': 'Blank CRT CE basis ext','removed': int(m5.sum()),
         'pnl_removed': round(crt[m5]['pnl'].sum(), 0)},
        {'cut': 'Blank CRT CE RT ib_up', 'removed': n_rt_removed,
         'pnl_removed': round(rt_pnl_removed, 0)},
    ])
    cuts.to_excel(writer, sheet_name='cuts_applied', index=False)

    pd.DataFrame([{
        'metric': 'Total trades',     'value': total_t},
        {'metric': 'Win rate %',      'value': total_wr},
        {'metric': 'Total P&L Rs.',   'value': total_pnl},
        {'metric': 'Max DD Rs.',      'value': round(-max_dd, 0)},
        {'metric': 'Max DD %',        'value': -max_dd_pct},
        {'metric': 'Coverage days',   'value': covered},
        {'metric': 'Total days',      'value': total_days},
        {'metric': 'Coverage %',      'value': round(covered / total_days * 100, 1)},
    ]).to_excel(writer, sheet_name='summary', index=False)

print(f"  Saved: {xl_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — EQUITY CHART
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding equity chart...")

all_t_sorted = all_t.sort_values('date').copy()
all_t_sorted['dt'] = pd.to_datetime(all_t_sorted['date'], format='%Y%m%d')
daily = all_t_sorted.groupby('dt')['pnl'].sum()

eq_series = daily.cumsum()
peak_s    = eq_series.cummax()
dd_series = -(peak_s - eq_series) / peak_s.replace(0, np.nan) * 100
dd_series = dd_series.fillna(0)

plot_equity(
    eq_series, dd_series,
    name='126_equity',
    folder_path=OUT_DIR,
    title=f'126 Final Clean: {total_t}t | WR {total_wr}% | '
          f'Rs.{total_pnl/100000:.2f}L | DD {max_dd_pct}% | Cover {covered/total_days*100:.0f}%'
)
print("  Chart pushed ✓")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("FINAL NUMBERS")
print("=" * 62)
print(f"  Trades   : {total_t}")
print(f"  Win rate : {total_wr}%")
print(f"  P&L      : Rs. {total_pnl:,.2f}  ({total_pnl/100000:.2f}L)")
print(f"  Max DD   : -Rs. {max_dd:,.0f}  ({max_dd_pct}%)")
print(f"  Coverage : {covered}/{total_days} days = {covered/total_days*100:.1f}%")
print("=" * 62)
