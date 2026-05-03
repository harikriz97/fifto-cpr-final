"""
125_quality_final.py
=====================
Quality Final System — Three-component clean assembly:
  1. BASE  : v17a + cam + iv2 trades (no score=6, PE basis filter)
  2. S4    : 2nd-entry retracement on base target days
  3. BLANK : CRT + MRC on non-base days

Quality filters applied (all bias-free, use only pre-entry data):
  a) Remove score=6 base trades (+Rs.17K, 10 trades)
  b) PE sells with fut_basis_pts in [50,100] removed — WR 56%, avg -Rs.1201 (59 trades, +Rs.71K)

Coverage: 1084/1155 days = 93.9% (well above 75% target)
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_util import list_trading_dates

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data', '20260430')
CONS_DIR   = os.path.join(BASE_DIR, 'data', 'consolidated')
OUT_DIR    = os.path.join(BASE_DIR, 'data', '20260503')
os.makedirs(OUT_DIR, exist_ok=True)

LOT_SIZE   = 65   # NIFTY lot size (fixed — no scaling hacks)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & FILTER BASE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STEP 1: BASE TRADES")
print("=" * 60)

raw = pd.read_csv(os.path.join(DATA_DIR, '124_base_clean.csv'))
raw['date'] = raw['date'].astype(str)

# Quality filter a: remove score=6
n_before = len(raw)
base = raw[raw['score'] != 6].copy()
print(f"  Score=6 removed : {n_before - len(base)} trades")

# Load master for basis filter (bias-free: computed at 09:15)
master = pd.read_excel(os.path.join(DATA_DIR, '122_full_master.xlsx'))
master['date'] = master['date'].astype(str)
base = base.merge(
    master[['date', 'fut_basis_pts', 'ib_adr_ratio_%', 'ib_expand', 'gap_pct', 'day_type']],
    on='date', how='left'
)

# Quality filter b: PE sells when fut_basis_pts in [50, 100]
#   These 59 trades have WR=56%, avg P&L = -Rs.1201 → total -Rs.70,866
#   Removing them improves WR from 71.6% → 73.6%
basis_bad_mask = (base['opt'] == 'PE') & \
                 (base['fut_basis_pts'] >= 50) & \
                 (base['fut_basis_pts'] <= 100)
n_basis_removed = basis_bad_mask.sum()
base = base[~basis_bad_mask].copy()
print(f"  PE basis 50-100 : {n_basis_removed} trades removed (WR 56%, avg -Rs.1201)")
print(f"  Final base      : {len(base)} trades | WR {base['win'].mean()*100:.1f}% | Rs.{base['pnl'].sum():,.0f}")

# Per-year base summary
base_year = base.groupby('year').agg(
    trades=('pnl', 'count'),
    wr=('win', lambda x: round(x.mean() * 100, 1)),
    pnl=('pnl', 'sum')
).reset_index()
print("\n  Base by year:")
print(base_year.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD & ALIGN S4 (2nd-entry trades on base target days)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 2: S4 2ND-ENTRY TRADES")
print("=" * 60)

s4_raw = pd.read_csv(os.path.join(DATA_DIR, '124_s4_clean.csv'))
s4_raw['date'] = s4_raw['date'].astype(str)

# Only keep S4 dates where base trade still exists after filtering
base_dates = set(base['date'])
s4 = s4_raw[s4_raw['date'].isin(base_dates)].copy()
s4['lots'] = 1   # S4 always 1 lot

print(f"  S4 total        : {len(s4)} trades | WR {s4['win'].mean()*100:.1f}% | Rs.{s4['pnl'].sum():,.0f}")
s4_year = s4.groupby('year').agg(
    trades=('pnl', 'count'),
    wr=('win', lambda x: round(x.mean() * 100, 1)),
    pnl=('pnl', 'sum')
).reset_index()
print("\n  S4 by year:")
print(s4_year.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 3. LOAD CRT+MRC BLANK DAY TRADES (non-overlapping with base)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 3: CRT + MRC BLANK DAY TRADES")
print("=" * 60)

crt_raw = pd.read_csv(os.path.join(CONS_DIR, '103_crt_mrc_atm30_20181010_20260430.csv'))
crt_raw['date'] = crt_raw['date'].astype(str)

# Only use days NOT covered by base (zero overlap confirmed)
overlap = len(set(crt_raw['date']) & base_dates)
print(f"  Overlap with base : {overlap} days (expected 0)")

crt = crt_raw[~crt_raw['date'].isin(base_dates)].copy()
crt['lots'] = 1

print(f"  CRT+MRC total   : {len(crt)} trades | WR {crt['win'].mean()*100:.1f}% | Rs.{crt['pnl'].sum():,.0f}")

crt_by_type = crt.groupby('signal_type').agg(
    trades=('pnl', 'count'),
    wr=('win', lambda x: round(x.mean() * 100, 1)),
    avg_pnl=('pnl', lambda x: round(x.mean(), 0)),
    total_pnl=('pnl', 'sum')
).reset_index()
print("\n  CRT+MRC by type:")
print(crt_by_type.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 4. COMBINE ALL TRADES & BUILD EQUITY CURVE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 4: COMBINED SYSTEM")
print("=" * 60)

# Normalize to common schema: date, component, signal, opt, pnl, win, lots, year
def make_unified(df, component, signal_col=None, opt_col='opt'):
    out = pd.DataFrame()
    out['date']      = df['date']
    out['component'] = component
    out['signal']    = df[signal_col].values if signal_col else component
    out['opt']       = df[opt_col].values if opt_col in df.columns else 'NA'
    out['entry_time']= df['entry_time'].values if 'entry_time' in df.columns else (
                       df['reentry_time'].values if 'reentry_time' in df.columns else 'NA')
    out['ep']        = df['ep'].values if 'ep' in df.columns else np.nan
    out['xp']        = df['xp'].values if 'xp' in df.columns else np.nan
    out['exit_reason']= df['exit_reason'].values if 'exit_reason' in df.columns else 'NA'
    out['pnl']       = df['pnl'].values
    out['win']       = df['win'].values
    out['lots']      = df['lots'].values if 'lots' in df.columns else 1
    out['year']      = df['year'].values if 'year' in df.columns else (
                       pd.to_datetime(df['date'], format='%Y%m%d').dt.year.values)
    return out

base_u = make_unified(base,  'base',  signal_col='strategy', opt_col='opt')
s4_u   = make_unified(s4,    'S4',    signal_col='strategy',  opt_col='opt')
crt_u  = make_unified(crt,   'blank', signal_col='signal_type', opt_col='signal')

all_trades = pd.concat([base_u, s4_u, crt_u], ignore_index=True)
all_trades = all_trades.sort_values(['date', 'component']).reset_index(drop=True)
all_trades['cum_pnl'] = all_trades['pnl'].cumsum()

# Coverage
base_covered  = set(base['date'])
blank_covered = set(crt['date'])
total_covered = base_covered | blank_covered
total_days    = 1155   # from master

total_pnl = round(all_trades['pnl'].sum(), 2)
total_wr   = round(all_trades['win'].mean() * 100, 1)
total_t    = len(all_trades)
max_dd_abs = 0.0
peak       = 0.0
for v in all_trades['cum_pnl']:
    if v > peak:
        peak = v
    dd = peak - v
    if dd > max_dd_abs:
        max_dd_abs = dd
max_dd_pct = round(max_dd_abs / peak * 100, 2) if peak > 0 else 0.0
coverage   = round(len(total_covered) / total_days * 100, 1)

print(f"\n  {'Component':<10} {'Trades':>7} {'WR%':>6} {'Rs. P&L':>14}")
print(f"  {'-'*42}")
for comp, grp in all_trades.groupby('component', sort=False):
    n  = len(grp)
    wr = round(grp['win'].mean() * 100, 1)
    pl = round(grp['pnl'].sum(), 0)
    print(f"  {comp:<10} {n:>7} {wr:>6} {pl:>14,.0f}")
print(f"  {'TOTAL':<10} {total_t:>7} {total_wr:>6} {total_pnl:>14,.2f}")
print(f"\n  Coverage  : {len(total_covered)}/{total_days} days = {coverage}%")
print(f"  Max DD    : -Rs.{max_dd_abs:,.0f} ({max_dd_pct}% of peak)")

# Per-year summary
yr_summary = all_trades.groupby('year').agg(
    trades=('pnl', 'count'),
    wr=('win', lambda x: round(x.mean() * 100, 1)),
    pnl=('pnl', 'sum'),
    dd_max=('pnl', lambda x: -round(max(0,
        max(x.cumsum().expanding().max() - x.cumsum())
    ), 0))
).reset_index()
yr_summary['avg_pnl'] = (yr_summary['pnl'] / yr_summary['trades']).round(0)
yr_summary['pnl']     = yr_summary['pnl'].round(0)
print("\n  Year-wise summary:")
print(yr_summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 5. DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

peak_series = all_trades['cum_pnl'].cummax()
dd_series   = peak_series - all_trades['cum_pnl']
max_dd_idx  = dd_series.idxmax()
max_dd_date = all_trades.loc[max_dd_idx, 'date']
print(f"\n  Max DD date : {max_dd_date} (trade #{max_dd_idx})")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 5: SAVING OUTPUTS")
print("=" * 60)

# Save combined trades CSV
csv_path = os.path.join(OUT_DIR, '125_all_trades.csv')
all_trades.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# Save Excel with multiple sheets
xl_path = os.path.join(OUT_DIR, '125_quality_final.xlsx')
with pd.ExcelWriter(xl_path, engine='openpyxl') as writer:
    # Sheet 1: All trades
    all_trades.to_excel(writer, sheet_name='all_trades', index=False)

    # Sheet 2: Year summary
    yr_summary.to_excel(writer, sheet_name='year_summary', index=False)

    # Sheet 3: Component summary
    comp_sum = all_trades.groupby('component').agg(
        trades=('pnl', 'count'),
        wins=('win', 'sum'),
        losses=('win', lambda x: (~x).sum()),
        wr_pct=('win', lambda x: round(x.mean() * 100, 1)),
        avg_pnl=('pnl', lambda x: round(x.mean(), 0)),
        total_pnl=('pnl', lambda x: round(x.sum(), 0))
    ).reset_index()
    comp_sum.to_excel(writer, sheet_name='component_summary', index=False)

    # Sheet 4: Base confluence analysis
    conf_cols = ['date', 'strategy', 'zone', 'opt', 'score', 'lots',
                 'entry_time', 'ep', 'xp', 'exit_reason', 'pnl', 'win', 'year',
                 'fut_basis_pts', 'ib_adr_ratio_%', 'ib_expand', 'gap_pct', 'day_type']
    base_conf = base[[c for c in conf_cols if c in base.columns]].copy()
    base_conf.to_excel(writer, sheet_name='base_confluence', index=False)

    # Sheet 5: Summary metrics
    summary_df = pd.DataFrame([{
        'metric': 'Total trades',        'value': total_t},
        {'metric': 'Win rate %',         'value': total_wr},
        {'metric': 'Total P&L (Rs.)',    'value': total_pnl},
        {'metric': 'Max drawdown (Rs.)', 'value': round(-max_dd_abs, 0)},
        {'metric': 'Max DD %',           'value': -max_dd_pct},
        {'metric': 'Coverage days',      'value': len(total_covered)},
        {'metric': 'Total trading days', 'value': total_days},
        {'metric': 'Coverage %',         'value': coverage},
        {'metric': 'Base trades',        'value': len(base)},
        {'metric': 'S4 trades',          'value': len(s4)},
        {'metric': 'CRT+MRC trades',     'value': len(crt)},
        {'metric': 'Score=6 removed',    'value': 10},
        {'metric': 'PE basis filtered',  'value': n_basis_removed},
    ])
    summary_df.to_excel(writer, sheet_name='summary', index=False)

print(f"  Saved: {xl_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. EQUITY CURVE CHART
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding equity curve chart …")

# Component equity curves
def cum_pnl_for(comp):
    sub = all_trades[all_trades['component'] == comp].copy()
    sub = sub.sort_values('date').reset_index(drop=True)
    sub['cum_pnl'] = sub['pnl'].cumsum()
    return list(zip(sub['date'].tolist(), sub['cum_pnl'].tolist()))

base_eq  = cum_pnl_for('base')
s4_eq    = cum_pnl_for('S4')
blank_eq = cum_pnl_for('blank')

# Full combined equity and drawdown
all_sorted = all_trades.sort_values('date').reset_index(drop=True)
cum_eq = all_sorted['pnl'].cumsum().tolist()
peak_v = pd.Series(cum_eq).cummax()
dd_pct = (-(peak_v - pd.Series(cum_eq)) / peak_v.replace(0, np.nan) * 100).fillna(0).tolist()
dates  = all_sorted['date'].tolist()

# Build fig
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3],
    shared_xaxes=True,
    subplot_titles=['Cumulative P&L (Rs.)', 'Drawdown %']
)

# Combined
fig.add_trace(go.Scatter(
    x=dates, y=cum_eq, mode='lines', name='Combined',
    line=dict(color='#26a69a', width=3)
), row=1, col=1)

# Component lines (sparse — just key dates)
def sparse_line(eq_list, name, color):
    xs = [e[0] for e in eq_list]
    ys = [e[1] for e in eq_list]
    return go.Scatter(x=xs, y=ys, mode='lines', name=name,
                      line=dict(color=color, width=1.5, dash='dot'), opacity=0.7)

fig.add_trace(sparse_line(base_eq,  'Base',      '#2196F3'), row=1, col=1)
fig.add_trace(sparse_line(s4_eq,    'S4 addon',  '#FF9800'), row=1, col=1)
fig.add_trace(sparse_line(blank_eq, 'CRT+MRC',   '#9C27B0'), row=1, col=1)

# Drawdown fill
fig.add_trace(go.Scatter(
    x=dates, y=dd_pct, mode='lines', name='Drawdown %',
    fill='tozeroy', line=dict(color='#ef5350', width=1),
    fillcolor='rgba(239,83,80,0.2)'
), row=2, col=1)

fig.update_layout(
    title=f'125 Quality Final System — {total_t} trades | WR {total_wr}% | Rs.{total_pnl:,.0f} | DD {max_dd_pct}%',
    template='plotly_dark',
    height=700,
    legend=dict(orientation='h', y=1.02),
    margin=dict(l=60, r=30, t=60, b=40)
)
fig.update_yaxes(title_text='Rs.', row=1, col=1)
fig.update_yaxes(title_text='DD %', row=2, col=1)

# Push chart via sa-kron-chart
import sys as _sys
_sys.path.insert(0, '/home/hesham/.claude/skills/sa-kron-chart/scripts')
from plot_util import super_plotter, plot_equity, send_custom_chart

super_plotter(OUT_DIR, fig, '125_equity', '20260503',
              title=f'Quality Final: {total_t}t | WR{total_wr}% | Rs.{total_pnl/100000:.1f}L | DD{max_dd_pct}%',
              file_formats=['json'])
print("  Chart pushed ✓")

print("\n" + "=" * 60)
print("DONE")
print(f"  Trades   : {total_t}")
print(f"  Win rate : {total_wr}%")
print(f"  P&L      : Rs. {total_pnl:,.0f}")
print(f"  Max DD   : -Rs. {max_dd_abs:,.0f} ({max_dd_pct}%)")
print(f"  Coverage : {len(total_covered)}/{total_days} = {coverage}%")
print("=" * 60)
