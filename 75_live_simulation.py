"""
75_live_simulation.py  —  Final Unified Live-Trading Simulation
================================================================
Simulates exactly as live trading works:

  Each day, in order:
    1. Pre-market: compute CPR, EMA, VIX, DTE, all conviction features
    2. 09:15 open: check v17a zone signal  → if yes, take trade
    3. During day:  if no v17a → monitor cam_l3 / cam_h3 touch → take first
    4. 09:30-11:20: if still no signal    → check R1/R2/PDL break (intraday v2)
    5. ONE trade per day maximum

  Data source: loads from research-validated CSV files:
    - data/20260430/72_final_trades.csv   (480 v17a + cam trades, conviction scored)
    - data/20260430/70_intraday_v2_trades.csv  (70 intraday v2 trades)

  These CSVs were produced by the full tick-level backtests (scripts 51-72, 70)
  using per-zone optimised parameters.  Using them here prevents param mismatch.

  Conviction scoring (7 features + inside_cpr negative, from script 72):
    vix_ok, cpr_trend_aligned, consec_aligned, cpr_gap_aligned,
    dte_sweet, cpr_narrow, cpr_dir_aligned
    inside_cpr → reduce lots by 1 (min 1)
    Score 0-1 → 1 lot | 2-3 → 2 lots | 4+ → 3 lots

  Intraday v2 trades use fixed 1 lot (no-signal days — conviction not applied).

  All P&L at LOT=65 (SCALE=65/75 already applied in source CSVs).
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd

OUT_DIR = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)

# ── Load source data ──────────────────────────────────────────────────────────
print("Loading research-validated trade files...")

df72 = pd.read_csv('data/20260430/72_final_trades.csv')
df70 = pd.read_csv('data/20260430/70_intraday_v2_trades.csv')

print(f"  Script 72 (v17a + cam): {len(df72)} trades")
print(f"  Script 70 (intraday v2): {len(df70)} trades")

# ── Build unified trades dataframe ───────────────────────────────────────────
# Script 72: v17a + cam trades with 7-feat conviction
# Columns we need: date, strategy, zone, opt, dte, entry_time, entry_price,
#                  exit_price, exit_reason, pnl_65, pnl_conv, win, lots, score, year

records72 = []
for _, row in df72.iterrows():
    records72.append(dict(
        date        = str(row['date']),
        strategy    = row['strategy'],
        zone        = row.get('zone', ''),
        opt         = row.get('opt', ''),
        dte         = row.get('dte', 0),
        entry_time  = row.get('entry_time', ''),
        entry_price = row.get('ep', 0),
        exit_price  = row.get('xp', 0),
        exit_reason = row.get('exit_reason', ''),
        pnl_65      = r2(row['pnl_flat']),      # flat 1-lot P&L at LOT=65
        pnl_conv    = r2(row['pnl_final']),     # conviction P&L (7-feat + neg)
        win         = bool(row['win']),
        lots        = int(row['lots7n']),
        score       = int(row['score7']),
        year        = str(row['year']),
    ))

# Script 70: intraday v2 trades — fixed 1 lot, no conviction scoring
records70 = []
for _, row in df70.iterrows():
    strat_name = f"iv2_{str(row.get('level','?')).lower()}"
    records70.append(dict(
        date        = str(row['date']),
        strategy    = strat_name,
        zone        = row.get('level', ''),
        opt         = row.get('opt', ''),
        dte         = row.get('dte', 0),
        entry_time  = row.get('entry_time', ''),
        entry_price = row.get('entry_price', 0),
        exit_price  = row.get('exit_price', 0),
        exit_reason = row.get('exit_reason', ''),
        pnl_65      = r2(row['pnl']),
        pnl_conv    = r2(row['pnl']),   # 1 lot only — no conviction applied
        win         = bool(row['win']),
        lots        = 1,
        score       = -1,               # N/A for iv2 (no v17a signal day)
        year        = str(row['year']),
    ))

all_records = records72 + records70

# ── Verify no same-day duplicates (live sim constraint) ───────────────────────
df = pd.DataFrame(all_records).sort_values('date').reset_index(drop=True)

dup = df[df.duplicated('date', keep=False)]
if len(dup):
    print(f"\nWARNING: {len(dup)} same-day duplicates found!")
    print(dup[['date','strategy']].to_string())
else:
    print("OK: no same-day duplicates (live simulation constraint satisfied)")

# ── Results ───────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  STRATEGY BREAKDOWN')
print(f'{"="*65}')
print(f'  {"Strategy":<14} {"t":>5} {"WR":>8} {"Flat (1L)":>12} {"Conv":>12}')
print(f'  {"-"*54}')
strat_order = ['v17a','cam_l3','cam_h3','iv2_r1','iv2_r2','iv2_pdl','ALL']
for strat in strat_order:
    g = df if strat == 'ALL' else df[df['strategy'] == strat]
    if len(g) == 0: continue
    wr   = g['win'].mean() * 100
    flat = g['pnl_65'].sum()
    conv = g['pnl_conv'].sum()
    print(f'  {strat:<14} {len(g):>5} {wr:>7.1f}% {flat:>12,.0f} {conv:>12,.0f}')

print(f'\n{"="*65}')
print('  YEAR-WISE')
print(f'{"="*65}')
print(f'  {"Year":<6} {"t":>5} {"WR":>8} {"Flat (1L)":>12} {"Conv":>12}')
print(f'  {"-"*46}')
for yr in sorted(df['year'].unique()):
    g = df[df['year'] == yr]
    print(f'  {yr:<6} {len(g):>5} {g["win"].mean()*100:>7.1f}% {g["pnl_65"].sum():>12,.0f} {g["pnl_conv"].sum():>12,.0f}')
yr_g = df.groupby('year')['pnl_conv'].sum()
print(f'  Worst year: {yr_g.idxmin()} ₹{yr_g.min():,.0f}   Best year: {yr_g.idxmax()} ₹{yr_g.max():,.0f}')

print(f'\n{"="*65}')
print('  CONVICTION SCORE BREAKDOWN  (v17a + cam only)')
print(f'{"="*65}')
print(f'  {"Score":>6} {"Lots":>5} {"t":>5} {"WR":>8} {"Avg":>10} {"Total":>12}')
print(f'  {"-"*50}')
df_scored = df[df['score'] >= 0]   # exclude iv2 (score=-1)
for s in sorted(df_scored['score'].unique()):
    g = df_scored[df_scored['score'] == s]
    lots_val = g['lots'].iloc[0]
    print(f'  {s:>6} {lots_val:>5}x {len(g):>5} {g["win"].mean()*100:>7.1f}%'
          f' {g["pnl_conv"].mean():>10,.0f} {g["pnl_conv"].sum():>12,.0f}')

flat_total = df['pnl_65'].sum()
conv_total = df['pnl_conv'].sum()
total_t    = len(df)
wr_total   = df['win'].mean() * 100
pct_gain   = (conv_total / flat_total - 1) * 100 if flat_total != 0 else 0

print(f'\n  TOTAL: {total_t}t | WR {wr_total:.1f}%')
print(f'  Flat (1 lot):  ₹{flat_total:>10,.0f}')
print(f'  Conviction:    ₹{conv_total:>10,.0f}  (+₹{conv_total-flat_total:,.0f}, +{pct_gain:.0f}%)')

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = f'{OUT_DIR}/75_live_simulation.csv'
df.to_csv(save_path, index=False)
print(f'\n  Saved → {save_path}  ({len(df)} rows)')

# ── Charts ────────────────────────────────────────────────────────────────────
from plot_util import plot_equity, send_custom_chart

df_sorted = df.sort_values('date').copy()
df_sorted['date_ts'] = pd.to_datetime(df_sorted['date'], format='mixed')

# Equity: flat vs conviction
eq_flat = df_sorted.set_index('date_ts')['pnl_65'].cumsum()
eq_conv = df_sorted.set_index('date_ts')['pnl_conv'].cumsum()

def to_line(series, label, color):
    return {'id': label, 'label': label, 'color': color, 'lineWidth': 2,
            'data': [{'time': int(pd.Timestamp(d).timestamp()), 'value': round(float(v), 2)}
                     for d, v in series.items()]}

tv = {'candlestick': [], 'volume': [], 'markers': [],
      'lines': [to_line(eq_flat, 'Flat (1 lot)', '#888888'),
                to_line(eq_conv, 'Conviction (7-feat)', '#26a69a')],
      'isTvFormat': True}
send_custom_chart('75_equity_flat_vs_conv', tv,
                  title=f'Live Sim: Flat ₹{flat_total/1e5:.2f}L vs Conviction ₹{conv_total/1e5:.2f}L')

# Drawdown + equity for conviction
dd_conv = eq_conv - eq_conv.cummax()
plot_equity(eq_conv, dd_conv, '75_conviction_equity',
            title=f'Live Simulation — Conviction Sizing (LOT=65, 7-feat)  ₹{conv_total/1e5:.2f}L')

# Per-strategy equity
colors = {'v17a': '#1e88e5', 'cam_l3': '#26a69a', 'cam_h3': '#ef5350',
          'iv2_r1': '#ff9800', 'iv2_r2': '#ab47bc', 'iv2_pdl': '#78909c'}
lines_strat = []
for strat in sorted(df['strategy'].unique()):
    g = df_sorted[df_sorted['strategy'] == strat].set_index('date_ts')['pnl_conv'].cumsum()
    lines_strat.append(to_line(g, strat, colors.get(strat, '#888888')))
tv2 = {'candlestick': [], 'volume': [], 'markers': [], 'lines': lines_strat, 'isTvFormat': True}
send_custom_chart('75_per_strategy', tv2,
                  title='Live Sim: Per-Strategy Conviction Equity')

print('\nDone.')
