"""
68_no_trade_days.py
Backtest: high-conviction no-trade days (797 missed days → find tradeable ones)

Strategy: On days where v17a CPR zone signal didn't trigger,
          but conviction score is high (≥3) AND CPR+EMA bias is clear:
          → sell ATM CE/PE at 09:16:02, same T/SL rules

Forward bias: all features use prev-day data only (CPR, EMA, VIX)
Expiry: auto-detected per date (Thu→Tue switch handled)
"""

import sys, os, glob, datetime, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from my_util import DATA_FOLDER, load_spot_data, list_expiry_dates, load_tick_data

DATA_PATH = DATA_FOLDER
LOT_SIZE  = 65
TARGET_PCT = 0.40
SL_PCT     = 0.50
ENTRY_TIME = '09:16:02'
EOD_EXIT   = '15:20:00'
MIN_SCORE  = 3   # minimum conviction to trade on no-signal day

# ── Existing v17a trade dates ─────────────────────────────────────────────────
trades = pd.read_csv('data/56_combined_trades.csv')
trades['date'] = trades['date'].astype(str).str.replace('-','').str[:8]
v17a_dates = set(trades[trades['strategy']=='v17a']['date'].unique())
print(f'v17a trade dates: {len(v17a_dates)}')

# ── Build OHLC + features for all days ───────────────────────────────────────
print('Loading OHLC...')
all_folders = sorted(glob.glob(f'{DATA_PATH}/20[2-9][0-9][0-9][0-9][0-9][0-9]'))
ohlc_rows = []
for folder in all_folders:
    date = os.path.basename(folder)
    if date < '20210101': continue
    df = load_spot_data(date, 'NIFTY')
    if df is None: continue
    day = df[(df['time'] >= '09:15:00') & (df['time'] <= '15:30:00')]
    if len(day) == 0: continue
    ohlc_rows.append({'date': date, 'open': day.iloc[0]['price'],
                      'high': day['price'].max(), 'low': day['price'].min(),
                      'close': day.iloc[-1]['price']})

ohlc = pd.DataFrame(ohlc_rows).sort_values('date').reset_index(drop=True)
print(f'  OHLC: {len(ohlc)}d')

# CPR
ohlc['pp']  = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
ohlc['bc']  = (ohlc['high'] + ohlc['low']) / 2
ohlc['tc']  = 2 * ohlc['pp'] - ohlc['bc']
ohlc['cpr_width_pct'] = (ohlc['tc'] - ohlc['bc']).abs() / ohlc['open'] * 100
ohlc['prev_tc']    = ohlc['tc'].shift(1)
ohlc['prev_bc']    = ohlc['bc'].shift(1)
ohlc['prev_close'] = ohlc['close'].shift(1)

# EMA(20) — 40+ days seed
ohlc['ema20'] = ohlc['close'].ewm(span=20, adjust=False).mean()
ohlc.loc[:39, 'ema20'] = np.nan
ohlc['prev_ema20'] = ohlc['ema20'].shift(1)

# Bias: prev_close vs prev CPR + EMA confirmation
ohlc['bias_pe'] = (ohlc['prev_close'] > ohlc['prev_tc']).astype(int)   # above CPR → sell PE
ohlc['bias_ce'] = (ohlc['prev_close'] < ohlc['prev_bc']).astype(int)   # below CPR → sell CE
ohlc['ema_bull'] = (ohlc['prev_close'] > ohlc['prev_ema20']).astype(int)
ohlc['bias_ok'] = (
    ((ohlc['bias_pe']==1) & (ohlc['ema_bull']==1)) |
    ((ohlc['bias_ce']==1) & (ohlc['ema_bull']==0))
).astype(int)
ohlc['direction'] = ohlc['bias_pe'].map({1: 'PE', 0: 'CE'})  # simple bias direction

# Conviction features (all prev-day)
cpr_dir = (ohlc['prev_close'] > ohlc['prev_tc'].shift(1)).astype(int)
ohlc['consec_aligned'] = ((cpr_dir == cpr_dir.shift(1)) & (cpr_dir == cpr_dir.shift(2))).astype(int)
ohlc['cpr_gap']    = ((ohlc['tc'].shift(1) < ohlc['bc']) | (ohlc['bc'].shift(1) > ohlc['tc'])).astype(int)
ohlc['cpr_narrow'] = ohlc['cpr_width_pct'].shift(1).between(0.10, 0.20).astype(int)

# VIX
print('Loading VIX...')
vix_rows = []
for folder in all_folders:
    date = os.path.basename(folder)
    fp = f'{folder}/INDIAVIX.csv'
    if os.path.exists(fp):
        df = pd.read_csv(fp, header=None, names=['d','t','p','v','oi'])
        vix_rows.append({'date': date, 'vix': df['p'].mean()})
vix_df = pd.DataFrame(vix_rows)
ohlc = ohlc.merge(vix_df, on='date', how='left')
ohlc['vix_ma20'] = ohlc['vix'].rolling(20).mean()
ohlc['vix_ok'] = (ohlc['vix'].shift(1) < ohlc['vix_ma20'].shift(1)).astype(int)

# Score
FEATS = ['vix_ok','consec_aligned','cpr_gap','cpr_narrow','bias_ok']
ohlc['score'] = ohlc[FEATS].fillna(0).astype(int).sum(axis=1)

# ── Filter: no-trade days with high conviction ────────────────────────────────
in_range = ohlc[(ohlc['date'] >= '20210503') & (ohlc['date'] <= '20260429')].copy()
no_trade = in_range[~in_range['date'].isin(v17a_dates)].copy()
candidates = no_trade[(no_trade['score'] >= MIN_SCORE) & (no_trade['bias_ok'] == 1)].copy()

print(f'\nNo-trade days: {len(no_trade)}')
print(f'Candidates (score>={MIN_SCORE} + bias_ok): {len(candidates)}')
print(f'  Score dist: {candidates["score"].value_counts().sort_index().to_dict()}')
print(f'  Direction: PE={candidates["direction"].eq("PE").sum()} CE={candidates["direction"].eq("CE").sum()}')

# ── Run tick-level backtest on candidates ─────────────────────────────────────
print(f'\nRunning tick backtest ({len(candidates)} dates)...')

def run_one(row):
    sys.path.insert(0, '.')
    from my_util import load_spot_data, list_expiry_dates, load_tick_data
    date      = row['date']
    direction = row['direction']   # CE or PE

    try:
        # ATM via spot open rounded to nearest 50
        spot_df = load_spot_data(date, 'NIFTY')
        if spot_df is None: return None
        first = spot_df[spot_df['time'] >= '09:15:00']
        if len(first) == 0: return None
        spot = first.iloc[0]['price']
        atm_strike = int(round(spot / 50) * 50)
        expiries = list_expiry_dates(date, 'NIFTY')
        if not expiries: return None
        expiry_6 = expiries[0]   # nearest expiry
    except Exception:
        return None

    instrument = f'NIFTY{expiry_6}{atm_strike}{direction}'
    try:
        ticks = load_tick_data(date, instrument, ENTRY_TIME)
    except Exception:
        return None
    if ticks is None or len(ticks) == 0:
        return None

    entry_ticks = ticks[ticks['time'] >= ENTRY_TIME]
    if len(entry_ticks) == 0:
        return None

    entry_price = round(entry_ticks.iloc[0]['price'], 2)
    if entry_price <= 5:   # skip illiquid / too cheap
        return None

    target_price = round(entry_price * (1 - TARGET_PCT), 2)
    sl_price     = round(entry_price * (1 + SL_PCT), 2)

    exit_price = None
    exit_reason = 'eod'
    for _, tick in entry_ticks.iterrows():
        p = tick['price']
        t = tick['time']
        if t > EOD_EXIT:
            exit_price  = round(p, 2)
            exit_reason = 'eod'
            break
        if p <= target_price:
            exit_price  = round(p, 2)
            exit_reason = 'target'
            break
        if p >= sl_price:
            exit_price  = round(p, 2)
            exit_reason = 'sl'
            break
    if exit_price is None:
        exit_price  = round(entry_ticks.iloc[-1]['price'], 2)
        exit_reason = 'eod'

    pnl = round((entry_price - exit_price) * LOT_SIZE, 2)

    def conv_lots(s):
        if s >= 4: return 3
        if s >= 2: return 2
        return 1

    lots    = conv_lots(int(row['score']))
    pnl_c   = round(pnl * lots, 2)

    return {
        'date':        date,
        'year':        date[:4],
        'direction':   direction,
        'strike':      atm_strike,
        'expiry':      expiry_6,
        'score':       int(row['score']),
        'lots':        lots,
        'vix_ok':      int(row['vix_ok']),
        'consec':      int(row['consec_aligned']),
        'cpr_gap':     int(row['cpr_gap']),
        'cpr_narrow':  int(row['cpr_narrow']),
        'entry_price': entry_price,
        'target':      target_price,
        'sl':          sl_price,
        'exit_price':  exit_price,
        'exit_reason': exit_reason,
        'pnl':         pnl,
        'pnl_conv':    pnl_c,
        'win':         pnl > 0,
    }

rows_list = [row for _, row in candidates.iterrows()]
with Pool(cpu_count()) as pool:
    results = pool.map(run_one, rows_list)

result_df = pd.DataFrame([r for r in results if r is not None])
print(f'\nTrades executed: {len(result_df)} / {len(candidates)} candidates')

if len(result_df) == 0:
    print('No trades — check data availability')
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
wr  = result_df['win'].mean() * 100
avg = result_df['pnl'].mean()
tot = result_df['pnl'].sum()
tot_c = result_df['pnl_conv'].sum()
dd  = (result_df['pnl'].cumsum() - result_df['pnl'].cumsum().cummax()).min()
dd_c= (result_df['pnl_conv'].cumsum() - result_df['pnl_conv'].cumsum().cummax()).min()

print(f'\n{"="*65}')
print(f'  NO-TRADE DAYS EXPANSION RESULTS')
print(f'{"="*65}')
print(f'  Flat 1-lot:   {len(result_df)}t | WR {wr:.1f}% | Tot Rs {tot:>10,.0f} | DD Rs {dd:>10,.0f}')
print(f'  Conviction:   {len(result_df)}t | WR {wr:.1f}% | Tot Rs {tot_c:>10,.0f} | DD Rs {dd_c:>10,.0f}')

print(f'\n  By Score:')
for s in sorted(result_df['score'].unique()):
    grp = result_df[result_df['score']==s]
    l   = int(grp.iloc[0]['lots'])
    wr2 = grp['win'].mean()*100
    t2  = grp['pnl_conv'].sum()
    print(f'    Score={s} ({l}x): {len(grp):3d}t | WR {wr2:.1f}% | Tot Rs {t2:>10,.0f}')

print(f'\n  By Exit:')
for r in ['target','sl','eod']:
    grp = result_df[result_df['exit_reason']==r]
    if len(grp): print(f'    {r:<8} {len(grp):3d}t | WR {grp["win"].mean()*100:.1f}%')

print(f'\n  By Direction:')
for d in ['CE','PE']:
    grp = result_df[result_df['direction']==d]
    if len(grp):
        wr2 = grp['win'].mean()*100
        t2  = grp['pnl_conv'].sum()
        print(f'    {d}: {len(grp):3d}t | WR {wr2:.1f}% | Tot Rs {t2:>10,.0f}')

print(f'\n  Year-wise:')
result_df['year'] = result_df['date'].str[:4]
for yr in sorted(result_df['year'].unique()):
    grp = result_df[result_df['year']==yr]
    wr2 = grp['win'].mean()*100
    t2  = grp['pnl_conv'].sum()
    print(f'    {yr}: {len(grp):3d}t | WR {wr2:.1f}% | Tot Rs {t2:>10,.0f}')

# ── Combined with v17a ────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print(f'  COMBINED: v17a + No-trade expansion')
print(f'{"="*65}')
v17a_pnl  = pd.read_csv('data/20260430/67_final_trades.csv')['pnl_conv'].sum()
v17a_n    = 356
combined_pnl = v17a_pnl + tot_c
print(f'  v17a conviction:          {v17a_n}t | Rs {v17a_pnl:>10,.0f}')
print(f'  No-trade expansion:       {len(result_df)}t | Rs {tot_c:>10,.0f}')
print(f'  Combined:                 {v17a_n+len(result_df)}t | Rs {combined_pnl:>10,.0f}')

# Save
save_folder = 'data/20260430'
os.makedirs(save_folder, exist_ok=True)
result_df.to_csv(f'{save_folder}/68_no_trade_trades.csv', index=False)
print(f'\n  Saved: {save_folder}/68_no_trade_trades.csv')

# ── Charts ────────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print(f'  CHARTS')
print(f'{"="*65}')

import sys as _sys
_sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import plot_equity, send_custom_chart
import plotly.graph_objects as go, json
from plotly.subplots import make_subplots

# Chart 1: Equity curve of expansion trades
result_df_sorted = result_df.sort_values('date').reset_index(drop=True)
eq  = result_df_sorted['pnl_conv'].cumsum()
dd2 = eq - eq.cummax()
date_idx = pd.to_datetime(result_df_sorted['date'].astype(str))
plot_equity(pd.Series(eq.values, index=date_idx),
            pd.Series(dd2.values, index=date_idx),
            '68_expansion_equity', save_folder,
            title=f'No-Trade Days Expansion | {len(result_df)}t | WR {wr:.1f}% | Rs {tot_c:,.0f}')
print('  Chart 1: Expansion equity sent')

# Chart 2: Combined equity (v17a + expansion)
v17a_df = pd.read_csv('data/20260430/67_final_trades.csv')[['date','pnl_conv']].copy()
v17a_df['source'] = 'v17a'
exp_df  = result_df_sorted[['date','pnl_conv']].copy()
exp_df['source'] = 'expansion'
v17a_df['date'] = v17a_df['date'].astype(str)
exp_df['date']  = exp_df['date'].astype(str)
combined = pd.concat([v17a_df, exp_df]).sort_values('date').reset_index(drop=True)
combined_eq = combined['pnl_conv'].cumsum()
combined_dd = combined_eq - combined_eq.cummax()
cdate_idx = pd.to_datetime(combined['date'].astype(str))
plot_equity(pd.Series(combined_eq.values, index=cdate_idx),
            pd.Series(combined_dd.values, index=cdate_idx),
            '68_combined_equity', save_folder,
            title=f'Combined v17a+Expansion | {len(combined)}t | Rs {combined_pnl:,.0f}')
print('  Chart 2: Combined equity sent')

# Chart 3: Year-wise comparison bar
v17a_yr = v17a_df.copy()
v17a_yr['year'] = v17a_yr['date'].str[:4]
v17a_yr_sum = v17a_yr.groupby('year')['pnl_conv'].sum()
exp_yr  = result_df_sorted.groupby('year')['pnl_conv'].sum()
years   = sorted(set(v17a_yr_sum.index) | set(exp_yr.index))

fig3 = go.Figure()
fig3.add_trace(go.Bar(name='v17a', x=years,
    y=[v17a_yr_sum.get(y, 0) for y in years], marker_color='#4BC0C0'))
fig3.add_trace(go.Bar(name='Expansion', x=years,
    y=[exp_yr.get(y, 0) for y in years], marker_color='#26a69a'))
fig3.update_layout(barmode='stack', showlegend=True, yaxis_title='PnL (Rs)')
tv3 = json.loads(fig3.to_json()); tv3['isTvFormat'] = False
send_custom_chart('68_yearwise_combined', tv3,
                  title='Year-wise PnL: v17a + No-Trade Expansion (stacked)')
print('  Chart 3: Year-wise stacked sent')

print('\nAll done!')
