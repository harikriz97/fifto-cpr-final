"""
69_combined_conviction.py
ALL strategies (v17a + cam_l3 + cam_h3) with conviction scoring

Script 67 only had v17a (356t). This adds cam_l3 (67t, 80.6% WR) and
cam_h3 (57t, 56.1% WR) with same conviction features.

Sections:
  A. Load all 480 trades + compute conviction features for all
  B. Strategy-wise breakdown with conviction
  C. Entry time filter (09:31 v17a = 58.7% WR → check if skip helps)
  D. Zone quality filter
  E. Final combined equity + charts
"""

import sys, os, glob, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from my_util import DATA_FOLDER, load_spot_data

DATA_PATH = DATA_FOLDER
SCALE     = 65 / 75
LOT_SIZE  = 65

# ── Load all trades ───────────────────────────────────────────────────────────
trades = pd.read_csv('data/56_combined_trades.csv')
trades.columns = [c.lower().replace(' ', '_') for c in trades.columns]
trades['date']     = trades['date'].astype(str).str.replace('-', '').str[:8]
trades['pnl_65']   = (trades['pnl'] * SCALE).round(2)
trades['win']      = trades['pnl_65'] > 0
trades['direction'] = trades['opt']

print(f'Loaded {len(trades)} trades: {trades["strategy"].value_counts().to_dict()}')

# ── Build OHLC features for all trading dates ─────────────────────────────────
print('Loading OHLC + VIX features...')
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

# CPR
ohlc['pp']  = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
ohlc['bc']  = (ohlc['high'] + ohlc['low']) / 2
ohlc['tc']  = 2 * ohlc['pp'] - ohlc['bc']
ohlc['cpr_width_pct'] = (ohlc['tc'] - ohlc['bc']).abs() / ohlc['open'] * 100
ohlc['prev_tc']    = ohlc['tc'].shift(1)
ohlc['prev_bc']    = ohlc['bc'].shift(1)
ohlc['prev_close'] = ohlc['close'].shift(1)

# EMA(20)
ohlc['ema20'] = ohlc['close'].ewm(span=20, adjust=False).mean()
ohlc.loc[:39, 'ema20'] = np.nan
ohlc['prev_ema20'] = ohlc['ema20'].shift(1)

# Features (all prev-day, no forward bias)
cpr_dir = (ohlc['prev_close'] > ohlc['prev_tc'].shift(1)).astype(int)
ohlc['consec_aligned'] = ((cpr_dir == cpr_dir.shift(1)) & (cpr_dir == cpr_dir.shift(2))).astype(int)
ohlc['cpr_gap']    = ((ohlc['tc'].shift(1) < ohlc['bc']) | (ohlc['bc'].shift(1) > ohlc['tc'])).astype(int)
ohlc['cpr_narrow'] = ohlc['cpr_width_pct'].shift(1).between(0.10, 0.20).astype(int)
ohlc['open_above'] = (ohlc['open'] > ohlc['prev_tc']).astype(int)
ohlc['open_below'] = (ohlc['open'] < ohlc['prev_bc']).astype(int)

# VIX
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
ohlc['vix_ok']   = (ohlc['vix'].shift(1) < ohlc['vix_ma20'].shift(1)).astype(int)

# DTE (expiry calendar)
import datetime
expiry_set = set()
for folder in all_folders:
    files = glob.glob(f'{folder}/NIFTY[0-9]*.csv')
    for f in files:
        name = os.path.basename(f)
        if name.endswith('CE.csv') or name.endswith('PE.csv'):
            exp6 = name[5:11]
            if exp6.isdigit():
                try:
                    full = '20' + exp6
                    dt = datetime.datetime.strptime(full, '%Y%m%d')
                    if 2021 <= dt.year <= 2026:
                        expiry_set.add(full)
                except: pass

all_expiries = sorted(expiry_set)
dte_rows = []
for folder in all_folders:
    td = os.path.basename(folder)
    if td < '20210101': continue
    upcoming = [e for e in all_expiries if e >= td]
    if upcoming:
        nearest = upcoming[0]
        dte = (datetime.datetime.strptime(nearest, '%Y%m%d') -
               datetime.datetime.strptime(td, '%Y%m%d')).days
        dte_rows.append({'date': td, 'dte': dte})
dte_df = pd.DataFrame(dte_rows)
ohlc = ohlc.merge(dte_df, on='date', how='left')
ohlc['dte_sweet'] = ohlc['dte'].between(3, 5).astype(int)

print(f'  Features built for {len(ohlc)} dates')

# ── Join features with all trades ─────────────────────────────────────────────
feat_cols = ['date','vix_ok','consec_aligned','cpr_gap','cpr_narrow',
             'open_above','open_below','dte_sweet','dte','prev_tc','prev_bc','prev_close']
t = trades.merge(ohlc[feat_cols], on='date', how='left')

# Direction-aware features
t['cpr_trend_aligned'] = (
    ((t['direction'] == 'CE') & (t['prev_close'] < t['prev_bc'])) |
    ((t['direction'] == 'PE') & (t['prev_close'] > t['prev_tc']))
).astype(int)

t['open_aligned'] = (
    ((t['direction'] == 'CE') & (t['open_below'] == 1)) |
    ((t['direction'] == 'PE') & (t['open_above'] == 1))
).astype(int)

t['cpr_gap_aligned'] = (
    ((t['cpr_gap'] == 1) & (t['direction'] == 'CE') & (t['open_below'] == 1)) |
    ((t['cpr_gap'] == 1) & (t['direction'] == 'PE') & (t['open_above'] == 1))
).astype(int)

# Conviction score (6 features)
CONV_FEATS = ['vix_ok','cpr_trend_aligned','consec_aligned','cpr_gap_aligned','dte_sweet','cpr_narrow']
t['score'] = t[CONV_FEATS].fillna(0).astype(int).sum(axis=1)

def conv_lots(s):
    if s >= 4: return 3
    if s >= 2: return 2
    return 1

t['lots']     = t['score'].apply(conv_lots)
t['pnl_conv'] = (t['pnl_65'] * t['lots']).round(2)
t['year']     = t['date'].str[:4]

t_valid = t.dropna(subset=['vix_ok']).copy()
print(f'  Trades with features: {len(t_valid)}')

# ─────────────────────────────────────────────────────────────────────────────
# A. STRATEGY BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  A. STRATEGY BREAKDOWN WITH CONVICTION')
print(f'{"="*65}')
print(f'  {"Strategy":<10} {"t":>5} {"WR":>8} {"Flat Rs":>12} {"Conv Rs":>12} {"Lift":>10}')
print(f'  {"-"*55}')
for strat in ['v17a', 'cam_l3', 'cam_h3', 'ALL']:
    if strat == 'ALL':
        g = t_valid
    else:
        g = t_valid[t_valid['strategy'] == strat]
    if len(g) == 0: continue
    wr   = g['win'].mean() * 100
    flat = g['pnl_65'].sum()
    conv = g['pnl_conv'].sum()
    lift = conv - flat
    print(f'  {strat:<10} {len(g):>5} {wr:>7.1f}% {flat:>12,.0f} {conv:>12,.0f} {lift:>+10,.0f}')

# ─────────────────────────────────────────────────────────────────────────────
# B. CONVICTION SCORE TABLE (all strategies)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  B. CONVICTION SCORE → WR → LOTS (ALL 480 trades)')
print(f'{"="*65}')
print(f'  {"Score":>6} {"Lots":>6} {"t":>6} {"WR":>8} {"Avg":>10} {"Total":>12}')
print(f'  {"-"*55}')
for s in sorted(t_valid['score'].unique()):
    g    = t_valid[t_valid['score'] == s]
    lots = conv_lots(s)
    wr   = g['win'].mean() * 100
    avg  = g['pnl_conv'].mean()
    tot  = g['pnl_conv'].sum()
    print(f'  {s:>6} {lots:>6}x {len(g):>5} {wr:>7.1f}% {avg:>10,.0f} {tot:>12,.0f}')

flat_total = t_valid['pnl_65'].sum()
conv_total = t_valid['pnl_conv'].sum()
flat_wr    = t_valid['win'].mean() * 100
conv_dd    = (t_valid['pnl_conv'].cumsum() - t_valid['pnl_conv'].cumsum().cummax()).min()
flat_dd    = (t_valid['pnl_65'].cumsum() - t_valid['pnl_65'].cumsum().cummax()).min()

print(f'\n  Flat 1-lot:  {len(t_valid)}t | WR {flat_wr:.1f}% | Rs {flat_total:>10,.0f} | DD Rs {flat_dd:>10,.0f}')
print(f'  Conviction:  {len(t_valid)}t | WR {flat_wr:.1f}% | Rs {conv_total:>10,.0f} | DD Rs {conv_dd:>10,.0f}')
print(f'  Improvement: Rs {conv_total-flat_total:>+10,.0f}  ({(conv_total/flat_total-1)*100:+.1f}%)')
print(f'  Lot dist: {dict(t_valid["lots"].value_counts().sort_index())}')

print(f'\n  Year-wise:')
print(f'  {"Year":>5} {"t":>5} {"WR":>8} {"Flat":>12} {"Conv":>12} {"Lift":>10}')
for yr in sorted(t_valid['year'].unique()):
    g    = t_valid[t_valid['year'] == yr]
    wr2  = g['win'].mean() * 100
    flat2= g['pnl_65'].sum()
    conv2= g['pnl_conv'].sum()
    print(f'  {yr:>5} {len(g):>5} {wr2:>7.1f}% {flat2:>12,.0f} {conv2:>12,.0f} {conv2-flat2:>+10,.0f}')

# ─────────────────────────────────────────────────────────────────────────────
# C. ENTRY TIME FILTER
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  C. ENTRY TIME FILTER (v17a 09:31 = weakest)')
print(f'{"="*65}')

def time_bucket(t_str):
    if str(t_str) <= '09:19': return '09:16-19'
    if str(t_str) <= '09:29': return '09:20-29'
    if str(t_str) <= '09:39': return '09:30-39'
    if str(t_str) <= '10:59': return '10:00-10:59'
    return '11:00+'

t_valid = t_valid.copy()
t_valid['time_bucket'] = t_valid['entry_time'].apply(time_bucket)

print(f'  {"Bucket":<14} {"t":>5} {"WR":>8} {"Conv Rs":>12}')
print(f'  {"-"*45}')
for b in ['09:16-19','09:20-29','09:30-39','10:00-10:59','11:00+']:
    g = t_valid[t_valid['time_bucket'] == b]
    if len(g) == 0: continue
    wr2  = g['win'].mean() * 100
    conv2= g['pnl_conv'].sum()
    print(f'  {b:<14} {len(g):>5} {wr2:>7.1f}% {conv2:>12,.0f}')

# Skip 09:30-39 v17a?
v17a_late  = t_valid[(t_valid['strategy']=='v17a') & (t_valid['time_bucket']=='09:30-39')]
v17a_rest  = t_valid[~((t_valid['strategy']=='v17a') & (t_valid['time_bucket']=='09:30-39'))]
print(f'\n  v17a 09:30-39 skip effect:')
print(f'    Remove {len(v17a_late)} late v17a trades (WR {v17a_late["win"].mean()*100:.1f}%)')
print(f'    Remaining: {len(v17a_rest)}t | WR {v17a_rest["win"].mean()*100:.1f}% | Conv Rs {v17a_rest["pnl_conv"].sum():,.0f}')

# ─────────────────────────────────────────────────────────────────────────────
# D. ZONE QUALITY FILTER
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  D. ZONE QUALITY (worst zones — skip?)')
print(f'{"="*65}')
zone_stats = t_valid.groupby('zone').agg(
    n=('win','count'), wr=('win','mean'), conv=('pnl_conv','sum')
).reset_index()
zone_stats['wr'] *= 100
zone_stats = zone_stats.sort_values('wr')
print(f'  {"Zone":<14} {"t":>5} {"WR":>8} {"Conv Rs":>12}')
print(f'  {"-"*45}')
for _, r in zone_stats.iterrows():
    marker = ' << WEAK' if r['wr'] < 60 else ''
    print(f'  {r["zone"]:<14} {int(r["n"]):>5} {r["wr"]:>7.1f}% {r["conv"]:>12,.0f}{marker}')

# skip weak zones (WR < 60%)
weak_zones = zone_stats[zone_stats['wr'] < 60]['zone'].tolist()
filtered   = t_valid[~t_valid['zone'].isin(weak_zones)]
print(f'\n  Skip zones WR<60%: {weak_zones}')
print(f'  After filter: {len(filtered)}t | WR {filtered["win"].mean()*100:.1f}% | Conv Rs {filtered["pnl_conv"].sum():,.0f}')
print(f'  vs all {len(t_valid)}t | WR {t_valid["win"].mean()*100:.1f}% | Conv Rs {t_valid["pnl_conv"].sum():,.0f}')

# ─────────────────────────────────────────────────────────────────────────────
# E. FINAL BEST: skip weak zones + v17a late entries
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  E. FINAL BEST COMBINATION')
print(f'{"="*65}')
print('  Rules: all 480 trades + conviction lots + skip WR<60% zones')

best = t_valid[~t_valid['zone'].isin(weak_zones)].copy()
best_flat = best['pnl_65'].sum()
best_conv = best['pnl_conv'].sum()
best_wr   = best['win'].mean() * 100
best_dd   = (best['pnl_conv'].cumsum() - best['pnl_conv'].cumsum().cummax()).min()

print(f'\n  Flat 1-lot:  {len(best)}t | WR {best_wr:.1f}% | Rs {best_flat:>10,.0f}')
print(f'  Conviction:  {len(best)}t | WR {best_wr:.1f}% | Rs {best_conv:>10,.0f} | DD Rs {best_dd:>10,.0f}')

print(f'\n  Score breakdown (final):')
for s in sorted(best['score'].unique()):
    g    = best[best['score'] == s]
    lots = conv_lots(s)
    wr2  = g['win'].mean() * 100
    tot2 = g['pnl_conv'].sum()
    print(f'    Score={s} ({lots}x): {len(g):3d}t | WR {wr2:.1f}% | Rs {tot2:,.0f}')

print(f'\n  Year-wise (final):')
for yr in sorted(best['year'].unique()):
    g    = best[best['year'] == yr]
    wr2  = g['win'].mean() * 100
    conv2= g['pnl_conv'].sum()
    flat2= g['pnl_65'].sum()
    print(f'    {yr}: {len(g):3d}t | WR {wr2:.1f}% | Flat Rs {flat2:>10,.0f} | Conv Rs {conv2:>10,.0f}')

# Save final trades
save_folder = 'data/20260430'
os.makedirs(save_folder, exist_ok=True)
best.to_csv(f'{save_folder}/69_final_trades.csv', index=False)
print(f'\n  Saved: {save_folder}/69_final_trades.csv')

# ─────────────────────────────────────────────────────────────────────────────
# F. CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  F. CHARTS')
print(f'{"="*65}')

import sys as _sys
_sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import plot_equity, send_custom_chart
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

DATE_STR = '20260430'

# Chart 1: Equity curve — final best
best_sorted = best.sort_values('date').reset_index(drop=True)
eq  = best_sorted['pnl_conv'].cumsum()
dd2 = eq - eq.cummax()
date_idx = pd.to_datetime(best_sorted['date'].astype(str))
plot_equity(pd.Series(eq.values, index=date_idx),
            pd.Series(dd2.values, index=date_idx),
            '69_final_equity', save_folder,
            title=f'Final Combined (v17a+cam) | {len(best)}t | WR {best_wr:.1f}% | Rs {best_conv:,.0f}')
print('  Chart 1: Final equity sent')

# Chart 2: Strategy contribution bar
strat_conv = best.groupby('strategy')['pnl_conv'].sum().reset_index()
fig2 = go.Figure(go.Bar(
    x=strat_conv['strategy'], y=strat_conv['pnl_conv'],
    marker_color=['#4BC0C0','#26a69a','#FF9F40'],
    text=[f'Rs {v:,.0f}' for v in strat_conv['pnl_conv']],
    textposition='outside'
))
fig2.update_layout(showlegend=False, yaxis_title='Conviction PnL (Rs)')
tv2 = json.loads(fig2.to_json()); tv2['isTvFormat'] = False
send_custom_chart('69_strat_contribution', tv2,
                  title='Strategy Contribution — v17a + cam_l3 + cam_h3 (Conviction)')
print('  Chart 2: Strategy contribution sent')

# Chart 3: Year-wise grouped bar (flat vs conviction)
yr_flat = best.groupby('year')['pnl_65'].sum()
yr_conv = best.groupby('year')['pnl_conv'].sum()
years   = sorted(yr_flat.index)
fig3 = go.Figure()
fig3.add_trace(go.Bar(name='Flat 1-lot', x=years,
    y=[yr_flat.get(y,0) for y in years], marker_color='#4BC0C0'))
fig3.add_trace(go.Bar(name='Conviction', x=years,
    y=[yr_conv.get(y,0) for y in years], marker_color='#26a69a'))
fig3.update_layout(barmode='group', showlegend=True, yaxis_title='PnL (Rs)')
tv3 = json.loads(fig3.to_json()); tv3['isTvFormat'] = False
send_custom_chart('69_yearwise', tv3,
                  title='Year-wise: Flat vs Conviction — All Strategies')
print('  Chart 3: Year-wise sent')

# Chart 4: Score WR bar
score_g = best.groupby('score').agg(n=('win','count'), wr=('win','mean'), conv=('pnl_conv','sum')).reset_index()
score_g['wr'] *= 100
colors = ['#ef5350','#ef5350','#FF9F40','#FF9F40','#26a69a','#26a69a','#26a69a']
fig4 = go.Figure(go.Bar(
    x=[f'S{int(s)}({conv_lots(int(s))}x)' for s in score_g['score']],
    y=score_g['wr'],
    marker_color=colors[:len(score_g)],
    text=[f'{w:.1f}%\nn={int(n)}' for w,n in zip(score_g['wr'],score_g['n'])],
    textposition='outside'
))
fig4.add_hline(y=best_wr, line_dash='dash', line_color='gray',
               annotation_text=f'avg {best_wr:.1f}%')
fig4.update_layout(showlegend=False, yaxis=dict(range=[40,100], title='Win Rate %'))
tv4 = json.loads(fig4.to_json()); tv4['isTvFormat'] = False
send_custom_chart('69_score_wr', tv4,
                  title='Conviction Score WR — All Strategies Combined')
print('  Chart 4: Score WR sent')

print('\nAll done!')
