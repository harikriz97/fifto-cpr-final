"""
67_final_combination.py
ALL backtests → best ML combination → full equity + trade details

Sections:
  A. Master summary table (all 20 backtests)
  B. Feature lift table (all 7 signals)
  C. Pairwise combination matrix (all pairs WR)
  D. ML conviction scoring (7 features → 1/2/3 lots)
  E. Full trade details (with all feature flags)
  F. Equity curve + year-wise bar + feature WR heatmap
"""

import sys, os, glob, datetime, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from my_util import DATA_FOLDER, load_spot_data

DATA_PATH = DATA_FOLDER
LOT_SIZE  = 65
SCALE     = 65 / 75   # rescale older LOT=75 trades

# ─────────────────────────────────────────────────────────────────────────────
# A. MASTER SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 70)
print('  A. MASTER SUMMARY — ALL BACKTESTS')
print('=' * 70)
rows = [
    ('54/56', 'v17a CPR + EMA (LOT=75)',             356,  71.9, 340119, 'BASE'),
    ('55/56', 'Camarilla L3/H3 touch (LOT=75)',      124,  70.2, 159062, 'ADD-ON'),
    ('56',    'Combined v17a+Cam (LOT=75)',           525,  72.2, 499181, 'COMBINED'),
    ('59',    'VIX filter improvement',              260,  75.8,   None, '+5.8% WR lift'),
    ('59',    'Narrow CPR filter',                   201,  73.6,   None, '+1.7% lift'),
    ('60',    'Gap fill strategy',                   462,  65.0,   None, 'NEGATIVE PNL'),
    ('61',    '9:20 contra + ORB buy',               None, None,   None, 'BOTH NEGATIVE'),
    ('62',    'Full report LOT=65',                  525,  72.2, 467028, 'LOT=65 base'),
    ('63',    'Conviction 1x/2x lots',               356,  71.9, 500978, '+47% vs base'),
    ('64/S1', 'DTE sweet spot DTE=3 Monday',          92,  80.2,   None, 'DTE=3 BEST'),
    ('64/S2', 'CPR narrow 0.10-0.20%',              146,  75.5,   None, 'WIDTH FILTER'),
    ('64/S3', 'Weekly ±250pt range filter',          None, 74.1,   None, 'RANGE FILTER'),
    ('64/S4', 'H4/L4 breakout buy',                 761,  42.2, 139451, 'WEAK - SKIP'),
    ('64/S6', 'CPR gap (virgin CPR) + VIX',         None, 81.3,   None, 'STRONG'),
    ('65/G',  '5-signal conviction 1/2/3 lots',      356,  71.9, 667098, 'BEST: +96%'),
    ('65/H',  '3-way OOS validation 2024+',          137,  73.7,   None, 'VALIDATED'),
    ('66/B',  'Max pain filter on v17a',              43,  74.4,  56055, '+3.6% lift'),
    ('66/C',  'PCR filter on v17a',                 143,  72.0,   None, '+1.3% lift'),
    ('66/D',  'Standalone max pain DTE1-3',          267,  55.1,  58130, 'WEAK'),
]
df_summary = pd.DataFrame(rows, columns=['Script', 'Strategy', 'Trades', 'WR%', 'PnL', 'Verdict'])
print(df_summary.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Build feature table for v17a trades
# ─────────────────────────────────────────────────────────────────────────────
print('\nLoading v17a trades...')
trades = pd.read_csv('data/56_combined_trades.csv')
trades.columns = [c.lower().replace(' ', '_') for c in trades.columns]
trades['date'] = trades['date'].astype(str).str.replace('-', '').str[:8]
trades = trades[trades['strategy'] == 'v17a'].copy()
trades['pnl_scaled'] = (trades['pnl'] * SCALE).round(2)
trades['win'] = trades['pnl_scaled'] > 0
print(f'  v17a: {len(trades)}t | WR {trades["win"].mean()*100:.1f}% | PnL Rs {trades["pnl_scaled"].sum():,.0f}')

# ── Load OHLC for CPR / EMA features ─────────────────────────────────────────
print('Loading OHLC data...')
all_dates = sorted(glob.glob(f'{DATA_PATH}/20[2-9][0-9][0-9][0-9][0-9][0-9]'))
ohlc_rows = []
for folder in all_dates:
    date = os.path.basename(folder)
    df = load_spot_data(date, 'NIFTY')
    if df is None: continue
    day = df[(df['time'] >= '09:15:00') & (df['time'] <= '15:30:00')]
    if len(day) == 0: continue
    ohlc_rows.append({
        'date': date,
        'open':  round(day.iloc[0]['price'], 2),
        'high':  round(day['price'].max(), 2),
        'low':   round(day['price'].min(), 2),
        'close': round(day.iloc[-1]['price'], 2),
    })
ohlc = pd.DataFrame(ohlc_rows).sort_values('date').reset_index(drop=True)
print(f'  OHLC: {len(ohlc)}d')

# ── CPR features ──────────────────────────────────────────────────────────────
ohlc['pp']  = ((ohlc['high'] + ohlc['low'] + ohlc['close']) / 3).round(2)
ohlc['bc']  = ((ohlc['high'] + ohlc['low']) / 2).round(2)
ohlc['tc']  = (2 * ohlc['pp'] - ohlc['bc']).round(2)
ohlc['cpr_width_pct'] = ((ohlc['tc'] - ohlc['bc']).abs() / ohlc['open'] * 100).round(4)
ohlc['cpr_direction'] = (ohlc['tc'] > ohlc['bc'].shift(1)).astype(int)

# ── EMA(20) — needs 40+ days seed ────────────────────────────────────────────
ohlc['ema20'] = ohlc['close'].ewm(span=20, adjust=False).mean()
ohlc.loc[:39, 'ema20'] = np.nan

# ── Day-before values (shift 1 = no forward bias) ────────────────────────────
ohlc['prev_tc']    = ohlc['tc'].shift(1)
ohlc['prev_bc']    = ohlc['bc'].shift(1)
ohlc['prev_pp']    = ohlc['pp'].shift(1)
ohlc['prev_high']  = ohlc['high'].shift(1)
ohlc['prev_low']   = ohlc['low'].shift(1)
ohlc['prev_close'] = ohlc['close'].shift(1)
ohlc['prev_ema20'] = ohlc['ema20'].shift(1)
ohlc['today_open'] = ohlc['open']

# ── Feature: CPR trend aligned (prev close above/below prev CPR → next day bias) ─
ohlc['cpr_trend_aligned_raw'] = (
    ((ohlc['prev_close'] > ohlc['prev_tc'])) |   # bullish → PE sell on up days
    ((ohlc['prev_close'] < ohlc['prev_bc']))       # bearish → CE sell on down days
).astype(int)

# ── Feature: consecutive aligned (3-day CPR trend streak) ────────────────────
cpr_dir = (ohlc['prev_close'] > ohlc['prev_pp']).astype(int)
ohlc['consec_aligned'] = (
    (cpr_dir == cpr_dir.shift(1)) & (cpr_dir == cpr_dir.shift(2))
).astype(int)

# ── Feature: CPR gap (today CPR doesn't overlap yesterday CPR) ───────────────
ohlc['cpr_gap'] = (
    (ohlc['tc'].shift(1) < ohlc['bc']) |
    (ohlc['bc'].shift(1) > ohlc['tc'])
).astype(int)

# ── Feature: open position vs prev-day CPR ───────────────────────────────────
ohlc['open_above_cpr'] = (ohlc['today_open'] > ohlc['prev_tc']).astype(int)
ohlc['open_below_cpr'] = (ohlc['today_open'] < ohlc['prev_bc']).astype(int)

# ── Feature: CPR narrow ───────────────────────────────────────────────────────
ohlc['cpr_narrow'] = ohlc['cpr_width_pct'].shift(1).between(0.10, 0.20).astype(int)

# ── Load VIX ──────────────────────────────────────────────────────────────────
print('Loading VIX...')
vix_rows = []
for folder in all_dates:
    date = os.path.basename(folder)
    fp = f'{folder}/INDIAVIX.csv'
    if not os.path.exists(fp): continue
    df = pd.read_csv(fp, header=None, names=['date','time','price','vol','oi'])
    vix_rows.append({'date': date, 'vix': round(df['price'].mean(), 2)})
vix_df = pd.DataFrame(vix_rows)
ohlc = ohlc.merge(vix_df, on='date', how='left')
ohlc['vix_ma20'] = ohlc['vix'].rolling(20).mean()
ohlc['vix_ok']   = (ohlc['vix'].shift(1) < ohlc['vix_ma20'].shift(1)).astype(int)
print(f'  VIX: {vix_df["date"].nunique()}d')

# ── Load max pain + PCR (multiprocessed) ─────────────────────────────────────
print('Building expiry calendar + max pain...')
expiry_set = set()
for folder in all_dates:
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
calendar = {}
trading_dates = [os.path.basename(f) for f in all_dates if 2021 <= int(os.path.basename(f)[:4]) <= 2026]
for td in trading_dates:
    upcoming = [e for e in all_expiries if e >= td]
    if upcoming:
        nearest = upcoming[0]
        dt_exp = datetime.datetime.strptime(nearest, '%Y%m%d')
        dt_td  = datetime.datetime.strptime(td, '%Y%m%d')
        dte = (dt_exp - dt_td).days
        calendar[td] = {'expiry': nearest, 'expiry_6': nearest[2:], 'dte': dte}

def _compute_mp(args):
    sys.path.insert(0, '.')
    date, expiry_6 = args
    folder = f'{DATA_PATH}/{date}'
    files = glob.glob(f'{folder}/NIFTY{expiry_6}*.csv')
    records = []
    for f in files:
        name = os.path.basename(f).replace('.csv','')
        suffix = name[len('NIFTY')+len(expiry_6):]
        opt_type = suffix[-2:]
        if opt_type not in ('CE','PE'): continue
        try:
            strike = int(suffix[:-2])
            df = pd.read_csv(f, header=None, names=['d','t','p','v','oi'], nrows=3)
            oi = df['oi'].iloc[0]
            records.append({'strike': strike, 'type': opt_type, 'oi': oi})
        except: continue
    if len(records) < 8: return date, None
    df2 = pd.DataFrame(records)
    pv = df2.pivot_table(index='strike', columns='type', values='oi', aggfunc='first').fillna(0)
    if 'CE' not in pv.columns or 'PE' not in pv.columns: return date, None
    strikes = pv.index.tolist()
    ce_arr = pv['CE'].values; pe_arr = pv['PE'].values
    pain = [np.sum(ce_arr * np.maximum(0, s - np.array(strikes))) +
            np.sum(pe_arr * np.maximum(0, np.array(strikes) - s)) for s in strikes]
    mp_strike = strikes[int(np.argmin(pain))]
    pcr = round(pe_arr.sum() / ce_arr.sum(), 4) if ce_arr.sum() > 0 else None
    return date, {'max_pain': mp_strike, 'pcr': pcr, 'dte': calendar[date]['dte']}

args_list = [(d, calendar[d]['expiry_6']) for d in trading_dates]
with Pool(cpu_count()) as pool:
    mp_results = pool.map(_compute_mp, args_list)

mp_rows = []
for date, info in mp_results:
    if info: mp_rows.append({'date': date, **info})
mp_df = pd.DataFrame(mp_rows)
print(f'  Max pain: {len(mp_df)}d computed')

# join with ohlc for spot_open
mp_df = mp_df.merge(ohlc[['date','today_open']], on='date', how='left')
mp_df['spot_vs_mp'] = (mp_df['today_open'] - mp_df['max_pain']).round(2)

# ── Join all features with trades ─────────────────────────────────────────────
feature_cols = ['date','cpr_trend_aligned_raw','consec_aligned','cpr_gap','cpr_narrow',
                'open_above_cpr','open_below_cpr','vix_ok','prev_tc','prev_bc','cpr_width_pct']
t = trades.merge(ohlc[feature_cols], on='date', how='left')
t = t.merge(mp_df[['date','max_pain','spot_vs_mp','pcr','dte']], on='date', how='left')

# ── Direction-aware features ──────────────────────────────────────────────────
t['direction'] = t['opt']   # CE or PE

# cpr_gap aligned: gap day + direction (CE when gap up, PE when gap down)
# gap_up = today_open > prev_tc (prev CPR doesn't overlap up)
# gap_dn = today_open < prev_bc
t['gap_up'] = t['today_open'] > t['prev_tc'] if 'today_open' in t.columns else (t['spot_vs_mp'] > 0)
t['today_open_x'] = t.merge(ohlc[['date','today_open']], on='date', how='left')['today_open_y'] if 'today_open_y' in t.merge(ohlc[['date','today_open']], on='date', how='left').columns else t.get('today_open', pd.Series([np.nan]*len(t)))

# simpler: re-derive from ohlc join
ohlc_sub = ohlc[['date','today_open']].copy()
t = t.drop(columns=[c for c in t.columns if c=='today_open'], errors='ignore')
t = t.merge(ohlc_sub, on='date', how='left')

t['cpr_gap_aligned'] = (
    ((t['cpr_gap'] == 1) & (t['today_open'] > t['prev_tc']) & (t['direction'] == 'PE')) |
    ((t['cpr_gap'] == 1) & (t['today_open'] < t['prev_bc']) & (t['direction'] == 'CE'))
).astype(int)

t['cpr_trend_aligned'] = (
    ((t['cpr_trend_aligned_raw'] == 1) & (t['direction'] == 'CE') & (t['prev_close_x'] < t['prev_bc'])) |
    ((t['cpr_trend_aligned_raw'] == 1) & (t['direction'] == 'PE') & (t['prev_close_x'] > t['prev_tc']))
).astype(int) if 'prev_close_x' in t.columns else t['cpr_trend_aligned_raw']

# if prev_close not yet in t, join it
if 'prev_close' not in t.columns:
    t = t.merge(ohlc[['date','prev_close']], on='date', how='left')

t['cpr_trend_aligned'] = (
    ((t['direction'] == 'CE') & (t['prev_close'] < t['prev_bc'])) |
    ((t['direction'] == 'PE') & (t['prev_close'] > t['prev_tc']))
).astype(int)

t['open_aligned'] = (
    ((t['direction'] == 'CE') & (t['open_below_cpr'] == 1)) |
    ((t['direction'] == 'PE') & (t['open_above_cpr'] == 1))
).astype(int)

t['mp_aligned'] = (
    ((t['direction'] == 'CE') & (t['spot_vs_mp'] < -50)) |
    ((t['direction'] == 'PE') & (t['spot_vs_mp'] >  50))
).astype(int)

t['pcr_aligned'] = (
    ((t['direction'] == 'PE') & (t['pcr'] > 1.2)) |
    ((t['direction'] == 'CE') & (t['pcr'] < 0.8))
).astype(int)

t['dte_sweet'] = t['dte'].between(3, 5).astype(int)

t_valid = t.dropna(subset=['vix_ok','cpr_gap_aligned','consec_aligned']).copy()
print(f'\n  Trades with full features: {len(t_valid)}')


# ─────────────────────────────────────────────────────────────────────────────
# B. FEATURE LIFT TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('  B. FEATURE LIFT TABLE (7 signals)')
print(f'{"="*70}')

FEATURES = ['vix_ok','cpr_trend_aligned','consec_aligned','cpr_gap_aligned',
            'open_aligned','mp_aligned','pcr_aligned','dte_sweet','cpr_narrow']

base_wr = t_valid['win'].mean() * 100
print(f'  Base WR: {base_wr:.1f}%  ({len(t_valid)}t)\n')
print(f'  {"Feature":<22} {"N_yes":>6} {"WR_yes":>8} {"WR_no":>8} {"Lift":>8}  Phi')
print(f'  {"-"*65}')

results = {}
for feat in FEATURES:
    if feat not in t_valid.columns: continue
    yes = t_valid[t_valid[feat] == 1]
    no  = t_valid[t_valid[feat] == 0]
    wr_yes = yes['win'].mean() * 100 if len(yes) > 0 else 0
    wr_no  = no['win'].mean()  * 100 if len(no)  > 0 else 0
    lift   = wr_yes - wr_no
    n = len(t_valid)
    n1 = (t_valid[feat] == 1).sum(); n0 = n - n1
    p1 = (t_valid[(t_valid[feat]==1)&(t_valid['win'])].shape[0])
    p0 = (t_valid[(t_valid[feat]==0)&(t_valid['win'])].shape[0])
    # phi coefficient
    a,b,c,d = p1, n1-p1, p0, n0-p0
    denom = np.sqrt((a+b)*(c+d)*(a+c)*(b+d)) if (a+b)*(c+d)*(a+c)*(b+d) > 0 else 1
    phi = round((a*d - b*c) / denom, 3)
    results[feat] = {'wr_yes': wr_yes, 'wr_no': wr_no, 'lift': lift, 'n_yes': len(yes), 'phi': phi}
    marker = ' <<' if lift > 3 else ''
    print(f'  {feat:<22} {len(yes):>6} {wr_yes:>7.1f}% {wr_no:>7.1f}% {lift:>+7.1f}%  {phi:+.3f}{marker}')


# ─────────────────────────────────────────────────────────────────────────────
# C. PAIRWISE COMBINATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('  C. PAIRWISE COMBINATION MATRIX (WR% | trades)')
print(f'{"="*70}')

TOP_FEATS = ['vix_ok','consec_aligned','cpr_gap_aligned','cpr_trend_aligned','mp_aligned','dte_sweet']

header = f'  {"":>18}' + ''.join(f'  {f[:10]:>12}' for f in TOP_FEATS)
print(header)
for f1 in TOP_FEATS:
    row = f'  {f1[:18]:>18}'
    for f2 in TOP_FEATS:
        if f1 == f2:
            grp = t_valid[t_valid[f1] == 1]
            wr = grp['win'].mean()*100 if len(grp) > 0 else 0
            row += f'  {wr:>5.1f}%({len(grp):>3d})'
        else:
            grp = t_valid[(t_valid[f1]==1) & (t_valid[f2]==1)]
            wr = grp['win'].mean()*100 if len(grp) > 0 else 0
            row += f'  {wr:>5.1f}%({len(grp):>3d})'
    print(row)


# ─────────────────────────────────────────────────────────────────────────────
# D. ML CONVICTION SCORING → 1/2/3 LOTS
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('  D. 6-FEATURE CONVICTION SCORING → 1/2/3 LOTS')
print(f'{"="*70}')

CONV_FEATS = ['vix_ok','cpr_trend_aligned','consec_aligned','cpr_gap_aligned','mp_aligned','dte_sweet']
t_valid = t_valid.copy()
t_valid['score'] = t_valid[CONV_FEATS].fillna(0).astype(int).sum(axis=1)

def conv_lots(s):
    if s >= 4: return 3
    if s >= 2: return 2
    return 1

t_valid['lots'] = t_valid['score'].apply(conv_lots)
t_valid['pnl_conv'] = (t_valid['pnl_scaled'] / 1 * t_valid['lots']).round(2)

print(f'  Conviction rules:')
print(f'    Score 0-1 → 1 lot | Score 2-3 → 2 lots | Score 4-6 → 3 lots')
print(f'\n  Score → WR → Lots:')
print(f'  {"Score":>6} {"Lots":>6} {"t":>6} {"WR":>8} {"Avg PnL":>10} {"Total":>12}')
print(f'  {"-"*55}')
for s in sorted(t_valid['score'].unique()):
    grp = t_valid[t_valid['score'] == s]
    lots = conv_lots(s)
    wr   = grp['win'].mean() * 100
    avg  = grp['pnl_conv'].mean()
    tot  = grp['pnl_conv'].sum()
    print(f'  {s:>6} {lots:>6}x {len(grp):>5} {wr:>7.1f}% {avg:>10,.0f} {tot:>12,.0f}')

flat_pnl = t_valid['pnl_scaled'].sum()
conv_pnl = t_valid['pnl_conv'].sum()
flat_wr  = t_valid['win'].mean() * 100
conv_dd  = (t_valid['pnl_conv'].cumsum() - t_valid['pnl_conv'].cumsum().cummax()).min()
flat_dd  = (t_valid['pnl_scaled'].cumsum() - t_valid['pnl_scaled'].cumsum().cummax()).min()

print(f'\n  Flat 1-lot:   {len(t_valid)}t | WR {flat_wr:.1f}% | PnL Rs {flat_pnl:>10,.0f} | DD Rs {flat_dd:>10,.0f}')
print(f'  Conviction:   {len(t_valid)}t | WR {flat_wr:.1f}% | PnL Rs {conv_pnl:>10,.0f} | DD Rs {conv_dd:>10,.0f}')
print(f'  Improvement:  Rs {conv_pnl - flat_pnl:>+10,.0f}  ({(conv_pnl/flat_pnl - 1)*100:+.1f}%)')
print(f'  Lot dist: {dict(t_valid["lots"].value_counts().sort_index())}')

print(f'\n  Year-wise:')
print(f'  {"Year":>6} {"t":>5} {"Flat WR":>9} {"Flat Rs":>12} {"Conv Rs":>12} {"Lift Rs":>12}')
t_valid['year'] = t_valid['date'].str[:4]
for yr in sorted(t_valid['year'].unique()):
    grp = t_valid[t_valid['year'] == yr]
    wr2  = grp['win'].mean() * 100
    fp   = grp['pnl_scaled'].sum()
    cp   = grp['pnl_conv'].sum()
    print(f'  {yr:>6} {len(grp):>5} {wr2:>8.1f}% {fp:>12,.0f} {cp:>12,.0f} {cp-fp:>+12,.0f}')


# ─────────────────────────────────────────────────────────────────────────────
# E. FULL TRADE DETAILS
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('  E. FULL TRADE DETAILS (saving to CSV)')
print(f'{"="*70}')

detail_cols = ['date','year','strategy','zone','bias','direction','strike_type',
               'entry_time','ep','xp','exit_reason',
               'pnl_scaled','win','score','lots','pnl_conv',
               'vix_ok','cpr_trend_aligned','consec_aligned','cpr_gap_aligned',
               'mp_aligned','dte_sweet','dte','cpr_width_pct']
detail_cols = [c for c in detail_cols if c in t_valid.columns]
detail_df   = t_valid[detail_cols].copy()
detail_df   = detail_df.sort_values('date').reset_index(drop=True)

save_folder = 'data/20260430'
os.makedirs(save_folder, exist_ok=True)
out_csv = f'{save_folder}/67_final_trades.csv'
detail_df.to_csv(out_csv, index=False)
print(f'  Saved {len(detail_df)} trades → {out_csv}')
print(f'  Columns: {detail_cols}')

print(f'\n  Sample (last 5 trades):')
print(detail_df.tail(5)[['date','zone','direction','ep','xp','pnl_conv','score','lots','win']].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# F. CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('  F. CHARTS')
print(f'{"="*70}')

import sys as _sys
_sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import plot_equity, send_custom_chart
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

DATE_STR = '20260430'

# Chart 1: Equity curve — flat vs conviction
flat_eq  = t_valid['pnl_scaled'].cumsum().reset_index(drop=True)
conv_eq  = t_valid['pnl_conv'].cumsum().reset_index(drop=True)
flat_dd  = (flat_eq - flat_eq.cummax())
conv_dd  = (conv_eq - conv_eq.cummax())

# use trade dates as index
date_idx = pd.to_datetime(t_valid['date'].astype(str)).reset_index(drop=True)
flat_s   = pd.Series(flat_eq.values, index=date_idx)
conv_s   = pd.Series(conv_eq.values, index=date_idx)
conv_dd_s= pd.Series(conv_dd.values, index=date_idx)

# send conviction equity
plot_equity(conv_s, conv_dd_s, '67_conviction_equity', save_folder,
            title=f'v17a Conviction (6-feature) | {len(t_valid)}t | WR {flat_wr:.1f}% | Rs {conv_pnl:,.0f}')
print('  Chart 1: Conviction equity sent')

# Chart 2: Flat vs Conviction equity overlay
eq_data_flat = [{'time': int(pd.Timestamp(t).timestamp()), 'value': round(float(v), 2)}
                for t, v in flat_s.items() if pd.notna(v)]
eq_data_conv = [{'time': int(pd.Timestamp(t).timestamp()), 'value': round(float(v), 2)}
                for t, v in conv_s.items() if pd.notna(v)]
tv2 = {
    'lines': [
        {'id': 'flat',      'label': f'Flat 1-lot Rs{flat_pnl:,.0f}',  'seriesType': 'baseline', 'baseValue': 0, 'color': '#4BC0C0', 'data': eq_data_flat},
        {'id': 'conviction','label': f'Conviction Rs{conv_pnl:,.0f}', 'seriesType': 'baseline', 'baseValue': 0, 'color': '#26a69a', 'data': eq_data_conv},
    ],
    'candlestick': [], 'volume': [], 'isTvFormat': False
}
send_custom_chart('67_flat_vs_conv', tv2,
                  title=f'Flat vs Conviction Equity | Rs {flat_pnl:,.0f} → Rs {conv_pnl:,.0f}')
print('  Chart 2: Flat vs Conviction sent')

# Chart 3: Year-wise PnL bars (conviction)
yr_data = t_valid.groupby('year')['pnl_conv'].sum().reset_index()
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=yr_data['year'], y=yr_data['pnl_conv'],
    marker_color=['#26a69a' if v >= 0 else '#ef5350' for v in yr_data['pnl_conv']],
    text=[f'Rs {v:,.0f}' for v in yr_data['pnl_conv']],
    textposition='outside', name='Conviction PnL'
))
fig3.update_layout(showlegend=False, yaxis_title='PnL (Rs)')
tv3 = json.loads(fig3.to_json()); tv3['isTvFormat'] = False
send_custom_chart('67_yearwise', tv3, title='Year-wise PnL — v17a Conviction (6-feature, 1/2/3 lots)')
print('  Chart 3: Year-wise PnL sent')

# Chart 4: Feature WR heatmap
feat_labels = ['vix_ok','cpr_trend','consec','cpr_gap','mp_align','dte_sweet']
feat_cols   = ['vix_ok','cpr_trend_aligned','consec_aligned','cpr_gap_aligned','mp_aligned','dte_sweet']
# build 6x6 matrix
z_matrix = []
for f1 in feat_cols:
    row = []
    for f2 in feat_cols:
        if f1 == f2:
            grp = t_valid[t_valid[f1] == 1]
        else:
            grp = t_valid[(t_valid[f1]==1) & (t_valid[f2]==1)]
        wr_val = round(grp['win'].mean() * 100, 1) if len(grp) >= 5 else None
        row.append(wr_val)
    z_matrix.append(row)

tv4 = {'lines': [{
    'id': 'wr_heatmap', 'label': 'Win Rate Heatmap', 'seriesType': 'heatmap',
    'z': z_matrix, 'xLabels': feat_labels, 'yLabels': feat_labels,
    'colorscale': [[0,'#ef5350'],[0.5,'#ffffff'],[1,'#26a69a']],
    'zmid': 71.9, 'textTemplate': '%{z:.1f}%',
    'xTitle': 'Feature', 'yTitle': 'Feature', 'colorbarTitle': 'WR%',
    'yReversed': False
}], 'candlestick': [], 'volume': [], 'isTvFormat': False}
send_custom_chart('67_feature_heatmap', tv4,
                  title='Feature Pair WR Heatmap — v17a Trades')
print('  Chart 4: Feature heatmap sent')

# Chart 5: Score distribution bar
score_stats = t_valid.groupby('score').agg(
    n=('win','count'), wr=('win','mean'), avg=('pnl_conv','mean')).reset_index()
score_stats['wr'] *= 100
fig5 = make_subplots(rows=1, cols=2,
                     subplot_titles=['WR% by Score', 'Avg P&L by Score (conviction lots)'])
colors_sc = ['#ef5350','#ef5350','#FF9F40','#FF9F40','#26a69a','#26a69a','#26a69a']
fig5.add_trace(go.Bar(x=[f'S{s}({conv_lots(s)}x)' for s in score_stats['score']],
                       y=score_stats['wr'],
                       marker_color=colors_sc[:len(score_stats)],
                       text=[f'{w:.1f}%<br>n={n}' for w,n in zip(score_stats['wr'],score_stats['n'])],
                       textposition='outside'), row=1, col=1)
fig5.add_trace(go.Bar(x=[f'S{s}({conv_lots(s)}x)' for s in score_stats['score']],
                       y=score_stats['avg'],
                       marker_color=colors_sc[:len(score_stats)],
                       text=[f'Rs{a:,.0f}' for a in score_stats['avg']],
                       textposition='outside'), row=1, col=2)
fig5.add_hline(y=71.9, row=1, col=1, line_dash='dash', line_color='gray', annotation_text='base 71.9%')
fig5.update_layout(showlegend=False)
tv5 = json.loads(fig5.to_json()); tv5['isTvFormat'] = False
send_custom_chart('67_score_breakdown', tv5,
                  title='Conviction Score — WR% & Avg P&L by Score (1/2/3 lots)')
print('  Chart 5: Score breakdown sent')

print('\nAll done!')
