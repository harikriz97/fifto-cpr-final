"""
72_updated_conviction.py
Updated conviction scoring with PDF insights

Changes from script 69:
  + Feature 7: cpr_dir_aligned  (+5.1% WR lift, cam_h3 +17%)
    CPR midpoint trending 3 days in direction of trade
  - Negative filter: inside_cpr reduces lots by 1 (min 1)
    Inside CPR = trending day expected → bad for option sellers (-7.4% WR)

Sections:
  A. Feature comparison: 6-feat vs 7-feat vs 7-feat+negative
  B. Score distribution + WR per score
  C. Year-wise equity
  D. Combined with intraday v2 (script 70) — final total
  E. Charts
"""
import sys, os, glob, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
import datetime
from my_util import DATA_FOLDER, load_spot_data

SCALE    = 65 / 75
LOT_SIZE = 65
OUT_DIR  = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load trades ───────────────────────────────────────────────────────────────
trades = pd.read_csv('data/56_combined_trades.csv')
trades.columns = [c.lower().replace(' ', '_') for c in trades.columns]
trades['date']     = trades['date'].astype(str).str.replace('-', '').str[:8]
trades['pnl_65']   = (trades['pnl'] * SCALE).round(2)
trades['win']      = trades['pnl_65'] > 0
trades['direction'] = trades['opt']
print(f'Loaded {len(trades)} trades')

# ── Build OHLC features ───────────────────────────────────────────────────────
print('Loading OHLC + features...')
all_folders = sorted(glob.glob(f'{DATA_FOLDER}/20[2-9][0-9][0-9][0-9][0-9][0-9]'))
ohlc_rows = []
for folder in all_folders:
    date = os.path.basename(folder)
    if date < '20210101': continue
    df = load_spot_data(date, 'NIFTY')
    if df is None: continue
    day = df[(df['time'] >= '09:15:00') & (df['time'] <= '15:30:00')]
    if len(day) == 0: continue
    ohlc_rows.append({
        'date': date,
        'open': day.iloc[0]['price'],
        'high': day['price'].max(),
        'low':  day['price'].min(),
        'close': day.iloc[-1]['price'],
    })

ohlc = pd.DataFrame(ohlc_rows).sort_values('date').reset_index(drop=True)

# CPR
ohlc['pp']  = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
ohlc['bc']  = (ohlc['high'] + ohlc['low']) / 2
ohlc['tc']  = 2 * ohlc['pp'] - ohlc['bc']
ohlc['cpr_mid']        = (ohlc['tc'] + ohlc['bc']) / 2
ohlc['cpr_width_pct']  = (ohlc['tc'] - ohlc['bc']).abs() / ohlc['open'] * 100
ohlc['prev_tc']        = ohlc['tc'].shift(1)
ohlc['prev_bc']        = ohlc['bc'].shift(1)
ohlc['prev_close']     = ohlc['close'].shift(1)

# EMA(20) with 40-day seed
ohlc['ema20']      = ohlc['close'].ewm(span=20, adjust=False).mean()
ohlc.loc[:39, 'ema20'] = np.nan
ohlc['prev_ema20'] = ohlc['ema20'].shift(1)

# === EXISTING 6 FEATURES ===
cpr_dir = (ohlc['prev_close'] > ohlc['prev_tc'].shift(1)).astype(int)
ohlc['consec_aligned']  = ((cpr_dir == cpr_dir.shift(1)) & (cpr_dir == cpr_dir.shift(2))).astype(int)
ohlc['cpr_gap']         = ((ohlc['tc'].shift(1) < ohlc['bc']) | (ohlc['bc'].shift(1) > ohlc['tc'])).astype(int)
ohlc['cpr_narrow']      = ohlc['cpr_width_pct'].shift(1).between(0.10, 0.20).astype(int)
ohlc['open_above']      = (ohlc['open'] > ohlc['prev_tc']).astype(int)
ohlc['open_below']      = (ohlc['open'] < ohlc['prev_bc']).astype(int)

# === NEW FEATURE 7: cpr_dir_aligned ===
# CPR midpoint moving in direction of trade for 3 consecutive days
ohlc['prev_mid']  = ohlc['cpr_mid'].shift(1)
ohlc['prev2_mid'] = ohlc['cpr_mid'].shift(2)
ohlc['prev3_mid'] = ohlc['cpr_mid'].shift(3)
ohlc['asc_cpr']   = ((ohlc['prev_mid'] > ohlc['prev2_mid']) &
                     (ohlc['prev2_mid'] > ohlc['prev3_mid'])).astype(int)
ohlc['desc_cpr']  = ((ohlc['prev_mid'] < ohlc['prev2_mid']) &
                     (ohlc['prev2_mid'] < ohlc['prev3_mid'])).astype(int)

# === NEGATIVE FILTER: inside_cpr (no forward bias) ===
# Does TODAY's CPR fit inside YESTERDAY's CPR?
# Today's CPR  = prev_tc (shift1), prev_bc (shift1) — based on yesterday's H,L,C ✓
# Yesterday's CPR = tc.shift(2), bc.shift(2) — based on day-before data ✓
ohlc['inside_cpr'] = (
    (ohlc['tc'].shift(1) < ohlc['tc'].shift(2)) &
    (ohlc['bc'].shift(1) > ohlc['bc'].shift(2))
).astype(int)

# === VIX ===
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

# === DTE ===
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

# ── Join features ─────────────────────────────────────────────────────────────
feat_cols = ['date','vix_ok','consec_aligned','cpr_gap','cpr_narrow',
             'open_above','open_below','dte_sweet','dte',
             'prev_tc','prev_bc','prev_close',
             'asc_cpr','desc_cpr','inside_cpr']
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

# NEW: cpr_dir_aligned — CPR trending in trade direction
t['cpr_dir_aligned'] = (
    ((t['direction'] == 'PE') & (t['asc_cpr']  == 1)) |
    ((t['direction'] == 'CE') & (t['desc_cpr'] == 1))
).astype(int)

FEATS_6 = ['vix_ok','cpr_trend_aligned','consec_aligned','cpr_gap_aligned','dte_sweet','cpr_narrow']
FEATS_7 = FEATS_6 + ['cpr_dir_aligned']

def conv_lots(s, inside=0):
    base = 3 if s >= 4 else (2 if s >= 2 else 1)
    if inside: base = max(1, base - 1)
    return base

t_valid = t.dropna(subset=['vix_ok']).copy()
t_valid['score6']  = t_valid[FEATS_6].fillna(0).astype(int).sum(axis=1)
t_valid['score7']  = t_valid[FEATS_7].fillna(0).astype(int).sum(axis=1)
t_valid['lots6']   = t_valid['score6'].apply(conv_lots)
t_valid['lots7']   = t_valid['score7'].apply(conv_lots)
t_valid['lots7n']  = t_valid.apply(lambda r: conv_lots(r['score7'], r['inside_cpr']), axis=1)
t_valid['pnl_flat']  = t_valid['pnl_65']
t_valid['pnl_conv6'] = (t_valid['pnl_65'] * t_valid['lots6']).round(2)
t_valid['pnl_conv7'] = (t_valid['pnl_65'] * t_valid['lots7']).round(2)
t_valid['pnl_conv7n']= (t_valid['pnl_65'] * t_valid['lots7n']).round(2)
t_valid['year']      = t_valid['date'].str[:4]

print(f'  Trades with features: {len(t_valid)}')

# ── A. FEATURE COMPARISON ─────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  A. CONVICTION COMPARISON: FLAT vs 6-FEAT vs 7-FEAT vs 7-FEAT+NEG')
print(f'{"="*65}')
print(f'  {"Version":<28} {"t":>5} {"WR":>8} {"Total Rs":>14}')
print(f'  {"-"*58}')
base_wr = t_valid['win'].mean()*100
for label, col in [
    ('Flat (1 lot always)',      'pnl_flat'),
    ('6-feature conviction',     'pnl_conv6'),
    ('7-feature (+cpr_dir)',     'pnl_conv7'),
    ('7-feat + inside_cpr neg',  'pnl_conv7n'),
]:
    total = t_valid[col].sum()
    wr = (t_valid['pnl_65'] > 0).mean()*100
    print(f'  {label:<28} {len(t_valid):>5} {wr:>7.1f}% {total:>14,.0f}')

# ── B. SCORE DISTRIBUTION ─────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  B. SCORE DISTRIBUTION — 7-FEATURE')
print(f'{"="*65}')
print(f'  {"Score":>6} {"Lots":>5} {"t":>5} {"WR":>8} {"Avg":>10} {"Total":>12}')
print(f'  {"-"*50}')
for s in sorted(t_valid['score7'].unique()):
    g    = t_valid[t_valid['score7'] == s]
    lots = conv_lots(s)
    wr   = g['win'].mean()*100
    avg  = g['pnl_conv7'].mean()
    tot  = g['pnl_conv7'].sum()
    print(f'  {s:>6} {lots:>5}x {len(g):>5} {wr:>7.1f}% {avg:>10,.0f} {tot:>12,.0f}')

# ── C. YEAR-WISE ──────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  C. YEAR-WISE (7-feat + inside_cpr negative)')
print(f'{"="*65}')
print(f'  {"Year":<6} {"t":>5} {"WR":>8} {"Flat":>10} {"6-feat":>12} {"7-feat+neg":>12}')
print(f'  {"-"*58}')
for yr in sorted(t_valid['year'].unique()):
    g   = t_valid[t_valid['year'] == yr]
    wr  = g['win'].mean()*100
    f0  = g['pnl_flat'].sum()
    f6  = g['pnl_conv6'].sum()
    f7n = g['pnl_conv7n'].sum()
    print(f'  {yr:<6} {len(g):>5} {wr:>7.1f}% {f0:>10,.0f} {f6:>12,.0f} {f7n:>12,.0f}')

# ── D. COMBINED WITH INTRADAY V2 ──────────────────────────────────────────────
print(f'\n{"="*65}')
print('  D. COMBINED WITH INTRADAY V2 (script 70)')
print(f'{"="*65}')
v2_path = 'data/20260430/70_intraday_v2_trades.csv'
if os.path.exists(v2_path):
    tv2 = pd.read_csv(v2_path)
    tv2['date2'] = pd.to_datetime(tv2['date']).dt.strftime('%Y%m%d')
    v2_total = tv2['pnl'].sum()
    base480  = t_valid['pnl_conv6'].sum()
    new7n    = t_valid['pnl_conv7n'].sum()
    print(f'  Script 69 (6-feat conviction, 480t):   ₹{base480:>10,.0f}')
    print(f'  Script 72 (7-feat + neg, 480t):        ₹{new7n:>10,.0f}  ({new7n-base480:+,.0f})')
    print(f'  Script 70 (intraday v2, 70t):          ₹{v2_total:>10,.0f}')
    print(f'  TOTAL (72 + 70):                       ₹{new7n + v2_total:>10,.0f}')

# ── Save ──────────────────────────────────────────────────────────────────────
t_valid['pnl_final'] = t_valid['pnl_conv7n']
save_path = f'{OUT_DIR}/72_final_trades.csv'
t_valid.to_csv(save_path, index=False)
print(f'\n  Saved → {save_path}')

# ── E. CHARTS ─────────────────────────────────────────────────────────────────
from plot_util import plot_equity, send_custom_chart

# Chart 1: 6-feat vs 7-feat+neg equity overlay
def make_equity_line(series_pnl, dates, color, label):
    eq = pd.Series(series_pnl.values, index=pd.to_datetime(dates.values.astype(str))).cumsum()
    pts = [{'time': int(pd.Timestamp(d).timestamp()), 'value': round(float(v), 2),
            'label': label} for d, v in eq.items()]
    return {'id': label, 'label': label, 'data': pts, 'color': color, 'lineWidth': 2}

t_sorted = t_valid.sort_values('date')
lines = [
    make_equity_line(t_sorted['pnl_flat'],   t_sorted['date'], '#888888', 'Flat'),
    make_equity_line(t_sorted['pnl_conv6'],  t_sorted['date'], '#1e88e5', '6-feature'),
    make_equity_line(t_sorted['pnl_conv7n'], t_sorted['date'], '#26a69a', '7-feat+neg'),
]
tv_json = {'candlestick': [], 'volume': [], 'lines': lines, 'markers': [], 'isTvFormat': True}
send_custom_chart('72_conviction_comparison', tv_json,
                  title='Conviction: Flat vs 6-feature vs 7-feature+InsideCPR-neg')

# Chart 2: combined equity (72 + 70)
if os.path.exists(v2_path):
    tv2['date_ts'] = pd.to_datetime(tv2['date'])
    v2_eq = tv2.sort_values('date_ts').set_index('date_ts')['pnl']

    t_eq = t_sorted.copy()
    t_eq['date_ts'] = pd.to_datetime(t_eq['date'].astype(str))
    t_eq = t_eq.set_index('date_ts')['pnl_conv7n']

    all_eq = pd.concat([t_eq, v2_eq]).sort_index().cumsum()
    dd_all = all_eq - all_eq.cummax()
    plot_equity(all_eq, dd_all, '72_combined_equity',
                title='Final Combined (480t conviction + 70t intraday v2)')

# Chart 3: year-wise bar as lines
yr_data = t_valid.groupby('year').agg(
    flat=('pnl_flat','sum'), conv6=('pnl_conv6','sum'), conv7n=('pnl_conv7n','sum')
).reset_index()

lines_yr = []
fake_ts  = {yr: 1609459200 + i*31536000 for i, yr in enumerate(sorted(yr_data['year'].unique()))}
for col, color, lbl in [('flat','#888888','Flat'),('conv6','#1e88e5','6-feat'),('conv7n','#26a69a','7-feat+neg')]:
    pts = [{'time': fake_ts[r['year']], 'value': round(float(r[col]),2), 'label': r['year']}
           for _, r in yr_data.iterrows()]
    lines_yr.append({'id': lbl, 'label': lbl, 'data': pts, 'color': color, 'lineWidth': 3})

tv_yr = {'candlestick': [], 'volume': [], 'lines': lines_yr, 'markers': [], 'isTvFormat': True}
send_custom_chart('72_yearwise', tv_yr, title='Year-wise PnL: Flat vs 6-feat vs 7-feat+neg')

print('\nDone.')
