"""
66_max_pain_backtest.py
Max Pain + PCR analysis on NIFTY options

Sections:
  A. Build max pain + PCR for all trading dates (multiprocessed)
  B. Max pain as filter on existing v17a trades
  C. PCR bias as filter on existing v17a trades
  D. Standalone max pain signal (DTE 1-3, sell at max pain strike)
  E. Charts (push to chat)

Forward bias: OI used is from first tick of 09:15 (prior accumulated OI) — safe.
Expiry: auto-detected per date, handles Thu→Tue switch + holiday shifts.
"""

import sys, os, glob, datetime, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from my_util import (
    DATA_FOLDER, load_spot_data, create_spot_ohlc, load_tick_data
)

DATA_PATH = DATA_FOLDER
LOT_SIZE  = 65
TARGET_MULT = 0.40   # 40% of premium
SL_MULT     = 0.50   # 50% of premium
ENTRY_TIME  = '09:16:02'

# ── Existing v17a trades ──────────────────────────────────────────────────────
TRADES_CSV = 'data/56_combined_trades.csv'


# =============================================================================
# Utility: build full expiry calendar
# =============================================================================
def build_expiry_calendar():
    """Return dict: trading_date (YYYYMMDD) → nearest expiry (YYYYMMDD)"""
    all_folders = sorted(glob.glob(f'{DATA_PATH}/20[2-3][0-9][0-9][0-9][0-9][0-9]'))

    # collect all unique NIFTY weekly expiries
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
                    except:
                        pass

    all_expiries = sorted(expiry_set)

    # map each trading date → nearest upcoming expiry
    calendar = {}
    trading_dates = [os.path.basename(f) for f in all_folders
                     if 2021 <= int(os.path.basename(f)[:4]) <= 2026]
    for td in trading_dates:
        upcoming = [e for e in all_expiries if e >= td]
        if upcoming:
            nearest = upcoming[0]
            dt_exp = datetime.datetime.strptime(nearest, '%Y%m%d')
            dt_td  = datetime.datetime.strptime(td, '%Y%m%d')
            dte = (dt_exp - dt_td).days
            calendar[td] = {'expiry': nearest, 'expiry_6': nearest[2:], 'dte': dte}
    return calendar


# =============================================================================
# Worker: compute max pain + PCR for one trading date
# =============================================================================
def compute_max_pain_for_date(args):
    sys.path.insert(0, '.')
    date, expiry_6 = args
    folder = f'{DATA_PATH}/{date}'
    files = glob.glob(f'{folder}/NIFTY{expiry_6}*.csv')

    records = []
    for f in files:
        name = os.path.basename(f).replace('.csv', '')
        suffix = name[len('NIFTY') + len(expiry_6):]
        opt_type = suffix[-2:]
        if opt_type not in ('CE', 'PE'):
            continue
        try:
            strike = int(suffix[:-2])
        except:
            continue
        try:
            df = pd.read_csv(f, header=None, names=['date', 'time', 'price', 'vol', 'oi'],
                             nrows=3)
            oi = df['oi'].iloc[0]
            records.append({'strike': strike, 'type': opt_type, 'oi': oi})
        except:
            continue

    if len(records) < 10:
        return date, None

    df = pd.DataFrame(records)
    pivot = df.pivot_table(index='strike', columns='type', values='oi', aggfunc='first').fillna(0)
    if 'CE' not in pivot.columns or 'PE' not in pivot.columns:
        return date, None

    strikes = pivot.index.tolist()
    ce_arr  = pivot['CE'].values
    pe_arr  = pivot['PE'].values

    # max pain: vectorised
    pain = []
    for s_idx, s in enumerate(strikes):
        s_arr = np.array(strikes)
        ce_pain = np.sum(ce_arr * np.maximum(0, s - s_arr))
        pe_pain = np.sum(pe_arr * np.maximum(0, s_arr - s))
        pain.append(ce_pain + pe_pain)

    min_idx = int(np.argmin(pain))
    max_pain_strike = strikes[min_idx]

    total_ce_oi = ce_arr.sum()
    total_pe_oi = pe_arr.sum()
    pcr = round(total_pe_oi / total_ce_oi, 4) if total_ce_oi > 0 else None

    return date, {
        'max_pain': max_pain_strike,
        'pcr': pcr,
        'total_ce_oi': int(total_ce_oi),
        'total_pe_oi': int(total_pe_oi),
        'n_strikes': len(strikes)
    }


# =============================================================================
# Section A: build max pain table for all dates
# =============================================================================
print('Building expiry calendar...')
calendar = build_expiry_calendar()
trading_dates = sorted(calendar.keys())
print(f'  {len(trading_dates)} trading dates | expiry calendar built')

args_list = [(d, calendar[d]['expiry_6']) for d in trading_dates]

print(f'Computing max pain (multiprocessed {cpu_count()} cores)...')
with Pool(cpu_count()) as pool:
    results = pool.map(compute_max_pain_for_date, args_list)

mp_rows = []
for date, info in results:
    if info is None:
        continue
    row = {'date': date, **info, **calendar[date]}
    mp_rows.append(row)

mp_df = pd.DataFrame(mp_rows)
mp_df['date'] = mp_df['date'].astype(str)

# load spot open to compare with max pain (9:15 first price)
print('Loading NIFTY spot open prices...')
spot_rows = []
for td in trading_dates:
    tdf = load_spot_data(td, 'NIFTY')
    if tdf is None or len(tdf) == 0:
        continue
    first = tdf[tdf['time'] >= '09:15:00'].head(1)
    if len(first) == 0:
        continue
    spot_rows.append({'date': td, 'spot_open': first.iloc[0]['price']})
spot_open = pd.DataFrame(spot_rows)
spot_open['date'] = spot_open['date'].astype(str)

mp_df = mp_df.merge(spot_open, on='date', how='left')
mp_df['spot_vs_mp'] = (mp_df['spot_open'] - mp_df['max_pain']).round(2)
mp_df['mp_aligned_ce'] = mp_df['spot_vs_mp'] > 50    # spot above MP → sell CE
mp_df['mp_aligned_pe'] = mp_df['spot_vs_mp'] < -50   # spot below MP → sell PE

print(f'\n{"="*65}')
print('  A. MAX PAIN SUMMARY')
print(f'{"="*65}')
print(f'  Dates computed: {len(mp_df)}')
print(f'  Avg spot_vs_mp: {mp_df.spot_vs_mp.mean():.1f} pts')
print(f'  PCR mean: {mp_df.pcr.mean():.3f}  median: {mp_df.pcr.median():.3f}')
print(f'  MP aligned CE days (spot>MP+50): {mp_df.mp_aligned_ce.sum()}')
print(f'  MP aligned PE days (spot<MP-50): {mp_df.mp_aligned_pe.sum()}')
print(f'  Within ±50pts: {((mp_df.spot_vs_mp.abs() <= 50)).sum()}')

# DTE distribution
print(f'\n  DTE distribution:')
for dte_val in sorted(mp_df['dte'].unique())[:8]:
    n = (mp_df['dte'] == dte_val).sum()
    print(f'    DTE={dte_val}: {n} days')


# =============================================================================
# Section B: Max pain filter on v17a trades
# =============================================================================
print(f'\n{"="*65}')
print('  B. MAX PAIN FILTER ON v17a TRADES')
print(f'{"="*65}')

trades = pd.read_csv(TRADES_CSV)
trades.columns = [c.lower().replace(' ', '_') for c in trades.columns]

# normalize date to YYYYMMDD
date_col = 'date' if 'date' in trades.columns else trades.columns[0]
trades['date'] = trades[date_col].astype(str).str.replace('-', '').str[:8]

# identify trade direction from 'opt' column (CE/PE)
if 'opt' in trades.columns:
    trades['direction'] = trades['opt']
elif 'option_type' in trades.columns:
    trades['direction'] = trades['option_type']
elif 'instrument' in trades.columns:
    trades['direction'] = trades['instrument'].apply(
        lambda x: 'CE' if str(x).endswith('CE') else 'PE')
else:
    trades['direction'] = 'CE'

# pnl column
pnl_col = next((c for c in trades.columns if 'pnl' in c or 'profit' in c), None)
if pnl_col is None:
    pnl_col = trades.columns[-1]

trades['pnl'] = pd.to_numeric(trades[pnl_col], errors='coerce').fillna(0)
trades['win'] = trades['pnl'] > 0

print(f'  Loaded {len(trades)} trades, pnl_col={pnl_col}')

# join with max pain
t = trades.merge(mp_df[['date', 'max_pain', 'spot_vs_mp', 'mp_aligned_ce', 'mp_aligned_pe', 'pcr', 'dte']], on='date', how='left')
t_valid = t.dropna(subset=['max_pain'])
print(f'  Trades with max pain data: {len(t_valid)} / {len(trades)}')

# Max pain aligned: CE trade + spot above MP, or PE trade + spot below MP
t_valid = t_valid.copy()
t_valid['mp_aligned'] = (
    ((t_valid['direction'] == 'CE') & (t_valid['mp_aligned_ce'])) |
    ((t_valid['direction'] == 'PE') & (t_valid['mp_aligned_pe']))
)

for label, mask in [('All trades', slice(None)),
                     ('MP aligned', t_valid['mp_aligned'] == True),
                     ('MP misaligned', t_valid['mp_aligned'] == False)]:
    grp = t_valid[mask] if isinstance(mask, pd.Series) else t_valid
    n = len(grp)
    wr = grp['win'].mean() * 100 if n > 0 else 0
    avg = grp['pnl'].mean() if n > 0 else 0
    tot = grp['pnl'].sum() if n > 0 else 0
    print(f'  {label:<20} {n:4d}t | WR {wr:5.1f}% | Avg {avg:7.0f} | Tot {tot:10,.0f}')

# MP lift
aligned_wr   = t_valid[t_valid['mp_aligned']]['win'].mean() * 100 if t_valid['mp_aligned'].sum() > 0 else 0
misaligned_wr= t_valid[~t_valid['mp_aligned']]['win'].mean() * 100 if (~t_valid['mp_aligned']).sum() > 0 else 0
print(f'\n  Lift: +{aligned_wr - misaligned_wr:.1f}% WR when MP aligned')

# By DTE bucket
print(f'\n  By DTE bucket (MP aligned):')
for dte_bucket, label in [((1, 2), 'DTE 1-2'), ((3, 5), 'DTE 3-5'), ((6, 99), 'DTE 6+')]:
    grp = t_valid[(t_valid['mp_aligned']) & (t_valid['dte'].between(*dte_bucket))]
    n = len(grp)
    if n == 0:
        continue
    wr = grp['win'].mean() * 100
    avg = grp['pnl'].mean()
    print(f'    {label:<10} {n:3d}t | WR {wr:5.1f}% | Avg {avg:7.0f}')


# =============================================================================
# Section C: PCR bias filter
# =============================================================================
print(f'\n{"="*65}')
print('  C. PCR BIAS FILTER ON v17a TRADES')
print(f'{"="*65}')

# PCR > 1.2 = bullish (put sellers dominate) → favor PE sell (market won't fall much)
# PCR < 0.8 = bearish → favor CE sell
t_valid = t_valid.copy()
t_valid['pcr_bull']     = t_valid['pcr'] > 1.2
t_valid['pcr_bear']     = t_valid['pcr'] < 0.8
t_valid['pcr_neutral']  = t_valid['pcr'].between(0.8, 1.2)

t_valid['pcr_aligned'] = (
    ((t_valid['direction'] == 'PE') & (t_valid['pcr_bull'])) |
    ((t_valid['direction'] == 'CE') & (t_valid['pcr_bear']))
)

print(f'  PCR distribution:')
print(f'    Bull (>1.2):    {t_valid["pcr_bull"].sum()} days')
print(f'    Neutral (0.8-1.2): {t_valid["pcr_neutral"].sum()} days')
print(f'    Bear (<0.8):    {t_valid["pcr_bear"].sum()} days')

for label, mask in [('PCR aligned', t_valid['pcr_aligned'] == True),
                     ('PCR misaligned', t_valid['pcr_aligned'] == False)]:
    grp = t_valid[mask]
    n = len(grp)
    if n == 0:
        continue
    wr = grp['win'].mean() * 100
    avg = grp['pnl'].mean()
    tot = grp['pnl'].sum()
    print(f'  {label:<20} {n:4d}t | WR {wr:5.1f}% | Avg {avg:7.0f} | Tot {tot:10,.0f}')

pcr_aligned_wr    = t_valid[t_valid['pcr_aligned']]['win'].mean() * 100 if t_valid['pcr_aligned'].sum() > 0 else 0
pcr_misaligned_wr = t_valid[~t_valid['pcr_aligned']]['win'].mean() * 100 if (~t_valid['pcr_aligned']).sum() > 0 else 0
print(f'\n  PCR lift: +{pcr_aligned_wr - pcr_misaligned_wr:.1f}% WR when PCR aligned')

# Combined: MP + PCR both aligned
t_valid['both_aligned'] = t_valid['mp_aligned'] & t_valid['pcr_aligned']
grp = t_valid[t_valid['both_aligned']]
n = len(grp)
wr = grp['win'].mean() * 100 if n > 0 else 0
avg = grp['pnl'].mean() if n > 0 else 0
tot = grp['pnl'].sum() if n > 0 else 0
print(f'\n  MP + PCR both aligned: {n}t | WR {wr:.1f}% | Avg {avg:.0f} | Tot {tot:,.0f}')


# =============================================================================
# Section D: Standalone max pain signal (DTE 1-3)
# =============================================================================
print(f'\n{"="*65}')
print('  D. STANDALONE MAX PAIN SIGNAL (DTE 1-3)')
print(f'{"="*65}')
print('  Signal: if spot_open > max_pain+100 → sell CE at max_pain strike')
print('          if spot_open < max_pain-100 → sell PE at max_pain strike')
print('  Entry: 09:16:02 | Target: 40% | SL: 50% | EOD exit: 15:20')

THRESHOLD   = 100   # pts separation required
DTE_FILTER  = (1, 3)
TARGET_PCT  = 0.40
SL_PCT      = 0.50
EOD_EXIT    = '15:20:00'

# filter dates: DTE 1-3, spot clearly above/below max pain
signal_dates = mp_df[
    (mp_df['dte'].between(*DTE_FILTER)) &
    (mp_df['spot_vs_mp'].abs() > THRESHOLD)
].copy()
signal_dates['opt_type'] = signal_dates['spot_vs_mp'].apply(
    lambda x: 'CE' if x > 0 else 'PE')

print(f'\n  Signal dates found: {len(signal_dates)} (DTE {DTE_FILTER[0]}-{DTE_FILTER[1]}, >{THRESHOLD}pt separation)')

standalone_trades = []

for _, row in signal_dates.iterrows():
    date     = row['date']
    expiry_6 = row['expiry_6']
    strike   = int(row['max_pain'])
    opt_type = row['opt_type']
    dte_val  = row['dte']

    # round strike to nearest 50
    strike = round(strike / 50) * 50

    # find valid strike file
    found_strike = None
    for adj in [0, 50, -50, 100, -100]:
        s_try = strike + adj
        tf = f'{DATA_PATH}/{date}/NIFTY{expiry_6}{s_try}{opt_type}.csv'
        if os.path.exists(tf):
            found_strike = s_try
            break
    if found_strike is None:
        continue
    strike = found_strike
    instrument_name = f'NIFTY{expiry_6}{strike}{opt_type}'

    try:
        ticks = load_tick_data(date, instrument_name, ENTRY_TIME)
    except Exception:
        continue

    if ticks is None or len(ticks) == 0:
        continue

    # find entry tick
    entry_ticks = ticks[ticks['time'] >= ENTRY_TIME]
    if len(entry_ticks) == 0:
        continue

    entry_price = round(entry_ticks.iloc[0]['price'], 2)
    if entry_price <= 0:
        continue

    target_price = round(entry_price * (1 - TARGET_PCT), 2)
    sl_price     = round(entry_price * (1 + SL_PCT), 2)

    exit_price = None
    exit_time  = None
    exit_reason = 'eod'

    for _, tick in entry_ticks.iterrows():
        p = tick['price']
        t = tick['time']
        if t > EOD_EXIT:
            exit_price  = round(p, 2)
            exit_time   = t
            exit_reason = 'eod'
            break
        if p <= target_price:
            exit_price  = round(p, 2)
            exit_time   = t
            exit_reason = 'target'
            break
        if p >= sl_price:
            exit_price  = round(p, 2)
            exit_time   = t
            exit_reason = 'sl'
            break

    if exit_price is None:
        exit_price  = round(entry_ticks.iloc[-1]['price'], 2)
        exit_time   = entry_ticks.iloc[-1]['time']
        exit_reason = 'eod'

    pnl = round((entry_price - exit_price) * LOT_SIZE, 2)

    standalone_trades.append({
        'date':        date,
        'expiry':      expiry_6,
        'dte':         dte_val,
        'strike':      strike,
        'opt_type':    opt_type,
        'spot_open':   row['spot_open'],
        'max_pain':    row['max_pain'],
        'spot_vs_mp':  row['spot_vs_mp'],
        'entry_price': entry_price,
        'target':      target_price,
        'sl':          sl_price,
        'exit_price':  exit_price,
        'exit_reason': exit_reason,
        'pnl':         pnl,
        'win':         pnl > 0,
    })

st_df = pd.DataFrame(standalone_trades)

if len(st_df) == 0:
    print('  No standalone trades found.')
else:
    wr  = st_df['win'].mean() * 100
    avg = st_df['pnl'].mean()
    tot = st_df['pnl'].sum()
    dd  = (st_df['pnl'].cumsum() - st_df['pnl'].cumsum().cummax()).min()
    print(f'\n  All standalone:  {len(st_df)}t | WR {wr:.1f}% | Avg {avg:.0f} | Tot {tot:,.0f} | DD {dd:,.0f}')

    print(f'\n  By exit reason:')
    for reason in ['target', 'sl', 'eod']:
        grp = st_df[st_df['exit_reason'] == reason]
        if len(grp) == 0:
            continue
        print(f'    {reason:<8} {len(grp):3d}t | WR {grp["win"].mean()*100:.1f}%')

    print(f'\n  By DTE:')
    for dte_val in sorted(st_df['dte'].unique()):
        grp = st_df[st_df['dte'] == dte_val]
        n   = len(grp)
        wr2 = grp['win'].mean() * 100
        avg2= grp['pnl'].mean()
        print(f'    DTE={dte_val}: {n:3d}t | WR {wr2:.1f}% | Avg {avg2:.0f}')

    print(f'\n  By opt_type:')
    for ot in ['CE', 'PE']:
        grp = st_df[st_df['opt_type'] == ot]
        n   = len(grp)
        if n == 0:
            continue
        wr2 = grp['win'].mean() * 100
        avg2= grp['pnl'].mean()
        tot2= grp['pnl'].sum()
        print(f'    {ot}: {n:3d}t | WR {wr2:.1f}% | Avg {avg2:.0f} | Tot {tot2:,.0f}')

    print(f'\n  Year-wise:')
    st_df['year'] = st_df['date'].str[:4]
    for yr in sorted(st_df['year'].unique()):
        grp = st_df[st_df['year'] == yr]
        n   = len(grp)
        wr2 = grp['win'].mean() * 100
        tot2= grp['pnl'].sum()
        print(f'    {yr}: {n:3d}t | WR {wr2:.1f}% | Tot {tot2:10,.0f}')

    # save trades
    save_folder = f'data/20260430'
    os.makedirs(save_folder, exist_ok=True)
    out_csv = f'{save_folder}/66_max_pain_trades.csv'
    st_df.to_csv(out_csv, index=False)
    print(f'\n  Saved: {out_csv}')


# =============================================================================
# Section E: Charts
# =============================================================================
print(f'\n{"="*65}')
print('  E. CHARTS')
print(f'{"="*65}')

try:
    from plot_util import super_plotter, plot_equity
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    CHART_FOLDER = f'data/20260430'
    DATE_STR     = '20260430'
    os.makedirs(CHART_FOLDER, exist_ok=True)

    # Chart 1: Max pain vs spot_open scatter by day (DTE=0)
    expiry_days = mp_df[mp_df['dte'] == 0].copy()
    if len(expiry_days) > 0:
        fig1 = go.Figure()
        diff = expiry_days['spot_open'] - expiry_days['max_pain']
        fig1.add_trace(go.Histogram(x=diff, nbinsx=30,
                                    name='Spot - MaxPain (expiry day)',
                                    marker_color='steelblue'))
        fig1.add_vline(x=0, line_dash='dash', line_color='red')
        fig1.update_layout(title='Spot vs Max Pain on Expiry Day',
                           xaxis_title='Spot - MaxPain (pts)',
                           yaxis_title='Count')
        super_plotter(CHART_FOLDER, fig1, '66_mp_dist', DATE_STR,
                      title='Max Pain — Spot vs Max Pain on Expiry Day',
                      file_formats=['json'])
        print('  Chart 1: MaxPain distribution sent')

    # Chart 2: PCR distribution
    pcr_valid = mp_df[mp_df['pcr'].notna() & (mp_df['pcr'] < 3)]
    if len(pcr_valid) > 0:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=pcr_valid['pcr'], nbinsx=40,
                                    marker_color='coral', name='PCR'))
        fig2.add_vline(x=1.0, line_dash='dash', line_color='green', annotation_text='PCR=1')
        fig2.add_vline(x=1.2, line_dash='dot',  line_color='blue',  annotation_text='Bull')
        fig2.add_vline(x=0.8, line_dash='dot',  line_color='red',   annotation_text='Bear')
        fig2.update_layout(title='Daily PCR Distribution (all dates)',
                           xaxis_title='Put-Call Ratio (OI)',
                           yaxis_title='Count')
        super_plotter(CHART_FOLDER, fig2, '66_pcr_dist', DATE_STR,
                      title='PCR Distribution — NIFTY 2021-2026',
                      file_formats=['json'])
        print('  Chart 2: PCR distribution sent')

    # Chart 3: Standalone equity curve
    if len(st_df) > 0:
        equity = st_df['pnl'].cumsum()
        dd_series = equity - equity.cummax()
        plot_equity(equity, dd_series, '66_standalone_equity',
                    title=f'Max Pain Standalone Equity | {len(st_df)}t | WR {wr:.1f}%')
        print('  Chart 3: Standalone equity sent')

    # Chart 4: WR comparison bar — base v17a vs MP-filtered vs PCR-filtered vs both
    categories = ['v17a Base', 'v17a+MP', 'v17a+PCR', 'v17a+MP+PCR']
    wr_vals = [
        t_valid['win'].mean() * 100,
        t_valid[t_valid['mp_aligned']]['win'].mean() * 100 if t_valid['mp_aligned'].sum() > 0 else 0,
        t_valid[t_valid['pcr_aligned']]['win'].mean() * 100 if t_valid['pcr_aligned'].sum() > 0 else 0,
        t_valid[t_valid['both_aligned']]['win'].mean() * 100 if t_valid['both_aligned'].sum() > 0 else 0,
    ]
    n_vals = [
        len(t_valid),
        t_valid['mp_aligned'].sum(),
        t_valid['pcr_aligned'].sum(),
        t_valid['both_aligned'].sum(),
    ]
    fig4 = go.Figure()
    colors = ['steelblue', 'seagreen', 'coral', 'gold']
    for i, (cat, wr_v, n_v) in enumerate(zip(categories, wr_vals, n_vals)):
        fig4.add_trace(go.Bar(name=cat, x=[cat], y=[wr_v],
                              marker_color=colors[i],
                              text=[f'{wr_v:.1f}%<br>n={n_v}'],
                              textposition='outside'))
    fig4.add_hline(y=71.9, line_dash='dash', line_color='gray',
                   annotation_text='Base WR 71.9%')
    fig4.update_layout(title='WR Comparison: v17a + Max Pain + PCR filters',
                       yaxis_title='Win Rate %', showlegend=False,
                       yaxis=dict(range=[60, 90]))
    super_plotter(CHART_FOLDER, fig4, '66_wr_comparison', DATE_STR,
                  title='v17a — WR Comparison with Max Pain & PCR Filters',
                  file_formats=['json'])
    print('  Chart 4: WR comparison sent')

except Exception as e:
    print(f'  Chart error: {e}')

print('\nAll done!')
