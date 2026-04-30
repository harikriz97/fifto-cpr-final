"""
71_pdf_concepts_check.py
Check which PDF concepts add real lift on our 480 trades

Concepts to test:
  1. virgin_cpr    — prev day price range didn't touch prev day's CPR (TC/BC)
  2. inside_cpr    — today's CPR inside yesterday's CPR (width + position)
  3. outside_cpr   — today's CPR outside/wider than yesterday's
  4. asc_cpr       — CPR ascending 3 days (bullish bias)
  5. desc_cpr      — CPR descending 3 days (bearish bias)
  6. od_pattern    — first 5-min candle closes above PDH or below PDL (Open Drive)
  7. wide_cpr_side — wide CPR + both CE/PE trade possible (sideways → straddle)

For each feature:
  - WR when feature ON vs OFF
  - Correlation with existing conviction features
  - Verdict: add to conviction / new trades / skip
"""
import sys, os, glob, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import DATA_FOLDER, load_spot_data

SCALE    = 65 / 75
LOT_SIZE = 65
OUT_DIR  = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load trades ───────────────────────────────────────────────────────────────
trades = pd.read_csv('data/56_combined_trades.csv')
trades.columns = [c.lower().replace(' ', '_') for c in trades.columns]
trades['date']    = trades['date'].astype(str).str.replace('-', '').str[:8]
trades['pnl_65']  = (trades['pnl'] * SCALE).round(2)
trades['win']     = trades['pnl_65'] > 0
trades['direction'] = trades['opt']
print(f'Loaded {len(trades)} trades')

# ── Build daily OHLC ──────────────────────────────────────────────────────────
print('Loading daily OHLC...')
all_folders = sorted(glob.glob(f'{DATA_FOLDER}/20[2-9][0-9][0-9][0-9][0-9][0-9]'))
rows = []
for folder in all_folders:
    date = os.path.basename(folder)
    if date < '20210101': continue
    df = load_spot_data(date, 'NIFTY')
    if df is None: continue
    day = df[(df['time'] >= '09:15:00') & (df['time'] <= '15:30:00')]
    if len(day) < 2: continue
    rows.append({
        'date': date,
        'open': day.iloc[0]['price'],
        'high': day['price'].max(),
        'low':  day['price'].min(),
        'close': day.iloc[-1]['price'],
        'open5': day[day['time'] <= '09:20:00']['price'].iloc[-1] if len(day[day['time'] <= '09:20:00']) > 0 else day.iloc[0]['price'],
    })

ohlc = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
print(f'  {len(ohlc)} days loaded')

# ── CPR calculations ──────────────────────────────────────────────────────────
ohlc['pp'] = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
ohlc['bc'] = (ohlc['high'] + ohlc['low']) / 2
ohlc['tc'] = 2 * ohlc['pp'] - ohlc['bc']
ohlc['cpr_mid']   = (ohlc['tc'] + ohlc['bc']) / 2
ohlc['cpr_width'] = (ohlc['tc'] - ohlc['bc']).abs()

# prev day values
ohlc['prev_tc']    = ohlc['tc'].shift(1)
ohlc['prev_bc']    = ohlc['bc'].shift(1)
ohlc['prev_mid']   = ohlc['cpr_mid'].shift(1)
ohlc['prev_width'] = ohlc['cpr_width'].shift(1)
ohlc['prev_high']  = ohlc['high'].shift(1)
ohlc['prev_low']   = ohlc['low'].shift(1)
ohlc['prev_close'] = ohlc['close'].shift(1)

# 2 days ago CPR mid for ascending/descending
ohlc['prev2_mid'] = ohlc['cpr_mid'].shift(2)
ohlc['prev3_mid'] = ohlc['cpr_mid'].shift(3)

# ── Feature 1: Virgin CPR ─────────────────────────────────────────────────────
# Prev day's price range (high/low) did NOT overlap with prev day's CPR (tc/bc)
# i.e. prev day high < prev day tc  OR  prev day low > prev day bc
# Use: shift(1) for CPR of prev day, shift(1) for price range of prev day
prev_day_high  = ohlc['high'].shift(1)
prev_day_low   = ohlc['low'].shift(1)
prev_day_tc    = ohlc['tc'].shift(1)
prev_day_bc    = ohlc['bc'].shift(1)
# CPR untouched = price stayed fully above or fully below CPR
ohlc['virgin_cpr'] = (
    (prev_day_high < prev_day_tc) | (prev_day_low > prev_day_bc)
).astype(int)

# ── Feature 2: Inside CPR ─────────────────────────────────────────────────────
# Today's CPR fits inside yesterday's CPR: today_tc < prev_tc AND today_bc > prev_bc
ohlc['inside_cpr'] = (
    (ohlc['tc'] < ohlc['prev_tc']) & (ohlc['bc'] > ohlc['prev_bc'])
).astype(int)

# ── Feature 3: Outside CPR ───────────────────────────────────────────────────
# Today's CPR is wider AND contains yesterday's CPR
ohlc['outside_cpr'] = (
    (ohlc['tc'] > ohlc['prev_tc']) & (ohlc['bc'] < ohlc['prev_bc'])
).astype(int)

# ── Feature 4 & 5: Ascending / Descending CPR ────────────────────────────────
# CPR mid moving up for 3 consecutive days = ascending (bullish)
ohlc['asc_cpr']  = (
    (ohlc['prev_mid'] > ohlc['prev2_mid']) &
    (ohlc['prev2_mid'] > ohlc['prev3_mid'])
).astype(int)
ohlc['desc_cpr'] = (
    (ohlc['prev_mid'] < ohlc['prev2_mid']) &
    (ohlc['prev2_mid'] < ohlc['prev3_mid'])
).astype(int)

# ── Feature 6: Open Drive ─────────────────────────────────────────────────────
# First 5-min candle (09:15-09:20) closes above PDH or below PDL
# open5 = price at 09:20 (first 5-min candle close)
ohlc['od_above_pdh'] = (ohlc['open5'] > ohlc['prev_high']).astype(int)
ohlc['od_below_pdl'] = (ohlc['open5'] < ohlc['prev_low']).astype(int)
ohlc['od_pattern']   = ((ohlc['od_above_pdh'] == 1) | (ohlc['od_below_pdl'] == 1)).astype(int)

# ── Feature 7: Wide CPR ───────────────────────────────────────────────────────
# Prev day CPR width in top 40th percentile (relative wide)
cpr_w_thresh = ohlc['prev_width'].quantile(0.60)
ohlc['wide_cpr'] = (ohlc['prev_width'] > cpr_w_thresh).astype(int)
# Narrow CPR (already in script 69 as cpr_narrow = between 0.10-0.20%)
# Wide CPR means sideways → option sellers can sell both sides (straddle)

# ── Join with trades ──────────────────────────────────────────────────────────
NEW_FEATS = ['date','virgin_cpr','inside_cpr','outside_cpr','asc_cpr','desc_cpr',
             'od_pattern','od_above_pdh','od_below_pdl','wide_cpr',
             'prev_tc','prev_bc','prev_high','prev_low']
t = trades.merge(ohlc[NEW_FEATS], on='date', how='left')

# Direction-aware features
t['virgin_aligned'] = (
    ((t['direction']=='PE') & (t['prev_low']  > t['prev_bc'])) |   # price above CPR → virgin from above → selling PE
    ((t['direction']=='CE') & (t['prev_high'] < t['prev_tc']))      # price below CPR → virgin from below → selling CE
).astype(int)

t['cpr_dir_aligned'] = (
    ((t['direction']=='PE') & (t['asc_cpr']==1)) |
    ((t['direction']=='CE') & (t['desc_cpr']==1))
).astype(int)

t['od_aligned'] = (
    ((t['direction']=='PE') & (t['od_above_pdh']==1)) |
    ((t['direction']=='CE') & (t['od_below_pdl']==1))
).astype(int)

t_valid = t.dropna(subset=['virgin_cpr']).copy()
print(f'Trades with features: {len(t_valid)}')

# ── Analysis ──────────────────────────────────────────────────────────────────
def wr_analysis(df, feat, label, direction_aware=False):
    f = feat
    on  = df[df[f] == 1]
    off = df[df[f] == 0]
    wr_on  = on['win'].mean()*100  if len(on)  > 0 else 0
    wr_off = off['win'].mean()*100 if len(off) > 0 else 0
    lift = wr_on - wr_off
    pnl_on  = on['pnl_65'].sum()
    pnl_off = off['pnl_65'].sum()
    print(f'  {label:<28} ON: {len(on):>4}t {wr_on:>5.1f}%  OFF: {len(off):>4}t {wr_off:>5.1f}%  '
          f'Lift: {lift:>+5.1f}%  PnL_ON: {pnl_on:>8,.0f}')

print(f'\n{"="*75}')
print('  FEATURE LIFT ANALYSIS (all 480 trades, WR when feature ON vs OFF)')
print(f'{"="*75}')
print(f'  Base WR: {t_valid["win"].mean()*100:.1f}%  |  Total trades: {len(t_valid)}')
print()

wr_analysis(t_valid, 'virgin_cpr',      '1. Virgin CPR (any dir)')
wr_analysis(t_valid, 'virgin_aligned',  '1b. Virgin CPR (dir-aware)')
wr_analysis(t_valid, 'inside_cpr',      '2. Inside CPR')
wr_analysis(t_valid, 'outside_cpr',     '3. Outside CPR')
wr_analysis(t_valid, 'asc_cpr',         '4. Ascending CPR (3-day)')
wr_analysis(t_valid, 'desc_cpr',        '5. Descending CPR (3-day)')
wr_analysis(t_valid, 'cpr_dir_aligned', '4b. Asc/Desc CPR dir-aware')
wr_analysis(t_valid, 'od_pattern',      '6. Open Drive (any dir)')
wr_analysis(t_valid, 'od_aligned',      '6b. Open Drive (dir-aware)')
wr_analysis(t_valid, 'wide_cpr',        '7. Wide CPR (sideways)')

# ── Per-strategy breakdown ────────────────────────────────────────────────────
print(f'\n{"="*75}')
print('  FEATURE LIFT BY STRATEGY')
print(f'{"="*75}')
for strat in ['v17a', 'cam_l3', 'cam_h3']:
    g = t_valid[t_valid['strategy'] == strat]
    print(f'\n  {strat} ({len(g)}t, base WR {g["win"].mean()*100:.1f}%):')
    for feat, lbl in [('virgin_aligned','virgin_aligned'), ('inside_cpr','inside_cpr'),
                      ('cpr_dir_aligned','cpr_dir_aligned'), ('od_aligned','od_aligned')]:
        on = g[g[feat]==1]; off = g[g[feat]==0]
        if len(on) < 5: continue
        lift = on['win'].mean()*100 - off['win'].mean()*100
        print(f'    {lbl:<22} ON:{len(on):>3}t {on["win"].mean()*100:>5.1f}%  OFF:{len(off):>3}t {off["win"].mean()*100:>5.1f}%  lift:{lift:>+5.1f}%')

# ── Correlation with existing conviction features ─────────────────────────────
print(f'\n{"="*75}')
print('  CORRELATION WITH EXISTING CONVICTION FEATURES')
print(f'{"="*75}')
existing = ['vix_ok','cpr_trend_aligned','consec_aligned','cpr_gap_aligned','dte_sweet','cpr_narrow']
new_feats = ['virgin_aligned','inside_cpr','cpr_dir_aligned','od_aligned']

# Load existing features from 69_final_trades
final_path = 'data/20260430/69_final_trades.csv'
if os.path.exists(final_path):
    t69 = pd.read_csv(final_path)
    t69.columns = [c.lower().replace(' ', '_') for c in t69.columns]
    t69['date'] = t69['date'].astype(str).str.replace('-','').str[:8]
    t_all = t69.merge(t_valid[['date','strategy','virgin_aligned','inside_cpr',
                                'cpr_dir_aligned','od_aligned','wide_cpr']],
                       on=['date','strategy'], how='left')
    avail = [f for f in existing if f in t_all.columns]
    all_cols = avail + [f for f in new_feats if f in t_all.columns]
    corr = t_all[all_cols].corr().round(2)
    print(f'\n  Correlation matrix (existing x new):')
    print(f'  {"":22}', end='')
    for nf in new_feats:
        if nf in t_all.columns:
            print(f'{nf[:14]:>16}', end='')
    print()
    for ef in avail:
        print(f'  {ef:<22}', end='')
        for nf in new_feats:
            if nf in t_all.columns:
                v = corr.loc[ef, nf] if ef in corr.index and nf in corr.columns else 0
                print(f'{v:>16.2f}', end='')
        print()
else:
    print('  69_final_trades.csv not found — skipping correlation')

# ── Open Drive standalone analysis ────────────────────────────────────────────
print(f'\n{"="*75}')
print('  OD PATTERN — STANDALONE (no v17a signal required)')
print(f'{"="*75}')
od_days = ohlc[ohlc['od_pattern'] == 1].copy()
print(f'  OD days in dataset: {len(od_days)}')
od_above = ohlc[ohlc['od_above_pdh']==1]
od_below = ohlc[ohlc['od_below_pdl']==1]
print(f'  OD above PDH: {len(od_above)} days | OD below PDL: {len(od_below)} days')

# Check on trade days vs no-trade days
trade_dates = set(t_valid['date'].unique())
od_with_trade = od_days[od_days['date'].isin(trade_dates)]
od_no_trade   = od_days[~od_days['date'].isin(trade_dates)]
print(f'  OD days WITH v17a/cam trade: {len(od_with_trade)}')
print(f'  OD days with NO existing trade: {len(od_no_trade)}  ← new trade opportunity')

# ── Virgin CPR standalone: no-trade days ─────────────────────────────────────
print(f'\n{"="*75}')
print('  VIRGIN CPR — standalone new-trade days')
print(f'{"="*75}')
virgin_days = ohlc[ohlc['virgin_cpr']==1]
virgin_no_trade = virgin_days[~virgin_days['date'].isin(trade_dates)]
print(f'  Virgin CPR days total: {len(virgin_days)}')
print(f'  Virgin CPR days with NO existing trade: {len(virgin_no_trade)}')
v2_dates = set()
try:
    t70 = pd.read_csv('data/20260430/70_intraday_v2_trades.csv')
    t70['date2'] = pd.to_datetime(t70['date']).dt.strftime('%Y%m%d')
    v2_dates = set(t70['date2'].unique())
except: pass
all_covered = trade_dates | v2_dates
virgin_truly_new = virgin_days[~virgin_days['date'].isin(all_covered)]
print(f'  Virgin CPR uncovered by ANY strategy: {len(virgin_truly_new)}')

# ── Inside CPR on v17a trades: conviction boost ───────────────────────────────
print(f'\n{"="*75}')
print('  INSIDE CPR — as 7th conviction feature test')
print(f'{"="*75}')
if os.path.exists(final_path):
    t_all['score6'] = t_all[avail].fillna(0).astype(int).sum(axis=1) if avail else 0
    t_all['inside_cpr_f'] = t_all['inside_cpr'].fillna(0).astype(int)
    t_all['score7'] = t_all['score6'] + t_all['inside_cpr_f']

    def conv_lots(s):
        if s >= 4: return 3
        if s >= 2: return 2
        return 1

    t_all['lots6'] = t_all['score6'].apply(conv_lots)
    t_all['lots7'] = t_all['score7'].apply(conv_lots)
    t_all['pnl_6f'] = (t_all['pnl_65'] * t_all['lots6']).round(2)
    t_all['pnl_7f'] = (t_all['pnl_65'] * t_all['lots7']).round(2)
    changed = (t_all['lots7'] != t_all['lots6']).sum()
    print(f'  Trades where inside_cpr changes lot size: {changed}')
    print(f'  6-feature conviction total: ₹{t_all["pnl_6f"].sum():,.0f}')
    print(f'  7-feature (+inside_cpr) total: ₹{t_all["pnl_7f"].sum():,.0f}')
    diff = t_all["pnl_7f"].sum() - t_all["pnl_6f"].sum()
    print(f'  Difference: {diff:+,.0f}')

# ── Summary table ─────────────────────────────────────────────────────────────
print(f'\n{"="*75}')
print('  VERDICT SUMMARY')
print(f'{"="*75}')
print(f'  Feature              | WR Lift | Count ON | Recommendation')
print(f'  {"-"*65}')
base_wr = t_valid['win'].mean()*100
for feat, lbl, rec in [
    ('virgin_aligned',  'Virgin CPR (dir-aware)', ''),
    ('inside_cpr',      'Inside CPR',             ''),
    ('cpr_dir_aligned', 'Asc/Desc CPR dir-aware', ''),
    ('od_aligned',      'Open Drive (dir-aware)',  ''),
    ('wide_cpr',        'Wide CPR (sideways)',     ''),
]:
    on = t_valid[t_valid[feat]==1]
    off = t_valid[t_valid[feat]==0]
    if len(on) == 0: continue
    lift = on['win'].mean()*100 - off['win'].mean()*100
    rec = 'CONVICTION +1' if lift > 3 else ('NEW TRADES' if lift > 0 else 'SKIP')
    print(f'  {lbl:<22} | {lift:>+5.1f}% | {len(on):>8} | {rec}')

print('\nDone.')
