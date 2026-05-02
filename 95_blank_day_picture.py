"""
95_blank_day_picture.py — Overall picture of 344 remaining blank days
======================================================================
These are days where:
  - Base strategy (v17a/cam/iv2) had NO signal
  - CRT Approach D also had NO signal

Goal: understand what's happening on these days to find tradeable patterns.

Dimensions analysed:
  1. Year/month/weekday distribution
  2. Gap behavior (open vs prev close)
  3. CPR relationship (open above TC / inside / below BC)
  4. CPR width → trending vs sideways indicator
  5. Day direction (close > open = bullish)
  6. Day range vs average
  7. OR (Opening Range 09:15–09:30) — breakout vs reversal
  8. Pattern combinations — which clusters show tradeable bias
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, list_trading_dates
from plot_util import send_custom_chart

OUT_DIR  = 'data/20260430'
YEARS    = 5

# ── Identify the 344 remaining blank days ─────────────────────────────────────
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]

base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base_dates = set(base['date'].astype(str).str.replace('-', ''))

crt = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank_dates = set(crt[crt['is_blank'] == True]['date'].astype(str))

remaining_blank = [d for d in dates_5yr if d not in base_dates and d not in crt_blank_dates]
print(f"Remaining blank days: {len(remaining_blank)}")

# ── Build CPR + daily OHLC for all 5yr dates ──────────────────────────────────
print("\nBuilding daily OHLC + CPR...")
import time
t0 = time.time()

extra = max(0, all_dates.index(dates_5yr[0]) - 5)
rows = []
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    rows.append({'date': d,
                 'o': day.iloc[0]['price'],
                 'h': day['price'].max(),
                 'l': day['price'].min(),
                 'c': day.iloc[-1]['price']})

df_all = pd.DataFrame(rows)
ph = df_all['h'].shift(1); pl = df_all['l'].shift(1); pc = df_all['c'].shift(1)
df_all['pvt'] = ((ph + pl + pc) / 3).round(2)
df_all['bc']  = ((ph + pl) / 2).round(2)
df_all['tc']  = (df_all['pvt'] + (df_all['pvt'] - df_all['bc'])).round(2)
df_all['r1']  = (2 * df_all['pvt'] - pl).round(2)
df_all['s1']  = (2 * df_all['pvt'] - ph).round(2)
df_all['cpr_width'] = (df_all['tc'] - df_all['bc']).round(2)
df_all['prev_c'] = df_all['c'].shift(1)
df_all = df_all.dropna().reset_index(drop=True)

# Keep only 5yr dates
df_5yr = df_all[df_all['date'].isin(dates_5yr)].reset_index(drop=True)
df_rb   = df_5yr[df_5yr['date'].isin(remaining_blank)].reset_index(drop=True)
print(f"  Done | {time.time()-t0:.0f}s | remaining blank rows: {len(df_rb)}")

# ── Add OR (Opening Range = first 15min candle) ────────────────────────────────
print("\nAdding Opening Range data...")
or_rows = []
for d in remaining_blank:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    c1 = tks[(tks['time'] >= '09:15:00') & (tks['time'] < '09:30:00')]
    if len(c1) < 2: continue
    or_rows.append({'date': d, 'or_h': c1['price'].max(), 'or_l': c1['price'].min()})

df_or = pd.DataFrame(or_rows)
df_rb = df_rb.merge(df_or, on='date', how='left')

# ── Feature engineering ────────────────────────────────────────────────────────
avg_range = df_5yr['h'].sub(df_5yr['l']).mean()
avg_cpr   = df_5yr['cpr_width'].mean()

df_rb['day_range']     = (df_rb['h'] - df_rb['l']).round(2)
df_rb['day_dir']       = (df_rb['c'] > df_rb['o']).map({True: 'Bull', False: 'Bear'})
df_rb['gap_pts']       = (df_rb['o'] - df_rb['prev_c']).round(2)
df_rb['gap_pct']       = (df_rb['gap_pts'] / df_rb['prev_c'] * 100).round(3)
df_rb['gap_type']      = pd.cut(df_rb['gap_pct'], bins=[-99, -0.3, 0.3, 99],
                                 labels=['GapDown', 'Flat', 'GapUp'])
df_rb['cpr_type']      = pd.cut(df_rb['cpr_width'],
                                 bins=[-1, avg_cpr * 0.5, avg_cpr * 1.5, 9999],
                                 labels=['Narrow', 'Normal', 'Wide'])
df_rb['open_vs_cpr']   = 'InsideCPR'
df_rb.loc[df_rb['o'] > df_rb['tc'], 'open_vs_cpr'] = 'AboveTC'
df_rb.loc[df_rb['o'] < df_rb['bc'], 'open_vs_cpr'] = 'BelowBC'
df_rb['range_type']    = pd.cut(df_rb['day_range'],
                                 bins=[0, avg_range * 0.7, avg_range * 1.3, 9999],
                                 labels=['Narrow', 'Normal', 'Wide'])
df_rb['year']          = df_rb['date'].str[:4]
df_rb['weekday']       = pd.to_datetime(df_rb['date'], format='%Y%m%d').dt.day_name()
df_rb['month']         = pd.to_datetime(df_rb['date'], format='%Y%m%d').dt.month

# OR breakout vs inside
df_rb['or_range']     = (df_rb['or_h'] - df_rb['or_l']).round(2)
df_rb['close_vs_or']  = 'InsideOR'
df_rb.loc[df_rb['c'] > df_rb['or_h'], 'close_vs_or'] = 'AboveOR'
df_rb.loc[df_rb['c'] < df_rb['or_l'], 'close_vs_or'] = 'BelowOR'

# ── Print analysis ─────────────────────────────────────────────────────────────
sep = '─' * 60
print(f"\n{'='*60}")
print(f"  BLANK DAY PICTURE — {len(df_rb)} remaining days (no base, no CRT)")
print(f"{'='*60}")

print(f"\n  Avg 5yr day range: {avg_range:.0f} pts | Avg CPR width: {avg_cpr:.1f} pts")

print(f"\n{sep}")
print("  YEAR-WISE DISTRIBUTION")
print(sep)
for yr, g in df_rb.groupby('year'):
    bull = (g['day_dir'] == 'Bull').sum()
    print(f"  {yr}: {len(g):>3} days | Bull {bull:>3} ({bull/len(g)*100:.0f}%) | "
          f"Bear {len(g)-bull:>3} ({(len(g)-bull)/len(g)*100:.0f}%)")

print(f"\n{sep}")
print("  WEEKDAY DISTRIBUTION")
print(sep)
wd_order = ['Monday','Tuesday','Wednesday','Thursday','Friday']
for wd in wd_order:
    g = df_rb[df_rb['weekday'] == wd]
    if g.empty: continue
    bull = (g['day_dir'] == 'Bull').sum()
    print(f"  {wd:<12}: {len(g):>3} days | Bull {bull/len(g)*100:.0f}%")

print(f"\n{sep}")
print("  GAP BEHAVIOR")
print(sep)
for gt, g in df_rb.groupby('gap_type', observed=True):
    bull = (g['day_dir'] == 'Bull').sum()
    avg_g = g['gap_pts'].mean()
    print(f"  {gt:<10}: {len(g):>3} days | Bull {bull/len(g)*100:.0f}% | Avg gap {avg_g:+.1f} pts")

print(f"\n{sep}")
print("  OPEN vs CPR")
print(sep)
for oc, g in df_rb.groupby('open_vs_cpr'):
    bull = (g['day_dir'] == 'Bull').sum()
    print(f"  {oc:<12}: {len(g):>3} days | Bull {bull/len(g)*100:.0f}%")

print(f"\n{sep}")
print("  CPR WIDTH (Narrow=trending bias, Wide=sideways)")
print(sep)
for ct, g in df_rb.groupby('cpr_type', observed=True):
    bull = (g['day_dir'] == 'Bull').sum()
    avg_w = g['cpr_width'].mean()
    print(f"  {ct:<8}: {len(g):>3} days | Bull {bull/len(g)*100:.0f}% | Avg width {avg_w:.1f} pts")

print(f"\n{sep}")
print("  CLOSE vs OPENING RANGE")
print(sep)
for cor, g in df_rb.groupby('close_vs_or'):
    print(f"  {cor:<12}: {len(g):>3} days ({len(g)/len(df_rb)*100:.0f}%)")

print(f"\n{sep}")
print("  DAY RANGE")
print(sep)
for rt, g in df_rb.groupby('range_type', observed=True):
    bull = (g['day_dir'] == 'Bull').sum()
    avg_r = g['day_range'].mean()
    print(f"  {rt:<8}: {len(g):>3} days | Bull {bull/len(g)*100:.0f}% | Avg range {avg_r:.0f} pts")

# ── COMBO: Gap + Open vs CPR ───────────────────────────────────────────────────
print(f"\n{sep}")
print("  COMBINATION: GAP + OPEN vs CPR (directional bias)")
print(sep)
print(f"  {'Combo':<28} | Days | Bull% | Bear%")
print(f"  {'-'*55}")
combos = df_rb.groupby(['gap_type', 'open_vs_cpr'], observed=True)
for (gt, oc), g in sorted(combos, key=lambda x: -len(x[1])):
    if len(g) < 8: continue
    bull = (g['day_dir'] == 'Bull').sum()
    label = f"{gt} + {oc}"
    print(f"  {label:<28} | {len(g):>4} | {bull/len(g)*100:>4.0f}% | {(len(g)-bull)/len(g)*100:>4.0f}%")

# ── COMBO: CPR width + Gap ──────────────────────────────────────────────────────
print(f"\n{sep}")
print("  COMBINATION: CPR WIDTH + GAP (trending days)")
print(sep)
print(f"  {'Combo':<28} | Days | Bull% | AvgRange")
print(f"  {'-'*55}")
for (ct, gt), g in sorted(df_rb.groupby(['cpr_type','gap_type'], observed=True), key=lambda x: -len(x[1])):
    if len(g) < 8: continue
    bull = (g['day_dir'] == 'Bull').sum()
    avg_r = g['day_range'].mean()
    label = f"{ct} CPR + {gt}"
    print(f"  {label:<28} | {len(g):>4} | {bull/len(g)*100:>4.0f}% | {avg_r:>6.0f} pts")

# ── STRONG PATTERNS (>65% or <35% bullish = directional bias) ─────────────────
print(f"\n{'='*60}")
print("  STRONG DIRECTIONAL PATTERNS (>65% or <35% bull)")
print(f"{'='*60}")
combos3 = df_rb.groupby(['gap_type', 'open_vs_cpr', 'cpr_type'], observed=True)
strong = []
for keys, g in combos3:
    if len(g) < 8: continue
    bull_pct = (g['day_dir'] == 'Bull').mean() * 100
    if bull_pct >= 65 or bull_pct <= 35:
        strong.append((keys, len(g), bull_pct))

strong.sort(key=lambda x: -abs(x[1]))
for keys, n, bp in strong:
    bias = 'BULL' if bp >= 65 else 'BEAR'
    print(f"  [{bias}] {keys[0]} + {keys[1]} + {keys[2]} CPR | {n} days | {bp:.0f}% bull")

# ── Save ───────────────────────────────────────────────────────────────────────
df_rb.to_csv(f'{OUT_DIR}/95_blank_remaining.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/95_blank_remaining.csv")

# ── Chart: year-wise blank day count + bull/bear split ────────────────────────
yr_data = df_rb.groupby('year').apply(
    lambda g: pd.Series({'total': len(g), 'bull': (g['day_dir']=='Bull').sum(),
                         'bear': (g['day_dir']=='Bear').sum()})
).reset_index()

bull_bars = [{"time": int(pd.Timestamp(f"{r.year}-01-01").timestamp()),
              "value": int(r.bull), "color": "#26a69a"} for _, r in yr_data.iterrows()]
bear_bars = [{"time": int(pd.Timestamp(f"{r.year}-01-01").timestamp()),
              "value": -int(r.bear), "color": "#ef5350"} for _, r in yr_data.iterrows()]

tv_json = {
    "isTvFormat": False,
    "candlestick": [], "volume": [],
    "lines": [
        {"id": "bull", "label": "Bull days", "seriesType": "bar", "data": bull_bars},
        {"id": "bear", "label": "Bear days", "seriesType": "bar", "data": bear_bars},
    ]
}
send_custom_chart("95_blank_yr", tv_json,
                  title=f"Remaining Blank Days — Year-wise Bull/Bear ({len(df_rb)} days)")

print("\nDone.")
