"""
96_blank_candle_patterns.py — Candlestick patterns on remaining 344 blank days
===============================================================================
Check if daily candlestick patterns at key CPR/pivot levels appear on the
343 remaining blank days (no base signal, no CRT signal).

Patterns checked (daily candle vs prev day):
  - Bullish Engulfing  : today body engulfs prev body, bullish close
  - Bearish Engulfing  : today body engulfs prev body, bearish close
  - Hammer             : small body top, long lower wick (>=2x body), near support
  - Shooting Star      : small body bottom, long upper wick (>=2x body), near resistance
  - Doji               : body < 10% of range

Level proximity: candle's body/wick touches or is within 0.3% of TC, BC, R1, S1, R2, S2

Output: pattern hit rate, directional accuracy, P&L potential
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, list_trading_dates
import time

OUT_DIR = 'data/20260430'
YEARS   = 5
PROX    = 0.003   # 0.3% proximity to level

# ── Load remaining blank days ─────────────────────────────────────────────────
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]

base      = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base_dates = set(base['date'].astype(str).str.replace('-', ''))
crt        = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank_dates = set(crt[crt['is_blank'] == True]['date'].astype(str))
remaining_blank = set(d for d in dates_5yr if d not in base_dates and d not in crt_blank_dates)

# ── Build daily OHLC + CPR for all 5yr ───────────────────────────────────────
print("Building daily OHLC + CPR...")
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

df = pd.DataFrame(rows)
ph = df['h'].shift(1); pl = df['l'].shift(1); pc = df['c'].shift(1); po = df['o'].shift(1)
df['pvt']    = ((ph + pl + pc) / 3).round(2)
df['bc']     = ((ph + pl) / 2).round(2)
df['tc']     = (df['pvt'] + (df['pvt'] - df['bc'])).round(2)
df['r1']     = (2 * df['pvt'] - pl).round(2)
df['s1']     = (2 * df['pvt'] - ph).round(2)
df['r2']     = (df['pvt'] + ph - pl).round(2)
df['s2']     = (df['pvt'] - ph + pl).round(2)
df['prev_o'] = po
df['prev_c'] = pc
df['prev_h'] = ph
df['prev_l'] = pl
df = df.dropna().reset_index(drop=True)
df_rb = df[df['date'].isin(remaining_blank)].reset_index(drop=True)
print(f"  {len(df_rb)} blank days | {time.time()-t0:.0f}s")

# ── Pattern detection ─────────────────────────────────────────────────────────
def body_size(o, c):    return abs(c - o)
def full_range(h, l):   return h - l
def upper_wick(o, c, h): return h - max(o, c)
def lower_wick(o, c, l): return min(o, c) - l
def near(price, level, pct=PROX): return abs(price - level) / level <= pct
def touches(o, c, h, l, level, pct=PROX):
    return near(h, level, pct) or near(l, level, pct) or \
           near(o, level, pct) or near(c, level, pct) or \
           (l <= level <= h)

records = []
for _, r in df_rb.iterrows():
    o, h, l, c     = r['o'], r['h'], r['l'], r['c']
    po, ph, pl, pc = r['prev_o'], r['prev_h'], r['prev_l'], r['prev_c']
    tc, bc, r1, s1, r2, s2 = r['tc'], r['bc'], r['r1'], r['s1'], r['r2'], r['s2']

    body   = body_size(o, c)
    rng    = full_range(h, l)
    uw     = upper_wick(o, c, h)
    lw     = lower_wick(o, c, l)
    pbody  = body_size(po, pc)
    bull   = c > o
    prev_bull = pc > po

    # ── Pattern flags ──────────────────────────────────────────────────────────
    # Engulfing: today body > prev body, direction flip
    bull_engulf = (bull and not prev_bull and
                   min(o, c) < min(po, pc) and max(o, c) > max(po, pc) and
                   pbody > 0 and body > pbody)
    bear_engulf = (not bull and prev_bull and
                   min(o, c) < min(po, pc) and max(o, c) > max(po, pc) and
                   pbody > 0 and body > pbody)

    # Hammer: small body at top, long lower wick >= 2x body, upper wick small
    hammer = (lw >= 2 * body and uw <= 0.3 * body and rng > 0)
    # Shooting Star: small body at bottom, long upper wick >= 2x body
    shooting_star = (uw >= 2 * body and lw <= 0.3 * body and rng > 0)
    # Doji: body < 10% of range
    doji = (rng > 0 and body / rng < 0.10)
    # Bullish Harami: small today inside large prev, bullish
    bull_harami = (bull and not prev_bull and
                   o > min(po, pc) and c < max(po, pc) and body < pbody * 0.6)
    # Bearish Harami
    bear_harami = (not bull and prev_bull and
                   o < max(po, pc) and c > min(po, pc) and body < pbody * 0.6)

    # ── Level proximity ────────────────────────────────────────────────────────
    levels = {'TC': tc, 'BC': bc, 'R1': r1, 'S1': s1, 'R2': r2, 'S2': s2}
    near_levels = [name for name, lv in levels.items() if touches(o, c, h, l, lv)]
    at_resistance = any(x in near_levels for x in ['TC', 'R1', 'R2'])
    at_support    = any(x in near_levels for x in ['BC', 'S1', 'S2'])

    # Assign pattern label (priority order)
    if bull_engulf:   pattern = 'BullEngulf'
    elif bear_engulf: pattern = 'BearEngulf'
    elif hammer:      pattern = 'Hammer'
    elif shooting_star: pattern = 'ShootingStar'
    elif bull_harami: pattern = 'BullHarami'
    elif bear_harami: pattern = 'BearHarami'
    elif doji:        pattern = 'Doji'
    else:             pattern = 'None'

    # Directional signal from pattern
    if pattern in ('BullEngulf', 'Hammer', 'BullHarami'):
        pat_signal = 'Bull'
    elif pattern in ('BearEngulf', 'ShootingStar', 'BearHarami'):
        pat_signal = 'Bear'
    elif pattern == 'Doji':
        pat_signal = 'Neutral'
    else:
        pat_signal = 'None'

    actual_dir = 'Bull' if c > o else 'Bear'

    records.append(dict(
        date=r['date'], year=r['date'][:4],
        pattern=pattern, pat_signal=pat_signal,
        at_resistance=at_resistance, at_support=at_support,
        near_levels='+'.join(near_levels) if near_levels else 'None',
        actual_dir=actual_dir,
        bull=bull, body=round(body, 2), rng=round(rng, 2),
        o=o, h=h, l=l, c=c
    ))

df_out = pd.DataFrame(records)

# ── Results ───────────────────────────────────────────────────────────────────
sep = '─' * 65
print(f"\n{'='*65}")
print(f"  CANDLESTICK PATTERNS — {len(df_out)} remaining blank days")
print(f"{'='*65}")

# Overall pattern distribution
print(f"\n{sep}")
print("  PATTERN DISTRIBUTION")
print(sep)
for pat, g in df_out.groupby('pattern'):
    acc = (g['pat_signal'] == g['actual_dir']).mean() * 100 if g['pat_signal'].iloc[0] != 'None' else float('nan')
    acc_str = f"{acc:.0f}% acc" if not np.isnan(acc) else "—"
    print(f"  {pat:<16}: {len(g):>4} days ({len(g)/len(df_out)*100:.0f}%)  |  {acc_str}")

# Pattern at key level
print(f"\n{sep}")
print("  PATTERN AT KEY LEVEL (proximity ±0.3%)")
print(sep)
has_pat = df_out[df_out['pattern'] != 'None']
at_level = has_pat[(has_pat['at_resistance']) | (has_pat['at_support'])]
print(f"  Total patterns found   : {len(has_pat)} / {len(df_out)} days ({len(has_pat)/len(df_out)*100:.0f}%)")
print(f"  At key level           : {len(at_level)} / {len(has_pat)} ({len(at_level)/len(has_pat)*100:.0f}% of patterns)")

print(f"\n{sep}")
print("  PATTERN + LEVEL COMBOS (min 5 days)")
print(sep)
print(f"  {'Pattern':<16} {'Level':<8} | Days | Signal | Actual Bull% | Acc%")
print(f"  {'-'*63}")
for (pat, lvl), g in df_out[df_out['pattern'] != 'None'].groupby(['pattern', 'near_levels']):
    if len(g) < 5: continue
    sig = g['pat_signal'].iloc[0]
    bull_pct = (g['actual_dir'] == 'Bull').mean() * 100
    acc = (g['pat_signal'] == g['actual_dir']).mean() * 100 if sig not in ('None','Neutral') else float('nan')
    acc_str = f"{acc:.0f}%" if not np.isnan(acc) else "—"
    print(f"  {pat:<16} {lvl:<8} | {len(g):>4} | {sig:<7} | {bull_pct:>5.0f}%        | {acc_str}")

# Most actionable: directional pattern at level with >60% accuracy
print(f"\n{'='*65}")
print("  ACTIONABLE: Directional pattern at key level (acc >= 60%)")
print(f"{'='*65}")
actionable = []
for (pat, lvl), g in df_out[df_out['pattern'] != 'None'].groupby(['pattern', 'near_levels']):
    sig = g['pat_signal'].iloc[0]
    if sig in ('None', 'Neutral'): continue
    if len(g) < 5: continue
    acc = (g['pat_signal'] == g['actual_dir']).mean() * 100
    if acc >= 60:
        actionable.append((pat, lvl, len(g), sig, acc))

actionable.sort(key=lambda x: (-x[4], -x[2]))
for pat, lvl, n, sig, acc in actionable:
    print(f"  [{sig.upper():<4}] {pat:<16} @ {lvl:<10} | {n:>3} days | {acc:.0f}% accurate")

if not actionable:
    print("  No pattern+level combo meets threshold — lowering to 55%")
    for (pat, lvl), g in df_out[df_out['pattern'] != 'None'].groupby(['pattern', 'near_levels']):
        sig = g['pat_signal'].iloc[0]
        if sig in ('None', 'Neutral'): continue
        if len(g) < 4: continue
        acc = (g['pat_signal'] == g['actual_dir']).mean() * 100
        if acc >= 55:
            print(f"  [{sig.upper():<4}] {pat:<16} @ {lvl:<10} | {n:>3} days | {acc:.0f}% accurate")

# ── Year-wise breakdown for top patterns ─────────────────────────────────────
print(f"\n{sep}")
print("  ENGULFING PATTERNS — YEAR-WISE")
print(sep)
eng = df_out[df_out['pattern'].isin(['BullEngulf', 'BearEngulf'])]
if not eng.empty:
    for yr, g in eng.groupby('year'):
        acc = (g['pat_signal'] == g['actual_dir']).mean() * 100
        print(f"  {yr}: {len(g):>3} | BullEngulf {len(g[g['pattern']=='BullEngulf'])} | BearEngulf {len(g[g['pattern']=='BearEngulf'])} | Acc {acc:.0f}%")

# Save
df_out.to_csv(f'{OUT_DIR}/96_blank_candle_patterns.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/96_blank_candle_patterns.csv")
print(f"\nDone.")
