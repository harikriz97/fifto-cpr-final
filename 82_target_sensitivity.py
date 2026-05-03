"""
82_target_sensitivity.py — Target % Sensitivity Analysis
=========================================================
Re-simulate all 550 trades with different target percentages:
  [10%, 15%, 20%, 25%, 30%, 40%, 50%]
Keep SL and trailing logic same. See which target maximises P&L.

Also: year-wise breakdown to see if 20% was optimal in recent years
or if something changed (which explains paper trade misses).
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates

LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
EOD_EXIT   = '15:20:00'
SL_PCT     = 1.00   # always 100% hard SL
TARGETS    = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
OUT_DIR    = 'data/20260430'

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def get_strike(s, opt, stype):
    atm = get_atm(s)
    if stype == 'ATM':  return atm
    if stype == 'OTM1': return atm + STRIKE_INT if opt == 'CE' else atm - STRIKE_INT
    if stype == 'ITM1': return atm - STRIKE_INT if opt == 'CE' else atm + STRIKE_INT
    return atm

def simulate_sell(date_str, expiry, strike, opt, entry_time, tgt_pct, sl_pct=1.00):
    """Sell option simulation with 3-tier trailing SL."""
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - tgt_pct))
    hsl = r2(ep * (1 + sl_pct)); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((ep - p) * LOT_SIZE), 'eod', r2(ep), r2(p), t
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep - p) * LOT_SIZE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep - p) * LOT_SIZE), 'lockin_sl' if sl < hsl else 'hard_sl', r2(ep), r2(p), t
    return r2((ep - ps[-1]) * LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

# ── Load existing trade plans ──────────────────────────────────────────────────
print("Loading trade plans...")
df72 = pd.read_csv(f'{OUT_DIR}/72_final_trades.csv')
df70 = pd.read_csv(f'{OUT_DIR}/70_intraday_v2_trades.csv')

# Normalise dates
df72['date_str'] = df72['date'].astype(str).str.replace('-', '')
df70['date_str'] = df70['date'].astype(str).str.replace('-', '')

print(f"  Script 72 trades: {len(df72)}")
print(f"  Script 70 trades: {len(df70)}")

# ── Re-simulate with multiple targets ─────────────────────────────────────────
print("\nRe-simulating all trades with multiple targets...")
t0 = time.time()

# Store pnl_65 per trade per target
results = []   # list of dicts: {trade_id, date, strategy, year, lots, tgt: pnl}

total = len(df72) + len(df70)
done  = 0

# ── Script 72 trades ──────────────────────────────────────────────────────────
for _, row in df72.iterrows():
    dstr       = row['date_str']
    opt        = row['opt']
    stype      = row['strike_type']
    entry_time = row['entry_time']
    lots       = row['lots7n']
    year       = str(row['year'])

    # Get spot at entry to derive strike
    spot_tks = load_spot_data(dstr, 'NIFTY')
    if spot_tks is None: done += 1; continue
    at_entry = spot_tks[spot_tks['time'] >= entry_time]
    if at_entry.empty: done += 1; continue
    spot_p = at_entry.iloc[0]['price']
    strike = get_strike(spot_p, opt, stype)

    expiries = list_expiry_dates(dstr)
    if not expiries: done += 1; continue
    expiry = expiries[0]

    rec = {'date': dstr, 'strategy': row['strategy'], 'year': year,
           'lots': lots, 'opt': opt}
    for tgt in TARGETS:
        res = simulate_sell(dstr, expiry, strike, opt, entry_time, tgt)
        if res:
            pnl75, reason, ep, xp, xt = res
            rec[f'tgt{int(tgt*100)}'] = r2(pnl75 * SCALE * lots)
            rec[f'reason{int(tgt*100)}'] = reason
        else:
            rec[f'tgt{int(tgt*100)}'] = None
            rec[f'reason{int(tgt*100)}'] = None
    results.append(rec)
    done += 1
    if done % 50 == 0:
        print(f"  {done}/{total} | {time.time()-t0:.0f}s")

# ── Script 70 trades ──────────────────────────────────────────────────────────
for _, row in df70.iterrows():
    dstr       = row['date_str']
    opt        = row['opt']
    strike     = int(row['strike'])
    entry_time = row['entry_time']
    lots       = 1   # iv2 always 1 lot
    year       = str(row['year'])

    expiries = list_expiry_dates(dstr)
    if not expiries: done += 1; continue
    expiry = expiries[0]

    rec = {'date': dstr, 'strategy': 'iv2', 'year': year,
           'lots': lots, 'opt': opt}
    for tgt in TARGETS:
        res = simulate_sell(dstr, expiry, strike, opt, entry_time, tgt)
        if res:
            pnl75, reason, ep, xp, xt = res
            rec[f'tgt{int(tgt*100)}'] = r2(pnl75 * SCALE * lots)
            rec[f'reason{int(tgt*100)}'] = reason
        else:
            rec[f'tgt{int(tgt*100)}'] = None
            rec[f'reason{int(tgt*100)}'] = None
    results.append(rec)
    done += 1
    if done % 50 == 0:
        print(f"  {done}/{total} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s | {len(results)} trades re-simulated")

df = pd.DataFrame(results).dropna(subset=[f'tgt{int(TARGETS[0]*100)}'])

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  TARGET SENSITIVITY — ALL 550 TRADES (conviction lots)")
print(f"{'='*65}")
print(f"  {'Target':>8} | {'Trades':>7} | {'WR%':>6} | {'Total P&L':>12} | {'Avg/trade':>10} | Exit reasons")
print(f"  {'-'*75}")

best_pnl = -999999999
best_tgt = 0
for tgt in TARGETS:
    col     = f'tgt{int(tgt*100)}'
    rcol    = f'reason{int(tgt*100)}'
    sub     = df[df[col].notna()]
    wins    = (sub[col] > 0).sum()
    wr      = wins / len(sub) * 100
    total   = sub[col].sum()
    avg     = sub[col].mean()
    reasons = sub[rcol].value_counts().to_dict()
    marker  = ' ◄ current' if tgt == 0.20 else (' ◄ BEST' if total > best_pnl else '')
    if total > best_pnl:
        best_pnl = total; best_tgt = tgt
    print(f"  {int(tgt*100):>7}% | {len(sub):>7} | {wr:>5.1f}% | Rs.{total:>11,.0f} | Rs.{avg:>8,.0f} | "
          f"tgt:{reasons.get('target',0)} eod:{reasons.get('eod',0)} sl:{reasons.get('hard_sl',0)+reasons.get('lockin_sl',0)}")

print(f"\n  Best target: {int(best_tgt*100)}% | Rs.{best_pnl:,.0f}")

# ── Year-wise breakdown ────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  YEAR-WISE — P&L at each target")
print(f"{'='*65}")
tgt_cols = [f'tgt{int(t*100)}' for t in TARGETS]
header = f"  {'Year':>6} | " + " | ".join(f"TGT{int(t*100):>2}%" for t in TARGETS)
print(header)
print(f"  {'-'*80}")
for yr in sorted(df['year'].unique()):
    g = df[df['year'] == yr]
    vals = [f"Rs.{g[c].sum():>8,.0f}" for c in tgt_cols]
    print(f"  {yr:>6} | " + " | ".join(vals))
total_row = [f"Rs.{df[c].sum():>8,.0f}" for c in tgt_cols]
print(f"  {'TOTAL':>6} | " + " | ".join(total_row))

# ── Win rate by year ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  YEAR-WISE WIN RATE at TGT20% (current)")
print(f"{'='*65}")
col20 = 'tgt20'
for yr in sorted(df['year'].unique()):
    g = df[df['year'] == yr]
    wr = (g[col20] > 0).mean() * 100
    tgt_hits = (g['reason20'] == 'target').sum()
    eod_hits = (g['reason20'] == 'eod').sum()
    sl_hits  = len(g) - tgt_hits - eod_hits
    print(f"  {yr}: {len(g):>3}t | WR {wr:.0f}% | tgt:{tgt_hits} eod:{eod_hits} sl:{sl_hits}")

# ── Recent months analysis (2025+) ────────────────────────────────────────────
print(f"\n{'='*65}")
print("  2025-2026 DETAILED — why target not hitting?")
print(f"{'='*65}")
recent = df[df['year'].isin(['2025','2026'])]
print(f"  Trades: {len(recent)}")
print(f"  Current TGT20%: WR {(recent['tgt20']>0).mean()*100:.0f}% | Rs.{recent['tgt20'].sum():,.0f}")
print(f"  Best target for recent period:")
best_r = max(TARGETS, key=lambda t: recent[f'tgt{int(t*100)}'].sum())
print(f"    TGT{int(best_r*100)}% = Rs.{recent[f'tgt{int(best_r*100)}'].sum():,.0f}")
print(f"\n  What actually happened to TGT20% trades in 2025-2026:")
r20 = recent['reason20'].value_counts()
print(f"    {dict(r20)}")

# Save
df.to_csv(f'{OUT_DIR}/82_target_sensitivity.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/82_target_sensitivity.csv")
print("\nDone.")
