"""
83_partial_booking.py — Partial Profit Booking with Multiple Lots
==================================================================
For trades with 2-3 lots, test staggered exits vs single target.

3-lot configs tested:
  Current:  all 3 at TGT20%
  Config A: lot1=15%, lot2=25%, lot3=trail
  Config B: lot1=20%, lot2=35%, lot3=trail
  Config C: lot1=20%, lot2=40%, lot3=trail
  Config D: lot1=25%, lot2=50%, lot3=trail
  Config E: lot1=20%, lot2=50%, lot3=trail
  Best50:   all 3 at TGT50% (from sensitivity analysis)

2-lot configs:
  Current:  both at TGT20%
  Config A: lot1=20%, lot2=40%
  Config B: lot1=20%, lot2=50%
  Config C: lot1=15%, lot2=40%

Trail lot = SL at 100%, trailing kicks in at 25/40/60% drops, no fixed target
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates

LOT_SIZE  = 75
SCALE     = 65 / 75
STRIKE_INT = 50
EOD_EXIT  = '15:20:00'
OUT_DIR   = 'data/20260430'

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def get_strike(s, opt, stype):
    atm = get_atm(s)
    if stype == 'ATM':  return atm
    if stype == 'OTM1': return atm + STRIKE_INT if opt == 'CE' else atm - STRIKE_INT
    if stype == 'ITM1': return atm - STRIKE_INT if opt == 'CE' else atm + STRIKE_INT
    return atm

def simulate_one_lot(tks_arr, tks_time, ep, tgt_pct, sl_pct=1.00, trail=False):
    """
    Simulate a single lot.
    If trail=True: no fixed target, use trailing SL only.
    Returns (pnl_per_lot, exit_reason, exit_price, exit_time)
    """
    tgt = r2(ep * (1 - tgt_pct)) if not trail else -999999
    hsl = r2(ep * (1 + sl_pct))
    sl  = hsl; md = 0.0
    ps  = tks_arr; ts = tks_time
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((ep - p) * LOT_SIZE), 'eod', r2(p), t
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if not trail and p <= tgt:
            return r2((ep - p) * LOT_SIZE), 'target', r2(p), t
        if p >= sl:
            reason = 'lockin_sl' if sl < hsl else 'hard_sl'
            return r2((ep - p) * LOT_SIZE), reason, r2(p), t
    return r2((ep - ps[-1]) * LOT_SIZE), 'eod', r2(ps[-1]), ts[-1]

def simulate_partial(date_str, expiry, strike, opt, entry_time, lot_targets):
    """
    lot_targets: list of (tgt_pct_or_None, trail_bool) per lot
    Returns total pnl for all lots combined.
    """
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    ps = tks['price'].values
    ts = tks['time'].values

    total_pnl = 0.0
    for (tgt_pct, trail) in lot_targets:
        pnl_lot, reason, xp, xt = simulate_one_lot(ps, ts, ep, tgt_pct, trail=trail)
        total_pnl += pnl_lot

    return r2(total_pnl * SCALE)  # scaled to 65 lot size, all lots combined

# ── Load trades ────────────────────────────────────────────────────────────────
print("Loading trade data...")
df72 = pd.read_csv(f'{OUT_DIR}/72_final_trades.csv')
df72['date_str'] = df72['date'].astype(str).str.replace('-', '')

# Define partial booking configurations
# Format: list of (tgt_pct, trail) tuples — one per lot
CONFIGS_3LOT = {
    'current_20':  [(0.20, False), (0.20, False), (0.20, False)],
    'A_15_25_trail': [(0.15, False), (0.25, False), (None, True)],
    'B_20_35_trail': [(0.20, False), (0.35, False), (None, True)],
    'C_20_40_trail': [(0.20, False), (0.40, False), (None, True)],
    'D_25_50_trail': [(0.25, False), (0.50, False), (None, True)],
    'E_20_50_trail': [(0.20, False), (0.50, False), (None, True)],
    'best_50':      [(0.50, False), (0.50, False), (0.50, False)],
}
CONFIGS_2LOT = {
    'current_20':  [(0.20, False), (0.20, False)],
    'A_20_40':     [(0.20, False), (0.40, False)],
    'B_20_50':     [(0.20, False), (0.50, False)],
    'C_15_40':     [(0.15, False), (0.40, False)],
    'D_20_trail':  [(0.20, False), (None, True)],
}

trades_3lot = df72[df72['lots7n'] == 3]
trades_2lot = df72[df72['lots7n'] == 2]
trades_1lot = df72[df72['lots7n'] == 1]

print(f"  3-lot trades: {len(trades_3lot)}")
print(f"  2-lot trades: {len(trades_2lot)}")
print(f"  1-lot trades: {len(trades_1lot)}")

# ── Process 3-lot trades ───────────────────────────────────────────────────────
print(f"\nProcessing 3-lot trades ({len(trades_3lot)} trades × {len(CONFIGS_3LOT)} configs)...")
t0 = time.time()
results_3 = {cfg: [] for cfg in CONFIGS_3LOT}

for i, (_, row) in enumerate(trades_3lot.iterrows()):
    dstr       = row['date_str']
    opt        = row['opt']
    stype      = row['strike_type']
    entry_time = row['entry_time']

    spot_tks = load_spot_data(dstr, 'NIFTY')
    if spot_tks is None: continue
    at_entry = spot_tks[spot_tks['time'] >= entry_time]
    if at_entry.empty: continue
    strike = get_strike(at_entry.iloc[0]['price'], opt, stype)

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]

    for cfg_name, lot_targets in CONFIGS_3LOT.items():
        res = simulate_partial(dstr, expiry, strike, opt, entry_time, lot_targets)
        if res is not None:
            results_3[cfg_name].append({'date': dstr, 'year': str(row['year']), 'pnl': res})

    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(trades_3lot)} | {time.time()-t0:.0f}s")

# ── Process 2-lot trades ───────────────────────────────────────────────────────
print(f"\nProcessing 2-lot trades ({len(trades_2lot)} trades × {len(CONFIGS_2LOT)} configs)...")
results_2 = {cfg: [] for cfg in CONFIGS_2LOT}

for i, (_, row) in enumerate(trades_2lot.iterrows()):
    dstr       = row['date_str']
    opt        = row['opt']
    stype      = row['strike_type']
    entry_time = row['entry_time']

    spot_tks = load_spot_data(dstr, 'NIFTY')
    if spot_tks is None: continue
    at_entry = spot_tks[spot_tks['time'] >= entry_time]
    if at_entry.empty: continue
    strike = get_strike(at_entry.iloc[0]['price'], opt, stype)

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]

    for cfg_name, lot_targets in CONFIGS_2LOT.items():
        res = simulate_partial(dstr, expiry, strike, opt, entry_time, lot_targets)
        if res is not None:
            results_2[cfg_name].append({'date': dstr, 'year': str(row['year']), 'pnl': res})

    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(trades_2lot)} | {time.time()-t0:.0f}s")

print(f"\nDone | {time.time()-t0:.0f}s")

# ── Results ────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  3-LOT PARTIAL BOOKING COMPARISON")
print(f"{'='*65}")
print(f"  {'Config':<20} | {'Trades':>7} | {'Total P&L':>12} | {'Avg/trade':>10} | vs current")
print(f"  {'-'*65}")
base_3 = sum(r['pnl'] for r in results_3['current_20'])
for cfg, recs in results_3.items():
    if not recs: continue
    total = sum(r['pnl'] for r in recs)
    avg   = total / len(recs)
    diff  = total - base_3
    marker = ' ◄ current' if cfg == 'current_20' else (f' +Rs.{diff:,.0f}' if diff > 0 else f' -Rs.{abs(diff):,.0f}')
    print(f"  {cfg:<20} | {len(recs):>7} | Rs.{total:>10,.0f} | Rs.{avg:>8,.0f} |{marker}")

print(f"\n  Year-wise for best configs (3-lot):")
best_3 = max((k for k in results_3 if k != 'current_20'),
             key=lambda k: sum(r['pnl'] for r in results_3[k]))
print(f"  {'Year':<6} | {'current_20':>12} | {best_3:>20}")
for yr in sorted(set(r['year'] for r in results_3['current_20'])):
    c20  = sum(r['pnl'] for r in results_3['current_20'] if r['year']==yr)
    best = sum(r['pnl'] for r in results_3[best_3] if r['year']==yr)
    print(f"  {yr:<6} | Rs.{c20:>10,.0f} | Rs.{best:>10,.0f}")

print(f"\n{'='*65}")
print("  2-LOT PARTIAL BOOKING COMPARISON")
print(f"{'='*65}")
print(f"  {'Config':<20} | {'Trades':>7} | {'Total P&L':>12} | {'Avg/trade':>10} | vs current")
print(f"  {'-'*65}")
base_2 = sum(r['pnl'] for r in results_2['current_20'])
for cfg, recs in results_2.items():
    if not recs: continue
    total = sum(r['pnl'] for r in recs)
    avg   = total / len(recs)
    diff  = total - base_2
    marker = ' ◄ current' if cfg == 'current_20' else (f' +Rs.{diff:,.0f}' if diff > 0 else f' -Rs.{abs(diff):,.0f}')
    print(f"  {cfg:<20} | {len(recs):>7} | Rs.{total:>10,.0f} | Rs.{avg:>8,.0f} |{marker}")

# Grand total comparison
print(f"\n{'='*65}")
print("  GRAND TOTAL — COMBINED (3-lot + 2-lot + 1-lot)")
print(f"{'='*65}")

# 1-lot trades always use single target (20% current, 50% best)
df_82 = pd.read_csv(f'{OUT_DIR}/82_target_sensitivity.csv')
iv2_pnl = pd.read_csv(f'{OUT_DIR}/70_intraday_v2_trades.csv')
# Use pnl directly scaled (iv2 is 1 lot, 65 scale)
iv2_total = iv2_pnl['pnl'].sum() * SCALE

lot1_current = df_82[df_82['lots']==1]['tgt20'].sum()
lot1_best50  = df_82[df_82['lots']==1]['tgt50'].sum()

# Get best 2-lot config
best_2 = max((k for k in results_2 if k != 'current_20'),
             key=lambda k: sum(r['pnl'] for r in results_2[k]))

grand_current = base_3 + base_2 + lot1_current + iv2_total
grand_best    = (sum(r['pnl'] for r in results_3[best_3]) +
                 sum(r['pnl'] for r in results_2[best_2]) +
                 lot1_best50 + iv2_total)

print(f"  Current (all 20%):              Rs.{grand_current:,.0f}")
print(f"  Best partial ({best_3} / {best_2}): Rs.{grand_best:,.0f}")
print(f"  Improvement:                    Rs.{grand_best-grand_current:,.0f} (+{(grand_best-grand_current)/grand_current*100:.1f}%)")

print(f"\n  Recommended rules:")
best_3_cfg = CONFIGS_3LOT[best_3]
best_2_cfg = CONFIGS_2LOT[best_2]
print(f"  3-lot trade: Lot1={int(best_3_cfg[0][0]*100)}% | Lot2={int(best_3_cfg[1][0]*100) if not best_3_cfg[1][1] else 'trail'}% | Lot3={'trail' if best_3_cfg[2][1] else str(int(best_3_cfg[2][0]*100))+'%'}")
print(f"  2-lot trade: Lot1={int(best_2_cfg[0][0]*100)}% | Lot2={int(best_2_cfg[1][0]*100) if not best_2_cfg[1][1] else 'trail'}%")
print(f"  1-lot trade: Single exit at 50%")

print("\nDone.")
