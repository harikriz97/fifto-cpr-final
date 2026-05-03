"""
84_reentry_after_target.py — Re-entry after 20% target hit
===========================================================
After existing trade hits 20% target and exits, check:
  1. How much further does premium fall after exit?
  2. What % of times could we have profited on re-entry?
  3. What would re-entry P&L look like?

Re-entry logic:
  - Same strike, same option
  - Enter at tick immediately after target exit (at current market price)
  - New target: 20% of re-entry price
  - New SL: 50% of re-entry price (tighter — already had one good move)
  - EOD: 15:20

Only on trades where exit_reason = 'target' (hit 20%)
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
OUT_DIR    = 'data/20260430'

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def get_strike(s, opt, stype):
    atm = get_atm(s)
    if stype == 'ATM':  return atm
    if stype == 'OTM1': return atm + STRIKE_INT if opt == 'CE' else atm - STRIKE_INT
    if stype == 'ITM1': return atm - STRIKE_INT if opt == 'CE' else atm + STRIKE_INT
    return atm

def simulate_first_and_reentry(date_str, expiry, strike, opt, entry_time,
                                 tgt1_pct=0.20, sl1_pct=1.00,
                                 tgt2_pct=0.20, sl2_pct=0.50):
    """
    Simulate first trade. If it hits target, simulate re-entry.
    Returns: (trade1_result, trade2_result_or_None)
    """
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None, None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None, None

    ep   = r2(tks.iloc[0]['price'])
    if ep <= 0: return None, None
    tgt1 = r2(ep * (1 - tgt1_pct))
    hsl1 = r2(ep * (1 + sl1_pct)); sl1 = hsl1; md = 0.0
    ps   = tks['price'].values
    ts   = tks['time'].values

    # ── Trade 1 ──────────────────────────────────────────────────────────────
    t1_result = None
    reentry_idx = None

    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            t1_result = (r2((ep-p)*LOT_SIZE), 'eod', r2(ep), r2(p), t)
            break
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl1 = min(sl1, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl1 = min(sl1, r2(ep*0.80))
        elif md >= 0.25: sl1 = min(sl1, ep)
        if p <= tgt1:
            t1_result = (r2((ep-p)*LOT_SIZE), 'target', r2(ep), r2(p), t)
            reentry_idx = i + 1   # next tick
            break
        if p >= sl1:
            reason = 'lockin_sl' if sl1 < hsl1 else 'hard_sl'
            t1_result = (r2((ep-p)*LOT_SIZE), reason, r2(ep), r2(p), t)
            break

    if t1_result is None:
        t1_result = (r2((ep-ps[-1])*LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1])

    # ── Trade 2 (re-entry) — only if trade 1 hit target ──────────────────────
    if t1_result[1] != 'target' or reentry_idx is None or reentry_idx >= len(ts):
        return t1_result, None

    # Re-enter at next tick after target exit
    ep2 = r2(ps[reentry_idx])
    if ep2 <= 0: return t1_result, None

    tgt2 = r2(ep2 * (1 - tgt2_pct))
    hsl2 = r2(ep2 * (1 + sl2_pct)); sl2 = hsl2; md2 = 0.0

    for i in range(reentry_idx, len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            t2_result = (r2((ep2-p)*LOT_SIZE), 'eod', r2(ep2), r2(p), t)
            return t1_result, t2_result
        d = (ep2 - p) / ep2
        if d > md2: md2 = d
        # simpler trailing for re-entry
        if   md2 >= 0.40: sl2 = min(sl2, r2(ep2*0.80))
        elif md2 >= 0.25: sl2 = min(sl2, ep2)
        if p <= tgt2:
            return t1_result, (r2((ep2-p)*LOT_SIZE), 'target', r2(ep2), r2(p), t)
        if p >= sl2:
            reason = 'lockin_sl' if sl2 < hsl2 else 'hard_sl'
            return t1_result, (r2((ep2-p)*LOT_SIZE), reason, r2(ep2), r2(p), t)

    return t1_result, (r2((ep2-ps[-1])*LOT_SIZE), 'eod', r2(ep2), r2(ps[-1]), ts[-1])

# ── Load trades ────────────────────────────────────────────────────────────────
print("Loading trade data...")
df72 = pd.read_csv(f'{OUT_DIR}/72_final_trades.csv')
df72['date_str'] = df72['date'].astype(str).str.replace('-', '')

print(f"  Total trades: {len(df72)}")
print(f"  Target hits (20%): {(df72['exit_reason']=='target').sum()}")

# ── Re-simulation ──────────────────────────────────────────────────────────────
print("\nSimulating first trade + re-entry...")
t0 = time.time()

records = []
for i, (_, row) in enumerate(df72.iterrows()):
    dstr       = row['date_str']
    opt        = row['opt']
    stype      = row['strike_type']
    entry_time = row['entry_time']
    lots       = row['lots7n']
    year       = str(row['year'])

    spot_tks = load_spot_data(dstr, 'NIFTY')
    if spot_tks is None: continue
    at_entry = spot_tks[spot_tks['time'] >= entry_time]
    if at_entry.empty: continue
    strike = get_strike(at_entry.iloc[0]['price'], opt, stype)

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]

    t1, t2 = simulate_first_and_reentry(dstr, expiry, strike, opt, entry_time)
    if t1 is None: continue

    pnl1_conv = r2(t1[0] * SCALE * lots)
    pnl2_conv = r2(t2[0] * SCALE * 1) if t2 else None  # re-entry always 1 lot

    records.append(dict(
        date=dstr, year=year, lots=lots, opt=opt,
        ep1=t1[2], xp1=t1[3], reason1=t1[1], exit_time1=t1[4],
        pnl1=pnl1_conv,
        has_reentry=(t2 is not None),
        ep2=t2[2] if t2 else None,
        xp2=t2[3] if t2 else None,
        reason2=t2[1] if t2 else None,
        exit_time2=t2[4] if t2 else None,
        pnl2=pnl2_conv,
        combined=r2(pnl1_conv + (pnl2_conv or 0))
    ))

    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(df72)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

df = pd.DataFrame(records)
target_trades = df[df['reason1'] == 'target']
reentry_trades = df[df['has_reentry']]

# ── Results ────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  TRADE 1 RESULTS (baseline)")
print(f"{'='*65}")
print(f"  Total trades:    {len(df)}")
print(f"  Target hits:     {len(target_trades)} ({len(target_trades)/len(df)*100:.0f}%)")
print(f"  Trade1 Total P&L: Rs.{df['pnl1'].sum():,.0f}")
print(f"  Exit reasons: {dict(df['reason1'].value_counts())}")

print(f"\n{'='*65}")
print("  RE-ENTRY ANALYSIS (after 20% target hit)")
print(f"{'='*65}")
print(f"  Target trades eligible for re-entry: {len(target_trades)}")
print(f"  Re-entries executed:                 {len(reentry_trades)}")

if not reentry_trades.empty:
    wr2 = (reentry_trades['pnl2'] > 0).mean() * 100
    print(f"  Re-entry Win Rate:  {wr2:.1f}%")
    print(f"  Re-entry Total P&L: Rs.{reentry_trades['pnl2'].sum():,.0f}")
    print(f"  Re-entry Avg/trade: Rs.{reentry_trades['pnl2'].mean():,.0f}")
    print(f"  Re-entry exit reasons: {dict(reentry_trades['reason2'].value_counts())}")

    print(f"\n  Year-wise re-entry performance:")
    for yr in sorted(reentry_trades['year'].unique()):
        g = reentry_trades[reentry_trades['year']==yr]
        wr = (g['pnl2']>0).mean()*100
        print(f"    {yr}: {len(g):>3}t | WR {wr:.0f}% | Rs.{g['pnl2'].sum():,.0f}")

    # How much further did premium fall after target exit?
    print(f"\n  Premium fall AFTER 20% exit (on re-entry trades):")
    reentry_trades = reentry_trades.copy()
    reentry_trades['further_pct'] = ((reentry_trades['ep2'] - reentry_trades['xp2']) /
                                      reentry_trades['ep2'] * 100).round(1)
    print(f"  Avg further fall: {reentry_trades['further_pct'].mean():.1f}%")
    print(f"  Fell another >10%: {(reentry_trades['further_pct']>10).sum()} trades")
    print(f"  Fell another >20%: {(reentry_trades['further_pct']>20).sum()} trades")
    print(f"  Recovered (negative fall): {(reentry_trades['further_pct']<0).sum()} trades")

print(f"\n{'='*65}")
print("  COMBINED: Trade1 + Re-entry vs Trade1 Only")
print(f"{'='*65}")
t1_only     = df['pnl1'].sum()
t1_plus_t2  = df['combined'].sum()
improvement = t1_plus_t2 - t1_only
print(f"  Trade1 only:          Rs.{t1_only:,.0f}")
print(f"  Trade1 + Re-entry:    Rs.{t1_plus_t2:,.0f}")
print(f"  Re-entry contribution: Rs.{reentry_trades['pnl2'].sum():,.0f} (+{improvement/t1_only*100:.1f}%)")

print(f"\n  Year-wise combined:")
for yr in sorted(df['year'].unique()):
    g  = df[df['year']==yr]
    p1 = g['pnl1'].sum()
    pc = g['combined'].sum()
    print(f"    {yr}: Trade1=Rs.{p1:,.0f} | +Reentry=Rs.{pc-p1:,.0f} | Total=Rs.{pc:,.0f}")

print("\nDone.")
