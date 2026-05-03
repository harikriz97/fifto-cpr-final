"""
76_more_trades.py — Can we add more selling trades?
=====================================================
Tests 3 additions to the baseline 550 trades:

  A. Relax bias filter:
     - pdh_to_r1 + bull EMA → sell PE (currently only bear)
     - pdl_to_bc + bear EMA → sell CE (currently only bull)

  B. 2nd trade on same day if first exits by 11:30 (target or SL)
     After exit → rescan for cam_l3/cam_h3/iv2 signal

  C. A + B combined

Baseline: 550 trades, 71.1% WR, ₹10.72L
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import (DATA_FOLDER, load_spot_data, load_tick_data,
                     list_expiry_dates, list_trading_dates, fetch_option_chain)

LOT_SIZE  = 75
SCALE     = 65 / 75
STRIKE_INT = 50
EMA_PERIOD = 20
EOD_EXIT  = '15:20:00'
YEARS     = 5
CAM_RATIO = 1.1682
OUT_DIR   = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)

# ── Reuse simulate function (3-tier trailing SL) ──────────────────────────────
def simulate(date_str, expiry, strike, opt, entry_time_str, tgt_pct, sl_pct):
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks   = load_tick_data(date_str, instr, entry_time_str)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time_str].reset_index(drop=True)
    if tks.empty: return None
    ep  = r2(tks.iloc[0]['price'])
    ps  = tks['price'].values
    ts  = tks['time'].values
    eod = EOD_EXIT
    tgt = r2(ep*(1-tgt_pct)); hsl = r2(ep*(1+sl_pct)); sl = hsl; md = 0.0
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= eod: return r2((ep-p)*LOT_SIZE), 'eod', r2(ep), r2(p), t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep-p)*LOT_SIZE), 'lockin_sl' if sl<hsl else 'hard_sl', r2(ep), r2(p), t
    return r2((ep-ps[-1])*LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

def get_atm(spot): return int(round(spot/STRIKE_INT)*STRIKE_INT)
def get_otm1(spot, opt):
    atm = get_atm(spot)
    return atm - STRIKE_INT if opt=='PE' else atm + STRIKE_INT
def get_itm1(spot, opt):
    atm = get_atm(spot)
    return atm + STRIKE_INT if opt=='PE' else atm - STRIKE_INT
def get_strike(spot, opt, stype):
    if stype=='ATM':  return get_atm(spot)
    if stype=='OTM1': return get_otm1(spot, opt)
    if stype=='ITM1': return get_itm1(spot, opt)

# ── Build daily OHLC + features ───────────────────────────────────────────────
print("Building daily OHLC + features...")
t0 = time.time()
all_dates  = list_trading_dates()
latest     = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr  = [d for d in all_dates
              if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
extra      = max(0, all_dates.index(dates_5yr[0]) - 60)

ohlc_rows = []
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    ohlc_rows.append({'date': d,
                      'o': day.iloc[0]['price'], 'h': day['price'].max(),
                      'l': day['price'].min(),   'c': day.iloc[-1]['price']})

ohlc = pd.DataFrame(ohlc_rows)
ohlc['ema'] = ohlc['c'].ewm(span=EMA_PERIOD, adjust=False).mean().shift(1)
ohlc['ema9']  = ohlc['c'].ewm(span=9,  adjust=False).mean().shift(1)
ohlc['ema21'] = ohlc['c'].ewm(span=21, adjust=False).mean().shift(1)
ohlc['pvt'] = ((ohlc['h'] + ohlc['l'] + ohlc['c']) / 3).round(2)
ohlc['bc']  = ((ohlc['h'] + ohlc['l']) / 2).round(2)
ohlc['tc']  = (ohlc['pvt'] + (ohlc['pvt'] - ohlc['bc'])).round(2)
ohlc['r1']  = (2*ohlc['pvt'] - ohlc['l']).round(2)
ohlc['r2']  = (ohlc['pvt'] + ohlc['h'] - ohlc['l']).round(2)
ohlc['s1']  = (2*ohlc['pvt'] - ohlc['h']).round(2)
ohlc['s2']  = (ohlc['pvt'] - ohlc['h'] + ohlc['l']).round(2)
ohlc['pdh'] = ohlc['h'].shift(1)
ohlc['pdl'] = ohlc['l'].shift(1)
ohlc['cam_h3'] = (ohlc['c'].shift(1) + (ohlc['h'].shift(1) - ohlc['l'].shift(1)) * CAM_RATIO).round(2)
ohlc['cam_l3'] = (ohlc['c'].shift(1) - (ohlc['h'].shift(1) - ohlc['l'].shift(1)) * CAM_RATIO).round(2)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

def classify_zone(op, prev_row):
    # Use prev_row's own h/l as pdh/pdl (shifted pdh in ohlc is D-2, not D-1)
    pvt=prev_row['pvt']; r1=prev_row['r1']; r2=prev_row['r2']
    tc=prev_row['tc'];   bc=prev_row['bc']
    pdh=prev_row['h'];   pdl=prev_row['l']   # prev day's actual high/low
    s1=prev_row['s1'];   s2=prev_row['s2']
    if   op > r2:   return 'r2_plus'
    elif op > r1:   return 'r1_to_r2'
    elif op > pdh:  return 'pdh_to_r1'
    elif op > tc:   return 'tc_to_pdh'
    elif op >= bc:  return 'within_cpr'
    elif op > pdl:  return 'pdl_to_bc'
    elif op > s1:   return 'pdl_to_s1'
    elif op > s2:   return 's1_to_s2'
    else:           return 'below_s2'

# ── Params ────────────────────────────────────────────────────────────────────
# Baseline v17a params (from research)
V17A_BASE = {
    'r2_plus':   ('PE','OTM1',0.20,1.00,'09:16:02'),
    'r1_to_r2':  ('PE','OTM1',0.20,1.00,'09:20:02'),
    'pdh_to_r1': ('PE','OTM1',0.20,1.00,'09:20:02'),  # bear only
    'tc_to_pdh': ('PE','ITM1',0.20,1.00,'09:25:02'),
    'within_cpr':(None,'ATM', 0.20,1.00,'09:20:02'),
    'pdl_to_bc': ('PE','OTM1',0.20,1.00,'09:31:02'),  # bull only
    'pdl_to_s1': ('CE','ITM1',0.20,1.00,'09:20:02'),
    's1_to_s2':  ('CE','ATM', 0.20,1.00,'09:16:02'),
    'below_s2':  ('CE','ITM1',0.20,1.00,'09:16:02'),
}

def v17a_sig_base(zone, bias):
    """Current signal — returns (opt, stype, tgt, sl, etime) or None."""
    p = V17A_BASE.get(zone)
    if p is None: return None
    opt, stype, tgt, sl, etime = p
    if zone == 'within_cpr': opt = 'PE' if bias=='bull' else 'CE'
    if zone == 'pdh_to_r1' and bias != 'bear': return None
    if zone == 'pdl_to_bc' and bias != 'bull': return None
    if zone in {'pdl_to_s1','s1_to_s2','below_s2'} and bias != 'bear': return None
    return opt, stype, tgt, sl, etime

def v17a_sig_relaxed(zone, bias):
    """RELAXED — also includes pdh_to_r1 bull and pdl_to_bc bear."""
    p = V17A_BASE.get(zone)
    if p is None: return None
    opt, stype, tgt, sl, etime = p
    if zone == 'within_cpr': opt = 'PE' if bias=='bull' else 'CE'
    # RELAXED: pdh_to_r1 bull → sell PE (countertrend from R1 area)
    if zone == 'pdh_to_r1':
        opt = 'PE'  # both biases sell PE
    # RELAXED: pdl_to_bc bear → sell CE (countertrend from S1 area)
    if zone == 'pdl_to_bc':
        opt = 'CE' if bias == 'bear' else 'PE'
    if zone in {'pdl_to_s1','s1_to_s2','below_s2'} and bias != 'bear': return None
    return opt, stype, tgt, sl, etime

# ── Run simulation ────────────────────────────────────────────────────────────
print("\nRunning simulations...")
t0 = time.time()

records_base    = []   # baseline (should match 75)
records_relax   = []   # A: relaxed bias
records_2nd     = []   # B: 2nd trade
records_combined= []   # A+B

for idx, row in ohlc_5yr.iterrows():
    if idx < 3: continue
    dstr  = row['date']
    op    = row['o']
    bias  = 'bull' if op > row['ema'] else 'bear'
    prev_row = ohlc_5yr.iloc[idx - 1]
    zone  = classify_zone(op, prev_row)

    # Get expiry + ATM
    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
           pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

    # ── Baseline signal ───────────────────────────────────────────────────
    sig = v17a_sig_base(zone, bias)
    if sig:
        opt, stype, tgt, sl, etime = sig
        strike = get_strike(op, opt, stype)
        res = simulate(dstr, expiry, strike, opt, etime, tgt, sl)
        if res:
            pnl75, reason, ep, xp, xt = res
            win = pnl75*SCALE > 0
            records_base.append(dict(date=dstr, zone=zone, bias=bias, opt=opt,
                strike=strike, dte=dte, entry_time=etime, ep=ep, xp=xp,
                exit_time=xt, exit_reason=reason,
                pnl_65=r2(pnl75*SCALE), win=win,
                is_extra=False, year=dstr[:4]))

    # ── Relaxed signal (A) ────────────────────────────────────────────────
    sig_r = v17a_sig_relaxed(zone, bias)
    if sig_r:
        opt_r, stype_r, tgt_r, sl_r, etime_r = sig_r
        strike_r = get_strike(op, opt_r, stype_r)
        res_r = simulate(dstr, expiry, strike_r, opt_r, etime_r, tgt_r, sl_r)
        if res_r:
            pnl75, reason, ep, xp, xt = res_r
            win = pnl75*SCALE > 0
            # is_extra = only if this is a NEW trade not in baseline
            is_extra = (sig is None) or (opt_r != sig[0])
            records_relax.append(dict(date=dstr, zone=zone, bias=bias, opt=opt_r,
                strike=strike_r, dte=dte, entry_time=etime_r, ep=ep, xp=xp,
                exit_time=xt, exit_reason=reason,
                pnl_65=r2(pnl75*SCALE), win=win,
                is_extra=is_extra, year=dstr[:4]))

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | {time.time()-t0:.0f}s")

print(f"Pass done | {time.time()-t0:.0f}s")

# ── 2nd trade: days where baseline exits by 11:30 via target/SL ──────────────
print("\nChecking 2nd trade opportunity...")
df_base = pd.DataFrame(records_base)
early_exit = df_base[
    (df_base['exit_reason'].isin(['target','hard_sl','lockin_sl'])) &
    (df_base['exit_time'] <= '11:30:00')
]
print(f"  Days with early exit (by 11:30): {len(early_exit)}")

records_2nd_only = []
t0 = time.time()
for _, tr in early_exit.iterrows():
    dstr   = tr['date']
    xt     = tr['exit_time']   # re-entry scan starts here
    row_match = ohlc_5yr[ohlc_5yr['date']==dstr]
    if row_match.empty: continue
    row = row_match.iloc[0]
    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte    = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
              pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

    # Scan spot ticks from exit_time+2s for cam signal
    tks = load_spot_data(dstr, 'NIFTY')
    if tks is None: continue

    # Calculate re-entry time (exit + 2 minutes buffer)
    xt_mins = int(xt[:2])*60 + int(xt[3:5])
    rescan_time = f"{(xt_mins+2)//60:02d}:{(xt_mins+2)%60:02d}:00"
    scan = tks[(tks['time'] >= rescan_time) & (tks['time'] <= '13:00:00')]
    if scan.empty: continue

    cam_h3 = row['cam_h3']; cam_l3 = row['cam_l3']
    r1_lvl = row['r1'];     r2_lvl = row['r2'];  pdl_lvl = row['pdl']
    l3_done=False; h3_done=False; r1d=False; r2d=False; pdld=False
    second_sig = None

    for _, tick in scan.iterrows():
        p = tick['price']; t = tick['time']
        # Cam check
        if not l3_done and p < cam_l3:
            second_sig = ('CE','ATM',0.20,1.00,t,'cam_l3'); l3_done=True; break
        if not h3_done and p > cam_h3:
            second_sig = ('PE','OTM1',0.20,1.00,t,'cam_h3'); h3_done=True; break
        # iv2 check (09:30–12:00)
        if '09:30:00' <= t <= '12:00:00':
            if not r2d and p > r2_lvl:
                second_sig = ('PE','ITM1',0.50,1.00,t,'iv2_r2'); r2d=True; break
            if not r1d and p > r1_lvl:
                second_sig = ('PE','ATM',0.20,0.50,t,'iv2_r1'); r1d=True; break
            if not pdld and p < pdl_lvl:
                second_sig = ('CE','ATM',0.30,2.00,t,'iv2_pdl'); pdld=True; break

    if second_sig is None: continue
    opt2, stype2, tgt2, sl2, etime2, strat2 = second_sig
    spot2 = scan[scan['time'] >= etime2]['price'].iloc[0] if len(scan[scan['time'] >= etime2]) else None
    if spot2 is None: continue
    strike2 = get_strike(spot2, opt2, stype2)
    # entry_time = next second
    etime2_entry = etime2[:6] + f"{int(etime2[6:8])+2:02d}" if len(etime2) > 5 else etime2
    res2 = simulate(dstr, expiry, strike2, opt2, etime2[:8], tgt2, sl2)
    if res2:
        pnl75, reason, ep, xp, xt2 = res2
        win = pnl75*SCALE > 0
        records_2nd_only.append(dict(date=dstr, zone=strat2, bias='2nd', opt=opt2,
            strike=strike2, dte=dte, entry_time=etime2, ep=ep, xp=xp,
            exit_time=xt2, exit_reason=reason,
            pnl_65=r2(pnl75*SCALE), win=win,
            is_extra=True, year=dstr[:4]))

print(f"  2nd trades found: {len(records_2nd_only)} | {time.time()-t0:.0f}s")

# ── Compile results ───────────────────────────────────────────────────────────
df_relax  = pd.DataFrame(records_relax)
df_2nd    = pd.DataFrame(records_2nd_only)

# Extra trades only from relaxed (trades not in baseline)
extra_relax = df_relax[df_relax['is_extra']==True] if not df_relax.empty else pd.DataFrame()

print(f"\n{'='*60}")
print("  RESULTS")
print(f"{'='*60}")
print(f"\nBaseline (current):  {len(df_base):>4} trades | WR {df_base['win'].mean()*100:.1f}% | Flat ₹{df_base['pnl_65'].sum():,.0f}")

if not extra_relax.empty:
    print(f"\nA. Relaxed bias extras:")
    print(f"   +{len(extra_relax):>3} trades | WR {extra_relax['win'].mean()*100:.1f}% | ₹{extra_relax['pnl_65'].sum():,.0f}")
    print(f"   Breakdown:")
    for zone in extra_relax['zone'].unique():
        g = extra_relax[extra_relax['zone']==zone]
        print(f"     {zone:<15}: {len(g):>3}t | WR {g['win'].mean()*100:.0f}% | ₹{g['pnl_65'].sum():,.0f}")

if not df_2nd.empty:
    print(f"\nB. 2nd trade extras:")
    print(f"   +{len(df_2nd):>3} trades | WR {df_2nd['win'].mean()*100:.1f}% | ₹{df_2nd['pnl_65'].sum():,.0f}")
    print(f"   By signal:")
    for z in df_2nd['zone'].unique():
        g = df_2nd[df_2nd['zone']==z]
        print(f"     {z:<15}: {len(g):>3}t | WR {g['win'].mean()*100:.0f}% | ₹{g['pnl_65'].sum():,.0f}")

# Combined total
all_extra = pd.concat([r for r in [extra_relax, df_2nd] if not (r is None or r.empty)], ignore_index=True) if (not extra_relax.empty or not df_2nd.empty) else pd.DataFrame()
base_flat = df_base['pnl_65'].sum() if not df_base.empty else 0
extra_flat = all_extra['pnl_65'].sum() if not all_extra.empty else 0

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Baseline:          {len(df_base):>4}t | ₹{base_flat:>10,.0f}")
if not all_extra.empty:
    total_new = len(df_base) + len(all_extra)
    total_pnl = base_flat + extra_flat
    print(f"  Extra trades (A+B): +{len(all_extra):>3}t | ₹{extra_flat:>10,.0f}")
    print(f"  Combined:          {total_new:>4}t | ₹{total_pnl:>10,.0f}")

# Save
if not all_extra.empty:
    all_extra.to_csv(f'{OUT_DIR}/76_extra_trades.csv', index=False)
    print(f"\n  Saved → {OUT_DIR}/76_extra_trades.csv")

print("\nDone.")
