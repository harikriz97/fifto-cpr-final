"""
91_crt_ltf_confirmation.py — CRT: 4 approaches compared (LTF + CPR combinations)
==================================================================================
Side-by-side comparison of all 4 CRT approaches:

  A: 15M CRT alone          — no CPR filter, no LTF confirmation
  B: 15M CRT + CPR (TC+R1)  — CPR filter only, no LTF
  C: 15M CRT + 5M LTF       — LTF confirmation only, no CPR
  D: 15M CRT + CPR + 5M LTF — FULL CRT rules (CPR + LTF both)

LTF (5M) confirmation logic:
  After 15M bearish CRT signal, in the 30-min window after C3 closes:
  Look for a 5M bearish Turtle Soup candle:
    high > open (wick up) AND close < open (bearish body)
  This = brief bounce/manipulation → reversal confirmed on 5M
  Enter at next 5M candle after this confirmation

All entries: next candle after signal/confirmation — zero forward bias
Bearish CRT only (resistance sweep → Sell CE OTM1)
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates

LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
CANDLE_MIN = 15
EOD_EXIT   = '15:20:00'
YEARS      = 5
OUT_DIR    = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)
def get_otm1(s, opt):
    atm = int(round(s / STRIKE_INT) * STRIKE_INT)
    return atm + STRIKE_INT if opt == 'CE' else atm - STRIKE_INT

def build_ohlc(tks, freq='15min', start='09:15:00', end='12:45:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample(freq).ohlc().dropna()
    ohlc.columns = ['o', 'h', 'l', 'c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

def simulate_sell(date_str, expiry, strike, opt, entry_time,
                  tgt_pct=0.20, sl_pct=1.00):
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep*(1-tgt_pct)); hsl = r2(ep*(1+sl_pct)); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT: return r2((ep-p)*LOT_SIZE), 'eod', r2(ep), r2(p), t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep-p)*LOT_SIZE), 'lockin_sl' if sl<hsl else 'hard_sl', r2(ep), r2(p), t
    return r2((ep-ps[-1])*LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

# ── Build daily OHLC + CPR ─────────────────────────────────────────────────
print("Building daily OHLC + CPR...")
t0 = time.time()
all_dates  = list_trading_dates()
latest     = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr  = [d for d in all_dates
              if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
extra      = max(0, all_dates.index(dates_5yr[0]) - 60)

rows = []
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')]
    if len(day)<2: continue
    rows.append({'date':d,'o':day.iloc[0]['price'],
                 'h':day['price'].max(),'l':day['price'].min(),'c':day.iloc[-1]['price']})

ohlc = pd.DataFrame(rows)
ph = ohlc['h'].shift(1); pl = ohlc['l'].shift(1); pc = ohlc['c'].shift(1)
ohlc['pvt'] = ((ph+pl+pc)/3).round(2)
ohlc['bc']  = ((ph+pl)/2).round(2)
ohlc['tc']  = (ohlc['pvt']+(ohlc['pvt']-ohlc['bc'])).round(2)
ohlc['r1']  = (2*ohlc['pvt']-pl).round(2)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)

df_sell_base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
sell_dates   = set(df_sell_base['date'].astype(str).str.replace('-',''))
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nScanning 15M CRT with/without CPR and 5M LTF confirmation...")
t0 = time.time()

rec_a, rec_b, rec_c, rec_d = [], [], [], []

for idx, row in ohlc_5yr.iterrows():
    dstr     = row['date']
    tc_val   = row['tc']
    r1_val   = row['r1']
    is_blank = dstr not in sell_dates

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue

    # Build 15M and 5M OHLC once per day
    c15 = build_ohlc(spot, '15min')
    c5  = build_ohlc(spot, '5min', end='13:00:00')
    if len(c15) < 3 or len(c5) < 3: continue

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]

    year = dstr[:4]

    # Scan 15M for bearish CRT signal
    signal_found = {'a': False, 'b': False, 'c': False, 'd': False}

    for ci in range(1, len(c15)-1):
        if all(signal_found.values()): break

        c1 = c15.iloc[ci-1]
        c2 = c15.iloc[ci]
        c3_idx = ci + 1
        if c3_idx >= len(c15): break
        c3     = c15.iloc[c3_idx]
        c3_time= c3['time']
        if c3_time > '12:00:00': break

        c2h = c2['h']; c3c = c3['c']

        # Generic 3-candle CRT (C2 wicks above C1.H, C3 closes back)
        crt_any = (c2h > c1['h'] and c3c < c1['h'])

        # CRT at CPR key levels (TC or R1)
        crt_tc  = (c2h > tc_val and c3c < tc_val)
        crt_r1  = (c2h > r1_val and c3c < r1_val)
        crt_cpr = crt_tc or crt_r1
        level_name = ('TC' if crt_tc else 'R1') if crt_cpr else 'any'

        # ── A: 15M CRT alone, next 15M candle entry ─────────────────────────
        if not signal_found['a'] and crt_any:
            em = int(c3_time[:2])*60 + int(c3_time[3:5]) + CANDLE_MIN + 1
            et = f"{em//60:02d}:{em%60:02d}:02"
            if et < EOD_EXIT:
                res = simulate_sell(dstr, expiry, get_otm1(c3c,'CE'), 'CE', et)
                if res:
                    pnl75, reason, ep, xp, xt = res
                    rec_a.append(dict(date=dstr, level=level_name, year=year,
                        is_blank=is_blank, ep=ep, xp=xp, exit_reason=reason,
                        pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE>0, entry_time=et))
                    signal_found['a'] = True

        # ── B: 15M CRT + CPR, next 15M candle entry ─────────────────────────
        if not signal_found['b'] and crt_cpr:
            em = int(c3_time[:2])*60 + int(c3_time[3:5]) + CANDLE_MIN + 1
            et = f"{em//60:02d}:{em%60:02d}:02"
            if et < EOD_EXIT:
                res = simulate_sell(dstr, expiry, get_otm1(c3c,'CE'), 'CE', et)
                if res:
                    pnl75, reason, ep, xp, xt = res
                    rec_b.append(dict(date=dstr, level=level_name, year=year,
                        is_blank=is_blank, ep=ep, xp=xp, exit_reason=reason,
                        pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE>0, entry_time=et))
                    signal_found['b'] = True

        # ── C: 15M CRT alone + 5M LTF confirmation ──────────────────────────
        if not signal_found['c'] and crt_any:
            # Search for 5M bearish Turtle Soup in 30-min window after C3 closes
            # Window: C3 close time to C3 close + 30 min
            c3_close_min = int(c3_time[:2])*60 + int(c3_time[3:5]) + CANDLE_MIN
            win_end_min  = c3_close_min + 30
            c3_close_t   = f"{c3_close_min//60:02d}:{c3_close_min%60:02d}:00"
            win_end_t    = f"{win_end_min//60:02d}:{win_end_min%60:02d}:00"

            c5_win = c5[(c5['time'] > c3_close_t) & (c5['time'] <= win_end_t)].reset_index(drop=True)
            ltf_entry = None
            for fi in range(len(c5_win)):
                fc = c5_win.iloc[fi]
                # Bearish TS: wick up (H > O) AND close down (C < O) — brief manipulation up then fall
                if fc['h'] > fc['o'] and fc['c'] < fc['o']:
                    # Enter at next 5M candle
                    fmin  = int(fc['time'][:2])*60 + int(fc['time'][3:5]) + 5 + 1
                    ftime = f"{fmin//60:02d}:{fmin%60:02d}:02"
                    if ftime < EOD_EXIT:
                        ltf_entry = (ftime, fc['time'])
                    break

            if ltf_entry:
                et, ltf_t = ltf_entry
                res = simulate_sell(dstr, expiry, get_otm1(c3c,'CE'), 'CE', et)
                if res:
                    pnl75, reason, ep, xp, xt = res
                    rec_c.append(dict(date=dstr, level=level_name, year=year,
                        is_blank=is_blank, ep=ep, xp=xp, exit_reason=reason,
                        pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE>0,
                        entry_time=et, ltf_candle=ltf_t))
                    signal_found['c'] = True

        # ── D: 15M CRT + CPR + 5M LTF confirmation ──────────────────────────
        if not signal_found['d'] and crt_cpr:
            c3_close_min = int(c3_time[:2])*60 + int(c3_time[3:5]) + CANDLE_MIN
            win_end_min  = c3_close_min + 30
            c3_close_t   = f"{c3_close_min//60:02d}:{c3_close_min%60:02d}:00"
            win_end_t    = f"{win_end_min//60:02d}:{win_end_min%60:02d}:00"

            c5_win = c5[(c5['time'] > c3_close_t) & (c5['time'] <= win_end_t)].reset_index(drop=True)
            ltf_entry = None
            for fi in range(len(c5_win)):
                fc = c5_win.iloc[fi]
                if fc['h'] > fc['o'] and fc['c'] < fc['o']:
                    fmin  = int(fc['time'][:2])*60 + int(fc['time'][3:5]) + 5 + 1
                    ftime = f"{fmin//60:02d}:{fmin%60:02d}:02"
                    if ftime < EOD_EXIT:
                        ltf_entry = (ftime, fc['time'])
                    break

            if ltf_entry:
                et, ltf_t = ltf_entry
                res = simulate_sell(dstr, expiry, get_otm1(c3c,'CE'), 'CE', et)
                if res:
                    pnl75, reason, ep, xp, xt = res
                    rec_d.append(dict(date=dstr, level=level_name, year=year,
                        is_blank=is_blank, ep=ep, xp=xp, exit_reason=reason,
                        pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE>0,
                        entry_time=et, ltf_candle=ltf_t))
                    signal_found['d'] = True

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | A:{len(rec_a)} B:{len(rec_b)} C:{len(rec_c)} D:{len(rec_d)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
sell_conv = df_sell_base['pnl_conv'].sum()

def show(label, recs):
    if not recs:
        print(f"\n  {label}: no trades"); return pd.DataFrame()
    df = pd.DataFrame(recs)
    wr  = df['win'].mean()*100
    pnl = df['pnl_65'].sum()
    avg = df['pnl_65'].mean()
    bl  = df[df['is_blank']]
    bwr = bl['win'].mean()*100 if not bl.empty else 0
    bpnl= bl['pnl_65'].sum()
    bavg= bl['pnl_65'].mean() if not bl.empty else 0

    print(f"\n  {'─'*65}")
    print(f"  {label}")
    print(f"  {'─'*65}")
    print(f"  All days:   {len(df):>4}t | WR {wr:>5.1f}% | Rs.{pnl:>9,.0f} | Avg Rs.{avg:>5,.0f}")
    print(f"  Blank days: {len(bl):>4}t | WR {bwr:>5.1f}% | Rs.{bpnl:>9,.0f} | Avg Rs.{bavg:>5,.0f}")
    print(f"  Exits: {dict(df['exit_reason'].value_counts())}")
    print(f"  Year-wise:")
    for yr in sorted(df['year'].unique()):
        g = df[df['year']==yr]
        blg = g[g['is_blank']]
        print(f"    {yr}: {len(g):>3}t WR {g['win'].mean()*100:.0f}% Rs.{g['pnl_65'].sum():>8,.0f}"
              f"  | blank: {len(blg):>3}t WR {blg['win'].mean()*100 if not blg.empty else 0:.0f}%"
              f" Rs.{blg['pnl_65'].sum():>8,.0f}")
    print(f"  + baseline → Total Rs.{sell_conv+bpnl:,.0f}")
    return df

df_a = show("A: 15M CRT alone (no CPR, no LTF)  — script 86 re-run", rec_a)
df_b = show("B: 15M CRT + CPR (TC+R1)           — clean (script 90 style)", rec_b)
df_c = show("C: 15M CRT + 5M LTF confirmation   — new", rec_c)
df_d = show("D: 15M CRT + CPR + 5M LTF          — FULL CRT rules", rec_d)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print("  SUMMARY: 4 APPROACHES vs BLANK DAYS")
print(f"{'='*75}")
print(f"  {'Approach':<40} | {'Blank t':>7} | {'WR':>6} | {'P&L':>10} | {'Avg':>7}")
print(f"  {'-'*73}")

for label, recs in [
    ("A: 15M CRT alone",              rec_a),
    ("B: 15M CRT + CPR (TC+R1)",      rec_b),
    ("C: 15M CRT + 5M LTF",           rec_c),
    ("D: 15M CRT + CPR + 5M LTF",     rec_d),
]:
    if not recs:
        print(f"  {label:<40} | {'—':>7} | {'—':>6} | {'—':>10} | {'—':>7}")
        continue
    df = pd.DataFrame(recs)
    bl = df[df['is_blank']]
    if bl.empty:
        print(f"  {label:<40} |       0 |     — |          — |       —")
        continue
    bwr  = bl['win'].mean()*100
    bpnl = bl['pnl_65'].sum()
    bavg = bl['pnl_65'].mean()
    total= sell_conv + bpnl
    flag = ' ✓' if bpnl > 0 else ' ✗'
    print(f"  {label:<40} | {len(bl):>7} | {bwr:>5.1f}% | Rs.{bpnl:>7,.0f} | Rs.{bavg:>5,.0f}{flag}")

print(f"\n  Selling baseline:         Rs.{sell_conv:,.0f}")
print(f"  Script 86 reference:      blank 593t 80% Rs.63,580")

# Save
for name, recs in [('A',rec_a),('B',rec_b),('C',rec_c),('D',rec_d)]:
    if recs:
        pd.DataFrame(recs).to_csv(f'{OUT_DIR}/91_crt_ltf_{name}.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/91_crt_ltf_A/B/C/D.csv")
print("\nDone.")
