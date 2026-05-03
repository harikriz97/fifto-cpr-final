"""
93_crt_spot_target.py — CRT Proper Target: Spot-based TP1/TP2 vs Option % target
==================================================================================
CRT slide says: "Profits = TP1 at 50%, TP2 at Low/High"

This means targets are SPOT PRICE levels, not option premium %:
  Bearish CRT:
    CRT High = C1.H  (reference range top)
    CRT Low  = C1.L  (reference range bottom)
    CRT Range = C1.H - C1.L

    TP1 (spot) = C1.H - (range × 50%)  ← midpoint of C1
    TP2 (spot) = C1.L                   ← full range (CRT Low)

  Exit: when SPOT price hits TP1 or TP2 → close CE option at that tick
  SL:   option hard SL at 100% (option doubles) — risk management unchanged

Compare 3 exit methods on same Approach D signals:
  Current:  option loses 20% of premium
  TP1:      spot hits C1 midpoint
  TP2:      spot hits C1.Low (full range)
  TP1+TP2:  exit half at TP1, rest at TP2 (2-lot style)
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates

LOT_SIZE = 75
SCALE    = 65 / 75
STRIKE_INT = 50
CANDLE_MIN = 15
EOD_EXIT = '15:20:00'
YEARS    = 5
OUT_DIR  = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)
def get_otm1(s):
    atm = int(round(s / STRIKE_INT) * STRIKE_INT)
    return atm + STRIKE_INT

def build_ohlc(tks, freq='15min', start='09:15:00', end='12:45:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample(freq).ohlc().dropna()
    ohlc.columns = ['o', 'h', 'l', 'c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

def simulate_opt_pct(date_str, expiry, strike, entry_time, tgt_pct=0.20):
    """Current method: exit when option loses tgt_pct of premium"""
    instr = f'NIFTY{expiry}{strike}CE'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep*(1-tgt_pct)); hsl = r2(ep*2.0); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT: return r2((ep-p)*LOT_SIZE*SCALE), 'eod', r2(ep), r2(p), t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE*SCALE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep-p)*LOT_SIZE*SCALE), 'sl', r2(ep), r2(p), t
    return r2((ep-ps[-1])*LOT_SIZE*SCALE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

def simulate_spot_target(date_str, expiry, strike, entry_time,
                          tp1_spot, tp2_spot, spot_tks):
    """New method: exit when SPOT hits TP1 or TP2 (CRT range targets)"""
    instr = f'NIFTY{expiry}{strike}CE'
    opt_tks = load_tick_data(date_str, instr, entry_time)
    if opt_tks is None or opt_tks.empty: return None, None
    opt_tks = opt_tks[opt_tks['time'] >= entry_time].reset_index(drop=True)
    if opt_tks.empty: return None, None

    ep = r2(opt_tks.iloc[0]['price'])
    if ep <= 0: return None, None
    hard_sl = r2(ep * 2.0)   # 100% SL on option

    # Use spot for target detection, option for SL and exit price
    spot_from = spot_tks[spot_tks['time'] >= entry_time].reset_index(drop=True)
    if spot_from.empty: return None, None

    # Build a time-indexed lookup for option price
    opt_price = opt_tks.set_index('time')['price']

    tp1_hit = None; tp2_hit = None
    sl_hit  = None

    for _, srow in spot_from.iterrows():
        t = srow['time']; sp = srow['price']
        if t >= EOD_EXIT: break

        # Check option SL at this time (approximate from option ticks)
        opt_now = opt_tks[opt_tks['time'] <= t]
        if not opt_now.empty:
            op = opt_now.iloc[-1]['price']
            if op >= hard_sl:
                sl_hit = (t, op); break

        if sp <= tp1_spot and tp1_hit is None:
            # Get option price at this time
            op_row = opt_tks[opt_tks['time'] <= t]
            op = op_row.iloc[-1]['price'] if not op_row.empty else ep
            tp1_hit = (t, r2(op))

        if sp <= tp2_spot and tp2_hit is None:
            op_row = opt_tks[opt_tks['time'] <= t]
            op = op_row.iloc[-1]['price'] if not op_row.empty else ep
            tp2_hit = (t, r2(op))
            break  # TP2 = full target, exit all

    return tp1_hit, tp2_hit, sl_hit, ep

def get_eod_option_price(date_str, expiry, strike, entry_time):
    instr = f'NIFTY{expiry}{strike}CE'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    eod = tks[tks['time'] <= EOD_EXIT]
    return r2(eod.iloc[-1]['price']) if not eod.empty else None

# ── Build daily OHLC + CPR ─────────────────────────────────────────────────
print("Building daily OHLC + CPR...")
t0 = time.time()
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
extra     = max(0, all_dates.index(dates_5yr[0]) - 60)

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
print("\nScanning Approach D signals + spot-based targets...")
t0 = time.time()

rec_pct   = []   # current: 20% option premium
rec_tp1   = []   # new: spot hits C1 midpoint
rec_tp2   = []   # new: spot hits C1.Low (full range)

for idx, row in ohlc_5yr.iterrows():
    dstr     = row['date']
    tc_val   = row['tc']
    r1_val   = row['r1']
    is_blank = dstr not in sell_dates
    year     = dstr[:4]

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue

    c15 = build_ohlc(spot, '15min')
    c5  = build_ohlc(spot, '5min', end='13:00:00')
    if len(c15) < 3 or len(c5) < 3: continue

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]

    fired = False
    for ci in range(1, len(c15)-1):
        if fired: break
        c1 = c15.iloc[ci-1]
        c2 = c15.iloc[ci]
        c3_idx = ci+1
        if c3_idx >= len(c15): break
        c3      = c15.iloc[c3_idx]
        c3_time = c3['time']
        if c3_time > '12:00:00': break

        c2h = c2['h']; c3c = c3['c']
        if not ((c2h > tc_val and c3c < tc_val) or (c2h > r1_val and c3c < r1_val)):
            continue

        level = 'TC' if (c2h > tc_val and c3c < tc_val) else 'R1'

        # 5M LTF confirmation
        c3_close_min = int(c3_time[:2])*60 + int(c3_time[3:5]) + 15
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
                    ltf_entry = ftime
                break
        if not ltf_entry: continue

        strike = get_otm1(c3c)

        # CRT spot targets from C1 range
        c1h = c1['h']; c1l = c1['l']
        crt_range = c1h - c1l
        tp1_spot  = r2(c1h - crt_range * 0.50)   # midpoint
        tp2_spot  = r2(c1l)                        # C1 Low = full range

        # ── METHOD 1: Current 20% option premium ──────────────────────────
        res = simulate_opt_pct(dstr, expiry, strike, ltf_entry, tgt_pct=0.20)
        if res:
            pnl, reason, ep, xp, xt = res
            rec_pct.append(dict(date=dstr, level=level, year=year, is_blank=is_blank,
                ep=ep, xp=xp, exit_reason=reason, pnl=r2(pnl), win=pnl>0,
                c1h=r2(c1h), c1l=r2(c1l), tp1=r2(tp1_spot), tp2=r2(tp2_spot)))

        # ── METHOD 2 & 3: Spot-based TP1 / TP2 ───────────────────────────
        result = simulate_spot_target(dstr, expiry, strike, ltf_entry,
                                       tp1_spot, tp2_spot, spot)
        if result[0] is not None or len(result) == 4:
            tp1_hit, tp2_hit, sl_hit, ep = result
            if ep is None: continue

            # TP1 exit
            if tp1_hit:
                xt, xp = tp1_hit
                pnl = r2((ep - xp) * LOT_SIZE * SCALE)
                rec_tp1.append(dict(date=dstr, level=level, year=year, is_blank=is_blank,
                    ep=ep, xp=xp, exit_reason='tp1_spot', pnl=r2(pnl), win=pnl>0,
                    tp1=r2(tp1_spot), tp2=r2(tp2_spot), c1_range=r2(crt_range)))
            elif sl_hit:
                xt, xp = sl_hit
                pnl = r2((ep - xp) * LOT_SIZE * SCALE)
                rec_tp1.append(dict(date=dstr, level=level, year=year, is_blank=is_blank,
                    ep=ep, xp=xp, exit_reason='sl', pnl=r2(pnl), win=pnl>0,
                    tp1=r2(tp1_spot), tp2=r2(tp2_spot), c1_range=r2(crt_range)))
            else:
                eod_p = get_eod_option_price(dstr, expiry, strike, ltf_entry)
                if eod_p:
                    pnl = r2((ep - eod_p) * LOT_SIZE * SCALE)
                    rec_tp1.append(dict(date=dstr, level=level, year=year, is_blank=is_blank,
                        ep=ep, xp=eod_p, exit_reason='eod', pnl=r2(pnl), win=pnl>0,
                        tp1=r2(tp1_spot), tp2=r2(tp2_spot), c1_range=r2(crt_range)))

            # TP2 exit
            if tp2_hit:
                xt, xp = tp2_hit
                pnl = r2((ep - xp) * LOT_SIZE * SCALE)
                rec_tp2.append(dict(date=dstr, level=level, year=year, is_blank=is_blank,
                    ep=ep, xp=xp, exit_reason='tp2_spot', pnl=r2(pnl), win=pnl>0,
                    tp1=r2(tp1_spot), tp2=r2(tp2_spot), c1_range=r2(crt_range)))
            elif sl_hit:
                xt, xp = sl_hit
                pnl = r2((ep - xp) * LOT_SIZE * SCALE)
                rec_tp2.append(dict(date=dstr, level=level, year=year, is_blank=is_blank,
                    ep=ep, xp=xp, exit_reason='sl', pnl=r2(pnl), win=pnl>0,
                    tp1=r2(tp1_spot), tp2=r2(tp2_spot), c1_range=r2(crt_range)))
            else:
                eod_p = get_eod_option_price(dstr, expiry, strike, ltf_entry)
                if eod_p:
                    pnl = r2((ep - eod_p) * LOT_SIZE * SCALE)
                    rec_tp2.append(dict(date=dstr, level=level, year=year, is_blank=is_blank,
                        ep=ep, xp=eod_p, exit_reason='eod', pnl=r2(pnl), win=pnl>0,
                        tp1=r2(tp1_spot), tp2=r2(tp2_spot), c1_range=r2(crt_range)))

        fired = True

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | pct:{len(rec_pct)} tp1:{len(rec_tp1)} tp2:{len(rec_tp2)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
sell_conv = df_sell_base['pnl_conv'].sum()

def show(label, recs):
    if not recs:
        print(f"\n  {label}: no trades"); return pd.DataFrame()
    df = pd.DataFrame(recs)
    wr   = df['win'].mean()*100
    pnl  = df['pnl'].sum()
    avg  = df['pnl'].mean()
    bl   = df[df['is_blank']]
    bwr  = bl['win'].mean()*100 if not bl.empty else 0
    bpnl = bl['pnl'].sum()
    bavg = bl['pnl'].mean() if not bl.empty else 0
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  All:   {len(df):>4}t | WR {wr:>5.1f}% | Rs.{pnl:>9,.0f} | Avg Rs.{avg:>6,.0f}")
    print(f"  Blank: {len(bl):>4}t | WR {bwr:>5.1f}% | Rs.{bpnl:>9,.0f} | Avg Rs.{bavg:>6,.0f}")
    print(f"  Exits: {dict(df['exit_reason'].value_counts())}")
    print(f"  Year-wise:")
    for yr in sorted(df['year'].unique()):
        g = df[df['year']==yr]; blg = g[g['is_blank']]
        print(f"    {yr}: {len(g):>3}t WR {g['win'].mean()*100:.0f}% Rs.{g['pnl'].sum():>8,.0f}"
              f"  | blank {len(blg):>3}t WR {blg['win'].mean()*100 if not blg.empty else 0:.0f}%"
              f" Rs.{blg['pnl'].sum():>8,.0f}")
    return df

df_pct = show("METHOD 1: Option 20% target (current)", rec_pct)
df_tp1 = show("METHOD 2: Spot TP1 — C1 midpoint (50% of range)", rec_tp1)
df_tp2 = show("METHOD 3: Spot TP2 — C1 Low (full range)", rec_tp2)

# Also show avg C1 range in points
if rec_tp1:
    df_r = pd.DataFrame(rec_tp1)
    print(f"\n  Avg C1 range: {df_r['c1_range'].mean():.0f} pts | "
          f"Min: {df_r['c1_range'].min():.0f} | Max: {df_r['c1_range'].max():.0f}")
    print(f"  Avg TP1 distance from C1.H: {df_r['c1_range'].mean()/2:.0f} pts")

print(f"\n{'='*65}")
print("  SUMMARY — Blank days")
print(f"{'='*65}")
print(f"  {'Method':<40} | {'Blank t':>7} | {'WR':>6} | {'P&L':>10} | {'Avg':>7}")
print(f"  {'-'*63}")
for label, recs in [
    ("Option 20% premium (current)", rec_pct),
    ("Spot TP1 — C1 midpoint",       rec_tp1),
    ("Spot TP2 — C1 Low (full)",     rec_tp2),
]:
    if not recs: continue
    df = pd.DataFrame(recs)
    bl = df[df['is_blank']]
    if bl.empty: continue
    print(f"  {label:<40} | {len(bl):>7} | {bl['win'].mean()*100:>5.1f}% | "
          f"Rs.{bl['pnl'].sum():>7,.0f} | Rs.{bl['pnl'].mean():>5,.0f}")

# Save
for name, recs in [('pct',rec_pct),('tp1',rec_tp1),('tp2',rec_tp2)]:
    if recs: pd.DataFrame(recs).to_csv(f'{OUT_DIR}/93_crt_spot_{name}.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/93_crt_spot_*.csv")
print("\nDone.")
