"""
92_crt_banknifty.py — CRT Approach D: NIFTY vs BANKNIFTY side by side
=======================================================================
Runs the same winning strategy (Approach D from script 91) on both indices:

  Approach D: 15M 3-candle CRT at TC/R1 + 5M LTF bearish confirmation
              Next candle entry — zero forward bias

NIFTY:      STRIKE_INT=50,  LOT_SIZE=75, SCALE=65/75
BANKNIFTY:  STRIKE_INT=500, LOT_SIZE=15, SCALE=1.0
            (lot was 20 before Nov 2021, using 15 throughout — conservative)

Both compared on: all days, blank days, year-wise
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates

EOD_EXIT = '15:20:00'
YEARS    = 5
OUT_DIR  = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = {
    'NIFTY':     dict(symbol='NIFTY',     strike_int=50,  lot_size=75, scale=65/75),
    'BANKNIFTY': dict(symbol='BANKNIFTY', strike_int=500, lot_size=15, scale=1.0),
}

def r2(v): return round(float(v), 2)

def get_otm1(spot, opt, strike_int):
    atm = int(round(spot / strike_int) * strike_int)
    return atm + strike_int if opt == 'CE' else atm - strike_int

def build_ohlc(tks, freq='15min', start='09:15:00', end='12:45:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample(freq).ohlc().dropna()
    ohlc.columns = ['o', 'h', 'l', 'c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

def simulate_sell(date_str, expiry, instrument, entry_time, lot_size, scale,
                  tgt_pct=0.20, sl_pct=1.00):
    tks = load_tick_data(date_str, instrument, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep*(1-tgt_pct)); hsl = r2(ep*(1+sl_pct)); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT: return r2((ep-p)*lot_size*scale), 'eod', r2(ep), r2(p), t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*lot_size*scale), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep-p)*lot_size*scale), 'lockin_sl' if sl<hsl else 'hard_sl', r2(ep), r2(p), t
    return r2((ep-ps[-1])*lot_size*scale), 'eod', r2(ep), r2(ps[-1]), ts[-1]

# ── Build daily OHLC + CPR for both indices ───────────────────────────────────
print("Building daily OHLC + CPR for NIFTY and BANKNIFTY...")
t0 = time.time()
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
extra     = max(0, all_dates.index(dates_5yr[0]) - 60)

def build_cpr(symbol):
    rows = []
    for d in all_dates[extra:]:
        tks = load_spot_data(d, symbol)
        if tks is None: continue
        day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')]
        if len(day)<2: continue
        rows.append({'date':d,'o':day.iloc[0]['price'],
                     'h':day['price'].max(),'l':day['price'].min(),'c':day.iloc[-1]['price']})
    df = pd.DataFrame(rows)
    ph = df['h'].shift(1); pl = df['l'].shift(1); pc = df['c'].shift(1)
    df['pvt'] = ((ph+pl+pc)/3).round(2)
    df['bc']  = ((ph+pl)/2).round(2)
    df['tc']  = (df['pvt']+(df['pvt']-df['bc'])).round(2)
    df['r1']  = (2*df['pvt']-pl).round(2)
    df = df.dropna().reset_index(drop=True)
    return df[df['date'].isin(dates_5yr)].reset_index(drop=True)

ohlc_nf = build_cpr('NIFTY')
ohlc_bn = build_cpr('BANKNIFTY')
print(f"  NIFTY: {len(ohlc_nf)} days | BANKNIFTY: {len(ohlc_bn)} days | {time.time()-t0:.0f}s")

df_sell_base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
sell_dates   = set(df_sell_base['date'].astype(str).str.replace('-',''))

# ── Scan function (Approach D) ────────────────────────────────────────────────
def run_approach_d(symbol, ohlc_days, strike_int, lot_size, scale):
    print(f"\nScanning {symbol} — Approach D (CPR + 5M LTF)...")
    t0 = time.time()
    records = []

    for idx, row in ohlc_days.iterrows():
        dstr     = row['date']
        tc_val   = row['tc']
        r1_val   = row['r1']
        is_blank = dstr not in sell_dates

        spot = load_spot_data(dstr, symbol)
        if spot is None: continue

        c15 = build_ohlc(spot, '15min')
        c5  = build_ohlc(spot, '5min', end='13:00:00')
        if len(c15) < 3 or len(c5) < 3: continue

        expiries = list_expiry_dates(dstr, index_name=symbol)
        if not expiries: continue
        expiry = expiries[0]

        year = dstr[:4]
        fired = False

        for ci in range(1, len(c15)-1):
            if fired: break
            c1 = c15.iloc[ci-1]
            c2 = c15.iloc[ci]
            c3_idx = ci + 1
            if c3_idx >= len(c15): break
            c3      = c15.iloc[c3_idx]
            c3_time = c3['time']
            if c3_time > '12:00:00': break

            c2h = c2['h']; c3c = c3['c']
            crt_tc = (c2h > tc_val and c3c < tc_val)
            crt_r1 = (c2h > r1_val and c3c < r1_val)
            if not (crt_tc or crt_r1): continue

            level_name = 'TC' if crt_tc else 'R1'

            # 5M LTF confirmation in 30-min window after C3 closes
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

            strike  = get_otm1(c3c, 'CE', strike_int)
            instr   = f'{symbol}{expiry}{strike}CE'
            res = simulate_sell(dstr, expiry, instr, ltf_entry, lot_size, scale)
            if res:
                pnl, reason, ep, xp, xt = res
                records.append(dict(
                    date=dstr, level=level_name, year=year,
                    is_blank=is_blank, ep=ep, xp=xp,
                    exit_reason=reason, pnl=r2(pnl),
                    win=pnl>0, entry_time=ltf_entry))
                fired = True

        if idx % 100 == 0:
            print(f"  {idx}/{len(ohlc_days)} | {len(records)} signals | {time.time()-t0:.0f}s")

    print(f"  Done | {len(records)} trades | {time.time()-t0:.0f}s")
    return records

rec_nf = run_approach_d('NIFTY',     ohlc_nf, 50,  75, 65/75)
rec_bn = run_approach_d('BANKNIFTY', ohlc_bn, 500, 15, 1.0)

# ── Results ───────────────────────────────────────────────────────────────────
sell_conv = df_sell_base['pnl_conv'].sum()

def show(symbol, records, lot_note=''):
    if not records:
        print(f"\n{symbol}: no trades"); return
    df = pd.DataFrame(records)
    wr   = df['win'].mean()*100
    pnl  = df['pnl'].sum()
    avg  = df['pnl'].mean()
    bl   = df[df['is_blank']]
    bwr  = bl['win'].mean()*100 if not bl.empty else 0
    bpnl = bl['pnl'].sum()
    bavg = bl['pnl'].mean() if not bl.empty else 0

    print(f"\n{'='*65}")
    print(f"  {symbol} — Approach D (CPR TC+R1 + 5M LTF)  {lot_note}")
    print(f"{'='*65}")
    print(f"  All days:   {len(df):>4}t | WR {wr:>5.1f}% | Rs.{pnl:>9,.0f} | Avg Rs.{avg:>6,.0f}")
    print(f"  Blank days: {len(bl):>4}t | WR {bwr:>5.1f}% | Rs.{bpnl:>9,.0f} | Avg Rs.{bavg:>6,.0f}")
    print(f"  Exits: {dict(df['exit_reason'].value_counts())}")
    print(f"  By level:")
    for lvl in ['TC','R1']:
        g = df[df['level']==lvl]
        blg = g[g['is_blank']]
        if g.empty: continue
        print(f"    {lvl}: {len(g):>3}t WR {g['win'].mean()*100:.0f}% Rs.{g['pnl'].sum():>8,.0f}"
              f"  | blank {len(blg):>3}t WR {blg['win'].mean()*100 if not blg.empty else 0:.0f}%"
              f" Rs.{blg['pnl'].sum():>8,.0f}")
    print(f"  Year-wise:")
    for yr in sorted(df['year'].unique()):
        g = df[df['year']==yr]
        blg = g[g['is_blank']]
        print(f"    {yr}: {len(g):>3}t WR {g['win'].mean()*100:.0f}% Rs.{g['pnl'].sum():>8,.0f}"
              f"  | blank {len(blg):>3}t WR {blg['win'].mean()*100 if not blg.empty else 0:.0f}%"
              f" Rs.{blg['pnl'].sum():>8,.0f}")

show('NIFTY',     rec_nf, '[75 qty → 65 qty scaled]')
show('BANKNIFTY', rec_bn, '[15 qty]')

# ── Side by side ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  SIDE BY SIDE — NIFTY vs BANKNIFTY (Approach D)")
print(f"{'='*70}")
print(f"  {'Metric':<30} | {'NIFTY':>15} | {'BANKNIFTY':>15}")
print(f"  {'-'*68}")

for label, fn, fb in [
    ("All trades",               lambda d: len(d),                        lambda d: len(d)),
    ("All WR %",                 lambda d: f"{d['win'].mean()*100:.1f}%", lambda d: f"{d['win'].mean()*100:.1f}%"),
    ("All P&L",                  lambda d: f"Rs.{d['pnl'].sum():,.0f}",   lambda d: f"Rs.{d['pnl'].sum():,.0f}"),
    ("All avg/trade",            lambda d: f"Rs.{d['pnl'].mean():,.0f}",  lambda d: f"Rs.{d['pnl'].mean():,.0f}"),
    ("Blank trades",             lambda d: len(d[d['is_blank']]),          lambda d: len(d[d['is_blank']])),
    ("Blank WR %",               lambda d: f"{d[d['is_blank']]['win'].mean()*100:.1f}%", lambda d: f"{d[d['is_blank']]['win'].mean()*100:.1f}%"),
    ("Blank P&L",                lambda d: f"Rs.{d[d['is_blank']]['pnl'].sum():,.0f}", lambda d: f"Rs.{d[d['is_blank']]['pnl'].sum():,.0f}"),
    ("Blank avg/trade",          lambda d: f"Rs.{d[d['is_blank']]['pnl'].mean():,.0f}", lambda d: f"Rs.{d[d['is_blank']]['pnl'].mean():,.0f}"),
    ("Hard SL count",            lambda d: len(d[d['exit_reason']=='hard_sl']), lambda d: len(d[d['exit_reason']=='hard_sl'])),
]:
    dfn = pd.DataFrame(rec_nf) if rec_nf else pd.DataFrame()
    dfb = pd.DataFrame(rec_bn) if rec_bn else pd.DataFrame()
    vn = fn(dfn) if not dfn.empty else '—'
    vb = fb(dfb) if not dfb.empty else '—'
    print(f"  {label:<30} | {str(vn):>15} | {str(vb):>15}")

# Save
if rec_nf: pd.DataFrame(rec_nf).to_csv(f'{OUT_DIR}/92_crt_nifty_d.csv', index=False)
if rec_bn: pd.DataFrame(rec_bn).to_csv(f'{OUT_DIR}/92_crt_banknifty_d.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/92_crt_nifty_d.csv + 92_crt_banknifty_d.csv")
print("\nDone.")
