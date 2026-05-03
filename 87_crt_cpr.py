"""
87_crt_cpr.py — CRT at CPR Levels
===================================
Instead of random candle H/L, use CPR levels as the reference.

Concept:
  CPR TC (Top) and BC (Bottom) are institutional levels.
  If price sweeps above TC then closes back below → CPR rejected as resistance
  If price sweeps below BC then closes back above → CPR rejected as support

CPR Bearish CRT:
  Any 15-min candle: H > TC  AND  C < TC  (wick above TC, close back below)
  → Sell CE OTM1 at next candle open

CPR Bullish CRT:
  Any 15-min candle: L < BC  AND  C > BC  (wick below BC, close back above)
  → Sell PE OTM1 at next candle open

Also test:
  Pivot level CRT: wick above/below Pivot, close back
  R1/S1 CRT: wick above R1 or below S1, close back

Scan: 09:15 – 12:00
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
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def get_otm1(s, opt):
    atm = get_atm(s)
    return atm + STRIKE_INT if opt == 'CE' else atm - STRIKE_INT

def build_15min(tks, start='09:15:00', end='12:30:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('15min').ohlc().dropna()
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
        if t >= EOD_EXIT: return r2((ep-p)*LOT_SIZE),'eod',r2(ep),r2(p),t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE),'target',r2(ep),r2(p),t
        if p >= sl:  return r2((ep-p)*LOT_SIZE),'lockin_sl' if sl<hsl else 'hard_sl',r2(ep),r2(p),t
    return r2((ep-ps[-1])*LOT_SIZE),'eod',r2(ep),r2(ps[-1]),ts[-1]

# ── Build daily OHLC + CPR ────────────────────────────────────────────────────
print("Building daily OHLC + CPR levels...")
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
ohlc['ema']  = ohlc['c'].ewm(span=20,adjust=False).mean().shift(1)
# CPR from previous day (shift 1)
ohlc['pvt']  = ((ohlc['h'].shift(1)+ohlc['l'].shift(1)+ohlc['c'].shift(1))/3).round(2)
ohlc['bc']   = ((ohlc['h'].shift(1)+ohlc['l'].shift(1))/2).round(2)
ohlc['tc']   = (ohlc['pvt']+(ohlc['pvt']-ohlc['bc'])).round(2)
ohlc['r1']   = (2*ohlc['pvt']-ohlc['l'].shift(1)).round(2)
ohlc['s1']   = (2*ohlc['pvt']-ohlc['h'].shift(1)).round(2)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)

df_sell_base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
sell_dates   = set(df_sell_base['date'].astype(str).str.replace('-',''))
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nScanning CRT at CPR / Pivot / R1-S1 levels...")
t0 = time.time()

rec_tc  = []   # CRT at TC
rec_bc  = []   # CRT at BC
rec_pvt = []   # CRT at Pivot
rec_r1  = []   # CRT at R1
rec_s1  = []   # CRT at S1

for idx, row in ohlc_5yr.iterrows():
    dstr = row['date']
    tc   = row['tc']; bc = row['bc']; pvt = row['pvt']
    r1   = row['r1']; s1 = row['s1']
    op   = row['o'];  ema = row['ema']

    is_blank    = dstr not in sell_dates
    ema_bias    = 'bull' if op > ema else 'bear'

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue
    candles = build_15min(spot)
    if candles.empty: continue

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
           pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

    fired = {'tc': False, 'bc': False, 'pvt_bear': False,
             'pvt_bull': False, 'r1': False, 's1': False}

    for ci, crow in candles.iterrows():
        ct = crow['time']
        if ct > '12:00:00': break
        ch = crow['h']; cl = crow['l']; cc = crow['c']

        h_str = ct[:2]; m_str = ct[3:5]
        entry_min = int(h_str)*60 + int(m_str) + CANDLE_MIN + 1
        entry_t   = f"{entry_min//60:02d}:{entry_min%60:02d}:02"
        if entry_t >= EOD_EXIT: break

        def record(rec_list, level_name, crt_dir, opt, level_val):
            strike = get_otm1(cc, opt)
            res = simulate_sell(dstr, expiry, strike, opt, entry_t)
            if res:
                pnl75, reason, ep, xp, xt = res
                pnl65 = r2(pnl75 * SCALE)
                rec_list.append(dict(
                    date=dstr, level=level_name, crt=crt_dir,
                    candle_time=ct, entry_time=entry_t,
                    level_val=r2(level_val), ch=r2(ch), cl=r2(cl), cc=r2(cc),
                    opt=opt, strike=strike, dte=dte,
                    ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                    pnl_65=pnl65, win=pnl65>0,
                    is_blank=is_blank, ema_bias=ema_bias,
                    year=dstr[:4]))

        # ── TC: wick above TC, close back below ──────────────────────────────
        if not fired['tc'] and ch > tc and cc < tc:
            record(rec_tc, 'TC', 'bearish', 'CE', tc)
            fired['tc'] = True

        # ── BC: wick below BC, close back above ──────────────────────────────
        if not fired['bc'] and cl < bc and cc > bc:
            record(rec_bc, 'BC', 'bullish', 'PE', bc)
            fired['bc'] = True

        # ── Pivot: wick above, close back below ──────────────────────────────
        if not fired['pvt_bear'] and ch > pvt and cc < pvt:
            record(rec_pvt, 'PVT_bear', 'bearish', 'CE', pvt)
            fired['pvt_bear'] = True

        # ── Pivot: wick below, close back above ──────────────────────────────
        if not fired['pvt_bull'] and cl < pvt and cc > pvt:
            record(rec_pvt, 'PVT_bull', 'bullish', 'PE', pvt)
            fired['pvt_bull'] = True

        # ── R1: wick above, close back below ─────────────────────────────────
        if not fired['r1'] and ch > r1 and cc < r1:
            record(rec_r1, 'R1', 'bearish', 'CE', r1)
            fired['r1'] = True

        # ── S1: wick below, close back above ─────────────────────────────────
        if not fired['s1'] and cl < s1 and cc > s1:
            record(rec_s1, 'S1', 'bullish', 'PE', s1)
            fired['s1'] = True

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | TC:{len(rec_tc)} BC:{len(rec_bc)} "
              f"PVT:{len(rec_pvt)} R1:{len(rec_r1)} S1:{len(rec_s1)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
def show(label, recs, show_blank=True):
    if not recs:
        print(f"\n{label}: no trades"); return pd.DataFrame()
    df = pd.DataFrame(recs)
    wr  = df['win'].mean()*100
    pnl = df['pnl_65'].sum()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:    {len(df)} | WR: {wr:.1f}% | Total: Rs.{pnl:,.0f} | Avg: Rs.{df['pnl_65'].mean():,.0f}")
    print(f"  Exits: {dict(df['exit_reason'].value_counts())}")
    if show_blank:
        bl = df[df['is_blank']]
        if not bl.empty:
            print(f"  Blank days: {len(bl)}t | WR {bl['win'].mean()*100:.0f}% | Rs.{bl['pnl_65'].sum():,.0f}")
    print(f"  Year-wise:")
    for yr in sorted(df['year'].unique()):
        g = df[df['year']==yr]
        print(f"    {yr}: {len(g):>3}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
    return df

df_tc  = show("CRT at TC  (wick above TC, close below)",   rec_tc)
df_bc  = show("CRT at BC  (wick below BC, close above)",   rec_bc)
df_pvt = show("CRT at PVT (wick above/below pivot, close back)", rec_pvt)
df_r1  = show("CRT at R1  (wick above R1, close below)",  rec_r1)
df_s1  = show("CRT at S1  (wick below S1, close above)",  rec_s1)

# ── Summary ───────────────────────────────────────────────────────────────────
sell_conv = df_sell_base['pnl_conv'].sum()
print(f"\n{'='*60}")
print("  SUMMARY — CRT at each CPR level")
print(f"{'='*60}")
print(f"  {'Level':<12} | {'All t':>6} | {'All WR':>7} | {'All P&L':>11} | {'Blank t':>7} | {'Blank WR':>9} | {'Blank P&L':>11}")
print(f"  {'-'*80}")
for label, recs in [('TC', rec_tc),('BC', rec_bc),('Pivot', rec_pvt),('R1', rec_r1),('S1', rec_s1)]:
    if not recs: continue
    df   = pd.DataFrame(recs)
    bl   = df[df['is_blank']]
    wr   = df['win'].mean()*100
    bwr  = bl['win'].mean()*100 if not bl.empty else 0
    print(f"  {label:<12} | {len(df):>6} | {wr:>6.0f}% | Rs.{df['pnl_65'].sum():>9,.0f} | "
          f"{len(bl):>7} | {bwr:>8.0f}% | Rs.{bl['pnl_65'].sum():>9,.0f}")

# Best combo for blank days
all_blank = []
for recs in [rec_tc, rec_bc, rec_pvt, rec_r1, rec_s1]:
    if recs:
        all_blank.extend([r for r in recs if r['is_blank']])

# Dedup by date (take first signal per date)
if all_blank:
    df_ab = pd.DataFrame(all_blank).sort_values(['date','candle_time']).drop_duplicates('date', keep='first')
    print(f"\n  Best blank days (first signal per day, any CPR level):")
    print(f"  {len(df_ab)}t | WR {df_ab['win'].mean()*100:.0f}% | Rs.{df_ab['pnl_65'].sum():,.0f}")
    print(f"  → Combined with baseline: Rs.{sell_conv + df_ab['pnl_65'].sum():,.0f}")

    # Save
    df_ab.to_csv(f'{OUT_DIR}/87_crt_cpr_blank.csv', index=False)
    print(f"\n  Saved → {OUT_DIR}/87_crt_cpr_blank.csv")

print("\nDone.")
