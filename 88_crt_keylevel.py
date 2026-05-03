"""
88_crt_keylevel.py — Proper CRT at Key Levels (3-candle + R1/TC filter)
=========================================================================
Combines script 86 (3-candle structure) + script 87 (key levels):

Learning from CRT PDF:
  "Wicking (manipulation) MUST happen at HTF Key Zones — SNR, OB, Supply/Demand"
  For Nifty: R1 and TC are the validated HTF key levels (script 87 proved this)

Strategy:
  Bearish CRT at R1:  C2 wicks above R1 AND C3 closes below R1  → Sell CE OTM1
  Bearish CRT at TC:  C2 wicks above TC AND C3 closes below TC  → Sell CE OTM1

Two entry models (from PDF screenshots):
  Model A — C3-open entry:  Enter at C3 open (as soon as C3 starts, after C2 sweep)
  Model B — Next candle:    Enter after C3 fully closes (current script 86 approach)

Scan window: 09:15 – 12:00 (C1 candle start)
Only bearish CRT (R1 + TC) — bull/support side showed no edge in script 87
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
        if t >= EOD_EXIT: return r2((ep-p)*LOT_SIZE), 'eod', r2(ep), r2(p), t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep-p)*LOT_SIZE), 'lockin_sl' if sl<hsl else 'hard_sl', r2(ep), r2(p), t
    return r2((ep-ps[-1])*LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

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
    rows.append({'date':d, 'o':day.iloc[0]['price'],
                 'h':day['price'].max(), 'l':day['price'].min(), 'c':day.iloc[-1]['price']})

ohlc = pd.DataFrame(rows)
ohlc['ema'] = ohlc['c'].ewm(span=20, adjust=False).mean().shift(1)
ohlc['pvt'] = ((ohlc['h'].shift(1)+ohlc['l'].shift(1)+ohlc['c'].shift(1))/3).round(2)
ohlc['bc']  = ((ohlc['h'].shift(1)+ohlc['l'].shift(1))/2).round(2)
ohlc['tc']  = (ohlc['pvt']+(ohlc['pvt']-ohlc['bc'])).round(2)
ohlc['r1']  = (2*ohlc['pvt']-ohlc['l'].shift(1)).round(2)
ohlc['s1']  = (2*ohlc['pvt']-ohlc['h'].shift(1)).round(2)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)

df_sell_base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
sell_dates   = set(df_sell_base['date'].astype(str).str.replace('-',''))
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nScanning 3-candle CRT at R1 + TC levels...")
t0 = time.time()

# Two separate record lists: Model A (C3-open entry) and Model B (next candle)
rec_a = []   # C3-open entry
rec_b = []   # Next candle entry (baseline, same as 86)

for idx, row in ohlc_5yr.iterrows():
    dstr = row['date']
    tc   = row['tc']
    r1   = row['r1']
    ema  = row['ema']
    op   = row['o']

    is_blank  = dstr not in sell_dates
    ema_bias  = 'bull' if op > ema else 'bear'

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue
    candles = build_15min(spot)
    if len(candles) < 3: continue

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
           pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

    fired_a = False
    fired_b = False

    for ci in range(1, len(candles)-1):
        if fired_a and fired_b: break

        c1 = candles.iloc[ci-1]
        c2 = candles.iloc[ci]
        c3_idx = ci + 1
        if c3_idx >= len(candles): break

        c3 = candles.iloc[c3_idx]
        c3_time = c3['time']
        if c3_time > '12:00:00': break

        c1h = c1['h']; c1l = c1['l']
        c2h = c2['h']; c2l = c2['l']
        c3c = c3['c']

        # ── Check: C2 wicks above R1 AND C3 closes below R1 (bearish CRT at R1) ──
        # ── Check: C2 wicks above TC AND C3 closes below TC (bearish CRT at TC) ──
        is_crt_r1 = (c2h > r1 and c3c < r1)
        is_crt_tc = (c2h > tc and c3c < tc)

        if not (is_crt_r1 or is_crt_tc):
            continue

        level_name = 'R1' if is_crt_r1 else 'TC'
        level_val  = r1  if is_crt_r1 else tc

        opt    = 'CE'
        strike = get_otm1(c3c, opt)

        # ── Model A: C3-open entry ─────────────────────────────────────────────
        # Enter at the open of C3 (right when C3 candle starts)
        if not fired_a:
            h_str = c3_time[:2]; m_str = c3_time[3:5]
            entry_a = f"{c3_time[:2]}:{c3_time[3:5]}:02"   # 2 sec into C3 open
            if entry_a < EOD_EXIT:
                res = simulate_sell(dstr, expiry, strike, opt, entry_a)
                if res:
                    pnl75, reason, ep, xp, xt = res
                    pnl65 = r2(pnl75 * SCALE)
                    rec_a.append(dict(
                        date=dstr, level=level_name, entry_model='C3_open',
                        c1_time=c1['time'], c2_time=c2['time'], c3_time=c3_time,
                        level_val=r2(level_val), c2h=r2(c2h), c3c=r2(c3c),
                        entry_time=entry_a, opt=opt, strike=strike, dte=dte,
                        ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                        pnl_65=pnl65, win=pnl65>0,
                        is_blank=is_blank, ema_aligned=(ema_bias=='bear'),
                        year=dstr[:4]))
                    fired_a = True

        # ── Model B: Next candle entry (after C3 closes) ──────────────────────
        if not fired_b:
            h_str = c3_time[:2]; m_str = c3_time[3:5]
            entry_min = int(h_str)*60 + int(m_str) + CANDLE_MIN + 1
            entry_b   = f"{entry_min//60:02d}:{entry_min%60:02d}:02"
            if entry_b < EOD_EXIT:
                res = simulate_sell(dstr, expiry, strike, opt, entry_b)
                if res:
                    pnl75, reason, ep, xp, xt = res
                    pnl65 = r2(pnl75 * SCALE)
                    rec_b.append(dict(
                        date=dstr, level=level_name, entry_model='next_candle',
                        c1_time=c1['time'], c2_time=c2['time'], c3_time=c3_time,
                        level_val=r2(level_val), c2h=r2(c2h), c3c=r2(c3c),
                        entry_time=entry_b, opt=opt, strike=strike, dte=dte,
                        ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                        pnl_65=pnl65, win=pnl65>0,
                        is_blank=is_blank, ema_aligned=(ema_bias=='bear'),
                        year=dstr[:4]))
                    fired_b = True

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | A:{len(rec_a)} B:{len(rec_b)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
def show(label, recs):
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
    bl = df[df['is_blank']]
    if not bl.empty:
        print(f"  Blank days: {len(bl)}t | WR {bl['win'].mean()*100:.0f}% | Rs.{bl['pnl_65'].sum():,.0f} | Avg Rs.{bl['pnl_65'].mean():,.0f}")
    ea = df[df['ema_aligned']]
    if not ea.empty:
        print(f"  EMA-aligned (bear): {len(ea)}t | WR {ea['win'].mean()*100:.0f}% | Rs.{ea['pnl_65'].sum():,.0f}")
    bea = df[df['is_blank'] & df['ema_aligned']]
    if not bea.empty:
        print(f"  Blank + EMA-bear: {len(bea)}t | WR {bea['win'].mean()*100:.0f}% | Rs.{bea['pnl_65'].sum():,.0f}")

    print(f"  By level:")
    for lvl in df['level'].unique():
        g = df[df['level']==lvl]
        bl_g = g[g['is_blank']]
        print(f"    {lvl}: {len(g)}t WR {g['win'].mean()*100:.0f}% Rs.{g['pnl_65'].sum():,.0f}"
              f"  | blank: {len(bl_g)}t WR {bl_g['win'].mean()*100 if not bl_g.empty else 0:.0f}% Rs.{bl_g['pnl_65'].sum():,.0f}")
    print(f"  Year-wise:")
    for yr in sorted(df['year'].unique()):
        g = df[df['year']==yr]
        print(f"    {yr}: {len(g):>3}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
    return df

df_a = show("Model A — C3-open entry (enter at C3 open, right after C2 sweep)", rec_a)
df_b = show("Model B — Next candle entry (enter after C3 closes = current 86 style)", rec_b)

# ── Head-to-head comparison ───────────────────────────────────────────────────
sell_conv = df_sell_base['pnl_conv'].sum()

print(f"\n{'='*60}")
print("  HEAD-TO-HEAD: C3-open vs Next-candle (same signals)")
print(f"{'='*60}")

# Match by date+level (same CRT signal, different entry)
if rec_a and rec_b:
    df_a2 = pd.DataFrame(rec_a).set_index(['date','level'])
    df_b2 = pd.DataFrame(rec_b).set_index(['date','level'])
    common = df_a2.index.intersection(df_b2.index)
    if len(common):
        a_sub = df_a2.loc[common]
        b_sub = df_b2.loc[common]
        print(f"  Same-signal trades: {len(common)}")
        print(f"  C3-open entry:    WR {a_sub['win'].mean()*100:.1f}% | Rs.{a_sub['pnl_65'].sum():,.0f} | Avg Rs.{a_sub['pnl_65'].mean():,.0f}")
        print(f"  Next-candle entry: WR {b_sub['win'].mean()*100:.1f}% | Rs.{b_sub['pnl_65'].sum():,.0f} | Avg Rs.{b_sub['pnl_65'].mean():,.0f}")
        print(f"  Avg entry price — C3-open: {a_sub['ep'].mean():.1f} | next-candle: {b_sub['ep'].mean():.1f}")

# ── Blank day combinations ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  BLANK DAY SUMMARY — best combo with baseline")
print(f"{'='*60}")
print(f"  Selling baseline: Rs.{sell_conv:,.0f}")

for label, df_x in [("Model A (C3-open)", df_a), ("Model B (next-candle)", df_b)]:
    if df_x.empty: continue
    bl = df_x[df_x['is_blank']]
    if bl.empty: continue
    print(f"  + {label} blank: {len(bl)}t | WR {bl['win'].mean()*100:.0f}% | Rs.{bl['pnl_65'].sum():,.0f}"
          f" → Total Rs.{sell_conv + bl['pnl_65'].sum():,.0f}")

# Reference from earlier scripts
print(f"\n  [Reference]")
print(f"  Script 86 (any level) blank: 593t | WR 80% | Rs.63,580 → Rs.{sell_conv+63580:,.0f}")
print(f"  Script 87 R1 blank:          168t | WR 83% | Rs.27,339")
print(f"  Script 87 TC blank:          225t | WR 81% | Rs.21,746")
print(f"  Script 87 R1+TC blank combined (est): ~Rs.49,085")

# Save
if rec_a:
    pd.DataFrame(rec_a).to_csv(f'{OUT_DIR}/88_crt_keylevel_modelA.csv', index=False)
if rec_b:
    pd.DataFrame(rec_b).to_csv(f'{OUT_DIR}/88_crt_keylevel_modelB.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/88_crt_keylevel_modelA.csv + modelB.csv")
print("\nDone.")
