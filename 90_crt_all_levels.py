"""
90_crt_all_levels.py — 3-Candle CRT at All Pivot Levels (Clean, No Bias)
=========================================================================
Two clean tests side by side:

  TEST 1 — With CPR levels (R1-R4, S1-S4, TC, BC):
    Bearish CRT at resistance (TC, R1-R4): C2 wicks above level, C3 closes below → Sell CE
    Bullish CRT at support  (BC, S1-S4):  C2 wicks below level, C3 closes above → Sell PE

  TEST 2 — Without CPR (bare 3-candle CRT):
    Reference: Script 86 results (593 blank days, 80% WR, +Rs.63,580)

Entry: NEXT candle after C3 closes (zero forward bias — C3.close is fully known at entry)
3-candle structure: C1=range, C2=sweep past level, C3=close back inside level

Pivot formulas (all from previous day OHLC):
  pvt = (H + L + C) / 3
  bc  = (H + L) / 2
  tc  = pvt + (pvt - bc)
  R1  = 2*pvt - L
  R2  = pvt + (H - L)
  R3  = H + 2*(pvt - L)
  R4  = R3 + (H - L)
  S1  = 2*pvt - H
  S2  = pvt - (H - L)
  S3  = L - 2*(H - pvt)
  S4  = S3 - (H - L)
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

# ── Build daily OHLC + ALL pivot levels ───────────────────────────────────────
print("Building daily OHLC + all pivot levels (R1-R4, S1-S4, TC, BC)...")
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

# Previous day values for pivot calculations
ph = ohlc['h'].shift(1)
pl = ohlc['l'].shift(1)
pc = ohlc['c'].shift(1)

ohlc['pvt'] = ((ph + pl + pc) / 3).round(2)
ohlc['bc']  = ((ph + pl) / 2).round(2)
ohlc['tc']  = (ohlc['pvt'] + (ohlc['pvt'] - ohlc['bc'])).round(2)

# Resistance levels
ohlc['r1']  = (2*ohlc['pvt'] - pl).round(2)
ohlc['r2']  = (ohlc['pvt'] + (ph - pl)).round(2)
ohlc['r3']  = (ph + 2*(ohlc['pvt'] - pl)).round(2)
ohlc['r4']  = (ohlc['r3'] + (ph - pl)).round(2)

# Support levels
ohlc['s1']  = (2*ohlc['pvt'] - ph).round(2)
ohlc['s2']  = (ohlc['pvt'] - (ph - pl)).round(2)
ohlc['s3']  = (pl - 2*(ph - ohlc['pvt'])).round(2)
ohlc['s4']  = (ohlc['s3'] - (ph - pl)).round(2)

ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)

df_sell_base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
sell_dates   = set(df_sell_base['date'].astype(str).str.replace('-',''))
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

# Level definitions: (name, col, direction, option)
RESIST_LEVELS = [('TC','tc'), ('R1','r1'), ('R2','r2'), ('R3','r3'), ('R4','r4')]
SUPPORT_LEVELS = [('BC','bc'), ('S1','s1'), ('S2','s2'), ('S3','s3'), ('S4','s4')]

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nScanning 3-candle CRT at all levels (next-candle entry, no bias)...")
t0 = time.time()

# Records per level — each level gets first signal per day
records = {lvl[0]: [] for lvl in RESIST_LEVELS + SUPPORT_LEVELS}

for idx, row in ohlc_5yr.iterrows():
    dstr = row['date']
    ema  = row['ema']
    op   = row['o']
    is_blank = dstr not in sell_dates
    ema_bias = 'bull' if op > ema else 'bear'

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue
    candles = build_15min(spot)
    if len(candles) < 3: continue

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
           pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

    # Track which levels already fired today
    fired = {lvl[0]: False for lvl in RESIST_LEVELS + SUPPORT_LEVELS}

    for ci in range(1, len(candles)-1):
        if all(fired.values()): break

        c1 = candles.iloc[ci-1]
        c2 = candles.iloc[ci]
        c3_idx = ci + 1
        if c3_idx >= len(candles): break
        c3 = candles.iloc[c3_idx]
        c3_time = c3['time']
        if c3_time > '12:00:00': break

        c2h = c2['h']; c2l = c2['l']
        c3c = c3['c']

        # Entry: next candle after C3 closes (clean, no forward bias)
        h_str = c3_time[:2]; m_str = c3_time[3:5]
        entry_min = int(h_str)*60 + int(m_str) + CANDLE_MIN + 1
        entry_t   = f"{entry_min//60:02d}:{entry_min%60:02d}:02"
        if entry_t >= EOD_EXIT: break

        # ── Resistance levels: C2 wicks above level, C3 closes below ──────────
        for lvl_name, lvl_col in RESIST_LEVELS:
            if fired[lvl_name]: continue
            level_val = row[lvl_col]
            if c2h > level_val and c3c < level_val:
                strike = get_otm1(c3c, 'CE')
                res = simulate_sell(dstr, expiry, strike, 'CE', entry_t)
                if res:
                    pnl75, reason, ep, xp, xt = res
                    pnl65 = r2(pnl75 * SCALE)
                    records[lvl_name].append(dict(
                        date=dstr, level=lvl_name, direction='bearish',
                        c3_time=c3_time, entry_time=entry_t,
                        level_val=r2(level_val), c2h=r2(c2h), c3c=r2(c3c),
                        opt='CE', strike=strike, dte=dte,
                        ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                        pnl_65=pnl65, win=pnl65>0,
                        is_blank=is_blank, ema_bear=(ema_bias=='bear'),
                        year=dstr[:4]))
                    fired[lvl_name] = True

        # ── Support levels: C2 wicks below level, C3 closes above ─────────────
        for lvl_name, lvl_col in SUPPORT_LEVELS:
            if fired[lvl_name]: continue
            level_val = row[lvl_col]
            if c2l < level_val and c3c > level_val:
                strike = get_otm1(c3c, 'PE')
                res = simulate_sell(dstr, expiry, strike, 'PE', entry_t)
                if res:
                    pnl75, reason, ep, xp, xt = res
                    pnl65 = r2(pnl75 * SCALE)
                    records[lvl_name].append(dict(
                        date=dstr, level=lvl_name, direction='bullish',
                        c3_time=c3_time, entry_time=entry_t,
                        level_val=r2(level_val), c2l=r2(c2l), c3c=r2(c3c),
                        opt='PE', strike=strike, dte=dte,
                        ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                        pnl_65=pnl65, win=pnl65>0,
                        is_blank=is_blank, ema_bear=(ema_bias=='bear'),
                        year=dstr[:4]))
                    fired[lvl_name] = True

    if idx % 100 == 0:
        counts = ' '.join(f"{n}:{len(records[n])}" for n,_ in RESIST_LEVELS+SUPPORT_LEVELS)
        print(f"  {idx}/{len(ohlc_5yr)} | {counts} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
def show_level(lvl_name, recs):
    if not recs:
        print(f"  {lvl_name:<4} | {'0':>5}t | {'—':>7} | {'—':>12} | {'—':>8} | {'—':>5}t {'—':>5} {'—':>10}")
        return
    df = pd.DataFrame(recs)
    wr  = df['win'].mean()*100
    pnl = df['pnl_65'].sum()
    avg = df['pnl_65'].mean()
    bl  = df[df['is_blank']]
    bwr = bl['win'].mean()*100 if not bl.empty else 0
    bpnl= bl['pnl_65'].sum()
    flag = ' ✓' if pnl > 0 and bpnl > 0 else (' ~' if pnl > 0 or bpnl > 0 else '')
    print(f"  {lvl_name:<4} | {len(df):>5}t | {wr:>6.1f}% | Rs.{pnl:>10,.0f} | Rs.{avg:>6,.0f} | "
          f"{len(bl):>4}t {bwr:>5.1f}% Rs.{bpnl:>9,.0f}{flag}")

print(f"\n{'='*90}")
print("  TEST 1: 3-CANDLE CRT AT CPR LEVELS (next-candle entry, no bias)")
print(f"{'='*90}")
print(f"  {'Level':<4} | {'Trades':>5} | {'WR':>7} | {'All P&L':>12} | {'Avg':>8} | {'Blank trades / WR / P&L'}")
print(f"  {'-'*88}")
print("  — RESISTANCE (Bearish CRT → Sell CE) —")
for lvl_name, _ in RESIST_LEVELS:
    show_level(lvl_name, records[lvl_name])
print("  — SUPPORT (Bullish CRT → Sell PE) —")
for lvl_name, _ in SUPPORT_LEVELS:
    show_level(lvl_name, records[lvl_name])

# ── Year-wise for each level ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  YEAR-WISE BREAKDOWN")
print(f"{'='*70}")
years = sorted(dates_5yr[0][:4] for _ in [1])
all_years = sorted(set(r['year'] for recs in records.values() for r in recs))
for lvl_name, _ in RESIST_LEVELS + SUPPORT_LEVELS:
    recs = records[lvl_name]
    if not recs: continue
    df = pd.DataFrame(recs)
    if df['pnl_65'].sum() <= 0: continue   # show only profitable levels
    row_str = f"  {lvl_name:<4}:"
    for yr in all_years:
        g = df[df['year']==yr]
        if g.empty: row_str += f"  {yr}:  0t    —"
        else: row_str += f"  {yr}: {len(g):>3}t {g['win'].mean()*100:.0f}% Rs.{g['pnl_65'].sum():>7,.0f}"
    print(row_str)

# ── Blank days summary ────────────────────────────────────────────────────────
sell_conv = df_sell_base['pnl_conv'].sum()
print(f"\n{'='*70}")
print("  BLANK DAYS SUMMARY — Top levels")
print(f"{'='*70}")
blank_rows = []
for lvl_name, _ in RESIST_LEVELS + SUPPORT_LEVELS:
    recs = records[lvl_name]
    if not recs: continue
    df = pd.DataFrame(recs)
    bl = df[df['is_blank']]
    if bl.empty: continue
    blank_rows.append((lvl_name, len(bl), bl['win'].mean()*100, bl['pnl_65'].sum(), bl['pnl_65'].mean()))

blank_rows.sort(key=lambda x: -x[3])
for lvl, cnt, wr, pnl, avg in blank_rows:
    flag = ' ✓' if pnl > 0 else ' ✗'
    print(f"  {lvl:<4} | {cnt:>4}t | WR {wr:>5.1f}% | Rs.{pnl:>9,.0f} | Avg Rs.{avg:>6,.0f}{flag}")

print(f"\n{'='*70}")
print("  TEST 2: WITHOUT CPR — bare 3-candle CRT (script 86 reference)")
print(f"{'='*70}")
print("  All days:   1106t | WR 78.0% | Rs.  88,384 | Avg Rs.   80")
print("  Blank days:  593t | WR 80.4% | Rs.  63,580 | Avg Rs.  107  ← baseline")
print(f"\n  Selling baseline: Rs.{sell_conv:,.0f}")
print(f"  + bare CRT blank:  593t Rs.63,580  → Total Rs.{sell_conv+63580:,.0f}")

top_blank_pnl = sum(pnl for _,_,_,pnl,_ in blank_rows if pnl > 0)
print(f"  + best CPR levels (blank, positive only): Rs.{top_blank_pnl:,.0f} → Total Rs.{sell_conv+top_blank_pnl:,.0f}")

# ── Save all records ──────────────────────────────────────────────────────────
all_recs = []
for lvl_name, _ in RESIST_LEVELS + SUPPORT_LEVELS:
    all_recs.extend(records[lvl_name])
if all_recs:
    pd.DataFrame(all_recs).to_csv(f'{OUT_DIR}/90_crt_all_levels.csv', index=False)
    print(f"\n  Saved → {OUT_DIR}/90_crt_all_levels.csv")

print("\nDone.")
