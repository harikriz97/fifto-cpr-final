"""
89_crt_buying.py — CRT at Key Levels: Buying PE (directional)
==============================================================
Same signal as script 88 (3-candle bearish CRT at R1/TC, C3-open entry)
But instead of selling CE → BUY PE directionally

Selling CE wins when market doesn't go up much (time decay + no move)
Buying PE wins when market falls significantly (directional)

After bearish CRT at R1/TC: market should fall (distribution)
→ Buy PE ATM or OTM1

Test configs:
  strike:  ATM, OTM1 (50 pts OTM)
  target:  50%, 80%, 100% (option doubles)
  SL:      30%, 50%
  entry:   C3-open (same as Model A in 88)

Compare buying vs selling on same signals.
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

def simulate_buy(date_str, expiry, strike, opt, entry_time,
                 tgt_pct=0.80, sl_pct=0.50):
    """Buy option: profit if option price rises tgt_pct, loss if falls sl_pct"""
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 + tgt_pct))
    hsl = r2(ep * (1 - sl_pct))
    ps = tks['price'].values
    ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((p - ep) * LOT_SIZE), 'eod', r2(ep), r2(p), t
        if p >= tgt:
            return r2((p - ep) * LOT_SIZE), 'target', r2(ep), r2(p), t
        if p <= hsl:
            return r2((p - ep) * LOT_SIZE), 'hard_sl', r2(ep), r2(p), t
    return r2((ps[-1] - ep) * LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

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
    rows.append({'date':d,'o':day.iloc[0]['price'],
                 'h':day['price'].max(),'l':day['price'].min(),'c':day.iloc[-1]['price']})

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

# ── Buy configs ───────────────────────────────────────────────────────────────
BUY_CONFIGS = [
    ('atm_50sl',  'ATM',  0.50, 0.50),
    ('atm_80sl',  'ATM',  0.80, 0.50),
    ('atm_100sl', 'ATM',  1.00, 0.50),
    ('otm_50sl',  'OTM1', 0.50, 0.50),
    ('otm_80sl',  'OTM1', 0.80, 0.50),
    ('otm_100sl', 'OTM1', 1.00, 0.50),
]

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nScanning 3-candle CRT at R1 + TC (bearish), C3-open entry...")
t0 = time.time()

# Collect all signals first (same signals as script 88 Model A)
signals = []

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

    for ci in range(1, len(candles)-1):
        c1 = candles.iloc[ci-1]
        c2 = candles.iloc[ci]
        c3_idx = ci + 1
        if c3_idx >= len(candles): break

        c3 = candles.iloc[c3_idx]
        c3_time = c3['time']
        if c3_time > '12:00:00': break

        c2h = c2['h']
        c3c = c3['c']

        is_crt_r1 = (c2h > r1 and c3c < r1)
        is_crt_tc = (c2h > tc and c3c < tc)
        if not (is_crt_r1 or is_crt_tc): continue

        level_name = 'R1' if is_crt_r1 else 'TC'
        level_val  = r1  if is_crt_r1 else tc
        entry_t    = f"{c3_time[:2]}:{c3_time[3:5]}:02"
        if entry_t >= EOD_EXIT: break

        spot_at_entry = c3c   # C3 close used as proxy for spot at C3 open
        signals.append(dict(
            date=dstr, expiry=expiry, level=level_name, level_val=r2(level_val),
            c3_time=c3_time, entry_t=entry_t, spot_c3c=r2(spot_at_entry), dte=dte,
            is_blank=is_blank, ema_bias=ema_bias, year=dstr[:4]))
        break   # first signal per day

print(f"  Collected {len(signals)} signals | {time.time()-t0:.0f}s")

# ── Simulate all configs ───────────────────────────────────────────────────────
print("\nSimulating buy configs + sell reference...")
t1 = time.time()

results = {cfg[0]: [] for cfg in BUY_CONFIGS}
results['sell_ref'] = []   # sell CE OTM1 (script 88 Model A reference)

for i, sig in enumerate(signals):
    dstr    = sig['date']
    expiry  = sig['expiry']
    entry_t = sig['entry_t']
    spot    = sig['spot_c3c']
    is_blank= sig['is_blank']
    year    = sig['year']
    level   = sig['level']
    ema_bear= sig['ema_bias'] == 'bear'

    atm  = get_atm(spot)

    # ── Sell CE OTM1 reference ────────────────────────────────────────────────
    strike_ce = atm + STRIKE_INT
    res = simulate_sell(dstr, expiry, strike_ce, 'CE', entry_t)
    if res:
        pnl75, reason, ep, xp, xt = res
        results['sell_ref'].append(dict(
            date=dstr, level=level, year=year, is_blank=is_blank,
            ema_bear=ema_bear, ep=ep, xp=xp, exit_reason=reason,
            pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE>0))

    # ── Buy PE configs ────────────────────────────────────────────────────────
    for cfg_name, strike_type, tgt_pct, sl_pct in BUY_CONFIGS:
        strike_pe = atm if strike_type == 'ATM' else atm - STRIKE_INT
        res = simulate_buy(dstr, expiry, strike_pe, 'PE', entry_t, tgt_pct, sl_pct)
        if res:
            pnl75, reason, ep, xp, xt = res
            results[cfg_name].append(dict(
                date=dstr, level=level, year=year, is_blank=is_blank,
                ema_bear=ema_bear, ep=ep, xp=xp, exit_reason=reason,
                pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE>0))

    if i % 100 == 0:
        print(f"  {i}/{len(signals)} | {time.time()-t1:.0f}s")

print(f"Done | {time.time()-t1:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
def show_cfg(label, recs, show_years=True):
    if not recs:
        print(f"\n{label}: no trades"); return
    df = pd.DataFrame(recs)
    wr  = df['win'].mean()*100
    pnl = df['pnl_65'].sum()
    bl  = df[df['is_blank']]
    print(f"  {label:<22} | {len(df):>4}t | WR {wr:>5.1f}% | Total Rs.{pnl:>10,.0f} | "
          f"Avg Rs.{df['pnl_65'].mean():>6,.0f} | "
          f"blank: {len(bl)}t WR {bl['win'].mean()*100 if not bl.empty else 0:.0f}% Rs.{bl['pnl_65'].sum():,.0f}")
    if show_years:
        for yr in sorted(df['year'].unique()):
            g = df[df['year']==yr]
            print(f"       {yr}: {len(g):>3}t WR {g['win'].mean()*100:.0f}% Rs.{g['pnl_65'].sum():,.0f}")

print(f"\n{'='*90}")
print("  CRT at R1/TC (bearish, C3-open entry) — BUY PE vs SELL CE")
print(f"{'='*90}")
print(f"  {'Config':<22} | {'Trades':>5} | {'WR':>7} | {'Total P&L':>16} | {'Avg':>9} | {'Blank days'}")
print(f"  {'-'*88}")

show_cfg("SELL CE OTM1 (ref)",    results['sell_ref'])
print(f"  {'-'*88}")
show_cfg("BUY PE ATM  TGT50%",    results['atm_50sl'])
show_cfg("BUY PE ATM  TGT80%",    results['atm_80sl'])
show_cfg("BUY PE ATM  TGT100%",   results['atm_100sl'])
print(f"  {'-'*88}")
show_cfg("BUY PE OTM1 TGT50%",    results['otm_50sl'])
show_cfg("BUY PE OTM1 TGT80%",    results['otm_80sl'])
show_cfg("BUY PE OTM1 TGT100%",   results['otm_100sl'])

# ── Blank-only summary ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  BLANK DAYS ONLY — Buy PE vs Sell CE")
print(f"{'='*70}")
for cfg_name, label in [
    ('sell_ref',  'SELL CE OTM1 (ref)'),
    ('atm_50sl',  'BUY PE ATM  TGT50%'),
    ('atm_80sl',  'BUY PE ATM  TGT80%'),
    ('atm_100sl', 'BUY PE ATM TGT100%'),
    ('otm_50sl',  'BUY PE OTM1 TGT50%'),
    ('otm_80sl',  'BUY PE OTM1 TGT80%'),
    ('otm_100sl', 'BUY PE OTM1TGT100%'),
]:
    recs = results[cfg_name]
    if not recs: continue
    df = pd.DataFrame(recs)
    bl = df[df['is_blank']]
    if bl.empty: continue
    print(f"  {label:<22} | {len(bl)}t | WR {bl['win'].mean()*100:.1f}% | Rs.{bl['pnl_65'].sum():,.0f} | Avg Rs.{bl['pnl_65'].mean():,.0f}")

sell_conv = df_sell_base['pnl_conv'].sum()
print(f"\n  Selling baseline: Rs.{sell_conv:,.0f}")
print(f"  Script 88 sell ref (blank): Rs.{pd.DataFrame(results['sell_ref'])[pd.DataFrame(results['sell_ref'])['is_blank']]['pnl_65'].sum():,.0f} → Total Rs.{sell_conv + pd.DataFrame(results['sell_ref'])[pd.DataFrame(results['sell_ref'])['is_blank']]['pnl_65'].sum():,.0f}")

print("\nDone.")
