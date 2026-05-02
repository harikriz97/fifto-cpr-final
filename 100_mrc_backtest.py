"""
100_mrc_backtest.py — MRC (Mean Reversion Concept) Backtest on Nifty
=====================================================================
Strategy (from bnf strategy.txt):
  Levels from PDH/PDL (PDH = 0%, PDL = 100%):
    l_0    = PDH
    l_25   = PDH - range × 0.25
    l_382  = PDH - range × 0.382   ← BUY zone (close ABOVE → bullish)
    l_50   = PDH - range × 0.50    ← Median / SL for both sides
    l_618  = PDH - range × 0.618   ← SELL zone (close BELOW → bearish)
    l_75   = PDH - range × 0.75
    l_100  = PDL

Signal: 5-min Heiken Ashi candle
  SELL: HA close < l_618 AND HA close < HA open (red candle) → sell CE OTM1
  BUY:  HA close > l_382 AND HA close > HA open (green candle) → sell PE OTM1
  Entry: next 5M candle open + 2s (no forward bias)
  SL: 50% option premium hard SL + trailing
  Target: 20% option premium
  Search window: 09:15–12:00 | EOD exit: 15:20
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates

EOD_EXIT   = '15:20:00'
YEARS      = 5
OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50

def r2(v): return round(float(v), 2)
def get_otm1(spot, opt):
    atm = int(round(spot / STRIKE_INT) * STRIKE_INT)
    return atm + STRIKE_INT if opt == 'CE' else atm - STRIKE_INT

def simulate_sell(date_str, instrument, entry_time, tgt_pct=0.20, sl_pct=1.00):
    tks = load_tick_data(date_str, instrument, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - tgt_pct)); hsl = r2(ep * (1 + sl_pct)); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((ep - p) * LOT_SIZE * SCALE), 'eod', r2(ep), r2(p), t
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep - p) * LOT_SIZE * SCALE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep - p) * LOT_SIZE * SCALE), 'lockin_sl' if sl < hsl else 'hard_sl', r2(ep), r2(p), t
    return r2((ep - ps[-1]) * LOT_SIZE * SCALE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

def compute_ha(ohlc):
    """Compute Heiken Ashi on a 5M OHLC DataFrame with columns o/h/l/c."""
    ha = ohlc.copy()
    ha['ha_c'] = ((ohlc['o'] + ohlc['h'] + ohlc['l'] + ohlc['c']) / 4).round(2)
    ha_o = [0.0] * len(ha)
    ha_o[0] = r2((ohlc['o'].iloc[0] + ohlc['c'].iloc[0]) / 2)
    for i in range(1, len(ha)):
        ha_o[i] = r2((ha_o[i-1] + ha['ha_c'].iloc[i-1]) / 2)
    ha['ha_o'] = ha_o
    ha['ha_h'] = ha[['h', 'ha_o', 'ha_c']].max(axis=1).round(2)
    ha['ha_l'] = ha[['l', 'ha_o', 'ha_c']].min(axis=1).round(2)
    return ha

def build_ohlc_5m(tks, start='09:15:00', end='12:00:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('5min').ohlc().dropna()
    ohlc.columns = ['o', 'h', 'l', 'c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

# ── Build daily OHLC for PDH/PDL ─────────────────────────────────────────────
print("Building daily OHLC (PDH/PDL)...")
t0 = time.time()
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]

extra = max(0, all_dates.index(dates_5yr[0]) - 3)
rows  = []
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    rows.append({'date': d, 'pdh': day['price'].max(), 'pdl': day['price'].min()})

df_d = pd.DataFrame(rows)
df_d['pdh_prev'] = df_d['pdh'].shift(1)
df_d['pdl_prev'] = df_d['pdl'].shift(1)
df_d = df_d.dropna().reset_index(drop=True)
df_5yr = df_d[df_d['date'].isin(dates_5yr)].reset_index(drop=True)
print(f"  {len(df_5yr)} days | {time.time()-t0:.0f}s")

base_df    = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base_dates = set(base_df['date'].astype(str).str.replace('-', ''))

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nScanning MRC signals (all days)...")
t0 = time.time()
records = []

for idx, row in df_5yr.iterrows():
    dstr     = row['date']
    is_blank = dstr not in base_dates
    year     = dstr[:4]

    pdh = row['pdh_prev']; pdl = row['pdl_prev']
    rng = pdh - pdl
    if rng < 50: continue   # skip tiny range days

    # MRC levels (PDH=0%, PDL=100%)
    l_382 = r2(pdh - rng * 0.382)   # BUY above this
    l_50  = r2(pdh - rng * 0.500)   # median / SL
    l_618 = r2(pdh - rng * 0.618)   # SELL below this

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue

    c5 = build_ohlc_5m(spot)
    if len(c5) < 3: continue

    ha = compute_ha(c5)

    expiries = list_expiry_dates(dstr, index_name='NIFTY')
    if not expiries: continue
    expiry = expiries[0]

    fired = False

    for ci in range(len(ha) - 1):
        if fired: break
        h = ha.iloc[ci]
        ct = h['time']
        if ct > '12:00:00': break

        # Next 5M candle entry time
        cmin  = int(ct[:2]) * 60 + int(ct[3:5]) + 5 + 1
        etime = f"{cmin//60:02d}:{cmin%60:02d}:02"
        if etime >= EOD_EXIT: break

        signal   = None
        level_hit = None

        # SELL: HA red candle closes BELOW 61.8% level
        if h['ha_c'] < l_618 and h['ha_c'] < h['ha_o']:
            signal    = 'CE'
            level_hit = 'l618_sell'

        # BUY: HA green candle closes ABOVE 38.2% level
        elif h['ha_c'] > l_382 and h['ha_c'] > h['ha_o']:
            signal    = 'PE'
            level_hit = 'l382_buy'

        if not signal: continue

        spot_ref = h['ha_c']
        strike   = get_otm1(spot_ref, signal)
        instr    = f'NIFTY{expiry}{strike}{signal}'

        res = simulate_sell(dstr, instr, etime)
        if res:
            pnl, reason, ep, xp, xt = res
            records.append(dict(
                date=dstr, year=year, is_blank=is_blank,
                signal=signal, level_hit=level_hit,
                pdh=pdh, pdl=pdl, l_382=l_382, l_50=l_50, l_618=l_618,
                ep=ep, xp=xp, exit_reason=reason,
                pnl=r2(pnl), win=pnl > 0, entry_time=etime, candle_time=ct
            ))
            fired = True

    if idx % 100 == 0:
        print(f"  {idx}/{len(df_5yr)} | {len(records)} trades | {time.time()-t0:.0f}s")

print(f"  Done | {len(records)} trades | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

def stats(g, label=''):
    if g.empty: print(f"  {label}: 0 trades"); return
    wr  = g['win'].mean() * 100
    pnl = g['pnl'].sum()
    avg = g['pnl'].mean()
    ex  = dict(g['exit_reason'].value_counts())
    print(f"  {label}: {len(g):>4}t | WR {wr:>5.1f}% | Rs.{pnl:>9,.0f} | Avg Rs.{avg:>6,.0f} | {ex}")

sep = '─' * 68
print(f"\n{'='*68}")
print(f"  MRC STRATEGY — NIFTY 5M Heiken Ashi (all {len(df_5yr)} days)")
print(f"{'='*68}")
stats(df,                  'All days ')
stats(df[df['is_blank']], 'Blank days')

print(f"\n{sep}")
print("  BY SIGNAL")
print(sep)
for sig, g in df.groupby('signal'):
    bl = g[g['is_blank']]
    stats(g,  f"  {sig} all  ")
    stats(bl, f"  {sig} blank")

print(f"\n{sep}")
print("  BY LEVEL HIT")
print(sep)
for lv, g in df.groupby('level_hit'):
    bl = g[g['is_blank']]
    stats(g,  f"  {lv} all  ")
    stats(bl, f"  {lv} blank")

print(f"\n{sep}")
print("  YEAR-WISE")
print(sep)
for yr, g in df.groupby('year'):
    bl = g[g['is_blank']]
    stats(g,  f"  {yr} all  ")
    stats(bl, f"  {yr} blank")

# ── vs CRT ────────────────────────────────────────────────────────────────────
crt = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank = crt[crt['is_blank'] == True]
mrc_blank = df[df['is_blank']]
crt_dates = set(crt_blank['date'].astype(str))
mrc_dates = set(mrc_blank['date'].astype(str))

print(f"\n{'='*68}")
print("  MRC vs CRT Approach D — BLANK DAYS")
print(f"{'='*68}")
print(f"  {'Strategy':<25} | Trades | WR     | P&L         | Avg")
print(f"  {'-'*63}")
for label, g, col in [
    ("CRT Approach D", crt_blank, 'pnl_65'),
    ("MRC (all blank)", mrc_blank, 'pnl'),
    ("MRC CE only",    mrc_blank[mrc_blank['signal']=='CE'], 'pnl'),
    ("MRC PE only",    mrc_blank[mrc_blank['signal']=='PE'], 'pnl'),
]:
    if g.empty: print(f"  {label:<25} | no trades"); continue
    print(f"  {label:<25} | {len(g):>6} | {g['win'].mean()*100:>5.1f}% | "
          f"Rs.{g[col].sum():>10,.0f} | Rs.{g[col].mean():>6,.0f}")

print(f"\n  Blank day coverage:")
print(f"    CRT only : {len(crt_dates - mrc_dates):>4} days")
print(f"    MRC only : {len(mrc_dates - crt_dates):>4} days")
print(f"    Both     : {len(crt_dates & mrc_dates):>4} days")

df.to_csv(f'{OUT_DIR}/100_mrc_trades.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/100_mrc_trades.csv")
print("\nDone.")
