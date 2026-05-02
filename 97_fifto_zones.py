"""
97_fifto_zones.py — FiFTO Dynamic Zone Rejection Backtest (All Days)
=====================================================================
Logic (from Pine Script):
  base   = today's open (daily / weekly)
  rng5   = SMA(5) of previous daily/weekly candle ranges (high-low)
  rng10  = SMA(10) of previous daily/weekly candle ranges

  daily_u1 = open + 0.5 * daily_rng5   ← tighter upper zone
  daily_l1 = open - 0.5 * daily_rng5   ← tighter lower zone
  daily_u2 = open + 0.5 * daily_rng10  ← wider upper zone
  daily_l2 = open - 0.5 * daily_rng10  ← wider lower zone
  (same for weekly)

Signal: 15M candle TOUCHES zone boundary and CLOSES back inside
  → bearish rejection at upper zone → sell CE OTM1
  → bullish rejection at lower zone → sell PE OTM1

Entry: next 15M candle open + 2s (no forward bias)
Target: 20% option premium | SL: 100% hard + trailing
Search window: 09:15–12:00 | EOD exit: 15:20

Breakdown: all days / blank days / year-wise / zone type / daily vs weekly
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
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)

def get_otm1(spot, opt):
    atm = int(round(spot / STRIKE_INT) * STRIKE_INT)
    return atm + STRIKE_INT if opt == 'CE' else atm - STRIKE_INT

def simulate_sell(date_str, instrument, entry_time, opt_type, tgt_pct=0.20, sl_pct=1.00):
    tks = load_tick_data(date_str, instrument, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - tgt_pct))
    hsl = r2(ep * (1 + sl_pct))
    sl  = hsl
    md  = 0.0
    ps  = tks['price'].values
    ts  = tks['time'].values
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

def build_ohlc_15m(tks, start='09:15:00', end='12:00:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('15min').ohlc().dropna()
    ohlc.columns = ['o', 'h', 'l', 'c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

# ── Build daily OHLC ──────────────────────────────────────────────────────────
print("Building daily OHLC...")
t0 = time.time()
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]

extra = max(0, all_dates.index(dates_5yr[0]) - 15)
rows = []
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    rows.append({'date': d,
                 'o': day.iloc[0]['price'],
                 'h': day['price'].max(),
                 'l': day['price'].min(),
                 'c': day.iloc[-1]['price']})

df_d = pd.DataFrame(rows)
df_d['range'] = df_d['h'] - df_d['l']

# Daily zones: SMA5/SMA10 of PREVIOUS bars' range (shift 1 so today's open uses past data)
df_d['rng5']  = df_d['range'].shift(1).rolling(5).mean().round(2)
df_d['rng10'] = df_d['range'].shift(1).rolling(10).mean().round(2)
# Zones anchored to today's open
df_d['d_u1']  = (df_d['o'] + 0.5 * df_d['rng5']).round(2)
df_d['d_u2']  = (df_d['o'] + 0.5 * df_d['rng10']).round(2)
df_d['d_l1']  = (df_d['o'] - 0.5 * df_d['rng5']).round(2)
df_d['d_l2']  = (df_d['o'] - 0.5 * df_d['rng10']).round(2)
df_d = df_d.dropna().reset_index(drop=True)

# ── Build weekly zones ────────────────────────────────────────────────────────
df_d['dt']       = pd.to_datetime(df_d['date'], format='%Y%m%d')
df_d['week_key'] = df_d['dt'].dt.isocalendar().year.astype(str) + '_' + \
                   df_d['dt'].dt.isocalendar().week.astype(str).str.zfill(2)
weekly = df_d.groupby('week_key').agg(
    w_open=('o', 'first'),
    w_high=('h', 'max'),
    w_low=('l', 'min'),
    w_date=('date', 'first')
).reset_index()
weekly['w_range'] = weekly['w_high'] - weekly['w_low']
weekly['w_rng5']  = weekly['w_range'].shift(1).rolling(5).mean().round(2)
weekly['w_rng10'] = weekly['w_range'].shift(1).rolling(10).mean().round(2)
weekly = weekly.dropna()

# Merge weekly zone params back to daily
df_d = df_d.merge(weekly[['week_key','w_open','w_rng5','w_rng10']], on='week_key', how='left')
df_d['w_u1'] = (df_d['w_open'] + 0.5 * df_d['w_rng5']).round(2)
df_d['w_u2'] = (df_d['w_open'] + 0.5 * df_d['w_rng10']).round(2)
df_d['w_l1'] = (df_d['w_open'] - 0.5 * df_d['w_rng5']).round(2)
df_d['w_l2'] = (df_d['w_open'] - 0.5 * df_d['w_rng10']).round(2)

# Keep only 5yr
df_5yr = df_d[df_d['date'].isin(dates_5yr)].reset_index(drop=True)
print(f"  {len(df_5yr)} days | {time.time()-t0:.0f}s")

# ── Blank day set ─────────────────────────────────────────────────────────────
base_df    = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base_dates = set(base_df['date'].astype(str).str.replace('-', ''))

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nScanning FiFTO zone rejections (all days)...")
t0 = time.time()
records = []

for idx, row in df_5yr.iterrows():
    dstr     = row['date']
    is_blank = dstr not in base_dates
    year     = dstr[:4]

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue

    c15 = build_ohlc_15m(spot)
    if len(c15) < 2: continue

    expiries = list_expiry_dates(dstr, index_name='NIFTY')
    if not expiries: continue
    expiry = expiries[0]

    # Zone levels for today
    d_u1 = row['d_u1']; d_l1 = row['d_l1']
    d_u2 = row['d_u2']; d_l2 = row['d_l2']
    w_u1 = row['w_u1']; w_l1 = row['w_l1']

    fired = False

    for ci in range(len(c15) - 1):
        if fired: break
        c = c15.iloc[ci]
        ch = c['h']; cl_price = c['l']; cc = c['c']; ct = c['time']
        if ct > '12:00:00': break

        # Next candle entry time
        cmin  = int(ct[:2]) * 60 + int(ct[3:5]) + 15 + 1
        etime = f"{cmin//60:02d}:{cmin%60:02d}:02"
        if etime >= EOD_EXIT: break

        signal    = None
        zone_type = None
        weekly_conf = False

        # Bearish rejection at upper daily zone (candle touches u1, closes BELOW u1)
        if ch >= d_u1 and cc < d_u1:
            signal    = 'CE'
            zone_type = 'daily_upper'
            # Weekly confluence: price also at/above weekly u1
            if not pd.isna(w_u1) and ch >= w_u1 * 0.998:
                weekly_conf = True

        # Bullish rejection at lower daily zone (candle touches l1, closes ABOVE l1)
        elif cl_price <= d_l1 and cc > d_l1:
            signal    = 'PE'
            zone_type = 'daily_lower'
            if not pd.isna(w_l1) and cl_price <= w_l1 * 1.002:
                weekly_conf = True

        # Weekly zone rejection (if no daily signal yet)
        elif not pd.isna(w_u1) and ch >= w_u1 and cc < w_u1:
            signal    = 'CE'
            zone_type = 'weekly_upper'
        elif not pd.isna(w_l1) and cl_price <= w_l1 and cc > w_l1:
            signal    = 'PE'
            zone_type = 'weekly_lower'

        if not signal: continue

        spot_ref = cc
        strike   = get_otm1(spot_ref, signal)
        instr    = f'NIFTY{expiry}{strike}{signal}'

        res = simulate_sell(dstr, instr, etime, signal)
        if res:
            pnl, reason, ep, xp, xt = res
            records.append(dict(
                date=dstr, year=year, is_blank=is_blank,
                signal=signal, zone_type=zone_type,
                weekly_conf=weekly_conf,
                ep=ep, xp=xp, exit_reason=reason,
                pnl=r2(pnl), win=pnl > 0,
                entry_time=etime, candle_time=ct
            ))
            fired = True

    if idx % 100 == 0:
        print(f"  {idx}/{len(df_5yr)} | {len(records)} trades | {time.time()-t0:.0f}s")

print(f"  Done | {len(records)} trades | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

def stats(g, label=''):
    if g.empty: return
    wr   = g['win'].mean() * 100
    pnl  = g['pnl'].sum()
    avg  = g['pnl'].mean()
    exits = dict(g['exit_reason'].value_counts())
    print(f"  {label}: {len(g):>4}t | WR {wr:>5.1f}% | Rs.{pnl:>9,.0f} | Avg Rs.{avg:>6,.0f} | {exits}")

sep = '─' * 65
print(f"\n{'='*65}")
print(f"  FIFTO ZONE REJECTION — NIFTY (all {len(df_5yr)} days)")
print(f"{'='*65}")
stats(df, 'All days')
stats(df[df['is_blank']], 'Blank days')

print(f"\n{sep}")
print("  BY ZONE TYPE")
print(sep)
for zt, g in df.groupby('zone_type'):
    bl = g[g['is_blank']]
    stats(g,  f"{zt:<20} all  ")
    stats(bl, f"{zt:<20} blank")

print(f"\n{sep}")
print("  WEEKLY CONFLUENCE (daily + weekly zone aligned)")
print(sep)
stats(df[df['weekly_conf']], 'Weekly conf all  ')
stats(df[df['weekly_conf'] & df['is_blank']], 'Weekly conf blank')

print(f"\n{sep}")
print("  BY SIGNAL (CE = bearish, PE = bullish)")
print(sep)
for sig, g in df.groupby('signal'):
    bl = g[g['is_blank']]
    stats(g,  f"{sig} all  ")
    stats(bl, f"{sig} blank")

print(f"\n{sep}")
print("  YEAR-WISE")
print(sep)
for yr, g in df.groupby('year'):
    bl = g[g['is_blank']]
    stats(g,  f"{yr} all  ")
    stats(bl, f"{yr} blank")

# ── CRT vs FiFTO comparison (blank days) ─────────────────────────────────────
crt_df   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank = crt_df[crt_df['is_blank'] == True]

fifto_blank = df[df['is_blank']]

print(f"\n{'='*65}")
print("  CRT Approach D vs FiFTO Zones — BLANK DAYS")
print(f"{'='*65}")
print(f"  {'Strategy':<25} | Trades | WR     | P&L        | Avg")
print(f"  {'-'*63}")
for label, g, pnl_col in [
    ("CRT Approach D",    crt_blank,   'pnl_65'),
    ("FiFTO Zones",       fifto_blank, 'pnl'),
]:
    wr  = g['win'].mean() * 100 if not g.empty else 0
    pnl = g[pnl_col].sum()    if not g.empty else 0
    avg = g[pnl_col].mean()   if not g.empty else 0
    print(f"  {label:<25} | {len(g):>6} | {wr:>5.1f}% | Rs.{pnl:>9,.0f} | Rs.{avg:>6,.0f}")

# Overlap: days covered by BOTH
crt_dates   = set(crt_blank['date'].astype(str))
fifto_dates = set(fifto_blank['date'].astype(str))
overlap     = crt_dates & fifto_dates
only_fifto  = fifto_dates - crt_dates
only_crt    = crt_dates   - fifto_dates

print(f"\n  Coverage on blank days:")
print(f"    CRT only  : {len(only_crt)} days")
print(f"    FiFTO only: {len(only_fifto)} days")
print(f"    Both      : {len(overlap)} days (same day, different signal)")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(f'{OUT_DIR}/97_fifto_zones.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/97_fifto_zones.csv")
print("\nDone.")
