"""
98_fifto_cpr.py — FiFTO Zones + CPR Bias Filter
=================================================
Add CPR daily bias to FiFTO zone rejection:

  CPR bias = open vs TC/BC:
    Open > TC  → Bullish bias  → only trade LOWER zone rejection (sell PE)
    Open < BC  → Bearish bias  → only trade UPPER zone rejection (sell CE)
    Open inside CPR → Neutral  → skip (or allow both with lower confidence)

Logic:
  Signal only fires when FiFTO zone direction AGREES with CPR bias.
  e.g. bearish CPR day + price bounces to upper zone → reject → sell CE ✓
       bearish CPR day + price drops to lower zone   → skip (fights bias)

Same entry/exit as script 97: OTM1, 20% target, trailing SL, EOD 15:20
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

def build_ohlc_15m(tks, start='09:15:00', end='12:00:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('15min').ohlc().dropna()
    ohlc.columns = ['o', 'h', 'l', 'c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

# ── Build daily OHLC + zones + CPR ───────────────────────────────────────────
print("Building daily OHLC + FiFTO zones + CPR...")
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

# FiFTO daily zones (anchored to today's open, using past ranges)
df_d['rng5']  = df_d['range'].shift(1).rolling(5).mean().round(2)
df_d['rng10'] = df_d['range'].shift(1).rolling(10).mean().round(2)
df_d['d_u1']  = (df_d['o'] + 0.5 * df_d['rng5']).round(2)
df_d['d_l1']  = (df_d['o'] - 0.5 * df_d['rng5']).round(2)

# CPR from previous day OHLC
ph = df_d['h'].shift(1); pl = df_d['l'].shift(1); pc = df_d['c'].shift(1)
df_d['pvt'] = ((ph + pl + pc) / 3).round(2)
df_d['bc']  = ((ph + pl) / 2).round(2)
df_d['tc']  = (df_d['pvt'] + (df_d['pvt'] - df_d['bc'])).round(2)
df_d['r1']  = (2 * df_d['pvt'] - pl).round(2)
df_d['s1']  = (2 * df_d['pvt'] - ph).round(2)

df_d = df_d.dropna().reset_index(drop=True)

# CPR bias
df_d['cpr_bias'] = 'neutral'
df_d.loc[df_d['o'] > df_d['tc'], 'cpr_bias'] = 'bull'
df_d.loc[df_d['o'] < df_d['bc'], 'cpr_bias'] = 'bear'

df_5yr = df_d[df_d['date'].isin(dates_5yr)].reset_index(drop=True)
print(f"  {len(df_5yr)} days | {time.time()-t0:.0f}s")
print(f"  CPR bias: bull={len(df_5yr[df_5yr['cpr_bias']=='bull'])} | "
      f"bear={len(df_5yr[df_5yr['cpr_bias']=='bear'])} | "
      f"neutral={len(df_5yr[df_5yr['cpr_bias']=='neutral'])}")

base_df    = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base_dates = set(base_df['date'].astype(str).str.replace('-', ''))

# ── Scan ─────────────────────────────────────────────────────────────────────
print("\nScanning FiFTO + CPR bias (all days)...")
t0 = time.time()
records = []

for idx, row in df_5yr.iterrows():
    dstr     = row['date']
    is_blank = dstr not in base_dates
    year     = dstr[:4]
    bias     = row['cpr_bias']

    # Neutral days → skip (no clear directional bias)
    if bias == 'neutral':
        continue

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue

    c15 = build_ohlc_15m(spot)
    if len(c15) < 2: continue

    expiries = list_expiry_dates(dstr, index_name='NIFTY')
    if not expiries: continue
    expiry = expiries[0]

    d_u1 = row['d_u1']; d_l1 = row['d_l1']
    tc   = row['tc'];   bc   = row['bc']
    r1   = row['r1'];   s1   = row['s1']

    fired = False

    for ci in range(len(c15) - 1):
        if fired: break
        c  = c15.iloc[ci]
        ch = c['h']; cl = c['l']; cc = c['c']; ct = c['time']
        if ct > '12:00:00': break

        cmin  = int(ct[:2]) * 60 + int(ct[3:5]) + 15 + 1
        etime = f"{cmin//60:02d}:{cmin%60:02d}:02"
        if etime >= EOD_EXIT: break

        signal = None
        zone_hit = None

        if bias == 'bear':
            # Bearish day: sell CE when price bounces to upper zone
            if ch >= d_u1 and cc < d_u1:
                signal   = 'CE'
                zone_hit = 'daily_upper'
            elif ch >= r1 and cc < r1:       # also check R1 as extra resistance
                signal   = 'CE'
                zone_hit = 'r1_level'

        elif bias == 'bull':
            # Bullish day: sell PE when price pulls back to lower zone
            if cl <= d_l1 and cc > d_l1:
                signal   = 'PE'
                zone_hit = 'daily_lower'
            elif cl <= s1 and cc > s1:       # also check S1 as extra support
                signal   = 'PE'
                zone_hit = 's1_level'

        if not signal: continue

        spot_ref = cc
        strike   = get_otm1(spot_ref, signal)
        instr    = f'NIFTY{expiry}{strike}{signal}'

        res = simulate_sell(dstr, instr, etime)
        if res:
            pnl, reason, ep, xp, xt = res
            records.append(dict(
                date=dstr, year=year, is_blank=is_blank,
                cpr_bias=bias, signal=signal, zone_hit=zone_hit,
                ep=ep, xp=xp, exit_reason=reason,
                pnl=r2(pnl), win=pnl > 0, entry_time=etime
            ))
            fired = True

    if idx % 100 == 0:
        print(f"  {idx}/{len(df_5yr)} | {len(records)} trades | {time.time()-t0:.0f}s")

print(f"  Done | {len(records)} trades | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

def stats(g, label=''):
    if g.empty: return
    wr  = g['win'].mean() * 100
    pnl = g['pnl'].sum()
    avg = g['pnl'].mean()
    ex  = dict(g['exit_reason'].value_counts())
    print(f"  {label}: {len(g):>4}t | WR {wr:>5.1f}% | Rs.{pnl:>9,.0f} | Avg Rs.{avg:>6,.0f} | {ex}")

sep = '─' * 68
print(f"\n{'='*68}")
print(f"  FIFTO ZONES + CPR BIAS — NIFTY (neutral days skipped)")
print(f"{'='*68}")
stats(df, 'All days ')
stats(df[df['is_blank']], 'Blank days')

print(f"\n{sep}")
print("  BY CPR BIAS")
print(sep)
for bias, g in df.groupby('cpr_bias'):
    bl = g[g['is_blank']]
    stats(g,  f"{bias} all  ")
    stats(bl, f"{bias} blank")

print(f"\n{sep}")
print("  BY ZONE HIT")
print(sep)
for zh, g in df.groupby('zone_hit'):
    bl = g[g['is_blank']]
    stats(g,  f"{zh:<15} all  ")
    stats(bl, f"{zh:<15} blank")

print(f"\n{sep}")
print("  YEAR-WISE")
print(sep)
for yr, g in df.groupby('year'):
    bl = g[g['is_blank']]
    stats(g,  f"{yr} all  ")
    stats(bl, f"{yr} blank")

# ── Compare all 3: Base / CRT / FiFTO+CPR ─────────────────────────────────────
crt_blank   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank   = crt_blank[crt_blank['is_blank'] == True]
fifto_blank = df[df['is_blank']]

crt_dates   = set(crt_blank['date'].astype(str))
fifto_dates = set(fifto_blank['date'].astype(str))
overlap     = crt_dates & fifto_dates
only_fifto  = fifto_dates - crt_dates
only_crt    = crt_dates   - fifto_dates

print(f"\n{'='*68}")
print("  CRT vs FiFTO+CPR — BLANK DAYS")
print(f"{'='*68}")
print(f"  {'Strategy':<25} | Trades | WR     | P&L         | Avg")
print(f"  {'-'*65}")
for label, g, col in [
    ("CRT Approach D",    crt_blank,   'pnl_65'),
    ("FiFTO + CPR bias",  fifto_blank, 'pnl'),
]:
    if g.empty: print(f"  {label:<25} | no trades"); continue
    print(f"  {label:<25} | {len(g):>6} | {g['win'].mean()*100:>5.1f}% | "
          f"Rs.{g[col].sum():>10,.0f} | Rs.{g[col].mean():>6,.0f}")

print(f"\n  Blank day coverage:")
print(f"    CRT only        : {len(only_crt):>4} days")
print(f"    FiFTO+CPR only  : {len(only_fifto):>4} days")
print(f"    Both (same day) : {len(overlap):>4} days")

# FiFTO-only days P&L (unique addition to CRT)
only_fifto_df = fifto_blank[fifto_blank['date'].astype(str).isin(only_fifto)]
print(f"\n  FiFTO+CPR unique blank days (not covered by CRT):")
stats(only_fifto_df, '  FiFTO-only blank')

df.to_csv(f'{OUT_DIR}/98_fifto_cpr.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/98_fifto_cpr.csv")
print("\nDone.")
