"""
81_crt_buying.py — CRT as Option BUYING strategy
==================================================
CRT gives a directional reversal signal — the sharp move after sweep+close-back
is better captured by BUYING options, not selling.

Signal (same as script 80):
  Bearish CRT: 15-min candle sweeps above PDH, closes back below PDH
               → Buy PE ATM at candle_close + 1min
  Bullish CRT: 15-min candle sweeps below PDL, closes back above PDL
               → Buy CE ATM at candle_close + 1min

Sweep window: 09:15 candle only (first candle is the cleanest manipulation)
Entry:        candle open + 15min + 1min (after 15-min candle fully closes)

Exit rules (buying):
  Target: +80% of premium
  SL:     -30% of premium (hard stop, no trailing for buying)
  EOD:    12:30 (theta kill — no afternoon hold)

Compare:
  A. CRT buying — all days
  B. CRT buying — blank days only
  C. CRT buying — first candle (09:15) only
  D. CRT selling (from script 80) — for direct comparison
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
EOD_BUY    = '12:30:00'
CANDLE_MIN = 15
TGT_PCT    = 0.80
SL_PCT     = 0.30
YEARS      = 5
OUT_DIR    = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)

def build_15min(tks, start='09:15:00', end='10:45:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('15min').ohlc().dropna()
    ohlc.columns = ['o', 'h', 'l', 'c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

def simulate_buy(date_str, expiry, strike, opt, entry_time):
    """Option BUY simulation — price rise = profit for CE, fall = profit for PE."""
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 + TGT_PCT))
    sl  = r2(ep * (1 - SL_PCT))
    ps  = tks['price'].values
    ts  = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_BUY: return r2((p - ep) * LOT_SIZE), 'eod', r2(ep), r2(p), t
        if p >= tgt:     return r2((p - ep) * LOT_SIZE), 'target', r2(ep), r2(p), t
        if p <= sl:      return r2((p - ep) * LOT_SIZE), 'hard_sl', r2(ep), r2(p), t
    return r2((ps[-1] - ep) * LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

def simulate_sell(date_str, expiry, strike, opt, entry_time, tgt_pct=0.20, sl_pct=1.00):
    """Sell simulation for comparison."""
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - tgt_pct))
    hsl = r2(ep * (1 + sl_pct)); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    EOD = '15:20:00'
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD: return r2((ep - p) * LOT_SIZE), 'eod', r2(ep), r2(p), t
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep - p) * LOT_SIZE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep - p) * LOT_SIZE), 'lockin_sl' if sl < hsl else 'hard_sl', r2(ep), r2(p), t
    return r2((ep - ps[-1]) * LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

# ── Build daily OHLC ──────────────────────────────────────────────────────────
print("Building daily OHLC...")
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
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    rows.append({'date': d, 'o': day.iloc[0]['price'],
                 'h': day['price'].max(), 'l': day['price'].min(),
                 'c': day.iloc[-1]['price']})

ohlc = pd.DataFrame(rows)
ohlc['ema']    = ohlc['c'].ewm(span=20, adjust=False).mean().shift(1)
ohlc['pvt']    = ((ohlc['h'] + ohlc['l'] + ohlc['c']) / 3).round(2)
ohlc['bc']     = ((ohlc['h'] + ohlc['l']) / 2).round(2)
ohlc['tc']     = (ohlc['pvt'] + (ohlc['pvt'] - ohlc['bc'])).round(2)
ohlc['pdh']    = ohlc['h'].shift(1)
ohlc['pdl']    = ohlc['l'].shift(1)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)

df_sell_base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
sell_dates   = set(df_sell_base['date'].astype(str).str.replace('-', ''))
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

# ── Main loop ─────────────────────────────────────────────────────────────────
rec_buy  = []   # CRT buying
rec_sell = []   # CRT selling (for comparison)

print("\nRunning CRT scan (buying + selling comparison)...")
t0 = time.time()

for idx, row in ohlc_5yr.iterrows():
    if idx < 3: continue
    dstr = row['date']
    prev = ohlc_5yr.iloc[idx - 1]
    pdh  = prev['h']
    pdl  = prev['l']
    op   = row['o']
    ema  = row['ema']

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
           pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue
    # Only first 15-min candle (09:15 candle)
    candles = build_15min(spot, start='09:15:00', end='09:30:00')
    if candles.empty: continue

    crow = candles.iloc[0]
    candle_time = crow['time']   # should be '09:15:00'
    ch = crow['h']; cl = crow['l']; cc = crow['c']

    # Entry after first candle closes: 09:15 + 15min + 1min = 09:31:02
    h_str = candle_time[:2]; m_str = candle_time[3:5]
    entry_min = int(h_str)*60 + int(m_str) + CANDLE_MIN + 1
    entry_t   = f"{entry_min//60:02d}:{entry_min%60:02d}:02"

    is_blank     = dstr not in sell_dates
    ema_aligned_bear = (op < ema)   # bear EMA agrees with bearish CRT
    ema_aligned_bull = (op > ema)   # bull EMA agrees with bullish CRT

    # Bearish CRT: swept above PDH, closed back below PDH
    if ch > pdh and cc < pdh:
        strike_buy  = get_atm(cc)          # buy PE ATM
        strike_sell = get_atm(cc) + STRIKE_INT  # sell CE OTM1

        res_buy = simulate_buy(dstr, expiry, strike_buy, 'PE', entry_t)
        if res_buy:
            pnl75, reason, ep, xp, xt = res_buy
            rec_buy.append(dict(
                date=dstr, crt='bearish', side='buy', opt='PE',
                strike=strike_buy, dte=dte, entry_time=entry_t,
                ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                is_blank=is_blank, ema_aligned=ema_aligned_bear,
                year=dstr[:4]))

        res_sell = simulate_sell(dstr, expiry, strike_sell, 'CE', entry_t)
        if res_sell:
            pnl75, reason, ep, xp, xt = res_sell
            rec_sell.append(dict(
                date=dstr, crt='bearish', side='sell', opt='CE',
                strike=strike_sell, dte=dte, entry_time=entry_t,
                ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                is_blank=is_blank, ema_aligned=ema_aligned_bear,
                year=dstr[:4]))

    # Bullish CRT: swept below PDL, closed back above PDL
    elif cl < pdl and cc > pdl:
        strike_buy  = get_atm(cc)          # buy CE ATM
        strike_sell = get_atm(cc) - STRIKE_INT  # sell PE OTM1

        res_buy = simulate_buy(dstr, expiry, strike_buy, 'CE', entry_t)
        if res_buy:
            pnl75, reason, ep, xp, xt = res_buy
            rec_buy.append(dict(
                date=dstr, crt='bullish', side='buy', opt='CE',
                strike=strike_buy, dte=dte, entry_time=entry_t,
                ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                is_blank=is_blank, ema_aligned=ema_aligned_bull,
                year=dstr[:4]))

        res_sell = simulate_sell(dstr, expiry, strike_sell, 'PE', entry_t)
        if res_sell:
            pnl75, reason, ep, xp, xt = res_sell
            rec_sell.append(dict(
                date=dstr, crt='bullish', side='sell', opt='PE',
                strike=strike_sell, dte=dte, entry_time=entry_t,
                ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                is_blank=is_blank, ema_aligned=ema_aligned_bull,
                year=dstr[:4]))

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | buy:{len(rec_buy)} sell:{len(rec_sell)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
def show(label, df):
    if df.empty: print(f"\n{label}: no trades"); return
    wr  = df['win'].mean() * 100
    pnl = df['pnl_65'].sum()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:    {len(df)}")
    print(f"  Win Rate:  {wr:.1f}%")
    print(f"  Total P&L: Rs.{pnl:,.0f}")
    print(f"  Avg/trade: Rs.{df['pnl_65'].mean():,.0f}")
    print(f"  Best:      Rs.{df['pnl_65'].max():,.0f}")
    print(f"  Worst:     Rs.{df['pnl_65'].min():,.0f}")
    print(f"  Exit reasons: {dict(df['exit_reason'].value_counts())}")
    for ct in df['crt'].unique():
        g = df[df['crt']==ct]
        print(f"  {ct}: {len(g)}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
    print(f"  Year-wise:")
    for yr in sorted(df['year'].unique()):
        g = df[df['year']==yr]
        print(f"    {yr}: {len(g):>3}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")

df_buy  = pd.DataFrame(rec_buy)
df_sell = pd.DataFrame(rec_sell)

print("\n" + "="*60)
print("  CRT BUYING vs SELLING — First candle (09:15) only")
print("="*60)

show("CRT BUYING — all days (TGT 80%, SL 30%, EOD 12:30)", df_buy)
show("CRT SELLING — all days (TGT 20%, SL 100%, EOD 15:20)", df_sell)

if not df_buy.empty:
    show("CRT BUYING — blank days only", df_buy[df_buy['is_blank']])
    show("CRT SELLING — blank days only", df_sell[df_sell['is_blank']])

# Summary comparison table
print(f"\n{'='*60}")
print("  BUYING vs SELLING — HEAD TO HEAD COMPARISON")
print(f"{'='*60}")
sell_conv = df_sell_base['pnl_conv'].sum()

configs = [
    ("BUY all",    df_buy,                                       False),
    ("BUY blank",  df_buy[df_buy['is_blank']]  if not df_buy.empty  else pd.DataFrame(), False),
    ("SELL all",   df_sell,                                      True),
    ("SELL blank", df_sell[df_sell['is_blank']] if not df_sell.empty else pd.DataFrame(), True),
]
for label, df, is_sell in configs:
    if df.empty: continue
    pnl = df['pnl_65'].sum()
    wr  = df['win'].mean()*100
    combo = sell_conv + pnl if is_sell else sell_conv + pnl
    print(f"  CRT {label}: {len(df):>3}t | WR {wr:.0f}% | Rs.{pnl:,.0f} | "
          f"Combined Rs.{combo:,.0f}")

# Save
if not df_buy.empty:
    df_buy.to_csv(f'{OUT_DIR}/81_crt_buying.csv', index=False)
    print(f"\n  Saved → {OUT_DIR}/81_crt_buying.csv")

print("\nDone.")
