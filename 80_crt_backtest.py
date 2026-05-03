"""
80_crt_backtest.py — CRT (Candle Range Theory) Backtest on Nifty
=================================================================
Logic:
  Reference range = Previous Day High (PDH) and Previous Day Low (PDL)

  Bearish CRT:
    1. Price sweeps ABOVE PDH (wicks beyond — takes out stops above prior high)
    2. A 15-min candle CLOSES BACK BELOW PDH (displacement confirmed)
    3. Entry: next tick after that candle's close + 2s → Sell CE OTM1
    4. Why: price faked the breakout, institutions now driving it down

  Bullish CRT:
    1. Price sweeps BELOW PDL
    2. A 15-min candle CLOSES BACK ABOVE PDL
    3. Entry: next tick after that candle's close + 2s → Sell PE OTM1

  Sweep window:  09:15 – 10:30 (morning manipulation window)
  Entry window:  09:30 – 10:45 (candle-close confirmed + 2s)
  EOD exit:      15:20

  Filters tested:
    A. All days (no filter)
    B. Blank days only (no v17a/cam/iv2 signal)
    C. EMA aligned (CRT direction agrees with EMA bias)

  SL/Target: 20% target, 100% hard SL, 3-tier trailing (same as v17a)
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
EOD_EXIT   = '15:20:00'
SWEEP_END  = '10:30:00'   # stop looking for sweep after this
CANDLE_MIN = 15           # 15-min candles
YEARS      = 5
OUT_DIR    = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def get_otm1(s, opt):
    atm = get_atm(s)
    return atm - STRIKE_INT if opt == 'PE' else atm + STRIKE_INT

def build_15min(tks, start='09:15:00', end='10:45:00'):
    """Build 15-min OHLC candles from tick data."""
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('15min').ohlc().dropna()
    ohlc.columns = ['o', 'h', 'l', 'c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

def simulate_sell(date_str, expiry, strike, opt, entry_time, tgt_pct=0.20, sl_pct=1.00):
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - tgt_pct))
    hsl = r2(ep * (1 + sl_pct))
    sl  = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((ep - p) * LOT_SIZE), 'eod', r2(ep), r2(p), t
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
ohlc['ema']  = ohlc['c'].ewm(span=20, adjust=False).mean().shift(1)
ohlc['pvt']  = ((ohlc['h'] + ohlc['l'] + ohlc['c']) / 3).round(2)
ohlc['bc']   = ((ohlc['h'] + ohlc['l']) / 2).round(2)
ohlc['tc']   = (ohlc['pvt'] + (ohlc['pvt'] - ohlc['bc'])).round(2)
ohlc['pdh']  = ohlc['h'].shift(1)
ohlc['pdl']  = ohlc['l'].shift(1)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)

df_sell    = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
sell_dates = set(df_sell['date'].astype(str).str.replace('-', ''))
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

# ── Main loop ─────────────────────────────────────────────────────────────────
records = []
print("\nRunning CRT scan...")
t0 = time.time()

for idx, row in ohlc_5yr.iterrows():
    if idx < 3: continue
    dstr = row['date']
    prev = ohlc_5yr.iloc[idx - 1]
    pdh  = prev['h']   # previous day actual high
    pdl  = prev['l']   # previous day actual low
    op   = row['o']
    ema  = row['ema']
    ptc  = prev['tc']
    pbc  = prev['bc']

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
           pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

    # Load spot ticks, build 15-min candles
    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue
    candles = build_15min(spot, start='09:15:00', end='10:45:00')
    if candles.empty: continue

    crt_fired = False
    for ci, crow in candles.iterrows():
        if crt_fired: break
        candle_time = crow['time']
        if candle_time > SWEEP_END: break   # only look up to 10:30 candle close

        ch = crow['h']; cl = crow['l']; cc = crow['c']

        # Bearish CRT: candle swept above PDH AND closed back below PDH
        if ch > pdh and cc < pdh:
            opt    = 'CE'
            # entry: AFTER the 15-min candle fully closes (candle_time is open time)
            # candle covers [candle_time, candle_time + 15min)
            # we enter 1 min after the candle closes = candle_open + 15 + 1 min
            h_str  = candle_time[:2]; m_str = candle_time[3:5]
            entry_min = int(h_str)*60 + int(m_str) + CANDLE_MIN + 1
            entry_t   = f"{entry_min//60:02d}:{entry_min%60:02d}:02"
            if entry_t >= EOD_EXIT: break

            strike = get_otm1(cc, opt)
            res    = simulate_sell(dstr, expiry, strike, opt, entry_t)
            if res:
                pnl75, reason, ep, xp, xt = res
                ema_aligned = (op < ema)  # bear EMA agrees with bearish CRT
                is_blank    = dstr not in sell_dates
                records.append(dict(
                    date=dstr, crt='bearish', opt=opt, strike=strike, dte=dte,
                    candle_time=candle_time, entry_time=entry_t,
                    sweep_high=r2(ch), pdh=r2(pdh), close_back=r2(cc),
                    ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                    pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                    ema_aligned=ema_aligned, is_blank=is_blank,
                    year=dstr[:4]))
                crt_fired = True

        # Bullish CRT: candle swept below PDL AND closed back above PDL
        elif cl < pdl and cc > pdl:
            opt    = 'PE'
            h_str  = candle_time[:2]; m_str = candle_time[3:5]
            entry_min = int(h_str)*60 + int(m_str) + CANDLE_MIN + 1
            entry_t   = f"{entry_min//60:02d}:{entry_min%60:02d}:02"
            if entry_t >= EOD_EXIT: break

            strike = get_otm1(cc, opt)
            res    = simulate_sell(dstr, expiry, strike, opt, entry_t)
            if res:
                pnl75, reason, ep, xp, xt = res
                ema_aligned = (op > ema)  # bull EMA agrees with bullish CRT
                is_blank    = dstr not in sell_dates
                records.append(dict(
                    date=dstr, crt='bullish', opt=opt, strike=strike, dte=dte,
                    candle_time=candle_time, entry_time=entry_t,
                    sweep_low=r2(cl), pdl=r2(pdl), close_back=r2(cc),
                    ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                    pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                    ema_aligned=ema_aligned, is_blank=is_blank,
                    year=dstr[:4]))
                crt_fired = True

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | trades:{len(records)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
if not records:
    print("No CRT trades found.")
else:
    df = pd.DataFrame(records)

    def show(label, sub):
        if sub.empty:
            print(f"\n{label}: no trades"); return
        wr  = sub['win'].mean() * 100
        pnl = sub['pnl_65'].sum()
        print(f"\n{'='*62}")
        print(f"  {label}")
        print(f"{'='*62}")
        print(f"  Trades:    {len(sub)}")
        print(f"  Win Rate:  {wr:.1f}%")
        print(f"  Total P&L: Rs.{pnl:,.0f}")
        print(f"  Avg/trade: Rs.{sub['pnl_65'].mean():,.0f}")
        print(f"  Best:      Rs.{sub['pnl_65'].max():,.0f}")
        print(f"  Worst:     Rs.{sub['pnl_65'].min():,.0f}")
        print(f"  Exit reasons: {dict(sub['exit_reason'].value_counts())}")
        for ct in sub['crt'].unique():
            g = sub[sub['crt']==ct]
            print(f"  {ct}: {len(g)}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
        print(f"  Year-wise:")
        for yr in sorted(sub['year'].unique()):
            g = sub[sub['year']==yr]
            print(f"    {yr}: {len(g):>3}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
        # Candle timing breakdown
        print(f"  Entry candle breakdown:")
        for ct in sorted(sub['candle_time'].unique()):
            g = sub[sub['candle_time']==ct]
            print(f"    {ct}: {len(g)}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")

    print(f"\n  CRT signal found on {len(df)} days out of {len(ohlc_5yr)}")
    show("ALL days — CRT (no filter)", df)
    show("Blank days only (no v17a/cam/iv2 signal)", df[df['is_blank']])
    show("EMA-aligned CRT only", df[df['ema_aligned']])
    show("Blank + EMA aligned", df[df['is_blank'] & df['ema_aligned']])

    sell_conv = df_sell['pnl_conv'].sum()
    print(f"\n{'='*62}")
    print("  CRT vs SELLING BASELINE")
    print(f"{'='*62}")
    print(f"  Selling baseline: 550t | Rs.{sell_conv:,.0f}")

    for label, sub in [
        ("All CRT",          df),
        ("CRT blank only",   df[df['is_blank']]),
        ("CRT EMA aligned",  df[df['ema_aligned']]),
        ("CRT blank+aligned",df[df['is_blank'] & df['ema_aligned']]),
    ]:
        if sub.empty: continue
        overlap = len(sub[~sub['is_blank']]) if label != "All CRT" else len(sub[~sub['is_blank']])
        combo   = sell_conv + sub['pnl_65'].sum()
        print(f"  + {label}: {len(sub):>3}t | WR {sub['win'].mean()*100:.0f}% | "
              f"Rs.{sub['pnl_65'].sum():,.0f} → Total Rs.{combo:,.0f}")

    df.to_csv(f'{OUT_DIR}/80_crt_trades.csv', index=False)
    print(f"\n  Saved → {OUT_DIR}/80_crt_trades.csv")

print("\nDone.")
