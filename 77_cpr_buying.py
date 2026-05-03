"""
77_cpr_buying.py — CPR + EMA(9,21) Option Buying Backtest
==========================================================
Strategy:
  Pre-market: Compute CPR, EMA(9), EMA(21) on daily closes
  Signal:
    Bull setup: EMA9 > EMA21 AND open above CPR TC → Buy CE ATM at 09:25:02
    Bear setup: EMA9 < EMA21 AND open below CPR BC → Buy PE ATM at 09:25:02

  Exit rules:
    Target: +80% of premium
    SL:     -30% of premium (hard stop — no trailing for buying)
    EOD:    12:30:00 (theta kill prevention — no afternoon hold)

  Only on days with NO v17a/cam sell signal (separate universe)
  OR test on ALL days (regardless of sell signal)

Lot size: 65 (LOT=75 scaled)
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import (load_spot_data, load_tick_data, list_expiry_dates,
                     list_trading_dates)

LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
EMA_PERIOD = 20
EOD_BUY    = '12:30:00'   # max hold for buying
TGT_PCT    = 0.80         # +80% target
SL_PCT     = 0.30         # -30% SL
YEARS      = 5
OUT_DIR    = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)
def get_atm(spot): return int(round(spot/STRIKE_INT)*STRIKE_INT)

def simulate_buy(date_str, expiry, strike, opt, entry_time_str, tgt_pct, sl_pct, eod_str):
    """Simulate option BUY trade (price rise = profit)."""
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks   = load_tick_data(date_str, instr, entry_time_str, eod_str)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time_str].reset_index(drop=True)
    if tks.empty: return None
    ep  = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep*(1+tgt_pct))
    sl  = r2(ep*(1-sl_pct))
    ps  = tks['price'].values
    ts  = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= eod_str: return r2((p-ep)*LOT_SIZE), 'eod', r2(ep), r2(p), t
        if p >= tgt:     return r2((p-ep)*LOT_SIZE), 'target', r2(ep), r2(p), t
        if p <= sl:      return r2((p-ep)*LOT_SIZE), 'hard_sl', r2(ep), r2(p), t
    return r2((ps[-1]-ep)*LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

# ── Build daily OHLC + EMAs ───────────────────────────────────────────────────
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
# EMA 20 for v17a bias
ohlc['ema20'] = ohlc['c'].ewm(span=20, adjust=False).mean().shift(1)
# EMA 9 and 21 for buying signal
ohlc['ema9']  = ohlc['c'].ewm(span=9,  adjust=False).mean().shift(1)
ohlc['ema21'] = ohlc['c'].ewm(span=21, adjust=False).mean().shift(1)
# CPR levels from previous day
ohlc['pvt']   = ((ohlc['h'].shift(1) + ohlc['l'].shift(1) + ohlc['c'].shift(1)) / 3).round(2)
ohlc['bc']    = ((ohlc['h'].shift(1) + ohlc['l'].shift(1)) / 2).round(2)
ohlc['tc']    = (ohlc['pvt'] + (ohlc['pvt'] - ohlc['bc'])).round(2)
ohlc['r1']    = (2*ohlc['pvt'] - ohlc['l'].shift(1)).round(2)
ohlc['r2']    = (ohlc['pvt'] + ohlc['h'].shift(1) - ohlc['l'].shift(1)).round(2)
ohlc['s1']    = (2*ohlc['pvt'] - ohlc['h'].shift(1)).round(2)
ohlc['s2']    = (ohlc['pvt'] - ohlc['h'].shift(1) + ohlc['l'].shift(1)).round(2)
ohlc['pdh']   = ohlc['h'].shift(1)
ohlc['pdl']   = ohlc['l'].shift(1)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

# Load existing sell signal days (to identify no-signal days)
df_sell = pd.read_csv('data/20260430/75_live_simulation.csv')
df_sell['date_str'] = df_sell['date'].astype(str).str.replace('-','')
sell_dates = set(df_sell['date_str'].tolist())

# ── Run buying backtest ───────────────────────────────────────────────────────
print("\nRunning buying backtest...")
t0 = time.time()
records_all     = []   # buy on ALL days
records_nosig   = []   # buy only on no-sell-signal days

for idx, row in ohlc_5yr.iterrows():
    if idx < 3: continue
    dstr   = row['date']
    op     = row['o']
    ema9   = row['ema9']
    ema21  = row['ema21']
    tc     = row['tc']
    bc     = row['bc']

    # EMA 9/21 signal
    bull = ema9 > ema21
    bear = ema9 < ema21

    # CPR confirmation
    above_cpr = op > tc    # price above CPR → strong bull setup
    below_cpr = op < bc    # price below CPR → strong bear setup

    # Signal conditions
    if bull and above_cpr:
        opt = 'CE'; sig = 'bull'
    elif bear and below_cpr:
        opt = 'PE'; sig = 'bear'
    else:
        continue  # within CPR or EMA-CPR mismatch → skip

    # Get expiry + strike
    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
           pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

    strike = get_atm(op)
    etime  = '09:25:02'

    res = simulate_buy(dstr, expiry, strike, opt, etime, TGT_PCT, SL_PCT, EOD_BUY)
    if res is None: continue
    pnl75, reason, ep, xp, xt = res
    pnl65 = r2(pnl75 * SCALE)
    win   = pnl65 > 0

    rec = dict(date=dstr, signal=sig, opt=opt, strike=strike, dte=dte,
               entry_time=etime, ep=ep, xp=xp, exit_time=xt,
               exit_reason=reason, pnl_65=pnl65, win=win, year=dstr[:4],
               has_sell_signal=dstr in sell_dates)

    records_all.append(rec)
    if dstr not in sell_dates:
        records_nosig.append(rec)

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

df_all   = pd.DataFrame(records_all)
df_nosig = pd.DataFrame(records_nosig)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  CPR + EMA(9,21) BUYING RESULTS")
print(f"{'='*65}")

for label, df in [("ALL days (incl. sell signal days)", df_all),
                  ("No-signal days only", df_nosig)]:
    if df.empty:
        print(f"\n{label}: no trades")
        continue
    wr   = df['win'].mean()*100
    flat = df['pnl_65'].sum()
    print(f"\n{label}:")
    print(f"  Trades:    {len(df)}")
    print(f"  Win Rate:  {wr:.1f}%")
    print(f"  Total P&L: Rs.{flat:,.0f}")
    print(f"  Avg/trade: Rs.{df['pnl_65'].mean():,.0f}")
    print(f"  Best:      Rs.{df['pnl_65'].max():,.0f}")
    print(f"  Worst:     Rs.{df['pnl_65'].min():,.0f}")
    print(f"  Exit reasons: {dict(df['exit_reason'].value_counts())}")

    # Year-wise
    print(f"  Year-wise:")
    for yr in sorted(df['year'].unique()):
        g = df[df['year']==yr]
        print(f"    {yr}: {len(g):>4}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")

    # Signal breakdown
    print(f"  Bull/Bear:")
    for sig in df['signal'].unique():
        g = df[df['signal']==sig]
        print(f"    {sig}: {len(g)}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")

# Comparison with selling
print(f"\n{'='*65}")
print("  BUYING vs SELLING COMPARISON")
print(f"{'='*65}")
sell_flat = df_sell['pnl_65'].sum() if 'pnl_65' in df_sell.columns else df_sell['pnl_flat'].sum()
sell_conv = df_sell['pnl_conv'].sum()
print(f"  Selling (baseline):      550t | 71.1% WR | Rs.{sell_conv:,.0f} (conviction)")
if not df_nosig.empty:
    combined_pnl = sell_conv + df_nosig['pnl_65'].sum()
    print(f"  Buying (no-signal days): {len(df_nosig):>3}t | {df_nosig['win'].mean()*100:.1f}% WR | Rs.{df_nosig['pnl_65'].sum():,.0f}")
    print(f"  Combined total:          {550+len(df_nosig):>3}t | Rs.{combined_pnl:,.0f}")

# Save
if not df_all.empty:
    df_all.to_csv(f'{OUT_DIR}/77_buying_trades.csv', index=False)
    print(f"\n  Saved → {OUT_DIR}/77_buying_trades.csv")

print("\nDone.")
