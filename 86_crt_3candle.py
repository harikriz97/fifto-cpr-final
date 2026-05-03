"""
86_crt_3candle.py — Proper 3-Candle CRT Backtest
=================================================
Classic CRT structure:
  Candle 1 (C1): Reference candle — defines the range
  Candle 2 (C2): Sweep candle — wick goes BEYOND C1 H or L
                 C2 does NOT need to close back (just the wick)
  Candle 3 (C3): Displacement — closes back INSIDE C1's range
                 Confirms institutional reversal
  Entry: next candle after C3 closes

Bearish CRT:
  C2 wicks above C1.High → C3 closes below C1.High
  → Sell CE OTM1 (market heading down)
  Target: C1.Low (full range)
  SL: C2.High + buffer

Bullish CRT:
  C2 wicks below C1.Low → C3 closes above C1.Low
  → Sell PE OTM1 (market heading up)
  Target: C1.High (full range)
  SL: C2.Low - buffer

Scan window: 09:15 – 12:00 (3 candles = up to 10:30 for C3 close → entry by 10:46)
Also test: Option TGT 20% SL 100% trailing (v17a style)
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
                  tgt_pct=0.20, sl_pct=1.00, eod=EOD_EXIT):
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
        if t >= eod: return r2((ep-p)*LOT_SIZE),'eod',r2(ep),r2(p),t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE),'target',r2(ep),r2(p),t
        if p >= sl:  return r2((ep-p)*LOT_SIZE),'lockin_sl' if sl<hsl else 'hard_sl',r2(ep),r2(p),t
    return r2((ep-ps[-1])*LOT_SIZE),'eod',r2(ep),r2(ps[-1]),ts[-1]

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
    day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')]
    if len(day)<2: continue
    rows.append({'date':d,'o':day.iloc[0]['price'],
                 'h':day['price'].max(),'l':day['price'].min(),'c':day.iloc[-1]['price']})

ohlc = pd.DataFrame(rows)
ohlc['ema'] = ohlc['c'].ewm(span=20,adjust=False).mean().shift(1)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)

df_sell_base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
sell_dates   = set(df_sell_base['date'].astype(str).str.replace('-',''))
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nScanning 3-candle CRT...")
t0 = time.time()
records = []

for idx, row in ohlc_5yr.iterrows():
    dstr = row['date']
    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue
    candles = build_15min(spot)
    if len(candles) < 3: continue

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]

    is_blank   = dstr not in sell_dates
    ema_bias   = 'bull' if row['o'] > row['ema'] else 'bear'

    crt_fired = False
    for ci in range(1, len(candles)-1):   # need C1, C2, C3
        if crt_fired: break

        c1 = candles.iloc[ci-1]
        c2 = candles.iloc[ci]
        c3_idx = ci + 1
        if c3_idx >= len(candles): break

        c3 = candles.iloc[c3_idx]
        c3_time = c3['time']
        if c3_time > '11:00:00': break    # stop at 11:00 candle

        c1h = c1['h']; c1l = c1['l']
        c2h = c2['h']; c2l = c2['l']
        c3c = c3['c']; c3o = c3['o']

        # Entry after C3 closes: C3_open_time + 15min + 1min
        h_str = c3_time[:2]; m_str = c3_time[3:5]
        entry_min = int(h_str)*60 + int(m_str) + CANDLE_MIN + 1
        entry_t   = f"{entry_min//60:02d}:{entry_min%60:02d}:02"
        if entry_t >= EOD_EXIT: break

        dte = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
               pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

        # ── Bearish CRT ────────────────────────────────────────────────────────
        # C2 sweeps above C1.High (wick above), C3 closes below C1.High
        if c2h > c1h and c3c < c1h:
            # Optional: C3 also closed below C3.Open (bearish C3)
            opt    = 'CE'
            strike = get_otm1(c3c, opt)
            res    = simulate_sell(dstr, expiry, strike, opt, entry_t)
            if res:
                pnl75, reason, ep, xp, xt = res
                pnl65 = r2(pnl75 * SCALE)
                records.append(dict(
                    date=dstr, crt='bearish', c1_time=c1['time'],
                    c2_time=c2['time'], c3_time=c3_time,
                    c1h=r2(c1h), c1l=r2(c1l), c2h=r2(c2h), c3c=r2(c3c),
                    entry_time=entry_t, opt=opt, strike=strike, dte=dte,
                    ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                    pnl_65=pnl65, win=pnl65>0,
                    is_blank=is_blank, ema_aligned=(ema_bias=='bear'),
                    year=dstr[:4]))
                crt_fired = True

        # ── Bullish CRT ────────────────────────────────────────────────────────
        # C2 sweeps below C1.Low (wick below), C3 closes above C1.Low
        elif c2l < c1l and c3c > c1l:
            opt    = 'PE'
            strike = get_otm1(c3c, opt)
            res    = simulate_sell(dstr, expiry, strike, opt, entry_t)
            if res:
                pnl75, reason, ep, xp, xt = res
                pnl65 = r2(pnl75 * SCALE)
                records.append(dict(
                    date=dstr, crt='bullish', c1_time=c1['time'],
                    c2_time=c2['time'], c3_time=c3_time,
                    c1h=r2(c1h), c1l=r2(c1l), c2l=r2(c2l), c3c=r2(c3c),
                    entry_time=entry_t, opt=opt, strike=strike, dte=dte,
                    ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                    pnl_65=pnl65, win=pnl65>0,
                    is_blank=is_blank, ema_aligned=(ema_bias=='bull'),
                    year=dstr[:4]))
                crt_fired = True

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | trades:{len(records)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
if not records:
    print("No CRT trades found."); exit()

df = pd.DataFrame(records)

def show(label, sub):
    if sub.empty: print(f"\n{label}: no trades"); return
    wr  = sub['win'].mean()*100
    pnl = sub['pnl_65'].sum()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:    {len(sub)}")
    print(f"  Win Rate:  {wr:.1f}%")
    print(f"  Total P&L: Rs.{pnl:,.0f}")
    print(f"  Avg/trade: Rs.{sub['pnl_65'].mean():,.0f}")
    print(f"  Best:      Rs.{sub['pnl_65'].max():,.0f}")
    print(f"  Worst:     Rs.{sub['pnl_65'].min():,.0f}")
    print(f"  Exits: {dict(sub['exit_reason'].value_counts())}")
    for ct in sub['crt'].unique():
        g = sub[sub['crt']==ct]
        print(f"  {ct}: {len(g)}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
    print(f"  C3 candle times: {dict(sub['c3_time'].value_counts().sort_index())}")
    print(f"  Year-wise:")
    for yr in sorted(sub['year'].unique()):
        g = sub[sub['year']==yr]
        print(f"    {yr}: {len(g):>3}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")

print(f"\n  Total 3-candle CRT signals: {len(df)} on {df['date'].nunique()} days")
show("ALL days — 3-candle CRT", df)
show("Blank days only",         df[df['is_blank']])
show("EMA aligned",             df[df['ema_aligned']])
show("Blank + EMA aligned",     df[df['is_blank'] & df['ema_aligned']])

# Compare with old 1-candle version (script 80)
sell_conv = df_sell_base['pnl_conv'].sum()
df_old = pd.read_csv(f'{OUT_DIR}/80_crt_trades.csv')
print(f"\n{'='*60}")
print("  3-CANDLE vs 1-CANDLE CRT COMPARISON")
print(f"{'='*60}")
print(f"  1-candle (script 80) blank days: {df_old['is_blank'].sum()}t | "
      f"WR {df_old[df_old['is_blank']]['win'].mean()*100:.0f}% | "
      f"Rs.{df_old[df_old['is_blank']]['pnl_65'].sum():,.0f}")
print(f"  3-candle (this)  blank days:     {df[df['is_blank']].shape[0]}t | "
      f"WR {df[df['is_blank']]['win'].mean()*100:.0f}% | "
      f"Rs.{df[df['is_blank']]['pnl_65'].sum():,.0f}")
print(f"\n  Selling baseline: Rs.{sell_conv:,.0f}")
for label, sub in [('3-candle all', df), ('3-candle blank', df[df['is_blank']]),
                   ('3-candle blank+ema', df[df['is_blank']&df['ema_aligned']])]:
    if sub.empty: continue
    print(f"  + {label}: {len(sub)}t | WR {sub['win'].mean()*100:.0f}% | "
          f"Rs.{sub['pnl_65'].sum():,.0f} → Total Rs.{sell_conv+sub['pnl_65'].sum():,.0f}")

df.to_csv(f'{OUT_DIR}/86_crt_3candle.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/86_crt_3candle.csv")
print("\nDone.")
