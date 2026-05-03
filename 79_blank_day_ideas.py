"""
79_blank_day_ideas.py — 3 new ideas tested on 619 blank days
=============================================================
All three use different triggers to find edge on days v17a/cam/iv2 skipped.

Idea 1: ORB Reversal (intraday trigger, v17a-style logic)
  - Open above CPR + ORB (09:15-09:30) breaks DOWN → sell CE OTM1
  - Open below CPR + ORB breaks UP → sell PE OTM1
  Entry: tick of break + 2s offset

Idea 2: Extreme Zone Momentum (sell the decaying side)
  - above_r1 + bull EMA → sell PE OTM1 at 09:25:02 (market so bullish PE decays)
  - below_s1 + bear EMA → sell CE OTM1 at 09:25:02
  Entry: fixed 09:25:02 (same as v17a)

Idea 3: Gap Continuation
  - Gap up >0.5% AND open above CPR TC → sell PE OTM1 (gap direction = strong bull)
  - Gap down >0.5% AND open below CPR BC → sell CE OTM1
  Entry: 09:25:02

All three use same SL/target as v17a: 20% target, 100% hard SL, trailing 3-tier
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
YEARS      = 5
OUT_DIR    = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s/STRIKE_INT)*STRIKE_INT)
def get_otm1(s, opt):
    atm = get_atm(s)
    return atm - STRIKE_INT if opt == 'PE' else atm + STRIKE_INT

def simulate_sell(date_str, expiry, strike, opt, entry_time, tgt_pct=0.20, sl_pct=1.00):
    """Sell option simulation with 3-tier trailing SL (same as v17a)."""
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - tgt_pct))
    hsl = r2(ep * (1 + sl_pct))
    sl  = hsl
    md  = 0.0  # max drop from entry
    ps  = tks['price'].values
    ts  = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((ep - p) * LOT_SIZE), 'eod', r2(ep), r2(p), t
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt:
            return r2((ep - p) * LOT_SIZE), 'target', r2(ep), r2(p), t
        if p >= sl:
            return r2((ep - p) * LOT_SIZE), 'lockin_sl' if sl < hsl else 'hard_sl', r2(ep), r2(p), t
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
ohlc['r1']   = (2 * ohlc['pvt'] - ohlc['l']).round(2)
ohlc['s1']   = (2 * ohlc['pvt'] - ohlc['h']).round(2)
ohlc['pdh']  = ohlc['h'].shift(1)
ohlc['pdl']  = ohlc['l'].shift(1)
ohlc['prev_c'] = ohlc['c'].shift(1)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)

df_sell  = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
sell_dates = set(df_sell['date'].astype(str).str.replace('-', ''))
print(f"  {len(ohlc_5yr)} days | blank: {len(set(ohlc_5yr['date']) - sell_dates)} | {time.time()-t0:.0f}s")

# ── Main loop ─────────────────────────────────────────────────────────────────
rec1 = []   # Idea 1: ORB reversal
rec2 = []   # Idea 2: extreme zone momentum
rec3 = []   # Idea 3: gap continuation

print("\nRunning backtest...")
t0 = time.time()

for idx, row in ohlc_5yr.iterrows():
    if idx < 3: continue
    dstr = row['date']
    if dstr in sell_dates: continue   # only blank days

    prev = ohlc_5yr.iloc[idx - 1]
    op   = row['o']
    ema  = row['ema']

    # CPR levels from previous day (no forward bias)
    ptc  = prev['tc']
    pbc  = prev['bc']
    pr1  = prev['r1']
    ps1  = prev['s1']
    pdh  = prev['h']
    pdl  = prev['l']
    prev_c = row['prev_c']

    bull_ema = op > ema
    bear_ema = op < ema

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte = (pd.Timestamp('20' + expiry[:2] + '-' + expiry[2:4] + '-' + expiry[4:6]) -
           pd.Timestamp(dstr[:4] + '-' + dstr[4:6] + '-' + dstr[6:])).days

    # ── Idea 1: ORB Reversal ──────────────────────────────────────────────────
    # Only for days where open is outside CPR (has a directional lean)
    if op > ptc or op < pbc:
        spot_tks = load_spot_data(dstr, 'NIFTY')
        if spot_tks is not None:
            orb = spot_tks[(spot_tks['time'] >= '09:15:00') &
                           (spot_tks['time'] <  '09:30:00')]
            post_orb = spot_tks[(spot_tks['time'] >= '09:30:00') &
                                (spot_tks['time'] <= '13:00:00')]
            if not orb.empty and not post_orb.empty:
                orb_h = orb['price'].max()
                orb_l = orb['price'].min()

                # Open above CPR → watch for ORB break DOWN → sell CE
                if op > ptc:
                    for _, tick in post_orb.iterrows():
                        p = tick['price']; t = tick['time']
                        if p < orb_l:
                            # ORB reversed down despite high open
                            entry_t = t
                            strike  = get_otm1(p, 'CE')
                            res = simulate_sell(dstr, expiry, strike, 'CE', entry_t)
                            if res:
                                pnl75, reason, ep, xp, xt = res
                                rec1.append(dict(date=dstr, idea='orb_rev_CE', opt='CE',
                                    strike=strike, dte=dte, entry_time=entry_t,
                                    ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                                    pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                                    year=dstr[:4]))
                            break

                # Open below CPR → watch for ORB break UP → sell PE
                elif op < pbc:
                    for _, tick in post_orb.iterrows():
                        p = tick['price']; t = tick['time']
                        if p > orb_h:
                            entry_t = t
                            strike  = get_otm1(p, 'PE')
                            res = simulate_sell(dstr, expiry, strike, 'PE', entry_t)
                            if res:
                                pnl75, reason, ep, xp, xt = res
                                rec1.append(dict(date=dstr, idea='orb_rev_PE', opt='PE',
                                    strike=strike, dte=dte, entry_time=entry_t,
                                    ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                                    pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                                    year=dstr[:4]))
                            break

    # ── Idea 2: Extreme Zone Momentum ─────────────────────────────────────────
    # above_r1 + bull EMA → sell PE OTM1 (strongly bullish day → PE will decay)
    if op > pr1 and bull_ema:
        strike = get_otm1(op, 'PE')
        res = simulate_sell(dstr, expiry, strike, 'PE', '09:25:02')
        if res:
            pnl75, reason, ep, xp, xt = res
            rec2.append(dict(date=dstr, idea='extreme_PE', opt='PE',
                strike=strike, dte=dte, entry_time='09:25:02',
                ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                year=dstr[:4]))

    # below_s1 + bear EMA → sell CE OTM1 (strongly bearish → CE will decay)
    elif op < ps1 and bear_ema:
        strike = get_otm1(op, 'CE')
        res = simulate_sell(dstr, expiry, strike, 'CE', '09:25:02')
        if res:
            pnl75, reason, ep, xp, xt = res
            rec2.append(dict(date=dstr, idea='extreme_CE', opt='CE',
                strike=strike, dte=dte, entry_time='09:25:02',
                ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                year=dstr[:4]))

    # ── Idea 3: Gap Continuation ──────────────────────────────────────────────
    # Gap up >0.5% + open above TC → sell PE OTM1 (strong bull momentum)
    if prev_c > 0:
        gap_pct = (op - prev_c) / prev_c * 100
        if gap_pct >= 0.5 and op > ptc:
            strike = get_otm1(op, 'PE')
            res = simulate_sell(dstr, expiry, strike, 'PE', '09:25:02')
            if res:
                pnl75, reason, ep, xp, xt = res
                rec3.append(dict(date=dstr, idea='gap_up_PE', opt='PE',
                    strike=strike, dte=dte, entry_time='09:25:02',
                    ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                    gap_pct=r2(gap_pct),
                    pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                    year=dstr[:4]))

        elif gap_pct <= -0.5 and op < pbc:
            strike = get_otm1(op, 'CE')
            res = simulate_sell(dstr, expiry, strike, 'CE', '09:25:02')
            if res:
                pnl75, reason, ep, xp, xt = res
                rec3.append(dict(date=dstr, idea='gap_down_CE', opt='CE',
                    strike=strike, dte=dte, entry_time='09:25:02',
                    ep=ep, xp=xp, exit_time=xt, exit_reason=reason,
                    gap_pct=r2(gap_pct),
                    pnl_65=r2(pnl75*SCALE), win=pnl75*SCALE > 0,
                    year=dstr[:4]))

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | Idea1:{len(rec1)} Idea2:{len(rec2)} Idea3:{len(rec3)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
def show(label, recs):
    if not recs:
        print(f"\n{label}: no trades")
        return pd.DataFrame()
    df = pd.DataFrame(recs)
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
    if 'idea' in df.columns:
        for i in df['idea'].unique():
            g = df[df['idea'] == i]
            print(f"  {i}: {len(g)}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
    print(f"  Exit reasons: {dict(df['exit_reason'].value_counts())}")
    print(f"  Year-wise:")
    for yr in sorted(df['year'].unique()):
        g = df[df['year'] == yr]
        print(f"    {yr}: {len(g):>3}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
    return df

df1 = show("Idea 1: ORB Reversal (open outside CPR + ORB breaks against open)",  rec1)
df2 = show("Idea 2: Extreme Zone Momentum (above_r1/below_s1 + EMA aligned)",     rec2)
df3 = show("Idea 3: Gap Continuation (gap >0.5% + CPR alignment)",                rec3)

# Summary
print(f"\n{'='*60}")
print("  SUMMARY vs baseline")
print(f"{'='*60}")
sell_conv = df_sell['pnl_conv'].sum()
print(f"  Selling baseline: 550t | Rs.{sell_conv:,.0f}")
for label, df in [("Idea1 ORB-rev", df1), ("Idea2 Extreme", df2), ("Idea3 Gap", df3)]:
    if df.empty: continue
    combined = sell_conv + df['pnl_65'].sum()
    print(f"  + {label}: {len(df):>3}t | WR {df['win'].mean()*100:.0f}% | Rs.{df['pnl_65'].sum():,.0f} → Total Rs.{combined:,.0f}")

# Save best
all_new = rec1 + rec2 + rec3
if all_new:
    pd.DataFrame(all_new).to_csv(f'{OUT_DIR}/79_blank_ideas.csv', index=False)
    print(f"\n  Saved → {OUT_DIR}/79_blank_ideas.csv")

print("\nDone.")
