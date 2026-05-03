"""
78_blank_day_patterns.py — Patterns on blank days (620 no-signal days)
=======================================================================
Tests 3 intraday patterns on days where v17a/cam/iv2 gave no signal:

  Pattern A: within_cpr intraday breakout
    Open between TC and BC → wait for price to break TC or BC intraday
    Break TC → sell CE OTM1 (price exiting CPR upward = bearish for CE?)
    Actually: break TC → bullish → sell PE OTM1
              break BC → bearish → sell CE OTM1
    Entry: tick after break + 2s

  Pattern B: Contradicting EMA retest of CPR edge
    Open above CPR (tc_to_pdh or pdh_to_r1) BUT EMA is BULL (no v17a signal fired)
    → Wait for price to pull back and TOUCH TC intraday → sell CE OTM1
    Open below CPR (pdl_to_bc or pdl_to_s1) BUT EMA is BEAR
    → Wait for price to bounce and TOUCH BC → sell PE OTM1
    Scan: 09:30 to 12:00 only

  Pattern C: Level retest (R1/PDL/R2 retest after break)
    Price breaks R1 → pulls back to R1 → holds → sell PE (retest of R1 as support)
    Price breaks PDL → pulls back to PDL → holds → sell CE (retest of PDL as resistance)
    More filtered than iv2 breakout entry
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates

LOT_SIZE  = 75
SCALE     = 65 / 75
STRIKE_INT = 50
EOD_EXIT  = '15:20:00'
YEARS     = 5
OUT_DIR   = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s/STRIKE_INT)*STRIKE_INT)
def get_otm1(s,opt): atm=get_atm(s); return atm-STRIKE_INT if opt=='PE' else atm+STRIKE_INT
def get_strike(s,opt,stype):
    if stype=='ATM':  return get_atm(s)
    if stype=='OTM1': return get_otm1(s,opt)
    if stype=='ITM1': atm=get_atm(s); return atm+STRIKE_INT if opt=='PE' else atm-STRIKE_INT

def simulate(date_str, expiry, strike, opt, entry_time, tgt_pct, sl_pct):
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    ps = tks['price'].values; ts = tks['time'].values
    tgt = r2(ep*(1-tgt_pct)); hsl = r2(ep*(1+sl_pct)); sl = hsl; md = 0.0
    for i in range(len(ts)):
        t=ts[i]; p=ps[i]
        if t >= EOD_EXIT: return r2((ep-p)*LOT_SIZE),'eod',r2(ep),r2(p),t
        d=(ep-p)/ep
        if d>md: md=d
        if   md>=0.60: sl=min(sl,r2(ep*(1-md*0.95)))
        elif md>=0.40: sl=min(sl,r2(ep*0.80))
        elif md>=0.25: sl=min(sl,ep)
        if p<=tgt: return r2((ep-p)*LOT_SIZE),'target',r2(ep),r2(p),t
        if p>=sl:  return r2((ep-p)*LOT_SIZE),'lockin_sl' if sl<hsl else 'hard_sl',r2(ep),r2(p),t
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
    rows.append({'date':d,'o':day.iloc[0]['price'],'h':day['price'].max(),
                 'l':day['price'].min(),'c':day.iloc[-1]['price']})

ohlc = pd.DataFrame(rows)
ohlc['ema'] = ohlc['c'].ewm(span=20,adjust=False).mean().shift(1)
ohlc['pvt'] = ((ohlc['h']+ohlc['l']+ohlc['c'])/3).round(2)
ohlc['bc']  = ((ohlc['h']+ohlc['l'])/2).round(2)
ohlc['tc']  = (ohlc['pvt']+(ohlc['pvt']-ohlc['bc'])).round(2)
ohlc['r1']  = (2*ohlc['pvt']-ohlc['l']).round(2)
ohlc['r2']  = (ohlc['pvt']+ohlc['h']-ohlc['l']).round(2)
ohlc['s1']  = (2*ohlc['pvt']-ohlc['h']).round(2)
ohlc['pdh'] = ohlc['h'].shift(1)
ohlc['pdl'] = ohlc['l'].shift(1)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)

# Blank days
df_sell = pd.read_csv('data/20260430/75_live_simulation.csv')
sell_dates = set(df_sell['date'].astype(str).str.replace('-',''))
blank_dates = set(ohlc_5yr['date']) - sell_dates
print(f"  {len(ohlc_5yr)} days | blank: {len(blank_dates)} | {time.time()-t0:.0f}s")

# ── Pattern runner ────────────────────────────────────────────────────────────
rec_a = []  # within_cpr breakout
rec_b = []  # CPR retest
rec_c = []  # level retest

print("\nRunning pattern scan...")
t0 = time.time()

for idx, row in ohlc_5yr.iterrows():
    if idx < 3: continue
    dstr = row['date']
    if dstr not in blank_dates: continue

    prev = ohlc_5yr.iloc[idx-1]
    op   = row['o']
    bias = 'bull' if op > row['ema'] else 'bear'

    # Pivot levels (from PREVIOUS day's actual OHLC)
    pvt = prev['pvt']; bc = prev['bc']; tc = prev['tc']
    r1  = prev['r1']
    pdh = prev['h'];   pdl = prev['l']

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]
    dte = (pd.Timestamp('20'+expiry[:2]+'-'+expiry[2:4]+'-'+expiry[4:6]) -
           pd.Timestamp(dstr[:4]+'-'+dstr[4:6]+'-'+dstr[6:])).days

    # Load spot ticks for intraday scan
    tks = load_spot_data(dstr, 'NIFTY')
    if tks is None: continue
    scan = tks[(tks['time']>='09:20:00')&(tks['time']<='13:00:00')].reset_index(drop=True)
    if scan.empty: continue

    # ── Pattern A: within_cpr intraday breakout ───────────────────────────────
    if bc <= op <= tc:  # open inside CPR
        tc_broken=False; bc_broken=False
        for _, tick in scan.iterrows():
            p=tick['price']; t=tick['time']
            if not tc_broken and p > tc:
                # Break above TC → bullish → sell PE OTM1
                strike = get_strike(p,'PE','OTM1')
                res = simulate(dstr,expiry,strike,'PE',t,0.20,1.00)
                if res:
                    pnl75,reason,ep,xp,xt = res
                    rec_a.append(dict(date=dstr,pattern='A_tc_break',opt='PE',
                        strike=strike,dte=dte,entry_time=t,ep=ep,xp=xp,
                        exit_reason=reason,pnl_65=r2(pnl75*SCALE),
                        win=pnl75*SCALE>0,year=dstr[:4]))
                tc_broken=True; break
            if not bc_broken and p < bc:
                # Break below BC → bearish → sell CE OTM1
                strike = get_strike(p,'CE','OTM1')
                res = simulate(dstr,expiry,strike,'CE',t,0.20,1.00)
                if res:
                    pnl75,reason,ep,xp,xt = res
                    rec_a.append(dict(date=dstr,pattern='A_bc_break',opt='CE',
                        strike=strike,dte=dte,entry_time=t,ep=ep,xp=xp,
                        exit_reason=reason,pnl_65=r2(pnl75*SCALE),
                        win=pnl75*SCALE>0,year=dstr[:4]))
                bc_broken=True; break

    # ── Pattern B: Contradicting EMA → CPR edge retest ───────────────────────
    # Open above CPR with BULL EMA → no v17a signal (bear needed for pdh_to_r1)
    # Wait for price to pull back to TC → sell CE OTM1 (reversal from TC resistance)
    if op > tc and bias == 'bull':
        tc_touched = False
        for _, tick in scan.iterrows():
            p=tick['price']; t=tick['time']
            if p <= tc:  # price pulled back to TC
                strike = get_strike(p,'CE','OTM1')
                res = simulate(dstr,expiry,strike,'CE',t,0.20,1.00)
                if res:
                    pnl75,reason,ep,xp,xt = res
                    rec_b.append(dict(date=dstr,pattern='B_tc_retest',opt='CE',
                        strike=strike,dte=dte,entry_time=t,ep=ep,xp=xp,
                        exit_reason=reason,pnl_65=r2(pnl75*SCALE),
                        win=pnl75*SCALE>0,year=dstr[:4]))
                tc_touched=True; break

    # Open below CPR with BEAR EMA → no v17a signal
    # Wait for price to bounce to BC → sell PE OTM1
    if op < bc and bias == 'bear':
        bc_touched = False
        for _, tick in scan.iterrows():
            p=tick['price']; t=tick['time']
            if p >= bc:  # price bounced to BC
                strike = get_strike(p,'PE','OTM1')
                res = simulate(dstr,expiry,strike,'PE',t,0.20,1.00)
                if res:
                    pnl75,reason,ep,xp,xt = res
                    rec_b.append(dict(date=dstr,pattern='B_bc_retest',opt='PE',
                        strike=strike,dte=dte,entry_time=t,ep=ep,xp=xp,
                        exit_reason=reason,pnl_65=r2(pnl75*SCALE),
                        win=pnl75*SCALE>0,year=dstr[:4]))
                bc_touched=True; break

    # ── Pattern C: R1/PDL level retest ───────────────────────────────────────
    # Price breaks R1, pulls back to R1, holds → sell PE ATM
    # Price breaks PDL, pulls back to PDL, holds → sell CE ATM
    r1_broken=False; pdl_broken=False
    prev_p = op
    scan2 = tks[(tks['time']>='09:30:00')&(tks['time']<='12:30:00')].reset_index(drop=True)
    retest_done_r1=False; retest_done_pdl=False

    for _, tick in scan2.iterrows():
        p=tick['price']; t=tick['time']
        # R1 retest
        if not retest_done_r1:
            if not r1_broken and p > r1:
                r1_broken = True
            elif r1_broken and p <= r1:  # pulled back to R1
                strike = get_strike(p,'PE','ATM')
                res = simulate(dstr,expiry,strike,'PE',t,0.20,0.50)
                if res:
                    pnl75,reason,ep,xp,xt = res
                    rec_c.append(dict(date=dstr,pattern='C_r1_retest',opt='PE',
                        strike=strike,dte=dte,entry_time=t,ep=ep,xp=xp,
                        exit_reason=reason,pnl_65=r2(pnl75*SCALE),
                        win=pnl75*SCALE>0,year=dstr[:4]))
                retest_done_r1=True
        # PDL retest
        if not retest_done_pdl:
            if not pdl_broken and p < pdl:
                pdl_broken = True
            elif pdl_broken and p >= pdl:  # bounced back to PDL
                strike = get_strike(p,'CE','ATM')
                res = simulate(dstr,expiry,strike,'CE',t,0.20,0.50)
                if res:
                    pnl75,reason,ep,xp,xt = res
                    rec_c.append(dict(date=dstr,pattern='C_pdl_retest',opt='CE',
                        strike=strike,dte=dte,entry_time=t,ep=ep,xp=xp,
                        exit_reason=reason,pnl_65=r2(pnl75*SCALE),
                        win=pnl75*SCALE>0,year=dstr[:4]))
                retest_done_pdl=True

    if idx % 100 == 0:
        elapsed = time.time()-t0
        print(f"  {idx}/{len(ohlc_5yr)} | A:{len(rec_a)} B:{len(rec_b)} C:{len(rec_c)} | {elapsed:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
def show(label, recs):
    if not recs:
        print(f"\n{label}: no trades found")
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    wr  = df['win'].mean()*100
    pnl = df['pnl_65'].sum()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:    {len(df)}")
    print(f"  Win Rate:  {wr:.1f}%")
    print(f"  Total P&L: Rs.{pnl:,.0f}")
    print(f"  Avg/trade: Rs.{df['pnl_65'].mean():,.0f}")
    if 'pattern' in df.columns:
        for pat in df['pattern'].unique():
            g = df[df['pattern']==pat]
            print(f"  {pat}: {len(g)}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
    print(f"  Year-wise:")
    for yr in sorted(df['year'].unique()):
        g = df[df['year']==yr]
        print(f"    {yr}: {len(g):>3}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")
    return df

df_a = show("Pattern A: within_cpr intraday breakout", rec_a)
df_b = show("Pattern B: Contradicting EMA → CPR retest", rec_b)
df_c = show("Pattern C: R1/PDL level retest", rec_c)

# Combined
all_recs = rec_a + rec_b + rec_c
if all_recs:
    df_all = pd.DataFrame(all_recs)
    # Check overlaps
    dup = df_all[df_all.duplicated('date',keep=False)]
    print(f"\nOverlapping dates across patterns: {len(dup)}")
    print(f"\n{'='*60}")
    print(f"  COMBINED (all 3 patterns, no dedup)")
    print(f"{'='*60}")
    print(f"  Trades:    {len(df_all)}")
    print(f"  Win Rate:  {df_all['win'].mean()*100:.1f}%")
    print(f"  Total P&L: Rs.{df_all['pnl_65'].sum():,.0f}")
    sell_conv = pd.read_csv('data/20260430/75_live_simulation.csv')['pnl_conv'].sum()
    combined = sell_conv + df_all['pnl_65'].sum()
    print(f"\n  Selling baseline: Rs.{sell_conv:,.0f}")
    print(f"  + New patterns:   Rs.{df_all['pnl_65'].sum():,.0f}")
    print(f"  = Grand total:    Rs.{combined:,.0f}")
    df_all.to_csv(f'{OUT_DIR}/78_blank_patterns.csv', index=False)
    print(f"\n  Saved → {OUT_DIR}/78_blank_patterns.csv")

print("\nDone.")
