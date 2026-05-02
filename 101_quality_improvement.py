"""
101_quality_improvement.py — Per-trade quality improvement test
================================================================
Test two fixes on CRT + MRC blank day trades:
  Fix 1: ATM strike instead of OTM1
  Fix 2: Target % → 20%, 25%, 30%, 35%

Matrix: 2 strikes × 4 targets × 2 strategies = 16 combos
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

def get_strike(spot, opt, use_atm=False):
    atm = int(round(spot / STRIKE_INT) * STRIKE_INT)
    if use_atm: return atm
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
            return r2((ep - p) * LOT_SIZE * SCALE), 'eod', r2(ep), r2(p)
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep - p) * LOT_SIZE * SCALE), 'target', r2(ep), r2(p)
        if p >= sl:  return r2((ep - p) * LOT_SIZE * SCALE), 'lockin_sl' if sl < hsl else 'hard_sl', r2(ep), r2(p)
    return r2((ep - ps[-1]) * LOT_SIZE * SCALE), 'eod', r2(ep), r2(ps[-1])

def compute_ha(ohlc):
    ha = ohlc.copy()
    ha['ha_c'] = ((ohlc['o'] + ohlc['h'] + ohlc['l'] + ohlc['c']) / 4).round(2)
    ha_o = [0.0] * len(ha)
    ha_o[0] = r2((ohlc['o'].iloc[0] + ohlc['c'].iloc[0]) / 2)
    for i in range(1, len(ha)):
        ha_o[i] = r2((ha_o[i-1] + ha['ha_c'].iloc[i-1]) / 2)
    ha['ha_o'] = ha_o
    ha['ha_h'] = ha[['h','ha_o','ha_c']].max(axis=1).round(2)
    ha['ha_l'] = ha[['l','ha_o','ha_c']].min(axis=1).round(2)
    return ha

def build_ohlc_5m(tks, end='12:00:00'):
    df = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('5min').ohlc().dropna()
    ohlc.columns = ['o','h','l','c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

def build_ohlc_15m(tks, end='12:00:00'):
    df = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('15min').ohlc().dropna()
    ohlc.columns = ['o','h','l','c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

# ── Load existing signal dates ────────────────────────────────────────────────
print("Loading signal dates...")
crt_raw  = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank = crt_raw[crt_raw['is_blank']==True].copy()
crt_blank['date'] = crt_blank['date'].astype(str)

mrc_raw  = pd.read_csv(f'{OUT_DIR}/100_mrc_trades.csv')
mrc_blank = mrc_raw[mrc_raw['is_blank']==True].copy()
mrc_blank['date'] = mrc_blank['date'].astype(str)
# MRC unique = not covered by CRT
crt_dates = set(crt_blank['date'])
mrc_unique = mrc_blank[~mrc_blank['date'].isin(crt_dates)].copy()

print(f"  CRT blank: {len(crt_blank)} days | MRC unique blank: {len(mrc_unique)} days")

# ── Build PDH/PDL for MRC levels ─────────────────────────────────────────────
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
extra = max(0, all_dates.index(dates_5yr[0]) - 3)
rows = []
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    rows.append({'date': d, 'pdh': day['price'].max(), 'pdl': day['price'].min()})
df_pdhl = pd.DataFrame(rows)
df_pdhl['pdh_prev'] = df_pdhl['pdh'].shift(1)
df_pdhl['pdl_prev'] = df_pdhl['pdl'].shift(1)
df_pdhl = df_pdhl.dropna()
pdhl = dict(zip(df_pdhl['date'], zip(df_pdhl['pdh_prev'], df_pdhl['pdl_prev'])))

# Build CPR for CRT LTF signals (need TC, R1)
rows2 = []
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    rows2.append({'date': d, 'h': day['price'].max(), 'l': day['price'].min(), 'c': day.iloc[-1]['price']})
df_cpr = pd.DataFrame(rows2)
ph = df_cpr['h'].shift(1); pl = df_cpr['l'].shift(1); pc = df_cpr['c'].shift(1)
df_cpr['pvt'] = ((ph+pl+pc)/3).round(2)
df_cpr['bc']  = ((ph+pl)/2).round(2)
df_cpr['tc']  = (df_cpr['pvt']+(df_cpr['pvt']-df_cpr['bc'])).round(2)
df_cpr['r1']  = (2*df_cpr['pvt']-pl).round(2)
df_cpr = df_cpr.dropna()
cpr_map = {r['date']: (r['tc'], r['r1']) for _, r in df_cpr.iterrows()}

TARGETS  = [0.20, 0.25, 0.30, 0.35]
STRIKES  = ['otm1', 'atm']
results  = []

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: CRT blank days — re-run with different strike + target
# ─────────────────────────────────────────────────────────────────────────────
print("\n[CRT] Scanning blank days with ATM + target variations...")
crt_days = sorted(crt_blank['date'].unique())

for strike_type in STRIKES:
    for tgt in TARGETS:
        t0 = time.time()
        trades = []
        for dstr in crt_days:
            row = crt_blank[crt_blank['date'] == dstr].iloc[0]
            entry_time = row['entry_time']
            signal = 'CE'  # CRT is always bearish CE sell

            spot = load_spot_data(dstr, 'NIFTY')
            if spot is None: continue
            spot_at_entry = spot[spot['time'] >= entry_time[:8]]
            if spot_at_entry.empty: continue
            spot_ref = spot_at_entry.iloc[0]['price']

            expiries = list_expiry_dates(dstr, index_name='NIFTY')
            if not expiries: continue
            strike = get_strike(spot_ref, signal, use_atm=(strike_type=='atm'))
            instr  = f'NIFTY{expiries[0]}{strike}{signal}'

            res = simulate_sell(dstr, instr, entry_time, tgt_pct=tgt)
            if res:
                pnl, reason, ep, xp = res
                trades.append({'pnl': pnl, 'win': pnl > 0, 'exit': reason, 'ep': ep})

        if trades:
            t_df = pd.DataFrame(trades)
            results.append({
                'strategy': 'CRT', 'strike': strike_type, 'target': tgt,
                'trades': len(t_df), 'wr': t_df['win'].mean()*100,
                'total_pnl': t_df['pnl'].sum(), 'avg_pnl': t_df['pnl'].mean(),
                'avg_ep': t_df['ep'].mean(),
                'target_hits': (t_df['exit']=='target').sum(),
                'hard_sl': (t_df['exit']=='hard_sl').sum()
            })
        print(f"  CRT {strike_type} tgt={int(tgt*100)}%: {len(trades)}t | "
              f"WR {pd.DataFrame(trades)['win'].mean()*100:.1f}% | "
              f"Rs.{pd.DataFrame(trades)['pnl'].sum():,.0f} | "
              f"Avg Rs.{pd.DataFrame(trades)['pnl'].mean():.0f} | {time.time()-t0:.0f}s")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: MRC unique blank days — re-run with different strike + target
# ─────────────────────────────────────────────────────────────────────────────
print("\n[MRC] Scanning unique blank days with ATM + target variations...")
mrc_days = sorted(mrc_unique['date'].unique())

for strike_type in STRIKES:
    for tgt in TARGETS:
        t0 = time.time()
        trades = []
        for dstr in mrc_days:
            row = mrc_unique[mrc_unique['date'] == dstr].iloc[0]
            entry_time = row['entry_time']
            signal     = row['signal']

            spot = load_spot_data(dstr, 'NIFTY')
            if spot is None: continue
            spot_at_entry = spot[spot['time'] >= entry_time[:8]]
            if spot_at_entry.empty: continue
            spot_ref = spot_at_entry.iloc[0]['price']

            expiries = list_expiry_dates(dstr, index_name='NIFTY')
            if not expiries: continue
            strike = get_strike(spot_ref, signal, use_atm=(strike_type=='atm'))
            instr  = f'NIFTY{expiries[0]}{strike}{signal}'

            res = simulate_sell(dstr, instr, entry_time, tgt_pct=tgt)
            if res:
                pnl, reason, ep, xp = res
                trades.append({'pnl': pnl, 'win': pnl > 0, 'exit': reason, 'ep': ep, 'signal': signal})

        if trades:
            t_df = pd.DataFrame(trades)
            results.append({
                'strategy': 'MRC', 'strike': strike_type, 'target': tgt,
                'trades': len(t_df), 'wr': t_df['win'].mean()*100,
                'total_pnl': t_df['pnl'].sum(), 'avg_pnl': t_df['pnl'].mean(),
                'avg_ep': t_df['ep'].mean(),
                'target_hits': (t_df['exit']=='target').sum(),
                'hard_sl': (t_df['exit']=='hard_sl').sum()
            })
        print(f"  MRC {strike_type} tgt={int(tgt*100)}%: {len(trades)}t | "
              f"WR {pd.DataFrame(trades)['win'].mean()*100:.1f}% | "
              f"Rs.{pd.DataFrame(trades)['pnl'].sum():,.0f} | "
              f"Avg Rs.{pd.DataFrame(trades)['pnl'].mean():.0f} | {time.time()-t0:.0f}s")

# ── Summary table ─────────────────────────────────────────────────────────────
df_r = pd.DataFrame(results)

print(f"\n{'='*75}")
print("  QUALITY IMPROVEMENT MATRIX")
print(f"{'='*75}")
print(f"  {'Strategy':<6} {'Strike':<6} {'TGT':>4} | {'Trades':>6} | {'WR':>6} | {'Total P&L':>12} | {'Avg/trade':>10} | {'Avg EP':>8}")
print(f"  {'-'*73}")
for _, r in df_r.sort_values(['strategy','strike','target']).iterrows():
    print(f"  {r['strategy']:<6} {r['strike']:<6} {int(r['target']*100):>3}% | "
          f"{r['trades']:>6} | {r['wr']:>5.1f}% | "
          f"Rs.{r['total_pnl']:>9,.0f} | Rs.{r['avg_pnl']:>7,.0f} | Rs.{r['avg_ep']:>5,.0f}")

# ── Best combo for each strategy ──────────────────────────────────────────────
print(f"\n{'='*75}")
print("  BEST COMBO PER STRATEGY (by avg P&L)")
print(f"{'='*75}")
for strat, g in df_r.groupby('strategy'):
    best = g.loc[g['avg_pnl'].idxmax()]
    baseline = g[(g['strike']=='otm1') & (g['target']==0.20)].iloc[0]
    print(f"\n  {strat} — best: {best['strike'].upper()} + {int(best['target']*100)}% target")
    print(f"    Baseline (OTM1 20%): {int(baseline['trades'])}t | {baseline['wr']:.1f}% WR | "
          f"Rs.{baseline['total_pnl']:,.0f} | Avg Rs.{baseline['avg_pnl']:.0f}")
    print(f"    Best     ({best['strike'].upper()} {int(best['target']*100)}%): {int(best['trades'])}t | {best['wr']:.1f}% WR | "
          f"Rs.{best['total_pnl']:,.0f} | Avg Rs.{best['avg_pnl']:.0f}")
    uplift = best['avg_pnl'] - baseline['avg_pnl']
    print(f"    Uplift avg: +Rs.{uplift:.0f}/trade")

# ── Combined total at best params ─────────────────────────────────────────────
print(f"\n{'='*75}")
print("  COMBINED TOTAL AT BEST PARAMS")
print(f"{'='*75}")
base_total = 1072048
for strike_type in STRIKES:
    for tgt in TARGETS:
        crt_r = df_r[(df_r['strategy']=='CRT') & (df_r['strike']==strike_type) & (df_r['target']==tgt)]
        mrc_r = df_r[(df_r['strategy']=='MRC') & (df_r['strike']==strike_type) & (df_r['target']==tgt)]
        if crt_r.empty or mrc_r.empty: continue
        combined = base_total + crt_r.iloc[0]['total_pnl'] + mrc_r.iloc[0]['total_pnl']
        print(f"  {strike_type.upper()} {int(tgt*100)}%: Base + CRT Rs.{crt_r.iloc[0]['total_pnl']:,.0f} "
              f"+ MRC Rs.{mrc_r.iloc[0]['total_pnl']:,.0f} = Total Rs.{combined:,.0f}")

df_r.to_csv(f'{OUT_DIR}/101_quality_matrix.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/101_quality_matrix.csv")
print("\nDone.")
