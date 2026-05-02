"""
102_final_equity.py — Final combined equity: Base + CRT(ATM 30%) + MRC(ATM 30%)
==================================================================================
Takes the best balanced params from quality improvement matrix:
  - Strike: ATM (vs OTM1 baseline)
  - Target: 30% (highest combined total P&L with acceptable WR)

Generates per-trade CSVs and final equity curve comparing:
  1. Baseline: Base + CRT(OTM1 20%) + MRC(OTM1 20%)
  2. Improved: Base + CRT(ATM 30%)  + MRC(ATM 30%)
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates
from plot_util import send_custom_chart

EOD_EXIT   = '15:20:00'
OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
TGT_PCT    = 0.30   # ATM 30% — best balanced

def r2(v): return round(float(v), 2)

def get_atm(spot):
    return int(round(spot / STRIKE_INT) * STRIKE_INT)

def simulate_sell(date_str, instrument, entry_time, tgt_pct=TGT_PCT):
    tks = load_tick_data(date_str, instrument, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - tgt_pct)); hsl = r2(ep * (1 + 1.00)); sl = hsl; md = 0.0
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

# ── Load signal dates ─────────────────────────────────────────────────────────
print("Loading signal dates...")
crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank = crt_raw[crt_raw['is_blank']==True].copy()
crt_blank['date'] = crt_blank['date'].astype(str)

mrc_raw   = pd.read_csv(f'{OUT_DIR}/100_mrc_trades.csv')
mrc_blank = mrc_raw[mrc_raw['is_blank']==True].copy()
mrc_blank['date'] = mrc_blank['date'].astype(str)
crt_dates  = set(crt_blank['date'])
mrc_unique = mrc_blank[~mrc_blank['date'].isin(crt_dates)].copy()

print(f"  CRT blank: {len(crt_blank)} days | MRC unique blank: {len(mrc_unique)} days")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: CRT blank days — ATM 30%
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[CRT] Scanning {len(crt_blank)} blank days with ATM + {int(TGT_PCT*100)}% target...")
t0 = time.time()
crt_trades = []
for dstr in sorted(crt_blank['date'].unique()):
    row = crt_blank[crt_blank['date'] == dstr].iloc[0]
    entry_time = row['entry_time']
    signal     = 'CE'

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue
    spot_ref_df = spot[spot['time'] >= entry_time[:8]]
    if spot_ref_df.empty: continue
    spot_ref = spot_ref_df.iloc[0]['price']

    expiries = list_expiry_dates(dstr, index_name='NIFTY')
    if not expiries: continue
    strike = get_atm(spot_ref)
    instr  = f'NIFTY{expiries[0]}{strike}{signal}'

    res = simulate_sell(dstr, instr, entry_time)
    if res:
        pnl, reason, ep, xp = res
        crt_trades.append({'date': dstr, 'signal': signal, 'pnl': pnl,
                           'win': pnl > 0, 'exit_reason': reason, 'ep': ep})

df_crt = pd.DataFrame(crt_trades)
wr = df_crt['win'].mean()*100
print(f"  {len(df_crt)}t | WR {wr:.1f}% | Rs.{df_crt['pnl'].sum():,.0f} | Avg Rs.{df_crt['pnl'].mean():.0f} | {time.time()-t0:.0f}s")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: MRC unique blank days — ATM 30%
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[MRC] Scanning {len(mrc_unique)} unique blank days with ATM + {int(TGT_PCT*100)}% target...")
t0 = time.time()
mrc_trades = []
for dstr in sorted(mrc_unique['date'].unique()):
    row = mrc_unique[mrc_unique['date'] == dstr].iloc[0]
    entry_time = row['entry_time']
    signal     = row['signal']

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue
    spot_ref_df = spot[spot['time'] >= entry_time[:8]]
    if spot_ref_df.empty: continue
    spot_ref = spot_ref_df.iloc[0]['price']

    expiries = list_expiry_dates(dstr, index_name='NIFTY')
    if not expiries: continue
    strike = get_atm(spot_ref)
    instr  = f'NIFTY{expiries[0]}{strike}{signal}'

    res = simulate_sell(dstr, instr, entry_time)
    if res:
        pnl, reason, ep, xp = res
        mrc_trades.append({'date': dstr, 'signal': signal, 'pnl': pnl,
                           'win': pnl > 0, 'exit_reason': reason, 'ep': ep})

df_mrc = pd.DataFrame(mrc_trades)
wr = df_mrc['win'].mean()*100
print(f"  {len(df_mrc)}t | WR {wr:.1f}% | Rs.{df_mrc['pnl'].sum():,.0f} | Avg Rs.{df_mrc['pnl'].mean():.0f} | {time.time()-t0:.0f}s")

# ── Save per-trade CSVs ────────────────────────────────────────────────────────
df_crt.to_csv(f'{OUT_DIR}/102_crt_atm30_trades.csv', index=False)
df_mrc.to_csv(f'{OUT_DIR}/102_mrc_atm30_trades.csv', index=False)

# ── Load base strategy ────────────────────────────────────────────────────────
base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
base_daily.columns = ['date', 'pnl']

# ── Build daily P&L timeline ──────────────────────────────────────────────────
def to_daily(df, label):
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'].astype(str), format='%Y%m%d')
    return df2.groupby('date')['pnl'].sum().reset_index().rename(columns={'pnl': label})

crt_daily = to_daily(df_crt, 'crt_pnl')
mrc_daily = to_daily(df_mrc, 'mrc_pnl')

# Baseline (OTM1 20%) from existing CSVs
crt_base = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_base = crt_base[crt_base['is_blank']==True].copy()
crt_base['date'] = pd.to_datetime(crt_base['date'].astype(str), format='%Y%m%d')
crt_base_daily = crt_base.groupby('date')['pnl_65'].sum().reset_index().rename(columns={'pnl_65': 'crt_base_pnl'})

mrc_base = pd.read_csv(f'{OUT_DIR}/100_mrc_trades.csv')
mrc_base = mrc_base[(mrc_base['is_blank']==True) & (~mrc_base['date'].isin([int(d) for d in crt_dates]))].copy()
mrc_base['date'] = pd.to_datetime(mrc_base['date'].astype(str), format='%Y%m%d')
mrc_base_daily = mrc_base.groupby('date')['pnl'].sum().reset_index().rename(columns={'pnl': 'mrc_base_pnl'})

all_dates = pd.DataFrame({'date': sorted(
    set(base_daily['date']) | set(crt_daily['date']) | set(mrc_daily['date']) |
    set(crt_base_daily['date']) | set(mrc_base_daily['date'])
)})

m = all_dates.copy()
for df_, col in [(base_daily, 'base_pnl'), (crt_daily, 'crt_pnl'), (mrc_daily, 'mrc_pnl'),
                 (crt_base_daily, 'crt_base_pnl'), (mrc_base_daily, 'mrc_base_pnl')]:
    src_col = df_.columns[1]  # second column after 'date'
    merged_col = src_col if src_col == col else src_col
    m = m.merge(df_.rename(columns={src_col: col}), on='date', how='left')
    m[col] = m[col].fillna(0)

m['improved_pnl'] = m['base_pnl'] + m['crt_pnl']     + m['mrc_pnl']
m['baseline_pnl'] = m['base_pnl'] + m['crt_base_pnl'] + m['mrc_base_pnl']

for col in ['base_pnl', 'crt_pnl', 'mrc_pnl', 'improved_pnl', 'baseline_pnl']:
    m[col+'_eq'] = m[col].cumsum()

# ── Stats ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FINAL COMBINED RESULTS")
print(f"{'='*70}")
print(f"  Base strategy:           Rs.{m['base_pnl'].sum():>10,.0f}  Avg/trade Rs.{base['pnl_conv'].mean():,.0f}")
print(f"  CRT blank (OTM1 20%):    Rs.{m['crt_base_pnl'].sum():>10,.0f}")
print(f"  MRC blank (OTM1 20%):    Rs.{m['mrc_base_pnl'].sum():>10,.0f}")
print(f"  BASELINE total:          Rs.{m['baseline_pnl'].sum():>10,.0f}")
print()
print(f"  CRT blank (ATM  30%):    Rs.{m['crt_pnl'].sum():>10,.0f}  "
      f"({len(df_crt)}t | {df_crt['win'].mean()*100:.1f}% WR | Avg Rs.{df_crt['pnl'].mean():.0f})")
print(f"  MRC blank (ATM  30%):    Rs.{m['mrc_pnl'].sum():>10,.0f}  "
      f"({len(df_mrc)}t | {df_mrc['win'].mean()*100:.1f}% WR | Avg Rs.{df_mrc['pnl'].mean():.0f})")
print(f"  IMPROVED total:          Rs.{m['improved_pnl'].sum():>10,.0f}")
uplift = m['improved_pnl'].sum() - m['baseline_pnl'].sum()
print(f"  Uplift vs baseline:     +Rs.{uplift:>10,.0f}")

base_dd = (m['base_pnl_eq']     - m['base_pnl_eq'].cummax()).min()
impr_dd = (m['improved_pnl_eq'] - m['improved_pnl_eq'].cummax()).min()
base_dd2= (m['baseline_pnl_eq'] - m['baseline_pnl_eq'].cummax()).min()
print(f"\n  Max drawdown:")
print(f"    Base alone:  Rs.{base_dd:,.0f}")
print(f"    Baseline:    Rs.{base_dd2:,.0f}")
print(f"    Improved:    Rs.{impr_dd:,.0f}")

# ── Build chart ────────────────────────────────────────────────────────────────
def eq_pts(series, dates):
    return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
            for d, v in zip(dates, series) if pd.notna(v)]

dates = m['date']

tv_json = {
    "isTvFormat": False,
    "candlestick": [],
    "volume": [],
    "lines": [
        {
            "id": "improved",
            "label": f"Improved — Base + CRT/MRC (ATM {int(TGT_PCT*100)}%)",
            "color": "#26a69a",
            "data": eq_pts(m['improved_pnl_eq'], dates),
            "seriesType": "line",
        },
        {
            "id": "baseline",
            "label": "Baseline — Base + CRT/MRC (OTM1 20%)",
            "color": "#0ea5e9",
            "data": eq_pts(m['baseline_pnl_eq'], dates),
            "seriesType": "line",
        },
        {
            "id": "base",
            "label": "Base Strategy Only",
            "color": "#9e9e9e",
            "data": eq_pts(m['base_pnl_eq'], dates),
            "seriesType": "line",
        },
        {
            "id": "improved_dd",
            "label": "Improved Drawdown",
            "color": "#ef5350",
            "data": eq_pts(m['improved_pnl_eq'] - m['improved_pnl_eq'].cummax(), dates),
            "seriesType": "baseline",
            "baseValue": 0,
            "isNewPane": True,
        },
    ]
}

send_custom_chart("102_final_equity", tv_json,
                  title=f"Final Equity — Base + CRT + MRC (ATM {int(TGT_PCT*100)}% target vs OTM1 20% baseline)")

print(f"\n  Saved → {OUT_DIR}/102_crt_atm30_trades.csv")
print(f"  Saved → {OUT_DIR}/102_mrc_atm30_trades.csv")
print("\nDone.")
